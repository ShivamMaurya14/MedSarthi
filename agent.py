import logging
import os
import asyncio
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe
from livekit import rtc
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import google as lkgoogle, sarvam, silero
import aiohttp
from typing import Annotated
from livekit.agents import llm

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# Global VAD model to avoid slow per-job initialization
try:
    VAD_MODEL = silero.VAD.load()
    logger.info("Silero VAD loaded globally.")
except Exception as e:
    logger.error(f"Failed to load VAD model: {e}")
    VAD_MODEL = None

# The system prompt was adopted from the original MedSarthi
# that defines the healthcare assistant roles, workflow, and instructions.
SYSTEM_PROMPT = """You are an empathetic, professional healthcare AI assistant built for an Indian regional hospital. Your default speaking language must be English. However, if the patient starts speaking in Hindi or another regional language, seamlessly switch to their language to make them comfortable.

### CONVERSATION WORKFLOW (MANDATORY):
You MUST follow these explicit steps in order during the conversation:
1. GREETING & INITIAL QUESTION: Start by saying exactly: "Hello! I am MedSarthi, your AI assistant. How can I help you with your health today?" After saying this, ask them how they would like to connect to the doctor today: via symptom diagnosis (telling you how they feel) OR via medical reports.
2. DATA GATHERING:
   - If Symptoms: Ask the patient ONE BY ONE the questions required to diagnose their condition. Do not ask a big list. Ask one question, wait for answer, then ask the next.
   - If Reports: Ask them to tell you their patient ID or their report findings.
3. ANALYSIS & RECOMMENDATIONS: Once you have enough symptoms OR the medical report analysis, provide the user with the possible condition/diagnosis. Then, explain the diet plan and precautions verbally.
4. OFFER WRITTEN PLAN: Ask the user if they want the diet plan and precautions sent to them in written format via SMS.
5. URGENCY & CONNECTION (CRITICAL): Assess the urgency of the patient's condition.
   - IF URGENT (severe symptoms, high pain, emergency): State that immediate attention is required and inform them that you are connecting them directly to the doctor or triage nurse.
   - IF NON-URGENT (routine checking, mild symptoms, general advice): State that medicines may be required and you will schedule a future visit with the doctor.
6. if not urgent then CLOSING: Finally, ask 'What more can I do for you today?' in the language the patient is speaking.

### STRICT GUARDRAILS:
1. NO FINAL MEDICAL CONCLUSIONS: Say 'Based on your symptoms, it could be [condition], but the doctor will confirm.' Do not prescribe medicine.
2. 8TH-GRADE READING LEVEL: Simplify all medical jargon.
3. CULTURAL RELEVANCE: Use contextually appropriate Indian analogies for diet.
4. ESCALATION: Transfer to human nurse/doctor immediately if severe symptoms are detected.
"""

class AssistantFnc:
    def __init__(self):
        pass

    @llm.function_tool(description="Forward the patient's symptoms and agent diagnosis to the human doctor.")
    async def forward_to_doctor(
        self,
        patient_name: Annotated[str, "Name of the patient"],
        symptoms: Annotated[str, "Summary of the symptoms or medical reports"],
        analysis: Annotated[str, "Your professional medical AI analysis using your brain"],
        diet_precautions_given: Annotated[str, "The diet plan and precautions you provided"]
    ):
        logger.info(f"Forwarding {patient_name} info to doctor.")
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "patient_name": patient_name,
                    "regtn_no": "TEMP-001",
                    "symptoms_or_report_summary": symptoms,
                    "agent_analysis": analysis,
                    "diet_precautions_given": diet_precautions_given
                }
                async with session.post("http://127.0.0.1:8000/forward-to-doctor", json=payload) as r:
                    res = await r.json()
                    return res.get("message", "Sent successfully")
        except Exception as e:
            return f"Failed to forward: {e}"

    @llm.function_tool(description="Emergency handoff to a human triage nurse due to severe symptoms.")
    async def handoff_to_human(
        self,
        patient_name: Annotated[str, "Patient name"],
        reason: Annotated[str, "Reason for emergency handoff"]
    ):
        logger.info(f"Escalating {patient_name} to human triage nurse.")
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"patient_name": patient_name, "reason": reason}
                async with session.post("http://127.0.0.1:8000/handoff-to-human", json=payload) as r:
                    res = await r.json()
                    return "Handoff initiated successfully."
        except Exception as e:
            return f"Failed to handoff: {e}"

    @llm.function_tool(description="Book a subsequent routine appointment with the human doctor.")
    async def book_appointment(
        self,
        patient_name: Annotated[str, "Patient name"],
        date: Annotated[str, "Preferred appointment date (e.g. 'Next Monday')"],
        time: Annotated[str, "Preferred appointment time (e.g. '10:00 AM')"]
    ):
        logger.info(f"Booking appointment for {patient_name}...")
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"patient_name": patient_name, "date": date, "time": time}
                async with session.post("http://127.0.0.1:8000/book-appointment", json=payload) as r:
                    res = await r.json()
                    return res.get("message", "Appointment booked successfully")
        except Exception as e:
            return f"Failed to book appointment: {e}"

class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
        )
    
    async def on_enter(self):
        self.session.say("Hello! I am MedSarthi, your AI assistant. How can I help you with your health today? Would you like symptom diagnosis or to analyze medical reports?")

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        logger.info(f"AGENT DETECTED USER TURN COMPLETED: {new_message.content}")

async def entrypoint(ctx: JobContext):
    """Main entry point - LiveKit calls this when a user connects"""
    logger.info(f"User connected to room: {ctx.room.name}")
    
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    logger.info(f"Connected to room {ctx.room.name}.")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"--- AGENT SUBSCRIBED TO AUDIO TRACK from {participant.identity} ---")

    # Create and start the agent session
    session = AgentSession(
        vad=VAD_MODEL,
        turn_detection="vad",
        stt=sarvam.STT(
            language="en-IN",
            model="saarika:v2.5", 
            mode="transcribe",
            api_key=os.getenv("SARVAM_API_KEY"),
            high_vad_sensitivity=True,
            flush_signal=True
        ),
        llm=lkgoogle.LLM(model="gemini-2.0-flash"),
        tts=sarvam.TTS(
            target_language_code="en-IN",
            api_key=os.getenv("SARVAM_API_KEY"),
        ),
        tools=llm.find_function_tools(AssistantFnc()),
        min_endpointing_delay=0.8, # Increased for more natural two-way
    )
    
    @session.on("user_turn_started")
    def _on_user_turn_started():
        logger.info("USER TURN STARTED (Activity detected)")

    @session.on("user_speech_committed")
    def _on_user_speech_committed(msg: llm.ChatMessage):
        logger.info(f"STT (Committed): {msg.content}")

    @session.on("agent_speech_started")
    def _on_agent_speech_started():
        logger.info("Agent STARTED speaking.")

    @session.on("agent_speech_interrupted")
    def _on_agent_speech_interrupted():
        logger.info("Agent speech was INTERRUPTED.")

    @session.on("agent_speech_committed")
    def _on_agent_speech_committed(msg: llm.ChatMessage):
        logger.info(f"LLM Response (Committed): {msg.content}")

    @session.on("error")
    def _on_error(error):
        logger.error(f"AgentSession Error: {error}")

    def on_user_turn_completed(turn_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        logger.info(f"User Turn Completed. Final text: {new_message.content}")

    session.on("user_turn_completed", on_user_turn_completed)

    agent = VoiceAgent()
    logger.info("VoiceAgent instance created.")
    
    await session.start(
        agent=agent,
        room=ctx.room
    )
    logger.info("AgentSession started.")

    logger.info("Keep-alive loop started.")
    # Keep the entrypoint alive while the participant is in the room
    while ctx.room.isconnected:
        await asyncio.sleep(0.5)
    
    logger.info("Room disconnected. Job ending.")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="medsarthi-agent"))
