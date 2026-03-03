import logging
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, sarvam

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# The system prompt was adopted from the original MedSarthi vapi_setup.py
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
6. CLOSING: Finally, ask 'What more can I do for you today?' in the language the patient is speaking.

### STRICT GUARDRAILS:
1. NO FINAL MEDICAL CONCLUSIONS: Say 'Based on your symptoms, it could be [condition], but the doctor will confirm.' Do not prescribe medicine.
2. 8TH-GRADE READING LEVEL: Simplify all medical jargon.
3. CULTURAL RELEVANCE: Use contextually appropriate Indian analogies for diet.
4. ESCALATION: Transfer to human nurse/doctor immediately if severe symptoms are detected.
"""

class VoiceAgent(Agent):
    def __init__(self) -> None:
        # Note: If you want to use function calling (like book_appointment),
        # you will need to add an fnc_ctx parameter with those tools defined here.
        super().__init__(
            instructions=SYSTEM_PROMPT,
            
            # Saaras v3 STT - Auto-detects the language or uses specifics
            stt=sarvam.STT(
                language="unknown",
                model="saaras:v3",
                mode="transcribe",
                flush_signal=True
            ),
            
            # OpenAI LLM
            llm=openai.LLM(model="gpt-4o"),
            
            # Bulbul TTS - Converts text to speech using an Indian voice
            tts=sarvam.TTS(
                target_language_code="en-IN",
                model="bulbul:v3",
                speaker="shubh"  # Female options: priya, simran, ishita, kavya | Male: aditya, anand, rohan, shubh
            ),
        )
    
    async def on_enter(self):
        """Called when user joins - agent starts the conversation"""
        logger.info("Agent entered the room. Initiating greeting sequence.")
        # We explicitly provide the greeting message to avoid an initial LLM call here,
        # which ensures the greeting is always delivered without API quota errors.
        self.session.say("Hello! I am MedSarthi, your AI assistant. How can I help you with your health today? Would you like symptom diagnosis or to analyze medical reports?")

async def entrypoint(ctx: JobContext):
    """Main entry point - LiveKit calls this when a user connects"""
    logger.info(f"User connected to room: {ctx.room.name}")
    
    # Create and start the agent session following Sarvam plugin best practices
    session = AgentSession(
        turn_detection="stt",
        min_endpointing_delay=0.07
    )
    await session.start(
        agent=VoiceAgent(),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
