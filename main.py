import os
import logging
import io
import requests
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Basic Gemini setup for chat memory
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    genai.configure(api_key=GEMINI_API_KEY)
    
    system_instruction = (
        "You are an empathetic, professional healthcare AI assistant built for an Indian regional hospital. "
        "Your default speaking language must be Hindi. However, if the patient starts speaking in English or another regional language, seamlessly switch to their language to make them comfortable.\n\n"
        "### CONVERSATION WORKFLOW (MANDATORY):\n"
        "You MUST follow these explicit steps in order during the conversation:\n"
        "1. GREETING & INITIAL QUESTION: Greet the user based on their language. Ask them how they would like to connect to the doctor today: via symptom diagnosis (telling you how they feel) OR via medical reports.\n"
        "2. DATA GATHERING:\n"
        "   - If Symptoms: Ask the patient ONE BY ONE the questions required to diagnose their condition. Do not ask a big list. Ask one question, wait for answer, then ask the next.\n"
        "   - If Reports: Ask them to upload/provide their report or tell you their patient ID so you can pull it using the `analyze_medical_report` tool.\n"
        "3. ANALYSIS & RECOMMENDATIONS: Once you have enough symptoms OR the medical report analysis, provide the user with the possible condition/diagnosis. Then, explain the diet plan and precautions verbally.\n"
        "4. OFFER WRITTEN PLAN: Ask the user if they want the diet plan and precautions sent to them in written format via SMS. If they say yes, use the `send_written_plan` tool.\n"
        "5. URGENCY & CONNECTION (CRITICAL): Assess the urgency of the patient's condition.\n"
        "   - IF URGENT (severe symptoms, high pain, emergency): State that immediate attention is required. Use the `forward_to_doctor` tool to send the report, AND immediately use the `handoff_to_human` tool to connect the call directly to the doctor or triage nurse.\n"
        "   - IF NON-URGENT (routine checking, mild symptoms, general advice): State that medicines may be required and you will forward their details. Use the `forward_to_doctor` tool to send the report, AND use the `book_appointment` tool to schedule a future visit with the doctor.\n"
        "6. CLOSING: Finally, ask 'What more can I do for you today?' in the language the patient is speaking.\n\n"
        "### STRICT GUARDRAILS:\n"
        "1. NO FINAL MEDICAL CONCLUSIONS: Say 'Based on your symptoms, it could be [condition], but the doctor will confirm.' Do not prescribe medicine.\n"
        "2. 8TH-GRADE READING LEVEL: Simplify all medical jargon.\n"
        "3. CULTURAL RELEVANCE: Use contextually appropriate Indian analogies for diet.\n"
        "4. ESCALATION: Transfer to human nurse/doctor immediately if severe symptoms are detected using `handoff_to_human`.\n\n"
        "Keep your responses extremely short and conversational, so it sounds great when spoken out loud via text-to-speech. "
        "Do not output markdown, bullet points, or special characters."
    )
    llm_model = genai.GenerativeModel("gemini-2.0-flash", system_instruction=system_instruction)
    chat_session = llm_model.start_chat()
else:
    chat_session = None

app = FastAPI(
    title="Healthcare AI Voice Agent Server",
    description="Backend to handle Vapi.ai server actions like scheduling and call handoffs.",
    version="1.0.0"
)

class AppointmentRequest(BaseModel):
    patient_name: str
    date: str
    time: str

class HandoffRequest(BaseModel):
    reason: str
    patient_name: str

class MedicalReportRequest(BaseModel):
    patient_id: str

class DietPrecautionRequest(BaseModel):
    condition: str
    patient_name: str

class SendWrittenPlanRequest(BaseModel):
    patient_name: str
    phone_number: str
    diet_plan: str
    precautions: str

class ForwardToDoctorRequest(BaseModel):
    patient_name: str
    regtn_no: str
    symptoms_or_report_summary: str
    agent_analysis: str
    diet_precautions_given: str

@app.get("/patient-profile/{patient_id}")
async def get_patient_profile(patient_id: str):
    """
    Mock endpoint: fetches a patient's historical records and newly uploaded reports for the Web UI.
    """
    logger.info(f"Fetching profile for {patient_id}")
    return {
        "status": "success",
        "data": {
            "name": "Virat Sharma",
            "age": 45,
            "patient_id": patient_id,
            "history": ["Hypertension (Diagnosed 2021)"],
            "latest_report": "Lipid Profile from 15 Oct - High LDL (160 mg/dL)",
            "upcoming_appointment": None
        }
    }

@app.get("/api/config")
async def get_config():
    """
    Provides the frontend with public keys safely so users don't need to be prompted.
    Never expose secret API keys here, only public IDs.
    """
    return {
        "vapi_public_key": os.getenv("VAPI_PUBLIC_KEY", ""),
        "vapi_assistant_id": os.getenv("VAPI_ASSISTANT_ID", "")
    }

@app.post("/book-appointment")
async def book_appointment(request: AppointmentRequest):
    """
    Simulates booking a slot in an Electronic Health Record (EHR) system.
    """
    logger.info(f"Received booking request: {request.patient_name} on {request.date} at {request.time}")
    
    # In a real-world scenario, you would integrate with your EHR DB here.
    return {
        "status": "success",
        "message": f"Appointment successfully scheduled for {request.patient_name} on {request.date} at {request.time}.",
        "confirmation_code": "EHR-9876"
    }

@app.post("/handoff-to-human")
async def handoff_to_human(request: HandoffRequest):
    """
    Triggers when the AI detects severe symptoms or high frustration.
    Returns the structured JSON to Vapi or Twilio to transfer the call to a human triage nurse.
    """
    logger.info(f"Initiating human handoff for {request.patient_name} due to: {request.reason}")
    triage_nurse_phone = os.getenv("TRIAGE_NURSE_PHONE", "+919876543210")
    
    # Vapi allows you to override the assistant's behavior from a custom tool.
    # By returning this specific structure, Vapi will execute a call transfer (forwarding).
    return {
        "results": [
            {
                "toolCallId": "handoff-to-human-tool",
                "result": f"Successfully initiated handoff for {request.patient_name}.",
                "instructions": {
                    "forward": {
                        "destination": {
                            "type": "number",
                            "number": triage_nurse_phone
                        },
                        "message": "Transferring you to a specialized triage nurse now. Please hold."
                    }
                }
            }
        ]
    }

@app.post("/analyze-medical-report")
async def analyze_medical_report(request: MedicalReportRequest):
    """
    Simulates fetching and analyzing a patient's recent medical report.
    """
    logger.info(f"Fetching medical report analysis for patient ID: {request.patient_id}")
    
    # Mock data representing a pulled and parsed EHR report
    return {
        "status": "success",
        "data": {
            "patient_id": request.patient_id,
            "recent_test": "Complete Blood Count (CBC) and Lipid Profile",
            "date": "2023-10-15",
            "findings": "Elevated LDL cholesterol at 160 mg/dL. Hemoglobin is normal. Fasting blood sugar is slightly high at 110 mg/dL.",
            "doctor_notes": "Patient needs to reduce saturated fats and begin mild aerobic exercise. Monitor for pre-diabetes."
        }
    }

@app.post("/diet-and-precautions")
async def diet_and_precautions(request: DietPrecautionRequest):
    """
    Provides specific diet recommendations and precautions based on the condition.
    """
    logger.info(f"Generating diet & precautions for {request.patient_name} with condition: {request.condition}")
    
    # Simple mock logic based on condition
    condition = request.condition.lower()
    
    if "cholesterol" in condition or "heart" in condition:
        diet = "Avoid saturated fats (like ghee, butter, and fried foods). Eat fiber-rich foods like oats, fruits, and green vegetables."
        precautions = "Exercise 30 mins daily. Avoid smoking and alcohol."
    elif "sugar" in condition or "diabetes" in condition:
        diet = "Low glycemic index foods. Replace white rice with brown rice or minor millets. Avoid sweets and sugary drinks."
        precautions = "Check blood sugar regularly. Do not skip meals, especially breakfast."
    else:
        diet = "Standard balanced diet focusing on fresh local vegetables and whole grains."
        precautions = "Stay hydrated and get enough rest."
    
    return {
        "status": "success",
        "recommendations": {
            "diet_plan": diet,
            "precautions": precautions
        }
    }

@app.post("/send-written-plan")
async def send_written_plan(request: SendWrittenPlanRequest):
    """
    Simulates sending the diet plan and precautions to the patient via SMS/WhatsApp.
    """
    logger.info(f"Sending written plan to {request.patient_name} at {request.phone_number}")
    
    # In a real app, you would use Twilio SMS API or WhatsApp API here.
    return {
        "status": "success",
        "message": f"Written plan successfully sent to {request.patient_name}'s phone."
    }

@app.post("/forward-to-doctor")
async def forward_to_doctor(request: ForwardToDoctorRequest):
    """
    Saves the aggregated call summary as a text file to act as a report,
    simulating forwarding it to a doctor.
    """
    logger.info(f"Forwarding data to doctor for {request.patient_name} (Reg: {request.regtn_no})")
    
    reports_dir = "doctor_reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Clean filename
    safe_name = "".join(c for c in request.patient_name if c.isalnum() or c in " _-").strip().replace(" ", "_")
    safe_reg = "".join(c for c in request.regtn_no if c.isalnum() or c in " _-").strip()
    
    filename = f"{safe_name}_{safe_reg}.txt"
    filepath = os.path.join(reports_dir, filename)
    
    report_content = f"--- PATIENT TRIAGE REPORT ---\n"
    report_content += f"Patient Name: {request.patient_name}\n"
    report_content += f"Registration No: {request.regtn_no}\n"
    report_content += f"\n--- Symptoms / Report Summary ---\n{request.symptoms_or_report_summary}\n"
    report_content += f"\n--- AI Agent Analysis ---\n{request.agent_analysis}\n"
    report_content += f"\n--- Diet / Precautions Given ---\n{request.diet_precautions_given}\n"
    report_content += f"-----------------------------\n"
    
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(report_content)
        
    return {
        "status": "success",
        "message": f"Report securely forwarded to the doctor. Saved as {filename}.",
        "file_path": filepath
    }

@app.post("/api/voice-chat")
async def process_voice_chat(audio: UploadFile = File(...)):
    """
    Core Voice Endpoint: STT (Sarvam) -> LLM (Gemini) -> TTS (Sarvam)
    """
    sarvam_key = os.getenv("SARVAM_API_KEY")
    if not sarvam_key or sarvam_key == "your_sarvam_api_key_here":
        raise HTTPException(status_code=500, detail="Sarvam API Key is missing")
    if not chat_session:
        raise HTTPException(status_code=500, detail="Gemini API Key is missing")

    try:
        # Read the raw webm audio from the browser
        audio_data = await audio.read()
        mime_type = audio.content_type if audio.content_type else "audio/webm"
        
        # 1. & 2. STT & LLM integrated natively into Gemini 2.0 Flash
        logger.info("Sending Audio to Gemini...")
        prompt = "Listen to the user's voice message, answer it accurately and conversationally. Do not output markdown."
        gemini_payload = [
            {"mime_type": mime_type, "data": audio_data},
            prompt
        ]
        
        gemini_response = chat_session.send_message(gemini_payload)
        ai_reply = gemini_response.text.strip()
        logger.info(f"AI Replied: {ai_reply}")

        # 3. TTS via Sarvam Bulbul
        url_tts = "https://api.sarvam.ai/text-to-speech"
        headers_tts = {"api-subscription-key": sarvam_key, "Content-Type": "application/json"}
        tts_payload = {
            "inputs": [ai_reply],
            "target_language_code": "hi-IN",
            "speaker": "shreya",
            "pitch": 0,
            "pace": 1.1,
            "loudness": 1.5,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": "bulbul:v3"
        }
        
        logger.info("Generating audio via Sarvam TTS...")
        tts_res = requests.post(url_tts, headers=headers_tts, json=tts_payload)
        
        if not tts_res.ok:
            logger.error(f"Sarvam TTS Error: {tts_res.text}")
            raise HTTPException(status_code=500, detail="Text-to-Speech conversion failed")
            
        audio_b64 = tts_res.json().get("audios", [])[0]
        
        return {
            "reply_text": ai_reply,
            "audio_base64": audio_b64
        }
    except Exception as e:
        logger.error(f"Voice Chat Pipeline Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (Web Interface) AFTER all API endpoints
app.mount("/", StaticFiles(directory="static", html=True), name="static")

