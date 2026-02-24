import os
import sys
import logging
import io
import requests
import base64
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import tempfile
from openai import AsyncOpenAI
from typing import Optional, List

# ── Disease Prediction Model Paths ──────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global caches
diabetes_model, diabetes_scaler = None, None
heart_model, heart_scaler = None, None
xray_model = None
mri_model = None

def load_ml_models():
    global diabetes_model, diabetes_scaler, heart_model, heart_scaler
    try:
        with open(os.path.join(MODELS_DIR, 'diabetes_model.pkl'), 'rb') as f:
            diabetes_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'diabetes_scaler.pkl'), 'rb') as f:
            diabetes_scaler = pickle.load(f)
            
        with open(os.path.join(MODELS_DIR, 'heart_model.pkl'), 'rb') as f:
            heart_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'heart_scaler.pkl'), 'rb') as f:
            heart_scaler = pickle.load(f)
        logger.info("Structured ML models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")

# Keras 3 compatibility patches from app.py
class FixedFlatten(tf.keras.layers.Flatten):
    def call(self, inputs, *args, **kwargs):
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            if not hasattr(inputs, 'shape'): inputs = inputs[0]
        while isinstance(inputs, (list, tuple)) and len(inputs) == 1:
            inputs = inputs[0]
        return super().call(inputs, *args, **kwargs)

class FixedPooling(tf.keras.layers.GlobalAveragePooling2D):
    def call(self, inputs, *args, **kwargs):
        if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            if not hasattr(inputs, 'shape'): inputs = inputs[0]
        while isinstance(inputs, (list, tuple)) and len(inputs) == 1:
            inputs = inputs[0]
        return super().call(inputs, *args, **kwargs)

import tempfile
import gdown

def load_dl_models():
    global xray_model, mri_model
    custom_objects = {'Flatten': FixedFlatten, 'GlobalAveragePooling2D': FixedPooling}
    try:
        xray_path = os.path.join(MODELS_DIR, 'xrays_pneumonia.keras')
        if os.path.exists(xray_path):
            xray_model = tf.keras.models.load_model(xray_path, compile=False, custom_objects=custom_objects)
            logger.info("X-Ray model loaded.")
    except Exception as e:
        logger.error(f"X-Ray model failed: {e}")
        
    try:
        mri_path = os.path.join(MODELS_DIR, 'brain_tumor_model.keras')
        
        # Download from Google Drive if missing
        if not os.path.exists(mri_path):
            logger.info("Brain tumor model missing. Downloading from Google Drive...")
            model_id = "12oBWm5zYq7az62TPq7w68iFz5IOTygrG"
            url = f'https://drive.google.com/uc?id={model_id}'
            os.makedirs(MODELS_DIR, exist_ok=True)
            gdown.download(url, mri_path, quiet=False)
            logger.info("Brain tumor model downloaded successfully.")
            
        if os.path.exists(mri_path):
            try:
                mri_model = tf.keras.models.load_model(mri_path, compile=False, custom_objects=custom_objects)
                logger.info("MRI model loaded.")
            except Exception as e:
                logger.warning(f"Standard MRI model loading failed: {e}. Attempting rebuild...")
                # Fallback to reconstructing the architecture and loading weights
                base_model = tf.keras.applications.Xception(weights=None, include_top=False, input_shape=(299, 299, 3), pooling='max')
                mri_model = tf.keras.Sequential([
                    base_model,
                    FixedFlatten(),
                    tf.keras.layers.Dropout(rate=0.3),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(rate=0.25),
                    tf.keras.layers.Dense(4, activation='softmax')
                ])
                mri_model.load_weights(mri_path)
                logger.info("MRI model rebuilt and loaded from weights successfully.")
    except Exception as e:
        logger.error(f"MRI model failed: {e}")

load_ml_models()
load_dl_models()


# Basic OpenAI setup for chat memory
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    system_instruction = (
        "You are an empathetic, professional healthcare AI assistant built for an Indian regional hospital. "
        "Your default speaking language must be English. However, if the patient starts speaking in Hindi or another regional language, seamlessly switch to their language to make them comfortable.\n\n"
        "### CONVERSATION WORKFLOW (MANDATORY):\n"
        "You MUST follow these explicit steps in order during the conversation:\n"
        "1. GREETING & INITIAL QUESTION: Greet the user based on their language. Ask them how they would like to connect to the doctor today: via symptom diagnosis (telling you how they feel) OR via medical reports.\n"
        "2. DATA GATHERING:\n"
        "   - If Symptoms: Ask the patient ONE BY ONE the questions required to diagnose their condition. Do not ask a big list. Ask one question, wait for answer, then ask the next.\n"
        "   - If Reports: Ask them to upload/provide their report or tell you their patient ID so you can pull it using the `analyze_medical_report` tool.\n"
        "3. ANALYSIS & RECOMMENDATIONS: Once you have enough symptoms OR the medical report analysis, provide the user with the possible condition/diagnosis. Then, explain the diet plan and precautions verbally.\n"
        "4. OFFER WRITTEN PLAN: Ask the user if they want the diet plan and precautions sent to them in written format via SMS. If they say yes, use the `send_written_plan` tool.\n"
        "5. URGENCY & CONNECTION (CRITICAL): Assess the urgency of the patient's condition.\n"
        "   - IF URGENT/SERIOUS (severe symptoms, high pain, emergency): State that immediate attention is required. Use the `forward_to_doctor` tool to send the report, AND use the `book_appointment` tool to book an *earlier/urgent* appointment (specify 'Urgent' or 'Today' as the date) to see the doctor right away. (Optionally handoff to a human nurse).\n"
        "   - IF NON-URGENT (routine checking, mild symptoms, general advice): State that medicines may be required and you will forward their details. Use the `forward_to_doctor` tool to send the report, AND use the `book_appointment` tool to schedule a regular future visit with the doctor.\n"
        "6. CLOSING: Finally, ask 'What more can I do for you today?' in the language the patient is speaking.\n\n"
        "### STRICT GUARDRAILS:\n"
        "1. NO FINAL MEDICAL CONCLUSIONS: Say 'Based on your symptoms, it could be [condition], but the doctor will confirm.' Do not prescribe medicine.\n"
        "2. 8TH-GRADE READING LEVEL: Simplify all medical jargon.\n"
        "3. CULTURAL RELEVANCE: Use contextually appropriate Indian analogies for diet.\n"
        "4. ESCALATION: Transfer to human nurse/doctor immediately if severe symptoms are detected using `handoff_to_human`.\n\n"
        "Keep your responses extremely short and conversational, so it sounds great when spoken out loud via text-to-speech. "
        "Do not output markdown, bullet points, or special characters."
    )
    # Store simple conversation memory in a global list (for demo purposes)
    chat_messages = [{"role": "system", "content": system_instruction}]
    
else:
    openai_client = None
    chat_messages = []

global_appointments = []
global_triage_reports = []

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
    
    global_appointments.insert(0, {
        "patient_name": request.patient_name,
        "date": request.date,
        "time": request.time,
        "status": "Urgent" if "urgent" in request.date.lower() or "urgent" in request.time.lower() else "Scheduled"
    })

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
        
    global_triage_reports.insert(0, {
        "patient_name": request.patient_name,
        "regtn_no": request.regtn_no,
        "symptoms": request.symptoms_or_report_summary,
        "analysis": request.agent_analysis,
        "file_path": filepath
    })
        
    return {
        "status": "success",
        "message": f"Report securely forwarded to the doctor. Saved as {filename}.",
        "file_path": filepath
    }

@app.get("/api/doctor-dashboard-data")
async def get_doctor_dashboard_data():
    return {
        "appointments": global_appointments,
        "reports": global_triage_reports
    }

@app.get("/api/greeting")
async def get_greeting():
    """
    Returns the initial greeting audio via OpenAI TTS.
    """
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API Key is missing")
        
    logger.info("Generating greeting audio via OpenAI TTS...")
    
    try:
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input="Hello! I am MedSarthi, your AI assistant. How can I help you with your health today?"
        )
        # Read the raw bytes and convert to base64
        audio_data = response.read()
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        
        return {"audio_base64": audio_b64, "text": "Hello! I am MedSarthi, your AI assistant. How can I help you with your health today?"}
    except Exception as e:
        logger.error(f"OpenAI TTS Greeting Error: {e}")
        raise HTTPException(status_code=500, detail="Greeting TTS failed")

@app.post("/api/upload-report-for-ai")
async def upload_report_for_ai(files: List[UploadFile] = File(...) ):
    """
    Accepts multiple medical report files from the patient dashboard and injects simulated OCR/Analysis
    context directly into the AI's conversation memory so the AI can discuss them.
    """
    filenames = [f.filename for f in files]
    logger.info(f"Received file uploads: {filenames}")
    
    names_str = ", ".join(filenames)
    simulated_findings = (
        f"The user has just uploaded {len(files)} medical report(s) named: {names_str}. "
        "The report indicates: Fasting Blood Sugar is 150 mg/dL (High), "
        "LDL Cholesterol is 160 mg/dL (Elevated), and Blood Pressure is 135/85. "
        "Please acknowledge these reports and advise the patient based on these findings if they ask."
    )
    
    # Inject it into the AI's short-term memory
    chat_messages.append({
        "role": "system",
        "content": simulated_findings
    })
    
    return {"status": "success", "message": f"{len(files)} report(s) analyzed and added to AI context."}

@app.post("/api/voice-chat")
async def process_voice_chat(audio: UploadFile = File(...)):
    """
    Core Voice Endpoint: STT (OpenAI Whisper) -> LLM (OpenAI GPT-4o-mini) -> TTS (OpenAI)
    """
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API Key is missing")

    try:
        # Read the raw webm audio from the browser
        audio_data = await audio.read()
        
        # 1. STT via OpenAI Whisper
        logger.info("Sending Audio to OpenAI Whisper for STT...")
        
        # Whisper requires a file-like object with a filename (like .webm)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=True, mode="wb") as temp_audio:
            temp_audio.write(audio_data)
            temp_audio.flush()
            
            with open(temp_audio.name, 'rb') as f:
                transcription = await openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
                
        user_text = transcription.text.strip()
        logger.info(f"User Transcribed: {user_text}")
        
        if not user_text:
            return {"reply_text": "", "audio_base64": None}

        # 2. LLM response via OpenAI GPT-4o-mini
        logger.info("Sending transcribed text to OpenAI GPT-4o-mini...")
        chat_messages.append({"role": "user", "content": user_text})
        
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_messages,
            temperature=0.3
        )
        
        ai_reply = completion.choices[0].message.content.strip()
        chat_messages.append({"role": "assistant", "content": ai_reply})
        
        logger.info(f"AI Replied: {ai_reply}")

        # 3. TTS via OpenAI TTS
        logger.info("Generating audio via OpenAI TTS...")
        tts_response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=ai_reply
        )
        
        audio_bytes = tts_response.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        return {
            "reply_text": ai_reply,
            "audio_base64": audio_b64
        }
    except Exception as e:
        logger.error(f"Voice Chat Pipeline Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Diagnostics Endpoints ---
class DiabetesRequest(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    dpf: float
    age: int

@app.post("/api/diagnose/diabetes")
async def diagnose_diabetes(request: DiabetesRequest):
    if not diabetes_model or not diabetes_scaler:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    def categorize_bmi_val(bmi_val):
        if bmi_val < 18.5: return 0
        elif 18.5 <= bmi_val < 25: return 1
        elif 25 <= bmi_val < 30: return 2
        else: return 3
        
    bmi_cat = categorize_bmi_val(request.bmi)
    input_data = np.array([[request.pregnancies, request.glucose, request.blood_pressure, 
                            request.skin_thickness, request.insulin, request.bmi, request.dpf, request.age, bmi_cat]])
    
    input_scaled = diabetes_scaler.transform(input_data)
    prediction = int(diabetes_model.predict(input_scaled)[0])
    probability = float(diabetes_model.predict_proba(input_scaled)[0][1])
    
    return {"prediction": prediction, "probability": probability}

class HeartRequest(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/api/diagnose/heart")
async def diagnose_heart(request: HeartRequest):
    if not heart_model or not heart_scaler:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    input_data = np.array([[request.age, request.sex, request.cp, request.trestbps, request.chol, 
                            request.fbs, request.restecg, request.thalach, request.exang, 
                            request.oldpeak, request.slope, request.ca, request.thal]])
                            
    input_scaled = heart_scaler.transform(input_data)
    prediction = int(heart_model.predict(input_scaled)[0])
    probability = float(heart_model.predict_proba(input_scaled)[0][1])
    
    return {"prediction": prediction, "probability": probability}

@app.post("/api/diagnose/xray")
async def diagnose_xray(file: UploadFile = File(...)):
    if not xray_model:
        raise HTTPException(status_code=500, detail="X-Ray Model not loaded")
        
    try:
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB': img = img.convert('RGB')
        img = img.resize((224, 224))
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        prediction = xray_model.predict(img_array)
        probability = float(prediction[0][1] if len(prediction[0]) > 1 else prediction[0][0])
        
        return {"probability": probability, "prediction": 1 if probability > 0.5 else 0}
    except Exception as e:
        logger.error(f"X-Ray analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diagnose/mri")
async def diagnose_mri(file: UploadFile = File(...)):
    if not mri_model:
        raise HTTPException(status_code=500, detail="MRI Model not loaded")
        
    try:
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB': img = img.convert('RGB')
        img = img.resize((299, 299))
        
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        prediction = mri_model.predict(img_array)
        class_indices = {0: 'Tumor glioma', 1: 'Tumor meningioma', 2: 'No_tumor', 3: 'Tumor pituitary'}
        class_idx = int(np.argmax(prediction))
        predicted_class = class_indices.get(class_idx, "Unknown")
        confidence = float(np.max(prediction))
        
        return {"class": predicted_class, "confidence": confidence}
    except Exception as e:
        logger.error(f"MRI analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (Web Interface) AFTER all API endpoints
app.mount("/", StaticFiles(directory="static", html=True), name="static")

