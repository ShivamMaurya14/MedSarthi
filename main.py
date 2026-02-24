import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

# Mount static files (Web Interface) AFTER all API endpoints
app.mount("/", StaticFiles(directory="static", html=True), name="static")

