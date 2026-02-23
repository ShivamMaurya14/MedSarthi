import os
import logging
from fastapi import FastAPI, HTTPException
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
