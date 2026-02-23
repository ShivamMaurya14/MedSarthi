import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

VAPI_API_KEY = os.getenv("VAPI_API_KEY")
VAPI_BASE_URL = "https://api.vapi.ai"

# Ensure API Key is available
if not VAPI_API_KEY or "your_vapi_api_key_here" in VAPI_API_KEY:
    print("WARNING: Please set a valid VAPI_API_KEY in your .env file.")

headers = {
    "Authorization": f"Bearer {VAPI_API_KEY}",
    "Content-Type": "application/json"
}

def create_healthcare_agent():
    # Use your Server URL where FastAPI is running (ngrok, Railway, Render, etc.)
    server_url = os.getenv("SERVER_URL", "https://your-ngrok-url.ngrok.app")
    
    url = f"{VAPI_BASE_URL}/assistant"

    # Define the System Prompt with robust guardrails
    system_prompt = (
        "You are an empathetic, professional healthcare AI assistant built for an Indian regional hospital. "
        "Your primary duties are to answer inbound patient calls, converse politely, note down symptoms, "
        "schedule routine medical checks, analyze recent medical reports if requested, provide basic diet and precaution recommendations securely, "
        "and identify if an immediate human triage nurse is needed.\n\n"
        "STRICT GUARDRAILS & TONE:\n"
        "1. NO MEDICAL HALLUCINATIONS: Under no circumstances should you diagnose conditions or prescribe medications. Do not offer final medical conclusions.\n"
        "2. 8TH-GRADE READING LEVEL: Simplify all medical jargon so a middle-schooler or a non-medical family member can easily understand it. Avoid academic medical terms.\n"
        "3. CULTURAL RELEVANCE: When giving general lifestyle recommendations or explaining general wellness, use contextually appropriate Indian analogies (e.g., recommend home-cooked meals like dal-chawal, kichdi instead of foreign diets).\n"
        "4. ESCALATION TO HUMAN: If the patient displays severe symptoms (like chest pain, heavy bleeding, difficulty breathing, slurred speech) or sounds highly frustrated, immediately use the handoff_to_human tool to transfer to a real nurse.\n"
        "5. LANGUAGE & ADAPTATION: Your default speaking language must be Hindi. However, if the patient starts speaking in English or another regional language, seamlessly switch to their language to make them comfortable.\n"
        "6. STRUCTURED MEDICAL REPLIES: When a user asks about their medical report or diet, always fetch the data using the tools, process the information yourself, and reply strictly in this exact order:\n"
        "   - First: Provide a simple, easy-to-understand summary of the medical report analysis.\n"
        "   - Second: List the mandatory precautions based on their condition.\n"
        "   - Third: Recommend the specific dietary plan."
    )

    # Construct the JSON payload required by Vapi
    payload = {
        "name": "Healthcare Triage AI Agent",
        "model": {
            "provider": "google",
            "model": "gemini-2.0-flash",
            "temperature": 0.2, # Lower temperature to prevent hallucinating medical facts
            "messages": [
                {
                    "content": system_prompt,
                    "role": "system"
                }
            ]
        },
        "voice": {
            "provider": "sarvam",
            "voiceId": "shreya" # Indian AI voice (alternative: 'shubh')
        },
        "transcriber": {
            "provider": "sarvam",
            "model": "saaras:v3",
            "language": "hi-IN" # Hindi as default, handles multilingual input well
        },
        "firstMessage": "नमस्ते! मैं अस्पताल का एआई असिस्टेंट हूँ। आज मैं आपके स्वास्थ्य में कैसे मदद कर सकता हूँ?",
        "tools": [
            {
                "type": "server",
                "messages": [
                    {
                        "type": "request-start",
                        "content": "Checking the Electronic Health Records for a slot...",
                    }
                ],
                "server": {
                    "url": f"{server_url}/book-appointment"
                },
                "function": {
                    "name": "book_appointment",
                    "description": "Books a medical appointment in the EHR system when a patient requests to see a doctor.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_name": {
                                "type": "string",
                                "description": "The full name of the patient calling."
                            },
                            "date": {
                                "type": "string",
                                "description": "The requested date for the appointment in YYYY-MM-DD format."
                            },
                            "time": {
                                "type": "string",
                                "description": "The requested time for the appointment in HH:MM format."
                            }
                        },
                        "required": ["patient_name", "date", "time"]
                    }
                }
            },
            {
                "type": "server",
                "messages": [
                    {
                        "type": "request-start",
                        "content": "Medical emergency detected or patient requested human. Forwarding the call to our triage nurse right away.",
                    }
                ],
                "server": {
                    "url": f"{server_url}/handoff-to-human"
                },
                "function": {
                    "name": "handoff_to_human",
                    "description": "Immediately transfers the call to a human triage nurse. Triggers on severe medical symptoms like chest pain, or patient extreme frustration.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "A short summary of why the call is being handed off to a human."
                            },
                            "patient_name": {
                                "type": "string",
                                "description": "The name of the patient."
                            }
                        },
                        "required": ["reason", "patient_name"]
                    }
                }
            },
            {
                "type": "server",
                "messages": [
                    {
                        "type": "request-start",
                        "content": "Pulling your latest medical report from the system now...",
                    }
                ],
                "server": {
                    "url": f"{server_url}/analyze-medical-report"
                },
                "function": {
                    "name": "analyze_medical_report",
                    "description": "Retrieves and summarizes a patient's recent medical report. Use this strictly when a patient asks to check or explain their recent report/bloods/scans.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "The patient ID or full name if ID is not known."
                            }
                        },
                        "required": ["patient_id"]
                    }
                }
            },
            {
                "type": "server",
                "messages": [
                    {
                        "type": "request-start",
                        "content": "Looking up recommended diet and precautions for your condition...",
                    }
                ],
                "server": {
                    "url": f"{server_url}/diet-and-precautions"
                },
                "function": {
                    "name": "diet_and_precautions",
                    "description": "Gets personalized diet recommendations and precautions for a specific condition. Use this when a user asks what to eat or what to avoid.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "condition": {
                                "type": "string",
                                "description": "The medical condition the patient is asking about (e.g., diabetes, high cholesterol, fever)."
                            },
                            "patient_name": {
                                "type": "string",
                                "description": "The patient's name."
                            }
                        },
                        "required": ["condition", "patient_name"]
                    }
                }
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Raise error on bad HTTP status
        
        agent_data = response.json()
        print("✅ Agent Created Successfully!")
        print("Agent ID:", agent_data.get("id"))
        print("\nSave this Agent ID to use in your frontend or telephony mapping.")
    except requests.exceptions.HTTPError as err:
        print(f"❌ HTTP Error creating assistant: {err}")
        print("Details:", response.text)
    except Exception as e:
        print(f"❌ Network or generic error: {e}")

if __name__ == "__main__":
    create_healthcare_agent()
