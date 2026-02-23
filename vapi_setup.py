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

    # Define the System Prompt with robust guardrails and exact conversation workflow
    system_prompt = (
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
        "5. CONNECTION STATUS & FORWARDING: State that based on the seriousness of the issue, medicines may be required and you will now connect/forward their details to the doctor. Use the `forward_to_doctor` tool to send their symptoms, your analysis, and the diet given to the doctor's system.\n"
        "6. CLOSING: Finally, ask 'What more can I do for you today?' in the language the patient is speaking.\n\n"
        "### STRICT GUARDRAILS:\n"
        "1. NO FINAL MEDICAL CONCLUSIONS: Say 'Based on your symptoms, it could be [condition], but the doctor will confirm.' Do not prescribe medicine.\n"
        "2. 8TH-GRADE READING LEVEL: Simplify all medical jargon.\n"
        "3. CULTURAL RELEVANCE: Use contextually appropriate Indian analogies for diet.\n"
        "4. ESCALATION: Transfer to human nurse immediately if severe symptoms are detected using `handoff_to_human`."
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
            },
            {
                "type": "server",
                "messages": [
                    {
                        "type": "request-start",
                        "content": "Sending the written diet plan and precautions to your phone...",
                    }
                ],
                "server": {
                    "url": f"{server_url}/send-written-plan"
                },
                "function": {
                    "name": "send_written_plan",
                    "description": "Simulates sending a written diet plan and precautions to the patient via SMS or WhatsApp if they request it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_name": {
                                "type": "string",
                                "description": "The patient's name."
                            },
                            "phone_number": {
                                "type": "string",
                                "description": "The patient's contact number."
                            },
                            "diet_plan": {
                                "type": "string",
                                "description": "The diet plan advised by the AI."
                            },
                            "precautions": {
                                "type": "string",
                                "description": "The precautions advised by the AI."
                            }
                        },
                        "required": ["patient_name", "phone_number", "diet_plan", "precautions"]
                    }
                }
            },
            {
                "type": "server",
                "messages": [
                    {
                        "type": "request-start",
                        "content": "Compiling your details and forwarding the report to the doctor now...",
                    }
                ],
                "server": {
                    "url": f"{server_url}/forward-to-doctor"
                },
                "function": {
                    "name": "forward_to_doctor",
                    "description": "Compiles the patient's symptoms, diagnosis, and given diet into a report and forwards it to the doctor.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_name": {
                                "type": "string",
                                "description": "The patient's name."
                            },
                            "regtn_no": {
                                "type": "string",
                                "description": "The patient's registration number or ID."
                            },
                            "symptoms_or_report_summary": {
                                "type": "string",
                                "description": "Summary of what symptoms the patient reported or what the AI found in the medical report."
                            },
                            "agent_analysis": {
                                "type": "string",
                                "description": "The AI's possible diagnosis or analysis of the situation."
                            },
                            "diet_precautions_given": {
                                "type": "string",
                                "description": "The exact diet plan and precautions explained to the patient during the call."
                            }
                        },
                        "required": ["patient_name", "regtn_no", "symptoms_or_report_summary", "agent_analysis", "diet_precautions_given"]
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
