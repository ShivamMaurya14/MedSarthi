import os
import requests
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("SARVAM_API_KEY")
if not key:
    print("NO KEY")
    exit()

url = "https://api.sarvam.ai/text-to-speech"
headers = {"api-subscription-key": key, "content-type": "application/json"}
payload = {
    "inputs": ["नमस्ते, मेरा नाम एआई डॉक्टर है। मैं आपकी क्या मदद कर सकता हूँ?"],
    "target_language_code": "hi-IN",
    "speaker": "meera",
    "pitch": 0,
    "pace": 1.1,
    "loudness": 1.5,
    "speech_sample_rate": 8000,
    "enable_preprocessing": True,
    "model": "bulbul:v1"
}
res = requests.post(url, headers=headers, json=payload)
if res.ok:
    print("TTS SUCCESS")
    audios = res.json().get('audios', [])
    print(f"Got {len(audios)} audios")
else:
    print("TTS FAILED", res.text)
