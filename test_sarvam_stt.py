import os
import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("SARVAM_API_KEY")

url = "https://api.sarvam.ai/speech-to-text-translate"
payload = {'model': 'saaras:v3'}
# Make a fake wav file
with open("test.wav", "wb") as f:
    f.write(b"fake audio data")

files=[
  ('file',('test.wav',open('test.wav','rb'),'audio/wav'))
]
headers = {
  'api-subscription-key': key
}
response = requests.request("POST", url, headers=headers, data=payload, files=files)
print(response.text)
