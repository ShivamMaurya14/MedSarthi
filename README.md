<div align="center">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/Event-Codecure_SPIRIT_2026-blue" alt="Event" />
  <img src="https://img.shields.io/badge/Institution-IIT_(BHU)_Varanasi-orange" alt="Institution" />
  
  <h1>🏥 MedSarthi</h1>
  <p><strong>Transforming Healthcare Delivery & Health Literacy through Voice-Native AI</strong></p>
</div>

<br />

## 🏆 Submission for Codecure (SPIRIT 2026)

**MedSarthi** is proud to be built as a flagship entry for **Codecure**, part of **SPIRIT 2026** at the **IIT (BHU) Varanasi Annual Techno-Pharma Conference**. 

This project was engineered to answer Codecure's challenge: celebrating the union of innovation and tradition by building an AI-driven health-tech solution that fundamentally enhances healthcare delivery, patient engagement, and health literacy. Deeply inspired by the impactful legacy of past partners such as **Sun Pharma, Marichi Ventures, and A2Z4.0**, MedSarthi delivers a scalable, highly accessible tool optimized for widespread societal impact—bridging the gap between specialized personalized treatment and proactive, preventive care.

---

## 🌟 The Vision & Societal Impact

In a landscape where language barriers and medical illiteracy prevent millions from understanding basic diagnostics, traditional healthcare delivery models often struggle. **MedSarthi** acts as a bridge:
1. **Health Literacy:** By utilizing conversational, multilingual AI (fluent in local dialects via Sarvam AI), patients can ask questions about complex medical terms or upload reports and receive simple, spoken explanations.
2. **Preventive Care:** Our integrated machine learning diagnostics module proactively assesses patient risk for severe ailments, moving the paradigm from reactive to preventive.
3. **Optimized Healthcare Delivery:** By automatically parsing patient symptoms before they even reach a doctor, MedSarthi dramatically reduces the administrative triage burden on medical professionals. 

---

## 🚀 Core Features

### 🎙️ 1. Real-Time Multilingual Patient Assistant
* **Direct Browser Calling:** Patients simply press a button to open a live voice link with MedSarthi.
* **Indic Language Support:** Integration with **Sarvam AI** allows for natural text-to-speech interaction designed for the linguistic nuances of India.
* **Intelligent Reasoning:** Powered by **Google Gemini 2.0 Flash**, MedSarthi evaluates medical contexts, explains prescriptions clearly, and actively manages conversational cadence with zero-latency streaming.
* **Emergency Handoffs:** Built-in safeguards automatically transfer the SIP telephony connection to a live human triage nurse immediately if critical symptoms (like severe chest pain) are detected or requested.

### 👨‍⚕️ 2. Doctor’s Command Dashboard
* **Automated Triage Alerts:** View synthesized patient conversation transcripts, severity classifications, and immediate risks.
* **Streamlined Scheduling:** Clean, dynamic overview UI for managing complex daily appointments without navigating cluttered hospital ERPs.

### 🔬 3. Machine Learning Diagnostic Predictors
* **Diabetes & Heart Disease Neural Engines:** Instant risk evaluations based on standard localized metric inputs.
* **Deep Learning X-Ray Analysis:** Direct patient/doctor upload pipelines to provide a secondary AI layer over chest imaging.

---

## 🛠️ Technical Architecture

Our stack is carefully selected to prioritize speed, security, and accessibility over varying network conditions.

* **Frontend Engine:** Semantic HTML5, Vanilla JavaScript, CSS3 featuring optimized glassmorphism and real-time DOM manipulation. 
* **Backend API & Orchestration:** `FastAPI` (Python) serving lightning-fast asynchronous requests and WebRTC endpoints.
* **LLM Engine:** `Google Gemini 2.0 Flash` configured with custom conversational safety guardrails.
* **TTS/STT Pipeline:** `Sarvam AI` engineered for native high-fidelity Indic dialects.
* **Machine Learning Context:** `TensorFlow`, `Keras`, `Scikit-Learn`, relying on `NumPy` and `Pandas` for robust pipeline scaling.

---

## ⚙️ Local Installation & Testing

Because MedSarthi utilizes live audio streaming and sensitive ML architectures, we strongly advise installing via a dedicated virtual environment.

### 1. Repository Setup
```bash
git clone https://github.com/shivamMaurya14/MedSarthi.git
cd MedSarthi
python -m venv venv
```

### 2. Activate Environment
* **Mac/Linux:** `source venv/bin/activate`
* **Windows:** `venv\Scripts\activate`

### 3. Install Core Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Secure Environment Variables
Create a root `.env` file and insert necessary provider keys:
```env
# Primary Intelligence Layer
GEMINI_API_KEY="your_google_gemini_api_key"

# Localized Voice Layer
SARVAM_API_KEY="your_sarvam_api_key"

# Telephony Layer (Optional for live phone dial-in)
TWILIO_ACCOUNT_SID="..."
TWILIO_AUTH_TOKEN="..."
```

### 5. Launch the Server Stack
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000` to interact with the MedSarthi Patient/Doctor portals locally.

---

<div align="center">
  <i>"Redefining healthcare accessibility, one conversation at a time."</i><br>
  <b>Built for Codecure | SPIRIT 2026 | IIT (BHU) Varanasi</b>
</div>
