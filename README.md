<div align="center">
  <h1>🏥 MedSarthi</h1>
  <p><strong>AI-Powered Multilingual Voice Health Assistant with ML Diagnostics</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Google_Gemini-8E75B2?logo=google&logoColor=white" alt="Gemini">
    <img src="https://img.shields.io/badge/Sarvam_AI-Indic_Voice-orange" alt="Sarvam">
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  </p>
</div>

---

## 🚨 The Problem

India's healthcare system is burdened by **language barriers, overcrowded clinics, and inaccessible medical jargon**. Patients — especially in rural and semi-urban areas — face:

- **Language exclusion:** Most digital health tools operate only in English.
- **Long wait times:** Overcrowded OPDs mean hours of waiting just for an initial triage.
- **Report confusion:** Patients cannot interpret their own lab reports or know what precautions to follow.
- **Delayed escalation:** Urgent cases often go undetected until it's too late.

---

## 💡 Our Solution — MedSarthi

**MedSarthi** is a full-stack AI healthcare platform that puts a **multilingual voice doctor assistant** in every patient's pocket — accessible via a single browser tap.

### What can a patient do?

| Feature | Description |
|---------|-------------|
| 🗣️ **Voice Conversation** | Talk to an AI doctor in **English, Hindi, or regional languages** — just like a real consultation. |
| 🩺 **Symptom Diagnosis** | The AI asks questions one by one, analyzes symptoms, and provides a preliminary assessment. |
| 📄 **Report Analysis** | Upload medical reports — the AI reads, interprets, and explains results in simple language. |
| 🥗 **Diet & Precautions** | Receive personalized, culturally relevant diet plans and lifestyle advice. |
| 🚑 **Smart Escalation** | AI assesses urgency — forwards critical cases to the doctor instantly, books routine appointments otherwise. |
| 👨‍⚕️ **Doctor Dashboard** | Doctors receive AI-generated triage reports, appointment schedules, and patient summaries. |

---

## 🖼️ Application Gallery

<div align="center">
  <table style="width: 100%; border-collapse: collapse;">
    <tr>
      <td align="center" style="padding: 10px;">
        <strong>Patient Dashboard</strong><br>
        <img src="doc_assets/patient_dashboard.png" width="100%" alt="Patient Dashboard">
      </td>
      <td align="center" style="padding: 10px;">
        <strong>Doctor Dashboard</strong><br>
        <img src="doc_assets/doctor_dashboard.png" width="100%" alt="Doctor Dashboard">
      </td>
    </tr>
    <tr>
      <td align="center" style="padding: 10px;">
        <strong>Diagnostics Page</strong><br>
        <img src="doc_assets/diagnostics_page.png" width="100%" alt="Diagnostics Page">
      </td>
      <td align="center" style="padding: 10px;">
        <strong>Login Portal</strong><br>
        <img src="doc_assets/login_page.png" width="100%" alt="Login Portal">
      </td>
    </tr>
  </table>
</div>

### 📺 Video Demonstration
[![Watch the Demo](https://img.shields.io/badge/Watch-Demo_Video-red?style=for-the-badge&logo=youtube)](doc_assets/dashboard_demo.mp4)

---

## 🏗️ Architecture & How It Works

```
Patient (Browser)
    │
    ├── 🎤 Records Voice (MediaRecorder API)
    │
    ▼
FastAPI Backend (main.py)
    │
    ├──► Sarvam AI STT ──► Transcript (Speech → Text)
    │
    ├──► Google Gemini 2.0 Pro ──► AI Response (LLM with conversation memory)
    │
    ├──► Sarvam AI TTS ──► Audio (Text → Speech)
    │
    └──► Response: { transcript, reply, audio_base64 }
              │
              ▼
        Patient hears AI doctor speak back
```

**Key flow:**
1. Patient taps "Connect to Doctor Assistant" — the AI **greets first** via TTS.
2. Patient speaks — browser records audio and POSTs to `/api/voice-chat`.
3. Backend pipeline: **Sarvam STT → Gemini LLM → Sarvam TTS** — returns audio response.
4. Conversation loops until the patient ends the session.
5. If urgent, the AI forwards the triage report to the doctor dashboard and books an appointment.

---

## 🚀 Key Features

### 🎙️ Multilingual Voice AI Assistant
- **Agent-First Greeting:** The AI initiates the conversation — no awkward silence.
- **Direct Voice Pipeline:** Browser mic → Sarvam STT → Gemini → Sarvam TTS → audio playback. No WebRTC or third-party call infra needed.
- **Indic Language Support:** Native STT/TTS for Hindi, English, Tamil, and more via **Sarvam AI**.
- **Conversational Memory:** Gemini maintains full chat history across the session.
- **Safety Guardrails:** Never prescribes medicine. Escalates to human doctor for emergencies.

### 👨‍⚕️ Doctor Dashboard
- AI-generated triage reports forwarded in real-time.
- Appointment management with urgency tagging.
- Patient vitals and history at a glance.

### 🔬 ML/DL Diagnostic Predictors
| Model | Task | Architecture |
|-------|------|-------------|
| Diabetes Risk | Blood sugar, BMI, insulin analysis | Scikit-Learn (Random Forest) |
| Heart Disease | 13-parameter cardiac risk scoring | Scikit-Learn (Random Forest) |
| Chest X-Ray | Pneumonia detection from X-ray images | TensorFlow/Keras CNN (224×224) |
| Brain MRI | 4-class tumor classification | Xception Transfer Learning (299×299) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Uvicorn, Python 3.10+ |
| **LLM** | Google Gemini 2.0 Pro (`google-genai` SDK) |
| **Voice (STT)** | Sarvam AI — `saarika:v2` |
| **Voice (TTS)** | Sarvam AI — `bulbul:v2`, Speaker: `anushka` |
| **ML/DL** | TensorFlow, Keras, Scikit-Learn, NumPy, Pandas |
| **Frontend** | Vanilla JS, CSS3 (Glassmorphism), HTML5 |
| **Deployment** | Single-process FastAPI server (no external infra) |

---

## ⚙️ Installation & Setup

### 1. Clone & Install
```bash
git clone https://github.com/shivamMaurya14/MedSarthi.git
cd MedSarthi
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables (`.env`)
```env
GOOGLE_API_KEY=your_google_gemini_api_key
SARVAM_API_KEY=your_sarvam_ai_api_key
```

### 3. Run
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser. That's it — **single command, single process**.

---

## 📂 Project Structure

```
MedSarthi/
├── main.py                  # FastAPI backend — voice pipeline, diagnostics, dashboards
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not committed)
├── models/
│   ├── diabetes_model.pkl   # Diabetes prediction model
│   ├── diabetes_scaler.pkl  # Feature scaler
│   ├── heart_model.pkl      # Heart disease model
│   ├── heart_scaler.pkl     # Feature scaler
│   ├── xrays_pneumonia.keras # Chest X-ray CNN
│   └── brain_tumor_model.keras # Brain MRI Xception
├── static/
│   ├── voice-app.js         # Frontend voice pipeline logic
│   ├── patient-dashboard.html
│   ├── doctor-dashboard.html
│   ├── diagnostics.html
│   ├── index.html           # Login page
│   ├── login.js
│   └── style.css
└── doc_assets/              # Screenshots & demo video
```

---

## 🔑 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/greeting` | AI greeting via Sarvam TTS (agent speaks first) |
| `POST` | `/api/voice-chat` | Core pipeline: audio → STT → Gemini → TTS → audio |
| `POST` | `/api/reset-chat` | Reset conversation memory |
| `POST` | `/api/upload-report-for-ai` | Upload medical reports to AI context |
| `POST` | `/api/diagnose/diabetes` | Diabetes risk prediction |
| `POST` | `/api/diagnose/heart` | Heart disease risk prediction |
| `POST` | `/api/diagnose/xray` | Chest X-ray pneumonia detection |
| `POST` | `/api/diagnose/mri` | Brain MRI tumor classification |
| `GET` | `/api/doctor-dashboard-data` | Doctor dashboard data feed |
| `POST` | `/forward-to-doctor` | Forward triage report to doctor |
| `POST` | `/book-appointment` | Book patient appointment |

---

## 🌍 Impact & Scalability

- **Zero infrastructure:** Runs on a single server — no WebRTC, no call providers, no vendor lock-in.
- **Language-first:** Built ground-up for Indic languages, not English-first with translation.
- **Offline-ready ML:** Diagnostic models run locally — no cloud dependency for predictions.
- **Extensible:** Plug in new Sarvam languages, swap Gemini for any LLM, add new diagnostic models.

---

## 👥 Team

Built with ❤️ for **accessible healthcare in India**.

---

<div align="center">
  <i>"Redefining healthcare accessibility — one conversation at a time."</i>
</div>
