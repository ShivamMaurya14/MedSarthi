<div align="center">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status" />
  
  <h1>🏥 MedSarthi</h1>
  <p><strong>Scalable AI Medical Infrastructure for Triage & Diagnostics</strong></p>
</div>

---

## 🚀 About the Project

**MedSarthi** is an intelligent medical infrastructure platform designed to bridge the gap between patient needs and efficient healthcare delivery. With an advanced multilingual voice agent, it is capable of diagnosing symptoms across various regional languages, providing comprehensive diet plans and taking necessary precautions, all while accurately assessing the urgency of cases to forward reports and book appointments efficiently.

---

## 🌟 Vision & Impact

*   **Health Literacy:** Multilingual voice agent (Hindi/English/Local) explains medical terms simply.
*   **Preventive Care:** Intelligent ML modules categorize risks before ailments escalate.
*   **Delivery Optimization:** Automated triage reduces administrative load on doctors.

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

## 🚀 Key Features

### 🎙️ 1. Multilingual Voice Assistant
*   **One-Tap Call:** Direct browser-based voice interaction powered by **LiveKit WebRTC**.
*   **Indic Support:** Native-fidelity Speech-to-Text (STT) & Text-to-Speech (TTS) via **Sarvam AI**.
*   **OpenAI GPT-4o Brain:** Real-time reasoning, prescription explaining, and medical guidance.
*   **Safety Handoff:** Auto-transfers to human triage for emergencies (e.g., chest pain).

### 👨‍⚕️ 2. Doctor Dashboard
*   **Instant Triage:** AI-synthesized transcripts and severity alerts.
*   **Unified View:** Clean appointment and patient vitals management.

### 🔬 3. Diagnostic Predictors
*   **ML Engines:** Instant risk assessment for Diabetes and Heart Disease.
*   **X-Ray Analysis:** Deep Learning pipeline for pneumonia/chest health.

---

## 🛠️ Tech Stack

*   **Logic:** FastAPI (Python), LiveKit (WebRTC), OpenAI GPT-4o, Sarvam AI (Indic STT & TTS).
*   **ML/DL:** TensorFlow, Keras, Scikit-Learn.
*   **Frontend:** Vanilla JS, CSS3 (Glassmorphism), HTML5.

---

## ⚙️ Installation

### 1. Setup
```bash
git clone https://github.com/shivamMaurya14/MedSarthi.git
cd MedSarthi
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment (`.env`)
```env
LIVEKIT_URL=wss://your-project-url.livekit.cloud
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret
SARVAM_API_KEY=your_sarvam_key
OPENAI_API_KEY=your_openai_key
```

### 3. Run

You will need to run both the FastAPI backend and the LiveKit agent process simultaneously.

**Terminal 1 (Backend Web Server):**
```bash
uvicorn main:app --reload
```
Visit `http://localhost:8000` to access the dashboards.

**Terminal 2 (LiveKit Voice Agent):**
```bash
python agent.py dev
```

---

<div align="center">
  <i>"Redefining healthcare accessibility with scalable AI."</i><br>
</div>
