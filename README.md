<div align="center">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/Event-Codecure_SPIRIT_2026-blue" alt="Event" />
  <img src="https://img.shields.io/badge/Institution-IIT_(BHU)_Varanasi-orange" alt="Institution" />
  
  <h1>🏥 MedSarthi</h1>
  <p><strong>Next-Gen Multilingual Voice AI for Healthcare Accessibility</strong></p>
</div>

---

## 🏆 Submission for Codecure (SPIRIT 2026)

**MedSarthi** is a flagship entry for **Codecure (SPIRIT 2026, IIT BHU)**. We leverage AI to bridge the gap between tradition and innovation, focusing on patient engagement, health literacy, and preventive care. Inspired by industry leaders like *Sun Pharma*, we aim for Scalable Societal Impact.

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
*   **One-Tap Call:** Direct browser-based voice interaction.
*   **Indic Support:** Native-fidelity TTS via **Sarvam AI**.
*   **Gemini 2.0 Brain:** Real-time reasoning, prescription explaining, and medical guidance.
*   **Safety Handoff:** Auto-transfers to human triage for emergencies (e.g., chest pain).

### 👨‍⚕️ 2. Doctor Dashboard
*   **Instant Triage:** AI-synthesized transcripts and severity alerts.
*   **Unified View:** Clean appointment and patient vitals management.

### 🔬 3. Diagnostic Predictors
*   **ML Engines:** Instant risk assessment for Diabetes and Heart Disease.
*   **X-Ray Analysis:** Deep Learning pipeline for pneumonia/chest health.

---

## 🛠️ Tech Stack

*   **Logic:** FastAPI (Python), Google Gemini 2.0 Flash, Sarvam AI (Indic TTS).
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
  <i>"Redefining healthcare accessibility."</i><br>
  <b>Built for Codecure | SPIRIT 2026 | IIT (BHU) Varanasi</b>
</div>
