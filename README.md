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

## 📺 Demo Video

<div align="center">
  <video width="100%" controls autoplay loop muted>
    <source src="https://raw.githubusercontent.com/ShivamMaurya14/MedSarthi/main/demo.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

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
GEMINI_API_KEY="your_key"
SARVAM_API_KEY="your_key"
```

### 3. Run
```bash
uvicorn main:app --reload
```
Visit `http://localhost:8000` to start.

---

<div align="center">
  <i>"Redefining healthcare accessibility."</i><br>
  <b>Built for Codecure | SPIRIT 2026 | IIT (BHU) Varanasi</b>
</div>
