// MedSarthi – Direct Voice Pipeline (no LiveKit)
// Flow: Mic → record → POST /api/voice-chat → play TTS audio → repeat

document.addEventListener("DOMContentLoaded", async () => {
    const patientId = "P-889012";
    await loadPatientData(patientId);
    setupVoiceChat();
});

async function loadPatientData(patientId) {
    try {
        const response = await fetch(`/patient-profile/${patientId}`);
        if (!response.ok) throw new Error("Network response was not ok");
        const resData = await response.json();
        const data = resData.data;

        document.getElementById("welcome-name").innerText = data.name.split(" ")[0];
        document.getElementById("user-name").innerText = data.name;
        document.getElementById("user-avatar").innerText = data.name.substring(0, 2).toUpperCase();
        document.getElementById("user-id").innerText = `ID: ${data.patient_id}`;
        document.getElementById("user-age").innerText = `${data.age} yrs`;
        document.getElementById("user-history").innerText = data.history[0];
        document.getElementById("report-title").innerText = data.latest_report.split("-")[0].trim();
    } catch (error) {
        console.error("Error loading patient data:", error);
    }
}

// ─── Voice Chat State ───────────────────────────────────────────────
let isSessionActive = false;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let callStartTime = 0;
let callTimerInterval = null;

function updateTimer(btnText) {
    if (!isSessionActive) return;
    const elapsed = Math.floor((Date.now() - callStartTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    btnText.innerText = `End (${mins}:${secs})`;
}

function playBase64Audio(base64Audio) {
    return new Promise((resolve) => {
        if (!base64Audio) { resolve(); return; }
        const audio = new Audio(`data:audio/wav;base64,${base64Audio}`);
        audio.onended = resolve;
        audio.onerror = resolve;
        audio.play().catch(() => resolve());
    });
}

function setupVoiceChat() {
    const btn = document.getElementById("vapi-btn");
    const statusText = document.getElementById("call-status");
    const btnText = btn.querySelector('.btn-text');
    const icon = btn.querySelector('.mic-icon');

    btn.addEventListener("click", async () => {
        if (!isSessionActive) {
            // ─── START SESSION ───
            isSessionActive = true;
            btn.classList.add('connecting');
            btnText.innerText = "Starting...";
            icon.innerText = "⏳";
            statusText.innerText = "Waking up AI Doctor...";

            try {
                // Upload pending reports first
                if (window.pendingReportsToAI && window.pendingReportsToAI.length > 0) {
                    statusText.innerText = "Analyzing attached reports...";
                    const formData = new FormData();
                    for (let i = 0; i < window.pendingReportsToAI.length; i++) {
                        formData.append('files', window.pendingReportsToAI[i]);
                    }
                    await fetch('/api/upload-report-for-ai', { method: 'POST', body: formData }).catch(console.error);
                    window.pendingReportsToAI = null;
                }

                // Reset chat for fresh session
                await fetch('/api/reset-chat', { method: 'POST' });

                // Play greeting (agent speaks first)
                statusText.innerText = "AI Doctor is greeting you...";
                const greetRes = await fetch('/api/greeting');
                const greetData = await greetRes.json();
                
                btn.classList.remove('connecting');
                btn.classList.add('active');
                icon.innerText = "⏹";
                callStartTime = Date.now();
                callTimerInterval = setInterval(() => updateTimer(btnText), 1000);
                updateTimer(btnText);

                statusText.innerText = "AI is speaking...";
                await playBase64Audio(greetData.audio_base64);

                // Now start listening loop
                startListening(statusText);

            } catch (err) {
                console.error("Session start failed:", err);
                isSessionActive = false;
                btn.classList.remove('connecting', 'active');
                btnText.innerText = "Connect to Doctor Assistant";
                icon.innerText = "🎤";
                statusText.innerText = "Failed to start. Try again.";
            }
        } else {
            // ─── END SESSION ───
            stopSession(btn, btnText, icon, statusText);
        }
    });
}

async function startListening(statusText) {
    if (!isSessionActive) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            if (!isSessionActive) return;

            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioChunks = [];

            // Skip very short recordings (noise)
            if (audioBlob.size < 1000) {
                startListening(statusText);
                return;
            }

            statusText.innerText = "Processing your speech...";

            try {
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.webm');

                const res = await fetch('/api/voice-chat', { method: 'POST', body: formData });
                const data = await res.json();

                if (data.transcript) {
                    statusText.innerText = "AI is speaking...";
                    await playBase64Audio(data.audio_base64);
                }
            } catch (err) {
                console.error("Voice chat error:", err);
                statusText.innerText = "Error processing. Try speaking again.";
            }

            // Continue listening
            if (isSessionActive) startListening(statusText);
        };

        // Record for up to 8 seconds, then auto-stop
        mediaRecorder.start();
        isRecording = true;
        statusText.innerText = "🎤 Listening... (speak now, auto-stops in 8s)";

        // Auto-stop after 8 seconds of recording
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                isRecording = false;
                // Stop all mic tracks to release mic between turns
                stream.getTracks().forEach(t => t.stop());
            }
        }, 8000);

    } catch (err) {
        console.error("Mic access error:", err);
        statusText.innerText = "Microphone access denied. Please allow mic.";
    }
}

function stopSession(btn, btnText, icon, statusText) {
    isSessionActive = false;
    isRecording = false;
    if (callTimerInterval) clearInterval(callTimerInterval);

    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }

    btn.classList.remove('connecting', 'active');
    btnText.innerText = "Connect to Doctor Assistant";
    icon.innerText = "🎤";
    statusText.innerText = "Call ended.";
}
