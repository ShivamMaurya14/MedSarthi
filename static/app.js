// import Vapi removed

// Wait for DOM to load
document.addEventListener("DOMContentLoaded", async () => {
    // 1. Mock "Login" to fetch patient dat
    const patientId = "P-889012";
    await loadPatientData(patientId);

    // 2. Initialize Vapi AI Voice Button
    setupVapi();
});

async function loadPatientData(patientId) {
    try {
        const response = await fetch(`/patient-profile/${patientId}`);
        if (!response.ok) throw new Error("Network response was not ok");

        const resData = await response.json();
        const data = resData.data;

        // Populate fields
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

// Custom Voice Pipeline Logic (Sarvam TTS + Gemini LLM native STT)
let mediaRecorder;
let audioChunks = [];
let isCallActive = false;
let isRecording = false;

function setupVapi() {
    const btn = document.getElementById("vapi-btn");
    const statusText = document.getElementById("call-status");
    const btnText = btn.querySelector('.btn-text');
    const icon = btn.querySelector('.mic-icon');

    btn.addEventListener("click", async () => {
        if (!isRecording) {
            // Start recording
            btn.classList.add('active');
            btn.style.background = "#EF4444";
            btnText.innerText = "Stop & Send Audio";
            icon.innerText = "⏺";

            await startRecordingSession(statusText);
        } else {
            // Stop recording and send
            btn.classList.remove('active');
            btn.style.background = "linear-gradient(135deg, var(--secondary), #818CF8)";
            btnText.innerText = "Talk to AI Doctor";
            icon.innerText = "🎤";

            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
            }
        }
    });
}

async function startRecordingSession(statusText) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstart = () => {
            isRecording = true;
            audioChunks = [];
            statusText.innerText = "Listening... (Speak now, click button again to send)";
        };

        mediaRecorder.onstop = async () => {
            isRecording = false;

            // Cleanup the media stream tracks so the red recording dot goes away
            stream.getTracks().forEach(track => track.stop());

            if (audioChunks.length === 0) {
                statusText.innerText = "No audio recorded.";
                return;
            }

            statusText.innerText = "AI is thinking...";

            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append("audio", audioBlob, "voice_input.webm");

            try {
                const response = await fetch('/api/voice-chat', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error("Backend Error");

                const data = await response.json();
                statusText.innerText = "AI is speaking...";

                // Play audio
                if (data.audio_base64) {
                    playBase64Audio(data.audio_base64, () => {
                        statusText.innerText = "Call ended.";
                    });
                } else {
                    statusText.innerText = "Call ended.";
                }

            } catch (err) {
                console.error(err);
                statusText.innerText = "Error communicating with AI.";
            }
        };

        // Start recording
        mediaRecorder.start();

    } catch (e) {
        console.error("Microphone access denied or error:", e);
        statusText.innerText = "Mic access blocked.";
    }
}

function playBase64Audio(base64str, onEndedCallback) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();

    // Convert base64 to array buffer
    const binaryString = window.atob(base64str);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    audioContext.decodeAudioData(bytes.buffer, (buffer) => {
        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(audioContext.destination);
        source.onended = onEndedCallback;
        source.start(0);
    }, (err) => {
        console.error("Error decoding audio data", err);
        onEndedCallback();
    });
}
