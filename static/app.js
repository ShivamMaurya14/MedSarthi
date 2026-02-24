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

// Custom Voice Pipeline Logic (Sarvam TTS + Gemini LLM native STT using VAD)
let mediaRecorder;
let audioChunks = [];
let isCallActive = false;
let isRecording = false;

// VAD setup
let audioContext;
let analyser;
let micStream;
let isSpeaking = false;
let silenceStart = 0;
let vadAnimationFrame;
let currentAudioSource = null;
let callStartTime = 0;
let callTimerInterval = null;

function updateTimer(btnText) {
    if (!isCallActive) return;
    const elapsed = Math.floor((Date.now() - callStartTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    btnText.innerText = `End (${mins}:${secs})`;
}
function setupVapi() {
    const btn = document.getElementById("vapi-btn");
    const statusText = document.getElementById("call-status");
    const btnText = btn.querySelector('.btn-text');
    const icon = btn.querySelector('.mic-icon');

    btn.addEventListener("click", async () => {
        if (!isCallActive) {
            isCallActive = true;
            btn.classList.remove('active');
            btn.classList.add('connecting');
            btn.style.background = ""; // let css handle it
            btnText.innerText = "Connecting...";
            icon.innerText = "⏳";
            statusText.innerText = "Waking up AI Doctor...";
            try {
                // If the user selected reports but hasn't uploaded them, send them right before call starts
                if (window.pendingReportsToAI && window.pendingReportsToAI.length > 0) {
                    statusText.innerText = "Analyzing attached reports...";
                    const formData = new FormData();
                    for (let i = 0; i < window.pendingReportsToAI.length; i++) {
                        formData.append('files', window.pendingReportsToAI[i]);
                    }
                    try {
                        await fetch('/api/upload-report-for-ai', { method: 'POST', body: formData });
                        window.pendingReportsToAI = null; // clear after successful upload
                    } catch (e) {
                        console.error("Failed to upload pending report", e);
                    }
                }

                statusText.innerText = "Waking up AI Doctor...";
                const greetingRes = await fetch('/api/greeting');
                if (!greetingRes.ok) throw new Error("Greeting failed");
                const greetingData = await greetingRes.json();

                statusText.innerText = "AI is speaking...";

                // Connection established, switch from green to red, start timer
                btn.classList.remove('connecting');
                btn.classList.add('active');
                icon.innerText = "⏹";
                callStartTime = Date.now();
                callTimerInterval = setInterval(() => updateTimer(btnText), 1000);
                updateTimer(btnText);

                if (greetingData.audio_base64) {
                    playBase64Audio(greetingData.audio_base64, async () => {
                        if (!isCallActive) return;
                        await startRecordingSession(statusText);
                    });
                } else {
                    await startRecordingSession(statusText);
                }
            } catch (err) {
                console.error("Failed to greet:", err);
                if (isCallActive) await startRecordingSession(statusText);
            }
        } else {
            // Stop Call completely
            isCallActive = false;
            if (callTimerInterval) clearInterval(callTimerInterval);

            btn.classList.remove('connecting', 'active');
            btn.style.background = ""; // let css handle it
            btnText.innerText = "Connect to Doctor Assistant";
            icon.innerText = "🎤";
            statusText.innerText = "Call ended.";

            if (vadAnimationFrame) cancelAnimationFrame(vadAnimationFrame);
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
            }
            if (currentAudioSource) {
                try { currentAudioSource.stop(); } catch (e) { }
                currentAudioSource = null;
            }
        }
    });
}

async function startRecordingSession(statusText) {
    if (!isCallActive) return;

    try {
        if (!micStream || !micStream.active) {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }

        mediaRecorder = new MediaRecorder(micStream);

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstart = () => {
            isRecording = true;
            audioChunks = [];
            statusText.innerText = "Listening... (Just talk, AI will reply automatically)";
            startVAD(statusText);
        };

        mediaRecorder.onstop = async () => {
            isRecording = false;
            if (vadAnimationFrame) cancelAnimationFrame(vadAnimationFrame);

            if (!isCallActive) {
                if (micStream) micStream.getTracks().forEach(track => track.stop());
                return;
            }

            if (audioChunks.length === 0 || !isSpeaking) {
                // False alarm or noise
                if (isCallActive) setTimeout(() => startRecordingSession(statusText), 100);
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
                        if (isCallActive) startRecordingSession(statusText);
                    });
                } else {
                    if (isCallActive) startRecordingSession(statusText);
                }
            } catch (err) {
                console.error(err);
                statusText.innerText = "Error communicating with AI.";
                if (isCallActive) setTimeout(() => startRecordingSession(statusText), 3000);
            }
        };

        mediaRecorder.start();

    } catch (e) {
        console.error("Microphone access denied or error:", e);
        statusText.innerText = "Mic access blocked.";
    }
}

function startVAD(statusText) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    const source = audioContext.createMediaStreamSource(micStream);
    analyser = audioContext.createAnalyser();
    source.connect(analyser);

    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    isSpeaking = false;
    silenceStart = Date.now();

    function checkSilence() {
        if (!isCallActive || !isRecording) return;

        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
            sum += dataArray[i];
        }
        let avg = sum / bufferLength;

        if (avg > 15) {
            isSpeaking = true;
            silenceStart = Date.now();
            statusText.innerText = "Hearing you... (Keep talking)";
        } else {
            if (isSpeaking && (Date.now() - silenceStart > 1500)) {
                // 1.5 seconds of silence stops recording
                if (mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                }
                return;
            } else if (!isSpeaking && avg < 5) {
                statusText.innerText = "Listening... (Just talk, AI will reply automatically)";
            }
        }
        vadAnimationFrame = requestAnimationFrame(checkSilence);
    }

    checkSilence();
}

function playBase64Audio(base64str, onEndedCallback) {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    try {
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
            source.onended = () => {
                currentAudioSource = null;
                onEndedCallback();
            };
            currentAudioSource = source; // Store reference to interrupt it
            source.start(0);
        }, (err) => {
            console.error("Error decoding audio data", err);
            onEndedCallback();
        });
    } catch (e) {
        onEndedCallback();
    }
}
