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

// Setup for LiveKit Voice Agent Connection
let room = null;
let isCallActive = false;
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
            btnText.innerText = "Connecting...";
            icon.innerText = "⏳";
            statusText.innerText = "Waking up AI Doctor...";

            try {
                // 1. Upload reports if the patient selected any before calling
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

                statusText.innerText = "Connecting to LiveKit Room...";
                // 2. Fetch LiveKit Token
                const res = await fetch('/api/livekit-token');
                if (!res.ok) throw new Error("Could not fetch LiveKit token from backend");
                const { token, url } = await res.json();

                // 3. Initialize LiveKit Room and Subscriptions
                room = new window.LivekitClient.Room();

                room.on(window.LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
                    // Play Agent Audio automatically when it's attached!
                    if (track.kind === window.LivekitClient.Track.Kind.Audio) {
                        const audioEl = track.attach();
                        document.body.appendChild(audioEl);
                    }
                });

                room.on(window.LivekitClient.RoomEvent.ActiveSpeakersChanged, (speakers) => {
                    // Update status for speaking
                    if (speakers.length > 0) {
                        const isLocalspeaking = speakers.some(s => s === room.localParticipant);
                        if (isLocalspeaking) {
                            statusText.innerText = "Hearing you... (Keep talking)";
                        } else {
                            statusText.innerText = "AI is speaking...";
                        }
                    } else {
                        statusText.innerText = "Listening... (Just talk, AI will reply automatically)";
                    }
                });

                // 4. Connect to Room
                await room.connect(url, token);

                // 5. Enable Local Microphone so Agent can hear us
                await room.localParticipant.setMicrophoneEnabled(true);

                // Visual Updates
                btn.classList.remove('connecting');
                btn.classList.add('active');
                icon.innerText = "⏹";
                statusText.innerText = "Connected! Start speaking.";

                callStartTime = Date.now();
                callTimerInterval = setInterval(() => updateTimer(btnText), 1000);
                updateTimer(btnText);

            } catch (err) {
                console.error("Failed to connect via LiveKit:", err);
                isCallActive = false;
                btn.classList.remove('connecting');
                btnText.innerText = "Connect to Doctor Assistant";
                icon.innerText = "🎤";
                statusText.innerText = "Connection failed. Please try again.";
            }
        } else {
            // Stop Call completely
            isCallActive = false;
            if (callTimerInterval) clearInterval(callTimerInterval);

            // Disconnect from LiveKit gracefully
            if (room) {
                await room.disconnect();
                room = null;
            }

            btn.classList.remove('connecting', 'active');
            btnText.innerText = "Connect to Doctor Assistant";
            icon.innerText = "🎤";
            statusText.innerText = "Call ended.";
        }
    });
}
