# streamlit_app.py
# Streamlit driver monitoring app — live camera only, processes every frame, robust alarm playback per detection type

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import os
import pygame
import queue
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------------------
# SETTINGS / THRESHOLDS
# ---------------------------
EAR_THRESHOLD = 0.2       # Drowsiness (eye aspect ratio)
MAR_THRESHOLD = 0.6       # Yawning
HTR_THRESHOLD = 0.5       # Head tilt
EYES_CLOSED_SECONDS = 3.0
ALERT_COOLDOWN = 5.0      # seconds before next same-type alert
DEFAULT_ALARM = "alarm-106447.mp3"

# ---------------------------
# AUDIO INIT (async-safe)
# ---------------------------
ALARM_QUEUE = queue.Queue()

def alarm_worker():
    """Background thread to handle audio playback safely."""
    try:
        pygame.mixer.init()
        if os.path.exists(DEFAULT_ALARM):
            sound = pygame.mixer.Sound(DEFAULT_ALARM)
        else:
            sound = None
    except Exception:
        sound = None

    while True:
        try:
            _ = ALARM_QUEUE.get()
            if sound:
                sound.play()
            time.sleep(1)  # prevent overlapping playback
        except Exception:
            pass

# Start background alarm thread
threading.Thread(target=alarm_worker, daemon=True).start()

def play_alarm_nonblocking():
    """Enqueue an alarm playback event."""
    try:
        ALARM_QUEUE.put_nowait(True)
    except queue.Full:
        pass


# ---------------------------
# METRIC FUNCTIONS
# ---------------------------
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_EAR(eye):
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    C = euclidean_distance(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C > 0 else None


def compute_metrics_from_landmarks(landmarks, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]

    try:
        left_eye = [pts[i] for i in LEFT_EYE_INDICES]
        right_eye = [pts[i] for i in RIGHT_EYE_INDICES]
        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        ear = (left_ear + right_ear) / 2.0 if (left_ear and right_ear) else None
    except Exception:
        ear = None

    try:
        mouth = [pts[i] for i in MOUTH_INDICES]
        A = euclidean_distance(mouth[1], mouth[7])
        B = euclidean_distance(mouth[2], mouth[6])
        C = euclidean_distance(mouth[3], mouth[5])
        D = euclidean_distance(mouth[0], mouth[4])
        mar = (A + B + C) / (2.0 * D) if D > 0 else None
    except Exception:
        mar = None

    try:
        left_cheek = pts[234]
        right_cheek = pts[454]
        nose_tip = pts[1]
        face_center_x = (left_cheek[0] + right_cheek[0]) / 2
        face_width = abs(right_cheek[0] - left_cheek[0])
        htr = abs(nose_tip[0] - face_center_x) / (face_width + 1e-6)
    except Exception:
        htr = None

    return ear, mar, htr, pts


def classify_driver_state(ear, mar, htr):
    ear_alert = ear is not None and ear <= EAR_THRESHOLD
    mar_alert = mar is not None and mar >= MAR_THRESHOLD
    htr_alert = htr is not None and htr >= HTR_THRESHOLD
    if htr_alert and not (ear_alert or mar_alert):
        return "Distracted"
    elif ear_alert or mar_alert:
        return "Drowsy"
    else:
        return "Alert"


# ---------------------------
# VIDEO TRANSFORMER
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh


class FaceTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # runtime state
        self.closed_start = None
        self.last_alert_time = {
            "eyes": 0.0,
            "yawn": 0.0,
            "tilt": 0.0,
        }
        self.eyes_closed_events = 0
        self.yawns = 0
        self.tilts = 0
        self.total_alerts = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Downscale for speed
        small = cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1])))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        frame_display = small.copy()
        ear = mar = htr = None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ear, mar, htr, pts = compute_metrics_from_landmarks(lm, small.shape[1], small.shape[0])
            for (x, y) in pts:
                cv2.circle(frame_display, (x, y), 1, (0, 255, 0), -1)

        now = time.time()
        unsafe = False

        # ---------------------------
        # Eyes closed (drowsiness)
        # ---------------------------
        if ear is not None and ear <= EAR_THRESHOLD:
            # mark that eyes are closed and track duration
            if self.closed_start is None:
                self.closed_start = now

            duration_closed = now - self.closed_start

            if duration_closed >= EYES_CLOSED_SECONDS:
                if now - self.last_alert_time["eyes"] >= ALERT_COOLDOWN:
                    self.eyes_closed_events += 1
                    self.total_alerts += 1
                    self.last_alert_time["eyes"] = now
                    play_alarm_nonblocking()
                unsafe = True
        else:
            # tolerate brief landmark losses (<0.5 sec)
            if self.closed_start is not None and (now - self.closed_start) < 0.5:
                pass  # don't reset yet
            else:
                self.closed_start = None


        # Yawning
        if mar is not None and mar >= MAR_THRESHOLD:
            if now - self.last_alert_time["yawn"] >= ALERT_COOLDOWN:
                self.yawns += 1
                self.total_alerts += 1
                self.last_alert_time["yawn"] = now
                play_alarm_nonblocking()
            unsafe = True

        # Head tilt
        if htr is not None and htr >= HTR_THRESHOLD:
            if now - self.last_alert_time["tilt"] >= ALERT_COOLDOWN:
                self.tilts += 1
                self.total_alerts += 1
                self.last_alert_time["tilt"] = now
                play_alarm_nonblocking()
            unsafe = True

        # Classification & overlay
        state = classify_driver_state(ear, mar, htr)
        color = (0, 0, 255) if unsafe else (0, 255, 0)

        cv2.putText(frame_display, f"State: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame_display, f"EAR:{ear if ear else 0:.2f} MAR:{mar if mar else 0:.2f} HTR:{htr if htr else 0:.2f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        counter_txt = f"E:{self.eyes_closed_events} Y:{self.yawns} T:{self.tilts} A:{self.total_alerts}"
        cv2.putText(frame_display, counter_txt, (10, frame_display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Driver Monitor", layout="wide")
st.title("🚗 Driver Monitoring — Live Camera Only")

col1, col2 = st.columns([2, 1])
with col2:
    st.header("🎛 Controls")
    start_btn = st.button("▶️ Start Monitoring")
    stop_btn = st.button("⏹ Stop")

with col1:
    stframe = st.empty()

if "running" not in st.session_state:
    st.session_state.running = False
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# ---------------------------
# MAIN LOOP (WebRTC only)
# ---------------------------
if st.session_state.running:
    ctx = webrtc_streamer(
        key="driver-monitor",
        video_transformer_factory=FaceTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )

    if ctx:
        # Create persistent sidebar placeholders
        st.sidebar.markdown("### Live Counters")
        ph_eyes = st.sidebar.empty()
        ph_yawns = st.sidebar.empty()
        ph_tilts = st.sidebar.empty()
        ph_alerts = st.sidebar.empty()

        # Wait until the video transformer is ready
        transformer = None
        wait_start = time.time()
        while st.session_state.running and ctx.video_transformer is None:
            # short wait for transformer to be created by webrtc
            time.sleep(0.05)
            # prevent infinite wait (optional)
            if time.time() - wait_start > 10:
                break

        transformer = ctx.video_transformer

        if transformer is None:
            st.info("Waiting for transformer... please allow camera access.")
        else:
            # Poll transformer counters and update UI placeholders until user stops
            try:
                while st.session_state.running and ctx.state.playing:
                    # read attributes from transformer (thread-safe-ish for simple ints)
                    ph_eyes.text(f"Eyes Closed: {getattr(transformer, 'eyes_closed_events', 0)}")
                    ph_yawns.text(f"Yawns: {getattr(transformer, 'yawns', 0)}")
                    ph_tilts.text(f"Tilts: {getattr(transformer, 'tilts', 0)}")
                    ph_alerts.text(f"Alerts: {getattr(transformer, 'total_alerts', 0)}")
                    # small sleep to avoid hogging the main thread / UI
                    time.sleep(0.4)
            except Exception:
                # transformer disappeared or stream stopped
                pass

else:
    st.info("Press Start to begin monitoring (allow camera access).")
