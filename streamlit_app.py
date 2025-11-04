import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import os
import pygame
from PIL import Image
import io
import tempfile

# ---------------------------
# SETTINGS
# ---------------------------
EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 0.6
HTR_THRESHOLD = 0.5
EYES_CLOSED_SECONDS = 3.0
ALERT_COOLDOWN = 5.0
DEFAULT_ALARM = "alarm-106447.mp3"

# ---------------------------
# INIT MEDIAPIPE + AUDIO
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

try:
    pygame.mixer.init()
    if os.path.exists(DEFAULT_ALARM):
        alarm_sound = pygame.mixer.Sound(DEFAULT_ALARM)
    else:
        alarm_sound = None
except Exception:
    alarm_sound = None


# ---------------------------
# FUNCTIONS
# ---------------------------
def play_alarm_nonblocking():
    if alarm_sound:
        threading.Thread(target=alarm_sound.play, daemon=True).start()


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_EAR(eye):
    A = euclidean_distance(eye[1], eye[5])
    B = euclidean_distance(eye[2], eye[4])
    C = euclidean_distance(eye[0], eye[3])
    if C == 0:
        return None
    return (A + B) / (2.0 * C)


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
        ear = (left_ear + right_ear) / 2.0
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
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Driver Monitor", layout="wide")
st.title("🚗 Driver Monitoring Dashboard")
st.markdown("**Real-time monitoring of drowsiness (eyes), yawning (mouth), and head tilt (distraction).**")

col1, col2 = st.columns([2, 1])

with col2:
    st.header("🎛 Controls")
    source_option = st.selectbox("Video Source", ("Streamlit Camera (Live)", "Upload Video File"), index=0)
    video_file = None
    if source_option == "Upload Video File":
        video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    process_every = st.slider("Process every Nth frame", 1, 20, 10, 1)
    start_btn = st.button("▶️ Start Monitoring")
    stop_btn = st.button("⏹ Stop Monitoring")

with col1:
    stframe = st.empty()
    status_text = st.empty()
    big_alert = st.empty()

if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.last_alert_time = 0.0
    st.session_state.closed_start = None
    st.session_state.eyes_closed_events = 0
    st.session_state.yawns = 0
    st.session_state.tilts = 0
    st.session_state.total_alerts = 0

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False


# ---------------------------
# FRAME PROCESSING FUNCTION
# ---------------------------
def process_frame(frame, now):
    small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ear = mar = htr = None
    frame_display = small.copy()

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        ear, mar, htr, pts = compute_metrics_from_landmarks(lm, small.shape[1], small.shape[0])
        for (x, y) in pts:
            cv2.circle(frame_display, (x, y), 1, (0, 255, 0), -1)

    unsafe = False
    unsafe_reasons = []

    # Eyes closed
    if ear is not None and ear <= EAR_THRESHOLD:
        if st.session_state.closed_start is None:
            st.session_state.closed_start = now
        if now - st.session_state.closed_start >= EYES_CLOSED_SECONDS:
            if now - st.session_state.last_alert_time >= ALERT_COOLDOWN:
                st.session_state.eyes_closed_events += 1
                st.session_state.total_alerts += 1
                st.session_state.last_alert_time = now
                play_alarm_nonblocking()
            unsafe = True
            unsafe_reasons.append("Eyes closed")
    else:
        st.session_state.closed_start = None

    # Yawning
    if mar is not None and mar >= MAR_THRESHOLD:
        if now - st.session_state.last_alert_time >= ALERT_COOLDOWN:
            st.session_state.yawns += 1
            st.session_state.total_alerts += 1
            st.session_state.last_alert_time = now
            play_alarm_nonblocking()
        unsafe = True
        unsafe_reasons.append("Yawning")

    # Head tilt
    if htr is not None and htr >= HTR_THRESHOLD:
        if now - st.session_state.last_alert_time >= ALERT_COOLDOWN:
            st.session_state.tilts += 1
            st.session_state.total_alerts += 1
            st.session_state.last_alert_time = now
            play_alarm_nonblocking()
        unsafe = True
        unsafe_reasons.append("Head tilt")

    # Classification
    state = classify_driver_state(ear, mar, htr)

    if unsafe:
        reason_str = ", ".join(unsafe_reasons)
        status_text.markdown(f"<h3 style='color:red'>🚨 DANGEROUS — {state}: {reason_str}</h3>", unsafe_allow_html=True)
        big_alert.markdown(f"<div style='background-color:#ff4d4d;padding:10px;border-radius:8px'><h3 style='color:white'>DANGER: {state}</h3></div>", unsafe_allow_html=True)
    else:
        if state == "Alert":
            status_text.markdown("<h3 style='color:green'>Status: ALERT</h3>", unsafe_allow_html=True)
        elif state == "Distracted":
            status_text.markdown("<h3 style='color:orange'>Status: DISTRACTED</h3>", unsafe_allow_html=True)
        else:
            status_text.markdown("<h3 style='color:yellow'>Status: DROWSY</h3>", unsafe_allow_html=True)
        big_alert.empty()

    stframe.image(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB), channels="RGB")


# ---------------------------
# MAIN LOOP
# ---------------------------
if st.session_state.running:
    frame_counter = 0

    if source_option == "Upload Video File" and video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)

        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1
            if frame_counter % process_every != 0:
                continue
            process_frame(frame, time.time())
            time.sleep(0.01)
        cap.release()

    elif source_option == "Streamlit Camera (Live)":
        camera_feed = st.camera_input("Live Camera Feed", key=f"camera_{int(time.time())}")
        if camera_feed is not None:
            image = Image.open(io.BytesIO(camera_feed.getvalue()))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frame_counter += 1
            if frame_counter % process_every == 0:
                process_frame(frame, time.time())
        # 🔁 Refresh automatically every 1s for live streaming effect
        time.sleep(1)
        st.experimental_rerun()

else:
    st.info("Press **Start Monitoring** to begin (allow camera access).")
