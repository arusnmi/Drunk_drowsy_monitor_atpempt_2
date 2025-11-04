# streamlit_app.py
# Real-time driver monitoring (EAR, MAR, HTR) with audible + visual alerts using Streamlit + Mediapipe + OpenCV + pygame

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import os
import pygame

# ---------------------------
# SETTINGS / THRESHOLDS
# ---------------------------
EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 0.6
HTR_THRESHOLD = 0.5

EYES_CLOSED_SECONDS = 3.0
ALERT_COOLDOWN = 5.0

DEFAULT_ALARM = "alarm-106447.mp3"

# ---------------------------
# Initialize mediapipe and pygame
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
    AUDIO_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Audio unavailable: {e}")
    AUDIO_AVAILABLE = False
    alarm_sound = None

# ---------------------------
# Utility functions
# ---------------------------
def play_alarm_nonblocking():
    if alarm_sound:
        try:
            alarm_sound.play()
        except Exception:
            pass

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
    try:
        left_eye = [pts[i] for i in LEFT_EYE_INDICES]
        right_eye = [pts[i] for i in RIGHT_EYE_INDICES]
        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)
        ear = (left_ear + right_ear) / 2.0 if (left_ear and right_ear) else None
    except Exception:
        ear = None

    MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]
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
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Driver Monitor", layout="wide")
st.title("Driver Monitoring Dashboard")
st.markdown("Real-time detection of drowsiness (eyes), yawning (mouth) and head tilt (distraction).")

col1, col2 = st.columns([2, 1])

with col2:
    st.header("Controls & Settings")
    source_option = st.selectbox("Video source", ("Webcam (0)", "Upload video file"), index=0)
    if source_option == "Upload video file":
        video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    process_every = st.slider("Process every Nth frame", 1, 20, 10, 1)
    start_btn = st.button("Start Monitoring")
    stop_btn = st.button("Stop")
    st.markdown("---")
    st.subheader("Thresholds (current)")
    st.write(f"Eye aspect ratio ≤ **{EAR_THRESHOLD:.3f}** → closed")
    st.write(f"Mouth aspect ratio ≥ **{MAR_THRESHOLD:.3f}** → yawning")
    st.write(f"Head tilt ratio ≥ **{HTR_THRESHOLD:.3f}** → head tilt")
    st.markdown("---")
    st.subheader("Event counters (session)")
    eyes_closed_count = st.empty()
    yawns_count = st.empty()
    tilt_count = st.empty()
    total_alerts = st.empty()

with col1:
    stframe = st.empty()
    status_text = st.empty()
    big_alert = st.empty()

if "running" not in st.session_state:
    st.session_state.running = False
    st.session_state.eyes_closed_events = 0
    st.session_state.yawns = 0
    st.session_state.tilts = 0
    st.session_state.total_alerts = 0
    st.session_state.closed_start = None
    st.session_state.last_alert_time = 0.0

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# ---------------------------
# Monitoring loop
# ---------------------------
def monitor_loop():
    if source_option == "Webcam (0)":
        cap = cv2.VideoCapture(0)
    else:
        if video_file is None:
            st.warning("Upload a video file to use this source.")
            return
        tfile = f"temp_upload_{int(time.time())}.mp4"
        with open(tfile, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(tfile)

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    eyes_closed_frame_threshold = max(1, int(EYES_CLOSED_SECONDS * fps))

    frame_count = 0

    try:
        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # ✅ Only process every Nth frame for optimization
            if frame_count % process_every != 0:
                # Optionally show unprocessed frame
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                continue

            small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            frame_display = small.copy()
            ear = mar = htr = None

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                ear, mar, htr, pts = compute_metrics_from_landmarks(lm, small.shape[1], small.shape[0])
                for (x, y) in pts:
                    cv2.circle(frame_display, (x, y), 1, (0, 255, 0), -1)

            now = time.time()
            unsafe = False
            unsafe_reasons = []

            if ear is not None and ear <= EAR_THRESHOLD:
                if st.session_state.closed_start is None:
                    st.session_state.closed_start = now
                duration = now - st.session_state.closed_start
                if duration >= EYES_CLOSED_SECONDS:
                    if now - st.session_state.last_alert_time >= ALERT_COOLDOWN:
                        st.session_state.eyes_closed_events += 1
                        st.session_state.total_alerts += 1
                        st.session_state.last_alert_time = now
                        threading.Thread(target=play_alarm_nonblocking, daemon=True).start()
                    unsafe = True
                    unsafe_reasons.append("Eyes closed")
            else:
                st.session_state.closed_start = None

            if mar is not None and mar >= MAR_THRESHOLD:
                if now - st.session_state.last_alert_time >= ALERT_COOLDOWN:
                    st.session_state.yawns += 1
                    st.session_state.total_alerts += 1
                    st.session_state.last_alert_time = now
                    threading.Thread(target=play_alarm_nonblocking, daemon=True).start()
                unsafe = True
                unsafe_reasons.append("Yawning")

            if htr is not None and htr >= HTR_THRESHOLD:
                if now - st.session_state.last_alert_time >= ALERT_COOLDOWN:
                    st.session_state.tilts += 1
                    st.session_state.total_alerts += 1
                    st.session_state.last_alert_time = now
                    threading.Thread(target=play_alarm_nonblocking, daemon=True).start()
                unsafe = True
                unsafe_reasons.append("Head tilt")

            state = classify_driver_state(ear, mar, htr)

            if unsafe:
                cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1]-1, frame_display.shape[0]-1), (0, 0, 255), 8)
                reason_str = ", ".join(unsafe_reasons)
                status_text.markdown(f"<h2 style='color:red'>DANGEROUS — {state}: {reason_str}</h2>", unsafe_allow_html=True)
                big_alert.markdown(f"<div style='background-color:#ff4d4d;padding:10px;border-radius:6px'><h2 style='color:white'>DANGER: {state}: {reason_str}</h2></div>", unsafe_allow_html=True)
            else:
                if state == "Alert":
                    status_text.markdown("<h2 style='color:green'>Status: ALERT</h2>", unsafe_allow_html=True)
                elif state == "Distracted":
                    status_text.markdown("<h2 style='color:orange'>Status: DISTRACTED</h2>", unsafe_allow_html=True)
                else:
                    status_text.markdown("<h2 style='color:yellow'>Status: DROWSY</h2>", unsafe_allow_html=True)
                big_alert.empty()

            info_lines = [
                f"EAR: {ear:.3f}" if ear is not None else "EAR: -",
                f"MAR: {mar:.3f}" if mar is not None else "MAR: -",
                f"HTR: {htr:.3f}" if htr is not None else "HTR: -"
            ]
            for i, ln in enumerate(info_lines):
                cv2.putText(frame_display, ln, (10, 60 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            stframe.image(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB), channels="RGB")

            eyes_closed_count.write(f"Eyes-closed events: {st.session_state.eyes_closed_events}")
            yawns_count.write(f"Yawns detected: {st.session_state.yawns}")
            tilt_count.write(f"Head tilt events: {st.session_state.tilts}")
            total_alerts.write(f"Total alerts: {st.session_state.total_alerts}")

            time.sleep(0.01)

    finally:
        try:
            cap.release()
        except Exception:
            pass

if st.session_state.running:
    monitor_loop()
else:
    st.info("Press 'Start Monitoring' to begin (ensure webcam permission allowed).")
