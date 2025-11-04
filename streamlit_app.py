# streamlit_app.py
# Streamlit driver monitoring using streamlit-webrtc (live browser webcam),
# plus upload-video fallback. Processes every Nth frame for optimization.

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import os
import pygame
from PIL import Image
import tempfile

# streamlit-webrtc imports
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings

# ---------------------------
# SETTINGS / THRESHOLDS
# ---------------------------
EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 0.6
HTR_THRESHOLD = 0.5

EYES_CLOSED_SECONDS = 3.0
ALERT_COOLDOWN = 5.0
DEFAULT_ALARM = "alarm-106447.mp3"  # optional server-side alarm sound

# ---------------------------
# INITIALIZE PYGAME (optional audio on server)
# ---------------------------
try:
    pygame.mixer.init()
    if os.path.exists(DEFAULT_ALARM):
        alarm_sound = pygame.mixer.Sound(DEFAULT_ALARM)
    else:
        alarm_sound = None
except Exception:
    alarm_sound = None


def play_alarm_nonblocking():
    if alarm_sound:
        try:
            alarm_sound.play()
        except Exception:
            pass


# ---------------------------
# UTILS (same logic as before)
# ---------------------------

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
        ear = (left_ear + right_ear) / 2.0 if (left_ear is not None and right_ear is not None) else None
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
# Video transformer for webrtc
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh

class FaceTransformer(VideoTransformerBase):
    def __init__(self, process_every=10, play_audio=False):
        self.process_every = int(process_every)
        self.play_audio = play_audio
        self.counter = 0

        # mediapipe face mesh instance
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                              max_num_faces=1,
                                              refine_landmarks=True,
                                              min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)
        # state / counters for display (kept inside transformer)
        self.closed_start = None
        self.last_alert_time = 0.0
        self.eyes_closed_events = 0
        self.yawns = 0
        self.tilts = 0
        self.total_alerts = 0

    def transform(self, frame):
        # frame is an av.VideoFrame
        img = frame.to_ndarray(format="bgr24")
        self.counter += 1

        # Optionally resize for speed (reduce CPU)
        small = cv2.resize(img, (320, int(320 * img.shape[0] / img.shape[1])))

        # Only process every Nth frame
        if self.counter % self.process_every != 0:
            # return unmodified (or a lightly scaled frame) to keep smoothness
            return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        frame_display = small.copy()
        ear = mar = htr = None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            ear, mar, htr, pts = compute_metrics_from_landmarks(lm, small.shape[1], small.shape[0])
            # draw landmarks (small green dots)
            for (x, y) in pts:
                cv2.circle(frame_display, (x, y), 1, (0, 255, 0), -1)

        now = time.time()
        unsafe = False
        unsafe_reasons = []

        # Eyes closed logic (duration-based)
        if ear is not None and ear <= EAR_THRESHOLD:
            if self.closed_start is None:
                self.closed_start = now
            duration = now - self.closed_start
            if duration >= EYES_CLOSED_SECONDS:
                if now - self.last_alert_time >= ALERT_COOLDOWN:
                    self.eyes_closed_events += 1
                    self.total_alerts += 1
                    self.last_alert_time = now
                    if self.play_audio:
                        play_alarm_nonblocking()
                unsafe = True
                unsafe_reasons.append("Eyes closed")
        else:
            self.closed_start = None

        # Yawning
        if mar is not None and mar >= MAR_THRESHOLD:
            if now - self.last_alert_time >= ALERT_COOLDOWN:
                self.yawns += 1
                self.total_alerts += 1
                self.last_alert_time = now
                if self.play_audio:
                    play_alarm_nonblocking()
            unsafe = True
            unsafe_reasons.append("Yawning")

        # Head tilt
        if htr is not None and htr >= HTR_THRESHOLD:
            if now - self.last_alert_time >= ALERT_COOLDOWN:
                self.tilts += 1
                self.total_alerts += 1
                self.last_alert_time = now
                if self.play_audio:
                    play_alarm_nonblocking()
            unsafe = True
            unsafe_reasons.append("Head tilt")

        # Classification (for on-frame label)
        state = classify_driver_state(ear, mar, htr)

        # Draw visual warnings on frame
        info_lines = [
            f"EAR: {ear:.3f}" if ear is not None else "EAR: -",
            f"MAR: {mar:.3f}" if mar is not None else "MAR: -",
            f"HTR: {htr:.3f}" if htr is not None else "HTR: -",
        ]
        for i, ln in enumerate(info_lines):
            cv2.putText(frame_display, ln, (10, 60 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # status box
        if unsafe:
            cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1] - 1, frame_display.shape[0] - 1),
                          (0, 0, 255), 6)
            reason_str = ", ".join(unsafe_reasons)
            cv2.putText(frame_display, f"DANGER: {state} - {reason_str}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            label_color = (0, 255, 0) if state == "Alert" else (0, 165, 255)
            cv2.putText(frame_display, f"Status: {state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

        # optional: overlay counters
        counter_txt = f"E:{self.eyes_closed_events} Y:{self.yawns} T:{self.tilts} A:{self.total_alerts}"
        cv2.putText(frame_display, counter_txt, (10, frame_display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # return RGB frame (webrtc expects RGB)
        return cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Driver Monitor (webrtc)", layout="wide")
st.title("Driver Monitoring — streamlit-webrtc (Live) + Upload fallback")

col1, col2 = st.columns([2, 1])

with col2:
    st.header("Controls")
    source_option = st.radio("Source", ("WebRTC Live Camera", "Upload Video File"), index=0)
    process_every = st.slider("Process every Nth frame", 1, 20, 10, 1)
    play_audio = st.checkbox("Play alarm on server (pygame)", value=False)
    start_btn = st.button("Start")
    stop_btn = st.button("Stop")

with col1:
    stframe = st.empty()

if "running" not in st.session_state:
    st.session_state.running = False

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# Display small help
st.markdown(
    "- Use **WebRTC Live Camera** for browser webcam (works in Streamlit Cloud and local). "
    "- Use **Upload Video File** to analyze a video file frame-by-frame."
)

# ---------------------------
# MAIN: WebRTC or Upload
# ---------------------------
if st.session_state.running:
    if source_option == "WebRTC Live Camera":
        # start a webrtc streamer and supply the transformer factory
        ctx = webrtc_streamer(
            key="driver-monitor",
            video_transformer_factory=lambda: FaceTransformer(process_every=process_every, play_audio=play_audio),
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
            client_settings=ClientSettings(
                # optional ice servers if needed on cloud.
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
        )

        # after starting, show simple controls / read transformer state if available
        if ctx and ctx.video_transformer:
            transformer = ctx.video_transformer
            # Expose simple read-only counters (manual refresh recommended)
            st.sidebar.markdown("**Session counters (read-only)**")
            st.sidebar.text(f"Eyes-closed events: {transformer.eyes_closed_events}")
            st.sidebar.text(f"Yawns: {transformer.yawns}")
            st.sidebar.text(f"Tilts: {transformer.tilts}")
            st.sidebar.text(f"Total alerts: {transformer.total_alerts}")
            st.sidebar.text(f"Frames processed (counter mod N): {transformer.counter}")
            st.sidebar.markdown("Tip: counters update live inside the video overlay; use sidebar refresh to fetch latest.")
        else:
            st.info("Starting WebRTC... allow camera access in your browser when prompted.")

    else:  # Upload Video File
        video_file = st.file_uploader("Upload video (mp4, mov, avi)", type=["mp4", "mov", "avi"])
        if video_file is None:
            st.warning("Upload a video file to use this mode.")
        else:
            # save to temp file and process in a local loop (same logic as earlier)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
            stframe = st.empty()

            # create a local FaceTransformer to reuse the logic
            local_transformer = FaceTransformer(process_every=process_every, play_audio=play_audio)

            while st.session_state.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # The transform function expects an av.VideoFrame, but we can call process logic directly:
                local_transformer.counter += 1
                if local_transformer.counter % local_transformer.process_every != 0:
                    # still show scaled frame
                    small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
                    stframe.image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                    # small sleep to allow UI update
                    time.sleep(0.01)
                    continue

                # run mediapipe process (copied from transform body)
                small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = local_transformer.face_mesh.process(rgb)
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

                # Eyes closed
                if ear is not None and ear <= EAR_THRESHOLD:
                    if local_transformer.closed_start is None:
                        local_transformer.closed_start = now
                    if now - local_transformer.closed_start >= EYES_CLOSED_SECONDS:
                        if now - local_transformer.last_alert_time >= ALERT_COOLDOWN:
                            local_transformer.eyes_closed_events += 1
                            local_transformer.total_alerts += 1
                            local_transformer.last_alert_time = now
                            if play_audio:
                                play_alarm_nonblocking()
                        unsafe = True
                        unsafe_reasons.append("Eyes closed")
                else:
                    local_transformer.closed_start = None

                if mar is not None and mar >= MAR_THRESHOLD:
                    if now - local_transformer.last_alert_time >= ALERT_COOLDOWN:
                        local_transformer.yawns += 1
                        local_transformer.total_alerts += 1
                        local_transformer.last_alert_time = now
                        if play_audio:
                            play_alarm_nonblocking()
                    unsafe = True
                    unsafe_reasons.append("Yawning")

                if htr is not None and htr >= HTR_THRESHOLD:
                    if now - local_transformer.last_alert_time >= ALERT_COOLDOWN:
                        local_transformer.tilts += 1
                        local_transformer.total_alerts += 1
                        local_transformer.last_alert_time = now
                        if play_audio:
                            play_alarm_nonblocking()
                    unsafe = True
                    unsafe_reasons.append("Head tilt")

                state = classify_driver_state(ear, mar, htr)

                # overlay info
                info_lines = [
                    f"EAR: {ear:.3f}" if ear is not None else "EAR: -",
                    f"MAR: {mar:.3f}" if mar is not None else "MAR: -",
                    f"HTR: {htr:.3f}" if htr is not None else "HTR: -"
                ]
                for i, ln in enumerate(info_lines):
                    cv2.putText(frame_display, ln, (10, 60 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if unsafe:
                    cv2.rectangle(frame_display, (0, 0), (frame_display.shape[1] - 1, frame_display.shape[0] - 1), (0, 0, 255), 6)
                    reason_str = ", ".join(unsafe_reasons)
                    cv2.putText(frame_display, f"ALERT: {reason_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if state == "Alert":
                        st_text_color = (0, 255, 0)
                    elif state == "Distracted":
                        st_text_color = (0, 165, 255)
                    else:
                        st_text_color = (0, 255, 255)
                    cv2.putText(frame_display, f"Status: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, st_text_color, 2)

                # counters overlay
                counter_txt = f"E:{local_transformer.eyes_closed_events} Y:{local_transformer.yawns} T:{local_transformer.tilts} A:{local_transformer.total_alerts}"
                cv2.putText(frame_display, counter_txt, (10, frame_display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                stframe.image(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB), channels="RGB")
                time.sleep(1.0 / (fps or 10))  # attempt to display at source fps

            cap.release()
            st.success("Finished processing uploaded video.")
else:
    st.info("Press Start to begin monitoring (allow camera access).")
