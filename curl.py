import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import os
import tempfile
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Load YOLO model once
model = YOLO("yolov8n-pose.pt")

# Speak using gTTS
def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

# Angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Repetition counter class
class BicepRepCounter:
    def __init__(self):
        self.left_counter = 0
        self.right_counter = 0
        self.left_stage = None
        self.right_stage = None

    def update(self, landmarks):
        feedback = []
        # Left Arm
        shoulder_l, elbow_l, wrist_l = landmarks[5], landmarks[7], landmarks[9]
        angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)

        if angle_l > 160:
            self.left_stage = "down"
        if angle_l < 40 and self.left_stage == "down":
            self.left_stage = "up"
            self.left_counter += 1
            feedback.append("Good rep on your left arm")

        # Right Arm
        shoulder_r, elbow_r, wrist_r = landmarks[6], landmarks[8], landmarks[10]
        angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)

        if angle_r > 160:
            self.right_stage = "down"
        if angle_r < 40 and self.right_stage == "down":
            self.right_stage = "up"
            self.right_counter += 1
            feedback.append("Good rep on your right arm")

        return feedback, angle_l, angle_r

rep_counter = BicepRepCounter()

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, verbose=False)
        kpts = results[0].keypoints

        if kpts is not None and len(kpts.xy) > 0:
            landmarks = kpts.xy[0].cpu().numpy().astype(np.int32)

            # Draw keypoints
            for pt in landmarks:
                cv2.circle(img, tuple(pt), 4, (0, 0, 255), -1)

            # Update logic
            feedback, angle_l, angle_r = rep_counter.update(landmarks)

            # UI
            cv2.putText(img, f"Left reps: {rep_counter.left_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(img, f"Right reps: {rep_counter.right_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(img, f"Left angle: {int(angle_l)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            cv2.putText(img, f"Right angle: {int(angle_r)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

            for f in feedback:
                speak(f)

        return img

st.title("🏋️ AI Trainer - Bicep Curl Counter")

speak("Hey! Rohan created me. I'm your AI Trainer. Let's count your bicep curls.")

# WebRTC with STUN fix
webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
