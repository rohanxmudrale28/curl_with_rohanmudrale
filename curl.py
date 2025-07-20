import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import os
from tempfile import NamedTemporaryFile
import math

# Load pose model
model = YOLO("yolov8n-pose.pt")

st.title("💪 Curl with Rohan - Bicep Rep Counter")

# Global variables
count = 0
direction = 0  # 0: down, 1: up

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle
    return angle

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.direction = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=0.5, classes=0, verbose=False)
        if results and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            # Use left arm: shoulder(5), elbow(7), wrist(9)
            try:
                shoulder = keypoints[5]
                elbow = keypoints[7]
                wrist = keypoints[9]

                angle = calculate_angle(shoulder, elbow, wrist)
                percentage = np.interp(angle, (50, 160), (100, 0))

                # Count logic
                if percentage == 100 and self.direction == 0:
                    self.counter += 1
                    self.direction = 1
                    say_count(self.counter)
                if percentage == 0 and self.direction == 1:
                    self.direction = 0

                # Draw
                cv2.putText(img, f"Reps: {self.counter}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)

            except Exception as e:
                pass

        return img

def say_count(count):
    """Speak count using gTTS"""
    tts = gTTS(text=f"{count}", lang='en')
    with NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
        tts.save(fp.name)
        os.system(f"mpg123 {fp.name}")

# Streamlit app interface
ctx = webrtc_streamer(
    key="repcounter",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
