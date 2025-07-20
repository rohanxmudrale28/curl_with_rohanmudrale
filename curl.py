import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import tempfile

model = YOLO("yolov8n-pose.pt")

st.title("💪 ROBOREP - Bicep Curl Counter")

# Text-to-Speech setup
def speak(text):
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        filename = fp.name
    engine.save_to_file(text, filename)
    engine.runAndWait()
    with open(filename, "rb") as file:
        st.audio(file.read(), format="audio/mp3")

class BicepCounter(VideoTransformerBase):
    def __init__(self):
        self.left_counter = 0
        self.right_counter = 0
        self.left_dir = 0
        self.right_dir = 0

    def bicep_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)[0]
        if results.keypoints is not None:
            for person in results.keypoints.data:
                keypoints = person.numpy()
                if len(keypoints) >= 7:
                    left_shoulder = keypoints[5][:2]
                    left_elbow = keypoints[7][:2]
                    left_wrist = keypoints[9][:2]

                    right_shoulder = keypoints[6][:2]
                    right_elbow = keypoints[8][:2]
                    right_wrist = keypoints[10][:2]

                    # Left arm angle
                    left_angle = self.bicep_angle(left_shoulder, left_elbow, left_wrist)
                    if left_angle > 160:
                        self.left_dir = 0
                    if left_angle < 30 and self.left_dir == 0:
                        self.left_counter += 1
                        self.left_dir = 1
                        speak(f"Left rep {self.left_counter}")

                    # Right arm angle
                    right_angle = self.bicep_angle(right_shoulder, right_elbow, right_wrist)
                    if right_angle > 160:
                        self.right_dir = 0
                    if right_angle < 30 and self.right_dir == 0:
                        self.right_counter += 1
                        self.right_dir = 1
                        speak(f"Right rep {self.right_counter}")

                    # Draw info
                    cv2.putText(img, f"Left: {self.left_counter}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    cv2.putText(img, f"Right: {self.right_counter}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        return img

webrtc_streamer(key="bicep", video_transformer_factory=BicepCounter)
