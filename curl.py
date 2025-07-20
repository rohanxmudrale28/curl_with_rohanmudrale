import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import os
import tempfile

# Load model
model = YOLO("yolov8n-pose.pt")

# Repetition counter class
class Counter:
    def __init__(self):
        self.count = 0
        self.up = False

    def detect(self, angle):
        if angle > 150:
            self.up = True
        if self.up and angle < 45:
            self.count += 1
            self.up = False
            return True
        return False

left_counter = Counter()
right_counter = Counter()

def say_text(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        os.system(f"mpg123 {fp.name}")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle if angle <= 180 else 360 - angle

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.prev_left = 0
        self.prev_right = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, verbose=False)
        for r in results:
            if r.keypoints is None: continue
            kp = r.keypoints.xy.cpu().numpy().squeeze()
            if kp.ndim != 2 or kp.shape[0] < 7: continue

            left_angle = calculate_angle(kp[5], kp[7], kp[9])
            right_angle = calculate_angle(kp[6], kp[8], kp[10])

            if left_counter.detect(left_angle):
                say_text(f"Left rep {left_counter.count}")
            if right_counter.detect(right_angle):
                say_text(f"Right rep {right_counter.count}")

            cv2.putText(img, f"Left: {left_counter.count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(img, f"Right: {right_counter.count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return img

# Streamlit UI
st.title("💪 ROBOREP - AI Bicep Curl Counter with Voice")
st.write("Counts bicep reps using webcam and speaks out the rep count.")

webrtc_streamer(
    key="rep-counter",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
