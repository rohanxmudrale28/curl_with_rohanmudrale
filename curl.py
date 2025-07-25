import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import os
import tempfile
import streamlit as st

# Load YOLO pose model
model = YOLO("yolov8n-pose.pt")

# Text-to-speech intro (uses gTTS)
def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        temp_path = fp.name
        tts.save(temp_path)
    st.audio(temp_path, format="audio/mp3")

speak("Hey! Rohan created me. I'm your AI Trainer. Let's count your bicep curls.")

# Angle calculator
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Draw skeleton on the frame
def draw_skeleton(frame, landmarks):
    skeleton_connections = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 12), (5, 11), (6, 12)
    ]
    for i, j in skeleton_connections:
        if i < len(landmarks) and j < len(landmarks):
            pt1 = tuple(landmarks[i].astype(int))
            pt2 = tuple(landmarks[j].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
    for point in landmarks:
        cv2.circle(frame, tuple(point.astype(int)), 4, (0, 0, 255), -1)

# Rep Counter Class
class BicepRepCounter:
    def __init__(self):
        self.left_counter = 0
        self.right_counter = 0
        self.left_stage = None
        self.right_stage = None

    def give_feedback(self, side, msg):
        speak(f"{msg} on your {side} arm")

    def update(self, landmarks, frame):
        try:
            # Left arm
            shoulder_l = landmarks[5]
            elbow_l = landmarks[7]
            wrist_l = landmarks[9]
            angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)

            if angle_l > 160:
                self.left_stage = "down"
            if angle_l < 40 and self.left_stage == "down":
                self.left_stage = "up"
                self.left_counter += 1
                self.give_feedback("left", "Good rep")

            # Right arm
            shoulder_r = landmarks[6]
            elbow_r = landmarks[8]
            wrist_r = landmarks[10]
            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)

            if angle_r > 160:
                self.right_stage = "down"
            if angle_r < 40 and self.right_stage == "down":
                self.right_stage = "up"
                self.right_counter += 1
                self.give_feedback("right", "Good rep")

            # Display angles
            cv2.putText(frame, f"{int(angle_l)}°", tuple(elbow_l.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(frame, f"{int(angle_r)}°", tuple(elbow_r.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # UI
            cv2.rectangle(frame, (10, 10), (310, 120), (50, 50, 50), -1)
            cv2.putText(frame, "LEFT", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Reps: {self.left_counter}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, f"Stage: {self.left_stage if self.left_stage else '-'}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            cv2.putText(frame, "RIGHT", (160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Reps: {self.right_counter}", (160, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Stage: {self.right_stage if self.right_stage else '-'}", (160, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        except Exception as e:
            print("Pose processing error:", e)

# Initialize webcam
cap = cv2.VideoCapture(0)
counter = BicepRepCounter()

print("Starting camera... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    kpts = results[0].keypoints

    if kpts is not None and len(kpts.xy) > 0:
        landmarks = kpts.xy[0].cpu().numpy()
        draw_skeleton(frame, landmarks)
        counter.update(landmarks, frame)

    cv2.imshow("AI Bicep Curl Counter - Rohan", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
