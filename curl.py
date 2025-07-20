import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import os
import tempfile

model = YOLO("yolov8n-pose.pt")  # Make sure this is downloaded

class BicepCounter:
    def __init__(self):
        self.left_counter = 0
        self.right_counter = 0
        self.prev_left_angle = 180
        self.prev_right_angle = 180
        self.left_up = False
        self.right_up = False

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arccos(np.clip(np.dot(b - a, c - b) /
                                    (np.linalg.norm(b - a) * np.linalg.norm(c - b)), -1.0, 1.0))
        angle = np.degrees(radians)
        return angle

    def count_reps(self, frame):
        results = model(frame, verbose=False)[0]
        if results.keypoints is None:
            return frame, self.left_counter, self.right_counter, None

        keypoints = results.keypoints.xy.cpu().numpy()[0]

        try:
            left_angle = self.calculate_angle(
                keypoints[11], keypoints[13], keypoints[15])
            right_angle = self.calculate_angle(
                keypoints[12], keypoints[14], keypoints[16])

            # Left arm logic
            if left_angle < 40 and not self.left_up:
                self.left_up = True
            if left_angle > 160 and self.left_up:
                self.left_counter += 1
                self.left_up = False
                self.speak("Left rep counted")

            # Right arm logic
            if right_angle < 40 and not self.right_up:
                self.right_up = True
            if right_angle > 160 and self.right_up:
                self.right_counter += 1
                self.right_up = False
                self.speak("Right rep counted")

        except Exception as e:
            pass

        return frame, self.left_counter, self.right_counter, None

    def speak(self, text):
        tts = gTTS(text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            os.system(f"mpg123 {fp.name} > /dev/null 2>&1")

counter = BicepCounter()

def video_stream(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_frame, left_count, right_count, _ = counter.count_reps(frame)
    return cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)

iface = gr.Interface(fn=video_stream,
                     inputs=gr.Image(source="webcam", streaming=True),
                     outputs=gr.Image(),
                     live=True,
                     title="ROBOREP - Bicep Counter with Voice",
                     description="Counts left and right bicep curls using webcam + gives voice feedback")

if __name__ == "__main__":
    iface.launch()
