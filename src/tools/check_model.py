from ultralytics import YOLO
import sys

try:
    model = YOLO("models/prey_V5.tflite")
    print("Model loaded successfully")
    print("Model names:", model.names)
except Exception as e:
    print("Error:", e)
