import cv2
import numpy as np
import sys
try:
    import ai_edge_litert
    sys.modules['tflite_runtime'] = ai_edge_litert
    import ai_edge_litert.interpreter
    sys.modules['tflite_runtime.interpreter'] = ai_edge_litert.interpreter
except ImportError:
    pass
from ultralytics import YOLO

class CatFlapPipeline:
    def __init__(self, detector_path="path/to/cat_face_detector.pt", classifier_path="path/to/prey_classifier.pt", apply_clahe_detector=False, apply_clahe_classifier=True):
        self.apply_clahe_detector = apply_clahe_detector
        self.apply_clahe_classifier = apply_clahe_classifier
        print(f"Loading Object Detector from: {detector_path}")
        try:
            self.detector = YOLO(detector_path, task='detect')
        except Exception as e:
            print(f"Warning: Failed to load detector. Make sure the path is correct. Error: {e}")
            self.detector = None
            
        print(f"Loading Classifier from: {classifier_path}")
        try:
            self.classifier = YOLO(classifier_path, task='classify')
        except Exception as e:
            print(f"Warning: Failed to load classifier. Make sure the path is correct. Error: {e}")
            self.classifier = None

    def run_detector(self, frame):
        if self.detector is None:
            return None
            
        if self.apply_clahe_detector:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_frame = clahe.apply(gray_frame)
            # Make 3 channels just in case YOLOv11 expects it for RGB models acting on gray
            # gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # Removed to fix 1-channel model crash
        else:
            gray_frame = frame
        
        try:
            results = self.detector.track(gray_frame, persist=True, tracker="bytetrack.yaml", conf=0.1, verbose=False)
            return results[0] 
        except ValueError as e:
            print(f"Tracking error (likely dimension mismatch): {e}")
            return None

    def run_classifier(self, frame, box):
        if self.classifier is None:
            return 0.0, None
            
        x1, y1, x2, y2 = map(int, box)
        face_w = x2 - x1
        face_h = y2 - y1
        
        if getattr(self, 'use_asymmetric_crop', True):
            pad_w = int(face_w * 0)
            pad_top = int(face_h * -0.35)
            pad_bottom = int(face_h * 0.3)
        else:
            pad_w = int(face_w * 0)
            pad_top = int(face_h * 0)
            pad_bottom = int(face_h * 0)
            
        x1 -= pad_w
        y1 -= pad_top
        x2 += pad_w
        y2 += pad_bottom
        
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        y1 = min(y2 - 1, y1)
        
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return 0.0, None
            
        ch, cw = crop.shape[:2]
        max_dim = max(ch, cw)
        
        top_pad = (max_dim - ch) // 2
        bottom_pad = max_dim - ch - top_pad
        left_pad = (max_dim - cw) // 2
        right_pad = max_dim - cw - left_pad
        
        square_crop = cv2.copyMakeBorder(
            crop, 
            top_pad, bottom_pad, left_pad, right_pad, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        if self.apply_clahe_classifier:
            gray_crop = cv2.cvtColor(square_crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_crop = clahe.apply(gray_crop)
            final_crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
        else:
            final_crop = square_crop
            
        results = self.classifier(final_crop, verbose=False)
        
        result = results[0]
        probs = result.probs
        
        if probs is None:
            return 0.0, final_crop
            
        class_idx = 1
        for i, name in result.names.items():
            if 'with' in name.lower() or 'prey' in name.lower():
                class_idx = i
                
        if probs.data.shape[0] > class_idx:
            prey_conf = probs.data[class_idx].item()
        else:
            prey_conf = probs.top1conf.item() if ('with' in result.names[probs.top1].lower()) else 0.0
            
        return prey_conf, final_crop

