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
import json
import zipfile
from crop import crop_region, CROP_PAD


def _export_metadata(model_path):
    """Exported models (e.g. .tflite) carry Ultralytics' metadata.json (task, channels, ...) as an embedded zip.
    Returns it as a dict, or {} for .pt files and files without metadata."""
    try:
        if model_path and zipfile.is_zipfile(model_path):
            with zipfile.ZipFile(model_path) as z:
                if "metadata.json" in z.namelist():
                    return json.loads(z.read("metadata.json"))
    except (OSError, ValueError):
        pass
    return {}

def _task_from_metadata(model_path):
    """Ultralytics cannot guess the task of exported models (e.g. .tflite) from the file name and falls back
    to 'detect', which decodes a segmentation output wrongly. Exported files carry their task in the embedded
    metadata.json, so read it from there. Returns None for .pt files, where Ultralytics infers it itself."""
    return _export_metadata(model_path).get("task")

def _input_channels(yolo_model, model_path=None):
    """Number of input channels of a YOLO model: from the model yaml (.pt) or from the export metadata (.tflite).
    3 if unknown."""
    try:
        return int(yolo_model.model.yaml.get("channels", 3))
    except AttributeError:
        return int(_export_metadata(model_path).get("channels", 3))


class CatFlapPipeline:
    def __init__(self, detector_path="path/to/cat_face_detector.pt", classifier_path="path/to/prey_classifier.pt", apply_clahe_detector=False, apply_clahe_classifier=True, color_mode_detector="grayscale", color_mode_classifier="grayscale"):
        self.apply_clahe_detector = apply_clahe_detector
        self.apply_clahe_classifier = apply_clahe_classifier
        self.color_mode_detector = color_mode_detector
        self.color_mode_classifier = color_mode_classifier
        self.detector_path = detector_path
        self.classifier_path = classifier_path
        print(f"Loading Object Detector from: {detector_path}")
        try:
            self.detector = YOLO(detector_path, task=_task_from_metadata(detector_path))
        except Exception as e:
            print(f"Warning: Failed to load detector. Make sure the path is correct. Error: {e}")
            self.detector = None
            
        print(f"Loading Prey Model from: {classifier_path}")
        try:
            self.classifier = YOLO(classifier_path, task=_task_from_metadata(classifier_path))
        except Exception as e:
            print(f"Warning: Failed to load prey model. Make sure the path is correct. Error: {e}")
            self.classifier = None

    def run_detector(self, frame, track=False):
        """Runs the cat face detector. With track=True, ByteTrack assigns persistent IDs (results.boxes.id)."""
        if self.detector is None:
            return None
            
        needs_gray = (self.color_mode_detector == "grayscale")
        
        if self.apply_clahe_detector:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_frame = clahe.apply(gray_frame)
            final_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        else:
            if needs_gray:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                final_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            else:
                final_frame = frame
        
        # Detectors trained with 'channels: 1' (e.g. cat_face.pt) expect a single-channel input
        if _input_channels(self.detector, self.detector_path) == 1 and final_frame.ndim == 3:
            final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2GRAY)[..., None]

        try:
            if track:
                results = self.detector.track(final_frame, conf=0.1, persist=True, tracker=getattr(self, 'tracker', "bytetrack.yaml"), verbose=False)
            else:
                results = self.detector.predict(final_frame, conf=0.1, verbose=False)
            return results[0] 
        except ValueError as e:
            print(f"Prediction error (likely dimension mismatch): {e}")
            return None

    def run_prey_detector(self, frame, box):
        if self.classifier is None:
            return 0.0, None
            
        h, w = frame.shape[:2]
        if getattr(self, 'use_asymmetric_crop', True):
            # Extension of the face box relative to its size (see crop.py); can be overridden via self.crop_pad
            (x1, y1, x2, y2), _ = crop_region(box, w, h, getattr(self, 'crop_pad', CROP_PAD))
        else:
            (x1, y1, x2, y2), _ = crop_region(box, w, h, (0.0, 0.0, 0.0))
        
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
        
        needs_gray = (self.color_mode_classifier == "grayscale")
        
        if self.apply_clahe_classifier:
            gray_crop = cv2.cvtColor(square_crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_crop = clahe.apply(gray_crop)
            final_crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
        else:
            if needs_gray:
                gray_crop = cv2.cvtColor(square_crop, cv2.COLOR_BGR2GRAY)
                final_crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
            else:
                final_crop = square_crop
            
        # Models trained with 'channels: 1' (e.g. prey_gray.pt) expect a single-channel input
        if _input_channels(self.classifier, self.classifier_path) == 1 and final_crop.ndim == 3:
            final_crop = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)[..., None]
            
        results = self.classifier(final_crop, verbose=False)
        
        result = results[0]
        
        prey_conf = 0.0
        
        # Object detection extraction logic
        if result.boxes is not None and len(result.boxes) > 0:
            for box_data in result.boxes:
                cls_id = int(box_data.cls[0].item())
                conf = box_data.conf[0].item()
                name = result.names[cls_id].lower()
                
                # Check if this box is classified as 'prey'
                if 'prey' in name or 'mouse' in name or 'bird' in name:
                    if conf > prey_conf:
                        prey_conf = conf
                        
            # Draw the bounding box of the prey on final_crop for visualization
            final_crop = result.plot()
            
        return prey_conf, final_crop
