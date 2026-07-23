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
        """
        Initializes the YOLO models.
        Currently using placeholders. Replace with actual paths when ready.
        """
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
        """
        Runs the cat face detector with ByteTrack.
        Returns the ultralytics results object.
        """
        if self.detector is None:
            return None
            
        # Convert to grayscale as the model name suggests it expects 1-channel input
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.apply_clahe_detector:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_frame = clahe.apply(gray_frame)
        
        # We re-enable ByteTrack here to preserve the track IDs for the State Machine.
        # The CenterCrop square padding fix we applied below is robust enough to handle 
        # any minor tracking jitter without losing the prey!
        try:
            results = self.detector.track(gray_frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            return results[0] # Return the first (and only) frame's results
        except ValueError as e:
            print(f"Tracking error (likely dimension mismatch): {e}")
            return None

    def run_classifier(self, frame, box):
        """
        Extracts the region of interest (ROI) from the frame based on the bounding box,
        and runs the prey classifier.
        
        box: [x1, y1, x2, y2]
        returns: float (confidence of 'prey')
        """
        if self.classifier is None:
            return 0.0
            
        x1, y1, x2, y2 = map(int, box)
        
        # Add asymmetrical padding (heavy on bottom for dangling prey)
        pad_w = 15
        pad_top = -10
        pad_bottom = 30
        x1 -= pad_w
        y1 -= pad_top
        x2 += pad_w
        y2 += pad_bottom
        
        # Ensure coordinates are within frame bounds and y1 is strictly less than y2
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        y1 = min(y2 - 1, y1)
        
        # Crop the image
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return 0.0
            
        # --- SQUARE PADDING LOGIC ---
        # YOLO Classification forces a CenterCrop if the image is not a perfect square.
        # We must manually pad the crop with black pixels to make it a perfect square
        # so the CenterCrop doesn't secretly chop off our asymmetrical dangling prey padding!
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
        # ----------------------------
            
        # Convert crop to grayscale, apply optional CLAHE, then back to 3-channel (g, g, g) as per training
        gray_crop = cv2.cvtColor(square_crop, cv2.COLOR_BGR2GRAY)
        
        if self.apply_clahe_classifier:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_crop = clahe.apply(gray_crop)
            
        gray_3ch_crop = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
            
        # Run classifier on the crop
        results = self.classifier(gray_3ch_crop, verbose=False)
        
        # Extract confidence for "prey"
        # Assuming the classifier has classes where one represents "prey" (e.g. class 1)
        # You may need to adjust the class index based on how your classifier was trained
        result = results[0]
        probs = result.probs
        
        # If probs is None, something went wrong with classification
        if probs is None:
            return 0.0
            
        # Example: assuming class index 1 is "prey", and 0 is "no prey"
        # Update this index '1' based on your model's names: result.names
        prey_class_idx = 1
        
        # Some models might have different class mappings. Let's find "prey" if possible:
        for idx, name in result.names.items():
            if name.lower() == 'prey':
                prey_class_idx = idx
                break
                
        # Return the confidence of the prey class
        return float(probs.data[prey_class_idx])
