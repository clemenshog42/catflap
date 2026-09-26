import argparse
import torch
import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

import cv2
import numpy as np

def inject_negatives(data_yaml, negatives_dir, num_negatives, face_model_path="models/cat_face_2609.pt"):
    """Finds cats in negative images, crops them, and copies them as background (negative) images."""
    if not negatives_dir or not num_negatives:
        return
        
    negatives_dir = Path(negatives_dir)
    if not negatives_dir.exists():
        print(f"❌ Negatives directory {negatives_dir} not found!")
        return
        
    if not os.path.exists(face_model_path):
        print(f"❌ Face model {face_model_path} not found! Cannot crop negatives.")
        return
        
    # Get all images in negatives_dir
    all_negs = []
    for ext in ['.jpg', '.jpeg', '.png']:
        all_negs.extend(negatives_dir.rglob(f'*{ext}'))
        
    if len(all_negs) == 0:
        print("❌ No images found in negatives directory!")
        return
        
    random.seed(42)
    random.shuffle(all_negs)
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
        
    base_dir = Path(data_yaml).parent
    
    # Resolve target directories
    def get_split_dir(split_name):
        if split_name not in data: return None
        split_path = base_dir / data[split_name]
        if not split_path.exists(): split_path = Path(data[split_name])
        if split_path.is_file(): return split_path.parent
        return split_path

    train_dir = get_split_dir('train')
    val_dir = get_split_dir('val')
    
    if not train_dir or not val_dir:
        print("❌ Could not resolve train/val directories from yaml!")
        return

    print(f"💉 Searching {len(all_negs)} negative images for cats using {face_model_path}...")
    face_model = YOLO(face_model_path)
    
    successful_crops = 0
    pad_w_ratio = 0.0  # From models.py
    pad_top_ratio = -0.35
    pad_bottom_ratio = 0.3
    
    for img_path in all_negs:
        if successful_crops >= num_negatives:
            break
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        results = face_model(img, verbose=False, conf=0.4)[0]
        if len(results.boxes) == 0:
            continue
            
        for box_idx, box in enumerate(results.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            h, w = img.shape[:2]
            
            face_w = x2 - x1
            face_h = y2 - y1
            
            x1_pad = max(0, x1 - int(face_w * pad_w_ratio))
            y1_pad = max(0, y1 - int(face_h * pad_top_ratio))
            x2_pad = min(w, x2 + int(face_w * pad_w_ratio))
            y2_pad = min(h, y2 + int(face_h * pad_bottom_ratio))
            y1_pad = min(y2_pad - 1, y1_pad)
            
            crop_img = img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            if crop_img.size == 0:
                continue
                
            ch, cw = crop_img.shape[:2]
            max_dim = max(ch, cw)
            top_pad = (max_dim - ch) // 2
            bottom_pad = max_dim - ch - top_pad
            left_pad = (max_dim - cw) // 2
            right_pad = max_dim - cw - left_pad
            
            crop_img = cv2.copyMakeBorder(
                crop_img, 
                top_pad, bottom_pad, left_pad, right_pad, 
                cv2.BORDER_CONSTANT, 
                value=[0, 0, 0]
            )
            
            # Save the cropped negative
            filename = f"{img_path.stem}_negcat{box_idx}{img_path.suffix}"
            target_dir = val_dir if random.random() < 0.2 else train_dir
            cv2.imwrite(str(target_dir / filename), crop_img)
            
            successful_crops += 1
            if successful_crops >= num_negatives:
                break
                
    print(f"✅ Successfully injected {successful_crops} cropped negative cat faces!")


def train_model(data_yaml, epochs=50, imgsz=224, batch=16, project="prey_detector", color_mode="rgb", apply_clahe=False, negatives_dir=None, num_negatives=0, face_model_path="models/cat_face_2609.pt"):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on device: {device}")
    
    inject_negatives(data_yaml, negatives_dir, num_negatives, face_model_path)
    
    # Let YOLO natively handle grayscale by modifying the yaml!
    if color_mode == "grayscale":
        with open(data_yaml, 'a') as f:
            f.write("\nchannels: 1\n")
        print("✅ Added 'channels: 1' to data.yaml for native YOLO grayscale support.")
        
    if apply_clahe:
        print("⚠️ Warning: YOLO does not natively support CLAHE via the yaml. If you need CLAHE, you must preprocess the images manually!")
    
    # Pretrained YOLOv11 Nano Segmentation model
    model = YOLO("yolo11n-seg.pt")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=10,
            project=project,
            mosaic=0.0,      # Disable 4-image stitching to preserve small prey features
            fliplr=0.5,      # Left/Right flip
            flipud=0.0,      # No upside-down cats
            degrees=10.0,    # Slight head tilts
            hsv_h=0.015,     # Slight hue variation
            hsv_s=0.7,       # Desaturates some images to mimic night-vision/IR
            hsv_v=0.4,       # Simulates bright daylight vs dark night lighting
            scale=0.2,
            name="train",
            cache=False
        )
        print("✅ Training complete!")
        print(f"Results and weights saved in the '{project}' directory.")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_yaml", required=True, help="Path to the data.yaml file from Roboflow")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="prey_detector")
    parser.add_argument("--color", choices=["rgb", "grayscale"], default="rgb", help="Color mode (grayscale will duplicate channels to 3)")
    parser.add_argument("--apply_clahe", action="store_true", help="Apply CLAHE")
    parser.add_argument("--negatives_dir", default=None, help="Directory containing images of cats with NO prey to use as background negatives.")
    parser.add_argument("--num_negatives", type=int, default=0, help="Number of negative images to randomly inject into the dataset.")
    parser.add_argument("--face_model", default="models/cat_face_2609.pt", help="Path to the YOLO face model to crop the negatives.")
    
    args = parser.parse_args()
    train_model(args.data_yaml, args.epochs, args.imgsz, args.batch, args.project, args.color, args.apply_clahe, args.negatives_dir, args.num_negatives, args.face_model)
