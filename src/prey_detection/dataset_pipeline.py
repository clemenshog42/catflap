import os
import shutil
import random
import yaml
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def process_and_split(prey_dir, clean_dir, output_dir, crop_model_path, pad_w=15, pad_top=-10, pad_bottom=30, color_mode="rgb", apply_clahe=False, val_ratio=0.2, seed=42):
    random.seed(seed)
    
    prey_path = Path(prey_dir)
    clean_path = Path(clean_dir)
    output_path = Path(output_dir)
    
    # Create output structure
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    for cls in ["prey", "clean"]:
        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)

    if not crop_model_path:
        raise ValueError("crop_model_path must be provided to crop the images")
    crop_model = YOLO(crop_model_path)
    
    # Get all valid files recursively
    prey_files = [f for f in prey_path.rglob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    clean_files = [f for f in clean_path.rglob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    # We will balance the dataset by counting SUCCESSFUL face crops, not raw input files.
    
    total_processed = 0
    successful_prey_count = 0
    successful_clean_count = 0
    
    # --- PHASE 1: Process Prey ---
    print(f"Found {len(prey_files)} raw prey images. Processing...")
    for f in tqdm(prey_files, desc="Processing prey"):
        img = cv2.imread(str(f))
        if img is None:
            continue
            
        results = crop_model(str(f), verbose=False)[0]
        if len(results.boxes) > 0:
            for box_idx, box in enumerate(results.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                h, w = img.shape[:2]
                
                # Add padding
                x1_pad = max(0, x1 - pad_w)
                y1_pad = max(0, y1 - pad_top)
                x2_pad = min(w, x2 + pad_w)
                y2_pad = min(h, y2 + pad_bottom)
                y1_pad = min(y2_pad - 1, y1_pad)
                
                crop_img = img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                
                # --- SQUARE PADDING LOGIC ---
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
                # ----------------------------
                
                # Colors/CLAHE
                if color_mode == "grayscale":
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    if apply_clahe:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        crop_img = clahe.apply(crop_img)
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
                else:
                    if apply_clahe:
                        lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        cl = clahe.apply(l)
                        limg = cv2.merge((cl,a,b))
                        crop_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                # Train/Val Split via probability
                dst_dir = val_dir / "prey" if random.random() < val_ratio else train_dir / "prey"
                filename = f"{f.stem}_cat{box_idx}{f.suffix}"
                cv2.imwrite(str(dst_dir / filename), crop_img)
                
                successful_prey_count += 1
                total_processed += 1
    
    print(f"Successfully generated {successful_prey_count} prey crops.")
    
    # --- PHASE 2: Process Clean (until balanced) ---
    print(f"Found {len(clean_files)} raw clean images. Sampling until we reach {successful_prey_count} successful crops...")
    random.shuffle(clean_files)
    
    for f in tqdm(clean_files, desc="Processing clean"):
        if successful_clean_count >= successful_prey_count:
            break # We have perfectly balanced the dataset!
            
        img = cv2.imread(str(f))
        if img is None:
            continue
            
        results = crop_model(str(f), verbose=False)[0]
        if len(results.boxes) > 0:
            for box_idx, box in enumerate(results.boxes.xyxy.cpu().numpy()):
                if successful_clean_count >= successful_prey_count:
                    break # Stop even if there are multiple cats in this final image
                    
                x1, y1, x2, y2 = map(int, box)
                h, w = img.shape[:2]
                
                # Add padding
                x1_pad = max(0, x1 - pad_w)
                y1_pad = max(0, y1 - pad_top)
                x2_pad = min(w, x2 + pad_w)
                y2_pad = min(h, y2 + pad_bottom)
                y1_pad = min(y2_pad - 1, y1_pad)
                
                crop_img = img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                
                # --- SQUARE PADDING LOGIC ---
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
                # ----------------------------
                
                # Colors/CLAHE
                if color_mode == "grayscale":
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    if apply_clahe:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        crop_img = clahe.apply(crop_img)
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
                else:
                    if apply_clahe:
                        lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        cl = clahe.apply(l)
                        limg = cv2.merge((cl,a,b))
                        crop_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                # Train/Val Split via probability
                dst_dir = val_dir / "clean" if random.random() < val_ratio else train_dir / "clean"
                filename = f"{f.stem}_cat{box_idx}{f.suffix}"
                cv2.imwrite(str(dst_dir / filename), crop_img)
                
                successful_clean_count += 1
                total_processed += 1

    print(f"✅ Processed and cropped {total_processed} images into train/val folders (Prey: {successful_prey_count}, Clean: {successful_clean_count})")
    
    # Create YAML for YOLO Classification
    yaml_data = {
        "train": str(train_dir),
        "val": str(val_dir),
        "nc": 2,
        "names": ["clean", "prey"]
    }
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)
        
    print(f"✅ YAML created at {yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prey_dir", required=True, help="Input directory containing cats with prey")
    parser.add_argument("--clean_dir", required=True, help="Input directory containing cats without prey")
    parser.add_argument("--output_dir", required=True, help="Output directory for train/val splits")
    parser.add_argument("--crop_model_path", required=True, help="Path to YOLO cat face model for cropping")
    parser.add_argument("--pad_w", type=int, default=10, help="Horizontal padding (pixels) around the bounding box")
    parser.add_argument("--pad_top", type=int, default=10, help="Vertical padding (pixels) above the bounding box")
    parser.add_argument("--pad_bottom", type=int, default=30, help="Vertical padding (pixels) below the bounding box (for prey)")
    parser.add_argument("--color", choices=["rgb", "grayscale"], default="rgb", help="Color mode (grayscale will duplicate channels to 3)")
    parser.add_argument("--apply_clahe", action="store_true", help="Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)")
    
    args = parser.parse_args()
    process_and_split(args.prey_dir, args.clean_dir, args.output_dir, args.crop_model_path, args.pad_w, args.pad_top, args.pad_bottom, args.color, args.apply_clahe)
