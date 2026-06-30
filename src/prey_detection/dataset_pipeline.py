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

def process_and_split(prey_dir, clean_dir, output_dir, crop_model_path, pad_w=15, pad_top=10, pad_bottom=50, color_mode="rgb", apply_clahe=False, val_ratio=0.2, seed=42):
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
    
    # Balance classes: Take all prey images, and sample an equal amount of clean images
    num_prey = len(prey_files)
    print(f"Found {num_prey} prey images.")
    
    if len(clean_files) < num_prey:
        print(f"⚠️ Warning: Found {len(clean_files)} clean images, which is less than the {num_prey} prey images. Using all clean images.")
        sampled_clean = clean_files
    else:
        print(f"Found {len(clean_files)} clean images. Sampling exactly {num_prey} to balance the dataset.")
        sampled_clean = random.sample(clean_files, num_prey)

    datasets = {
        "prey": prey_files,
        "clean": sampled_clean
    }
    
    total_processed = 0
    
    for cls, files in datasets.items():
        random.shuffle(files)
        val_count = max(1, int(len(files) * val_ratio)) if len(files) > 1 else 0
        
        for i, f in enumerate(tqdm(files, desc=f"Processing {cls}")):
            img = cv2.imread(str(f))
            if img is None:
                continue
                
            # Perform cropping
            results = crop_model(str(f), verbose=False)[0]
            if len(results.boxes) > 0:
                for box_idx, box in enumerate(results.boxes.xyxy.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)
                    
                    h, w = img.shape[:2]
                    
                    # Add asymmetrical padding (heavy on bottom for dangling prey)
                    x1_pad = max(0, x1 - pad_w)
                    y1_pad = max(0, y1 - pad_top)
                    x2_pad = min(w, x2 + pad_w)
                    y2_pad = min(h, y2 + pad_bottom)
                    
                    # Create the crop on a fresh copy so we don't destroy the original image for the next loop
                    crop_img = img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                    
                    # Handle color mode and optional CLAHE
                    if color_mode == "grayscale":
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        if apply_clahe:
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            crop_img = clahe.apply(crop_img)
                        # Duplicate channels to make it 3-channel for pretrained models
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
                    else:
                        if apply_clahe:
                            lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
                            l, a, b = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                            cl = clahe.apply(l)
                            limg = cv2.merge((cl,a,b))
                            crop_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                    # Determine split
                    dst_dir = val_dir / cls if i < val_count else train_dir / cls
                    
                    # Append _cat0, _cat1, etc. to the filename to avoid overwriting if there are multiple cats
                    filename = f"{f.stem}_cat{box_idx}{f.suffix}"
                    dst_path = dst_dir / filename
                    
                    cv2.imwrite(str(dst_path), crop_img)
                    total_processed += 1
            else:
                # If the face detector misses, we skip the image to ensure the classifier only sees cropped faces
                continue

    print(f"✅ Processed and cropped {total_processed} images into train/val folders")
    
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
