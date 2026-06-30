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

def process_and_split(prey_dir, clean_dir, output_dir, crop_model_path, pad_w=10, pad_h=10, color_mode="rgb", val_ratio=0.2, seed=42):
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
    
    # Get all valid files
    prey_files = [f for f in prey_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    clean_files = [f for f in clean_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
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
                box = results.boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                h, w = img.shape[:2]
                
                # Add padding
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(w, x2 + pad_w)
                y2 = min(h, y2 + pad_h)
                
                img = img[y1:y2, x1:x2]
            else:
                # If the face detector misses, we skip the image to ensure the classifier only sees cropped faces
                continue
            
            # Handle color mode
            if color_mode == "grayscale":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Duplicate channels to make it 3-channel for pretrained models
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Determine split
            dst_dir = val_dir / cls if i < val_count else train_dir / cls
            dst_path = dst_dir / f.name
            
            cv2.imwrite(str(dst_path), img)
            total_processed += 1

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
    parser.add_argument("--pad_h", type=int, default=10, help="Vertical padding (pixels) around the bounding box")
    parser.add_argument("--color", choices=["rgb", "grayscale"], default="rgb", help="Color mode (grayscale will duplicate channels to 3)")
    
    args = parser.parse_args()
    process_and_split(args.prey_dir, args.clean_dir, args.output_dir, args.crop_model_path, args.pad_w, args.pad_h, args.color)
