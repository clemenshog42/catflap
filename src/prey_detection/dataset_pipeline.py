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

def process_and_split(input_dir, output_dir, crop_model_path=None, pre_cropped=False, color_mode="rgb", val_ratio=0.2, seed=42):
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    classes = [d.name for d in input_path.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")
    
    # Create output structure
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    
    for cls in classes:
        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)

    crop_model = None
    if not pre_cropped:
        if not crop_model_path:
            raise ValueError("crop_model_path must be provided if not pre_cropped")
        crop_model = YOLO(crop_model_path)
    
    total_processed = 0
    
    for cls in classes:
        src_dir = input_path / cls
        files = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        random.shuffle(files)
        
        if not files:
            continue
            
        val_count = max(1, int(len(files) * val_ratio)) if len(files) > 1 else 0
        
        for i, f in enumerate(tqdm(files, desc=f"Processing {cls}")):
            img = cv2.imread(str(f))
            if img is None:
                continue
                
            # Handle cropping
            if not pre_cropped and crop_model is not None:
                results = crop_model(str(f), verbose=False)[0]
                if len(results.boxes) > 0:
                    box = results.boxes.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    # Add small padding
                    h, w = img.shape[:2]
                    x1 = max(0, x1 - 10)
                    y1 = max(0, y1 - 10)
                    x2 = min(w, x2 + 10)
                    y2 = min(h, y2 + 10)
                    img = img[y1:y2, x1:x2]
                else:
                    # Skip image if no face found or you can just use original
                    # Here we use original if no face is found
                    pass
            
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

    print(f"✅ Processed {total_processed} images into train/val folders")
    
    # Create YAML for YOLO Classification (optional, but good for tracking)
    yaml_data = {
        "train": str(train_dir),
        "val": str(val_dir),
        "nc": len(classes),
        "names": classes
    }
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)
        
    print(f"✅ YAML created at {yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Input directory containing class folders")
    parser.add_argument("--output_dir", required=True, help="Output directory for train/val splits")
    parser.add_argument("--crop_model_path", help="Path to YOLO cat face model for cropping")
    parser.add_argument("--pre_cropped", action="store_true", help="Skip cropping if images are already cropped")
    parser.add_argument("--color", choices=["rgb", "grayscale"], default="rgb", help="Color mode (grayscale will duplicate channels to 3)")
    
    args = parser.parse_args()
    process_and_split(args.input_dir, args.output_dir, args.crop_model_path, args.pre_cropped, args.color)
