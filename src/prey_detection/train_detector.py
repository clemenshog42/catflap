import argparse
import torch
import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

def inject_negatives(data_yaml, negatives_dir, num_negatives):
    """Copies negative images (background only, no prey) into the YOLO dataset."""
    if not negatives_dir or not num_negatives:
        return
        
    negatives_dir = Path(negatives_dir)
    if not negatives_dir.exists():
        print(f"❌ Negatives directory {negatives_dir} not found!")
        return
        
    # Get all images in negatives_dir
    all_negs = []
    for ext in ['.jpg', '.jpeg', '.png']:
        all_negs.extend(negatives_dir.rglob(f'*{ext}'))
        
    if len(all_negs) == 0:
        print("❌ No images found in negatives directory!")
        return
        
    # Sample the negatives
    random.seed(42)
    sampled_negs = random.sample(all_negs, min(num_negatives, len(all_negs)))
    print(f"💉 Injecting {len(sampled_negs)} negative images into the dataset...")
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
        
    base_dir = Path(data_yaml).parent
    
    # Calculate splits (80% train, 20% val)
    train_count = int(len(sampled_negs) * 0.8)
    train_negs = sampled_negs[:train_count]
    val_negs = sampled_negs[train_count:]
    
    def copy_negs(negs, split_name):
        if split_name not in data:
            return
        
        # Resolve target images directory
        split_path = base_dir / data[split_name]
        if not split_path.exists():
            split_path = Path(data[split_name])
            
        if split_path.is_file():  # sometimes it's a txt file list
            target_img_dir = split_path.parent
        else:
            target_img_dir = split_path
            
        for img in negs:
            # YOLO treats images without matching .txt files as background (negative) images!
            shutil.copy(img, target_img_dir / img.name)
            
    copy_negs(train_negs, 'train')
    copy_negs(val_negs, 'val')
    print("✅ Negatives successfully injected!")


def train_model(data_yaml, epochs=50, imgsz=224, batch=16, project="prey_detector", color_mode="rgb", apply_clahe=False, negatives_dir=None, num_negatives=0):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on device: {device}")
    
    inject_negatives(data_yaml, negatives_dir, num_negatives)
    
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
    
    args = parser.parse_args()
    train_model(args.data_yaml, args.epochs, args.imgsz, args.batch, args.project, args.color, args.apply_clahe, args.negatives_dir, args.num_negatives)
