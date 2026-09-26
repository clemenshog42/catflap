import argparse
import torch
import os
from pathlib import Path
from ultralytics import YOLO


def train_model(data_yaml, epochs=50, imgsz=224, batch=16, project="prey_detector", color_mode="rgb", apply_clahe=False):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on device: {device}")
    
    # Let YOLO natively handle grayscale by modifying the yaml!
    if color_mode == "grayscale":
        with open(data_yaml, 'a') as f:
            f.write("\nchannels: 1\n")
        print("✅ Added 'channels: 1' to data.yaml for native YOLO grayscale support.")
        
    if apply_clahe:
        print("⚠️ Warning: YOLO does not natively support CLAHE via the yaml. If you need CLAHE, you must preprocess the images manually!")
    
    # Pretrained YOLO26 Nano object detection model
    model = YOLO("yolo26n.pt")
    
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
    
    args = parser.parse_args()
    train_model(args.data_yaml, args.epochs, args.imgsz, args.batch, args.project, args.color, args.apply_clahe)
