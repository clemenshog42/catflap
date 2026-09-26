import argparse
import torch
import cv2
import os
import yaml
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def preprocess_dataset(data_yaml_path, color_mode="rgb", apply_clahe=False):
    """Preprocesses the Roboflow dataset images in-place for grayscale and CLAHE."""
    if color_mode == "rgb" and not apply_clahe:
        return
        
    print(f"Applying preprocessing: color={color_mode}, clahe={apply_clahe}")
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    base_dir = Path(data_yaml_path).parent
    
    for split in ['train', 'val', 'test']:
        if split not in data:
            continue
            
        split_path = base_dir / data[split]
        if not split_path.exists():
            # Sometimes paths in yaml are absolute or different
            split_path = Path(data[split])
            if not split_path.exists():
                continue
                
        # Find all images
        images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            images.extend(split_path.rglob(f'*{ext}'))
            
        for img_path in tqdm(images, desc=f"Processing {split}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            if color_mode == "grayscale":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if apply_clahe:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    gray = clahe.apply(gray)
                final_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                if apply_clahe:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl,a,b))
                    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                else:
                    final_img = img
                    
            cv2.imwrite(str(img_path), final_img)


def train_model(data_yaml, epochs=50, imgsz=224, batch=16, project="prey_detector", color_mode="rgb", apply_clahe=False):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on device: {device}")
    
    # Preprocess dataset first!
    preprocess_dataset(data_yaml, color_mode, apply_clahe)
    
    # Pretrained YOLO26 Nano object detection model
    model = YOLO("yolo11n.pt")
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=10,
            # Object Detection Augmentations
            fliplr=0.5,     # Safe: horizontally flip
            degrees=10.0,   # Safe: slight rotation
            translate=0.1,  # Safe: translation
            shear=0.5,      # Safe: slight shear
            scale=0.1,      # Slight scaling
            hsv_v=0.2,      # Brightness augmentation
            erasing=0.0,    # Disabled per user request
            project=project,
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
