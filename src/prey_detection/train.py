import argparse
import torch
from ultralytics import YOLO

def train_model(data_dir, epochs=50, imgsz=224, batch=16, project="clean_prey_yolov11n_cls"):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🧩 Training on device: {device}")
    
    # Pretrained YOLOv11 Nano classification model
    model = YOLO("yolo11n-cls.pt")
    
    try:
        results = model.train(
            data=data_dir,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=10,
            lr0=0.01,
            lrf=0.01,
            optimizer="Adam",
            augment=True,
            hsv_v=0.5, # Boost brightness augmentation for dark features
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
    parser.add_argument("--data_dir", required=True, help="Path to the directory containing train/ and val/ folders")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="clean_prey_yolov11n_cls")
    
    args = parser.parse_args()
    train_model(args.data_dir, args.epochs, args.imgsz, args.batch, args.project)
