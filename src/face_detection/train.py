import argparse
import torch
from ultralytics import YOLO

def train_model(data_yaml, epochs=100, imgsz=640, batch=16, project="yolov11_catmouth"):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting model training on {device}...")
    
    model = YOLO('yolo11n.pt')
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=project,
            device=device,
            patience=40,
            save=True,
            plots=True,
        )
        print("✅ Training complete!")
        print(f"Results and weights saved in the '{project}' directory.")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_yaml", required=True, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="yolov11_catmouth")
    
    args = parser.parse_args()
    train_model(args.data_yaml, args.epochs, args.imgsz, args.batch, args.project)
