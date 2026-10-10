# ==============================================================
# 🧠 YOLOv11 Binary Classification: clean vs prey (with Augmentation)
# ==============================================================
%rm -r /kaggle/working/data
%rm -r /kaggle/working/
# 1. Install dependencies
!pip install --upgrade ultralytics

import os
import shutil
import random
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from pathlib import Path


# ==============================================================
# 2. Set paths
# ==============================================================

INPUT_DIR = Path("/kaggle/input/cleanpreyfacesrgb")
DATASET_DIR = INPUT_DIR / "CleanPreyData_v1"  # <-- adjust if your dataset folder is named differently
CLEAN_SRC = DATASET_DIR / "clean"
PREY_SRC = DATASET_DIR / "prey"

WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "data"

# Create train/val structure
train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"
for d in [train_dir / "clean", train_dir / "prey", val_dir / "clean", val_dir / "prey"]:
    d.mkdir(parents=True, exist_ok=True)

def split_and_copy(src_dir, dst_train, dst_val, val_ratio=0.2, seed=42):
    random.seed(seed)
    files = [f for f in src_dir.iterdir() if f.is_file()]
    random.shuffle(files)

    if not files:
        print(f"⚠️ No files found in {src_dir}")
        return

    val_count = max(1, int(len(files) * val_ratio)) if len(files) > 1 else 0
    val_files = files[:val_count]
    train_files = files[val_count:]

    print(f"📂 {src_dir.name}: {len(train_files)} train, {len(val_files)} val")

    for f in train_files:
        shutil.copy(f, dst_train / f.name)
    for f in val_files:
        shutil.copy(f, dst_val / f.name)

# Split both classes
split_and_copy(CLEAN_SRC, train_dir / "clean", val_dir / "clean")
split_and_copy(PREY_SRC, train_dir / "prey", val_dir / "prey")

print("✅ Data successfully copied and split into train/val folders")

# ==============================================================
# 4. Create YOLO data.yaml
# ==============================================================

yaml_data = {
    "train": str(train_dir),
    "val": str(val_dir),
    "nc": 2,
    "names": ["clean", "prey"]
}

yaml_path = WORK_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(yaml_data, f)

print("✅ data.yaml created:")
print(open(yaml_path).read())

# ==============================================================
# 5. Train YOLOv11 Nano classification model
# ==============================================================

device = 0 if torch.cuda.is_available() else "cpu"
print(f"🧩 Training on device: {device}")

model = YOLO("yolo11n-cls.pt")  # Pretrained YOLOv11 Nano classification model

results = model.train(
    data="/kaggle/working/data",
    epochs=50,
    imgsz=224,
    batch=16,
    device=device,
    patience=10,
    lr0=0.01,       # initial learning rate
    lrf=0.01,       # final LR fraction
    optimizer="Adam", # Adam works well for classification
    augment=True,   # enable built-in augmentations
    project=str(WORK_DIR),
    name="clean_prey_yolov11n_cls",
    cache=False
)

print("✅ Training complete!")
print("Best model saved at:", results.best)
print("Metrics:", results.metrics)


!pip install --upgrade numpy
!pip install --upgrade matplotlib
!pip install --upgrade polars
!pip install --upgrade pyyaml
!pip install --upgrade Pillow
!pip install --upgrade psutil
!pip install --upgrade requests>=2.23.0
!pip install --upgrade torch>=1.8.0 # Or specific version/variant
!pip install --upgrade torchvision>=0.9.0 # Or specific version/variant