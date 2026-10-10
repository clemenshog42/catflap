%rm -r /kaggle/working/cat_yolo_dataset
import os
import cv2
import numpy as np
import shutil
import random
from tqdm import tqdm
import yaml

# ========== CONFIG ==========
DATASET_DIR = "/kaggle/input"   # adjust if dataset folder name differs
OUTPUT_BASE = "/kaggle/working/cat_yolo_dataset"
SPLITS = (0.8, 0.1, 0.1)
CLASS_ID = 0
CLASS_NAME = "cat_face"

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_BASE, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE, split, "labels"), exist_ok=True)

def to_yolo_bbox(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = abs(x2 - x1) / img_w
    h = abs(y2 - y1) / img_h
    return x_center, y_center, w, h


# === Collect (image, annotation) pairs ===
pairs = []
for root, dirs, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith(".jpg"):
            img_path = os.path.join(root, f)
            cat_path = img_path + ".cat"
            if os.path.exists(cat_path):
                pairs.append((img_path, cat_path))

print(f"Found {len(pairs)} images with annotations.")

# === Shuffle and split ===
random.shuffle(pairs)
n_total = len(pairs)
n_train = int(n_total * SPLITS[0])
n_val = int(n_total * SPLITS[1])
splits = {
    "train": pairs[:n_train],
    "val": pairs[n_train:n_train + n_val],
    "test": pairs[n_train + n_val:]
}

# === Process ===
for split_name, split_pairs in splits.items():
    for img_path, cat_path in tqdm(split_pairs, desc=f"Processing {split_name}"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        try:
            with open(cat_path) as f:
                data = list(map(int, f.read().split()))
        except Exception:
            continue

        coords = np.array(data[1:]).reshape(-1, 2)
        if len(coords) != 9:
            continue

        # === Extract key points ===
        mouth = coords[2]  # index 2 = mouth
        left_ear_pts = coords[3:6]
        right_ear_pts = coords[6:9]

        # Compute bounding region
        left_x = np.min(np.append(left_ear_pts[:, 0], mouth[0]))
        right_x = np.max(np.append(right_ear_pts[:, 0], mouth[0]))
        top_y = np.min(np.vstack((left_ear_pts, right_ear_pts))[:, 1])
        bottom_y = mouth[1] + 50

        # Basic bounding box
        x1, x2 = left_x, right_x
        y1, y2 = top_y, bottom_y
        # Convert to YOLO format
        x_c, y_c, bw, bh = to_yolo_bbox(x1, y1, x2, y2, w, h)

        # === Save label ===
        label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(OUTPUT_BASE, split_name, "labels", label_name)
        with open(label_path, "w") as f:
            f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

        # === Copy image ===
        dst_img = os.path.join(OUTPUT_BASE, split_name, "images", os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)

print("✅ All images and labels processed.")

# === Create dataset.yaml ===
yaml_path = os.path.join(OUTPUT_BASE, "dataset.yaml")
yaml_data = {
    "train": os.path.join(OUTPUT_BASE, "train", "images"),
    "val": os.path.join(OUTPUT_BASE, "val", "images"),
    "test": os.path.join(OUTPUT_BASE, "test", "images"),
    #"channels": 1, #remove comment to use gray scaling
    "nc": 1,
    "names": [CLASS_NAME]
}
with open(yaml_path, "w") as f:
    yaml.dump(yaml_data, f)

print(f"✅ Dataset YAML created at: {yaml_path}")


import os
import yaml
CLASS_NAME = "cat_face"
dirr = "/kaggle/input/catsyoloannotated/content/cat_dataset"
data_yaml = {
    "train": os.path.join(dirr, "train", "images"),
    "val": os.path.join(dirr, "val", "images"),
    "test": os.path.join(dirr, "test", "images"),
    #"channels": 1,
    "nc": 1,
    "names": [CLASS_NAME]
}

with open("/kaggle/working/data.yaml", "w") as f:
    yaml.dump(data_yaml, f)


%cp -r /kaggle/input/cat-dataset /kaggle/working/cat-dataset

!pip install -U ultralytics

%cp /kaggle/input/cat-face-2000/pytorch/default/1/model.pt /kaggle/working/model.pt
%cp /kaggle/input/cat-dataset/CAT_00/00000001_000.jpg /kaggle/working/1.jpg
import os
import shutil
import random
from tqdm import tqdm
import yaml
from ultralytics import YOLO
import cv2

# ========== CONFIG ==========
DATASET_DIR = "/kaggle/input"  # adjust if dataset folder name differs
OUTPUT_BASE = "/kaggle/working/cat_yolo_dataset"
SPLITS = (0.8, 0.1, 0.1)
CLASS_ID = 0
CLASS_NAME = "cat_face"
YOLO_MODEL_PATH = "/kaggle/input/cat-face-2000/pytorch/default/1/model.pt"  # path to your trained YOLO model

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_BASE, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_BASE, split, "labels"), exist_ok=True)

# Load YOLO model
model = YOLO("/kaggle/working/model.pt")
results = model("/kaggle/working/1.jpg")
#model.export(format="openvino")
#model = YOLO("/kaggle/working/model_openvino_model/")
# === Collect images ===
image_paths = []
for root, dirs, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith(".jpg"):
            image_paths.append(os.path.join(root, f))

print(f"Found {len(image_paths)} images.")

# === Shuffle and split ===
random.shuffle(image_paths)
n_total = len(image_paths)
n_train = int(n_total * SPLITS[0])
n_val = int(n_total * SPLITS[1])
splits = {
    "train": image_paths[:n_train],
    "val": image_paths[n_train:n_train + n_val],
    "test": image_paths[n_train + n_val:]
}

# === Process images ===
count = 0
for split_name, split_images in splits.items():
    for img_path in tqdm(split_images, desc=f"Processing {split_name}"):
        results = model(str(img_path))[0]  # take first result
        img_h, img_w = results.orig_shape[:2]

        # Skip images without predictions
        if len(results.boxes) == 0:
            continue

        # Save labels
        label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(OUTPUT_BASE, split_name, "labels", label_name)
        with open(label_path, "w") as f:
            for box in results.boxes.xyxy:  # xyxy format
                x1, y1, x2, y2 = box.cpu().numpy()
                # Convert to YOLO format
                x_c = ((x1 + x2) / 2) / img_w
                y_c = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                f.write(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        # Copy image
        dst_img = os.path.join(OUTPUT_BASE, split_name, "images", os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)
        count += 1
        if count % 500 == 0:
            print(count)

print("✅ All images and labels processed.")

# === Create dataset.yaml ===
yaml_path = os.path.join(OUTPUT_BASE, "dataset.yaml")
yaml_data = {
    "train": os.path.join(OUTPUT_BASE, "train", "images"),
    "val": os.path.join(OUTPUT_BASE, "val", "images"),
    "test": os.path.join(OUTPUT_BASE, "test", "images"),
    "channels": 1,
    "nc": 1,
    "names": [CLASS_NAME]
}
with open(yaml_path, "w") as f:
    yaml.dump(yaml_data, f)

print(f"✅ Dataset YAML created at: {yaml_path}")


import cv2
import os
import shutil
import random
import yaml
%cp /kaggle/input/catface-haarcascade/other/default/1/haarcascade_frontalcatface_extended.xml /kaggle/working/cascade.xml

CLASS_ID = 0
CLASS_NAME = "cat_face"
cascade_path = "/kaggle/working/cascade.xml"
input_folder = "/kaggle/input/cat-dataset"
output_image_folder = "/kaggle/working/cat_yolo_cascade/images"
output_label_folder = "/kaggle/working/cat_yolo_cascade/labels"
debug_bbox_folder = "/kaggle/working/debug_bboxes"

# Create output folders if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)
os.makedirs(debug_bbox_folder, exist_ok=True)

# Load Haar cascade
cat_cascade = cv2.CascadeClassifier(cascade_path)

# YOLO class ID for 'cat face'
cat_class_id = 0
count = 0
det_count = 0

# Walk recursively through input folder
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            count += 1
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Failed to load {filename}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.3,
                                                 minNeighbors=2, minSize=(75, 75))

            if len(faces) == 0:
                continue

            img_height, img_width = img.shape[:2]
            label_lines = []

            # Copy for bounding box visualization
            img_with_boxes = img.copy()

            for (x, y, w, h) in faces:
                # YOLO normalized format
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height

                label_lines.append(f"{cat_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                # Draw rectangle for debug image
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save original image (undrawn)
            output_img_path = os.path.join(output_image_folder, filename)
            cv2.imwrite(output_img_path, img)

            # Save YOLO label file
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            output_label_path = os.path.join(output_label_folder, txt_filename)
            with open(output_label_path, "w") as f:
                f.write("\n".join(label_lines))

            # Save debug image with bounding boxes
            debug_img_path = os.path.join(debug_bbox_folder, filename)
            cv2.imwrite(debug_img_path, img_with_boxes)
            det_count += 1
            if count % 100 == 0:
                print(f"✅ Detected and saved {filename} ({count} images processed)")

print(det_count)



%rm -r /kaggle/working/cat_yolo_dataset_split
import os
import shutil
import random
import yaml
input_dir = "/kaggle/working/cat_yolo_cascade"
output_dir = "/kaggle/working/cat_yolo_dataset"  # original dataset folder
images_dir = os.path.join(input_dir, "images")
labels_dir = os.path.join(input_dir, "labels")
CLASS_NAME = "cat_face"
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
seed = 42  # for reproducibility

yaml_file_path = os.path.join(output_dir, "dataset.yaml")

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

all_images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
all_images.sort()
random.seed(seed)
random.shuffle(all_images)

total = len(all_images)
print(total)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

splits = {
    "train": all_images[:train_end],
    "val": all_images[train_end:val_end],
    "test": all_images[val_end:]
}

for split, files in splits.items():
    for f in files:
        # Copy image
        shutil.copy(os.path.join(images_dir, f),
                    os.path.join(output_dir, split, "images", f))
        
        # Copy label (assumes same filename with .txt extension)
        label_file = os.path.splitext(f)[0] + ".txt"
        shutil.copy(os.path.join(labels_dir, label_file),
                    os.path.join(output_dir, split,"labels", label_file))

data_yaml = {
    "train": os.path.join(output_dir, "train", "images"),
    "val": os.path.join(output_dir, "val", "images"),
    "test": os.path.join(output_dir, "test", "images"),
    "channels": 1,
    "nc": 1,
    "names": [CLASS_NAME]
}

with open(yaml_file_path, "w") as f:
    yaml.dump(data_yaml, f)

print("Dataset split complete!")
print(f"YAML file saved to {yaml_file_path}")


import os
import cv2
import random
import matplotlib.pyplot as plt

# Path to your processed dataset
OUTPUT_BASE = "/kaggle/working/cat_yolo_dataset"
SPLIT = "train"  # or "val", "test"
NUM_SAMPLES = 20

# Gather all image paths
img_dir = os.path.join(OUTPUT_BASE, SPLIT, "images")
label_dir = os.path.join(OUTPUT_BASE, SPLIT, "labels")

images = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
sampled = random.sample(images, min(NUM_SAMPLES, len(images)))

plt.figure(figsize=(16, 16))

for i, img_name in enumerate(sampled):
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None or not os.path.exists(label_path):
        continue

    h, w = img.shape[:2]

    # Read YOLO label(s)
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_c, y_c, bw, bh = map(float, parts)
            x_c, y_c, bw, bh = x_c * w, y_c * h, bw * w, bh * h

            # Convert back to pixel coordinates
            x1 = int(x_c - bw / 2)
            y1 = int(y_c - bh / 2)
            x2 = int(x_c + bw / 2)
            y2 = int(y_c + bh / 2) + 30

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ✅ Correct function name
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(5, 4, i + 1)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(img_name[:15])

plt.tight_layout()
plt.show()


!pip install virtualenv
!virtualenv myenv
!source venv\bin\activate
!myenv/bin/pip install -U ultralytics
!myenv/bin/pip install ipykernel
!myenv/bin/python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

from ultralytics import YOLO
import cv2
from pathlib import Path

# --- Easy-to-change training parameters ---
NUM_EPOCHS = 100 
IMG_SIZE = 640
BATCH_SIZE = 16
PROJECT_NAME = 'yolov11_catmouth'

model = YOLO('yolo11n.pt')

# 2. Train the model with your custom augmentations
print("🚀 Starting model training with custom augmentations...")
try:
    results = model.train(
        # Core parameters
        data='/kaggle/working/data.yaml',
        #data='/kaggle/working/cat_yolo_dataset/dataset.yaml',
        epochs=NUM_EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=PROJECT_NAME,
        patience=40,  # Early stopping patience
        save=True,
        plots=True,
    )
    print("✅ Training complete!")
    
except Exception as e:
    print(f"❌ Training failed with error: {e}")
    print("\n💡 Try these solutions:")
    print("1. Run validate_images() to find corrupted images")
    print("2. Check your data.yaml paths")
    print("3. Reduce batch size if out of memory")
    raise



%rm -r /kaggle/working/runs/detect/yolov11_catmouth/weights/best_openvino_model
from ultralytics import YOLO
path_to_best_weights = '/kaggle/working/runs/detect/yolov11_catmouth/weights/best.pt'

model = YOLO(path_to_best_weights)
    
model.export(format="openvino", half=True)


import subprocess

# Convert ONNX to TensorFlow SavedModel first
result = subprocess.run([
    'onnx2tf', 
    '-i', '/kaggle/working/runs/detect/yolov11_catmouth/weights/best.onnx',
    '-o', '/kaggle/working/tf_model',
    '-b', '1',
    '-ois', 'images:1,1,640,640'
], capture_output=True, text=True)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)

%cat /kaggle/working/data.yaml