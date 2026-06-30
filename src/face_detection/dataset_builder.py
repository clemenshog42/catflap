import os
import cv2
import numpy as np
import shutil
import random
import yaml
import argparse
from tqdm import tqdm
from ultralytics import YOLO

def to_yolo_bbox(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = abs(x2 - x1) / img_w
    h = abs(y2 - y1) / img_h
    return x_center, y_center, w, h

def build_dataset(input_dir, output_dir, source, model_path=None, color_mode="rgb", apply_clahe=False, splits=(0.8, 0.1, 0.1), class_id=0, class_name="cat_face"):
    # Create directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    items = []
    
    if source == "annotation":
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, f)
                    cat_path = img_path + ".cat"
                    if os.path.exists(cat_path):
                        items.append((img_path, cat_path))
        print(f"Found {len(items)} images with .cat annotations.")
    elif source == "model":
        if not model_path:
            raise ValueError("model_path must be provided when source='model'")
        model = YOLO(model_path)
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    items.append(os.path.join(root, f))
        print(f"Found {len(items)} images to process with model.")

    random.shuffle(items)
    n_total = len(items)
    n_train = int(n_total * splits[0])
    n_val = int(n_total * splits[1])
    
    dataset_splits = {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:]
    }

    channels = 1 if color_mode == "grayscale" else 3

    for split_name, split_items in dataset_splits.items():
        for item in tqdm(split_items, desc=f"Processing {split_name}"):
            if source == "annotation":
                img_path, cat_path = item
            else:
                img_path = item

            img = cv2.imread(img_path)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Grayscale conversion and optional CLAHE
            if color_mode == "grayscale":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if apply_clahe:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    img = clahe.apply(img)
            else:
                if apply_clahe:
                    # Apply CLAHE to the L channel in LAB color space
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    cl = clahe.apply(l)
                    limg = cv2.merge((cl,a,b))
                    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            x_c, y_c, bw, bh = None, None, None, None
            
            if source == "annotation":
                try:
                    with open(cat_path) as f:
                        data = list(map(int, f.read().split()))
                    coords = np.array(data[1:]).reshape(-1, 2)
                    if len(coords) == 9:
                        mouth = coords[2]
                        left_ear_pts = coords[3:6]
                        right_ear_pts = coords[6:9]

                        left_x = np.min(np.append(left_ear_pts[:, 0], mouth[0]))
                        right_x = np.max(np.append(right_ear_pts[:, 0], mouth[0]))
                        top_y = np.min(np.vstack((left_ear_pts, right_ear_pts))[:, 1])
                        bottom_y = mouth[1] + 50
                        
                        x_c, y_c, bw, bh = to_yolo_bbox(left_x, top_y, right_x, bottom_y, w, h)
                except Exception:
                    continue
            elif source == "model":
                results = model(str(img_path), verbose=False)[0]
                if len(results.boxes) > 0:
                    box = results.boxes.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = box
                    x_c = ((x1 + x2) / 2) / w
                    y_c = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

            if x_c is not None:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(output_dir, split_name, "labels", f"{base_name}.txt")
                with open(label_path, "w") as f:
                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                
                dst_img = os.path.join(output_dir, split_name, "images", os.path.basename(img_path))
                cv2.imwrite(dst_img, img)

    yaml_data = {
        "train": os.path.join(output_dir, "train", "images"),
        "val": os.path.join(output_dir, "val", "images"),
        "test": os.path.join(output_dir, "test", "images"),
        "channels": channels,
        "nc": 1,
        "names": [class_name]
    }
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)
        
    print(f"✅ Dataset YAML created at: {yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Input directory of images")
    parser.add_argument("--output_dir", required=True, help="Output directory for YOLO dataset")
    parser.add_argument("--source", choices=["annotation", "model"], required=True, help="Source of annotations")
    parser.add_argument("--model_path", help="Path to YOLO model if source is 'model'")
    parser.add_argument("--color", choices=["rgb", "grayscale"], default="rgb", help="Color mode")
    parser.add_argument("--apply_clahe", action="store_true", help="Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)")
    
    args = parser.parse_args()
    build_dataset(args.input_dir, args.output_dir, args.source, args.model_path, args.color, args.apply_clahe)
