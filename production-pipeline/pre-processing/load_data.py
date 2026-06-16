import json
import os
import requests
from pathlib import Path

# ==========================
# Configuration
# ==========================
ANNOTATION_FILE = "TACO-raw-annotations/data/annotations.json"
OUTPUT_DIR = "production-pipeline/dataset_v1"

TARGET_CLASS = 5  # TACO class to keep
YOLO_CLASS = 0     # Output class id
# ==========================

images_dir = Path(OUTPUT_DIR) / "images"
labels_dir = Path(OUTPUT_DIR) / "labels"

images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Load COCO annotations
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

# Group ONLY target annotations by image
annotations_by_image = {}

for ann in coco["annotations"]:
    if ann["category_id"] != TARGET_CLASS:
        continue

    image_id = ann["image_id"]
    annotations_by_image.setdefault(image_id, []).append(ann)

# Keep only images that contain class 59
images = [
    img for img in coco["images"]
    if img["id"] in annotations_by_image
]

print(f"Found {len(images)} images containing class {TARGET_CLASS}")

for img in images:

    image_id = img["id"]

    image_url = img.get("flickr_url") or img.get("coco_url")

    if not image_url:
        print(f"Skipping image {image_id}: no URL")
        continue

    relative_path = Path(img["file_name"])

    image_path = images_dir / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        with open(image_path, "wb") as f:
            f.write(response.content)

        width = img["width"]
        height = img["height"]

        # Create YOLO label file
        label_path = labels_dir / relative_path.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)

        yolo_lines = []

        for ann in annotations_by_image[image_id]:

            x, y, w, h = ann["bbox"]

            # COCO -> YOLO
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            yolo_lines.append(
                f"{YOLO_CLASS} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{w_norm:.6f} "
                f"{h_norm:.6f}"
            )

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        print(f"Downloaded and labeled: {relative_path}")

    except Exception as e:
        print(f"Failed {relative_path}: {e}")

# Create dataset.yaml
yaml_content = f"""path: {Path(OUTPUT_DIR).resolve()}

train: images
val: images

names:
  {YOLO_CLASS}: target_object
"""

with open(Path(OUTPUT_DIR) / "dataset.yaml", "w") as f:
    f.write(yaml_content)

print("Done!")