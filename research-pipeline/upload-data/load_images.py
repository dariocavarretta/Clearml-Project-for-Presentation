import json
import os
import requests
from pathlib import Path

# ==========================
# Configuration
# ==========================
ANNOTATION_FILE = "TACO-raw-annotations/data/annotations.json"  
OUTPUT_DIR = "downloaded_data"
# ==========================

# Create folders
images_dir = Path(OUTPUT_DIR) / "images"
labels_dir = Path(OUTPUT_DIR) / "labels"

images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Load COCO annotations
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

images = coco["images"]

# Group annotations by image_id
annotations_by_image = {}

for ann in coco.get("annotations", []):
    image_id = ann["image_id"]
    annotations_by_image.setdefault(image_id, []).append(ann)

print(f"Downloading {len(images)} images...")

for img in images:

    image_id = img["id"]

    # Prefer flickr_url, fallback to coco_url
    image_url = img.get("flickr_url") or img.get("coco_url")

    if not image_url:
        print(f"Skipping image {image_id}: no URL")
        continue

    # Save image using original filename
    filename = os.path.basename(img["file_name"])

    relative_path = Path(img["file_name"])
    image_path = images_dir / relative_path

    image_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        with open(image_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded: {filename}")

        # Save labels for this image
        label_path = labels_dir / relative_path.with_suffix(".json")

        label_path.parent.mkdir(parents=True, exist_ok=True)

        label_data = {
            "image": img,
            "annotations": annotations_by_image.get(image_id, [])
        }

        with open(label_path, "w") as f:
            json.dump(label_data, f, indent=2)

    except Exception as e:
        print(f"Failed {filename}: {e}")

print("Done!")