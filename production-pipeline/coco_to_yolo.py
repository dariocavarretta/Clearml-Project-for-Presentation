"""
Convert Plastic Bottles COCO dataset -> YOLO format (dataset_v2)

Output structure mirrors dataset_v1:
  dataset_v2/
    images/batch_1/  batch_2/  batch_3/
    labels/batch_1/  batch_2/  batch_3/
    dataset.yaml
"""

import json
import shutil
import random
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
COCO_ROOT = Path(__file__).parent / "Plastic Bottles.coco copy"
OUT_ROOT  = Path(__file__).parent / "dataset_v2"
SPLITS    = ["train", "valid", "test"]
N_BATCHES = 3
SEED      = 42
# ─────────────────────────────────────────────────────────────────────────────

def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x_min, y_min, w, h] -> YOLO [cx, cy, w, h] normalised."""
    x, y, w, h = bbox
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def load_split(split_dir: Path):
    """Return list of (img_path, [(cls_id, cx, cy, w, h), ...])."""
    ann_file = split_dir / "_annotations.coco.json"
    with open(ann_file) as f:
        data = json.load(f)

    # single class dataset — always map to 0 regardless of COCO category id
    cat_map = {c["id"]: 0 for c in data["categories"]}

    img_meta = {img["id"]: img for img in data["images"]}

    # group annotations by image
    anns_by_img: dict = {img["id"]: [] for img in data["images"]}
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid in anns_by_img:
            cls = cat_map[ann["category_id"]]
            yolo = coco_bbox_to_yolo(ann["bbox"], img_meta[iid]["width"], img_meta[iid]["height"])
            anns_by_img[iid].append((cls, *yolo))

    records = []
    for iid, img in img_meta.items():
        img_path = split_dir / img["file_name"]
        records.append((img_path, anns_by_img[iid]))
    return records


def main():
    # collect all images across all splits
    all_records = []
    for split in SPLITS:
        split_dir = COCO_ROOT / split
        if split_dir.exists():
            all_records.extend(load_split(split_dir))

    print(f"Total images collected: {len(all_records)}")

    random.seed(SEED)
    random.shuffle(all_records)

    # split into N_BATCHES equal chunks
    chunks = [[] for _ in range(N_BATCHES)]
    for i, record in enumerate(all_records):
        chunks[i % N_BATCHES].append(record)

    for b_idx, chunk in enumerate(chunks, start=1):
        batch_name = f"batch_{b_idx}"
        img_dir = OUT_ROOT / "images" / batch_name
        lbl_dir = OUT_ROOT / "labels" / batch_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, boxes in chunk:
            shutil.copy(img_path, img_dir / img_path.name)
            label_path = lbl_dir / (img_path.stem + ".txt")
            with open(label_path, "w") as f:
                for box in boxes:
                    f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*box))

        print(f"  {batch_name}: {len(chunk)} images")

    # dataset.yaml
    yaml_path = OUT_ROOT / "dataset.yaml"
    yaml_path.write_text(
        f"path: {OUT_ROOT.resolve()}\n\n"
        "train: images\n"
        "val: images\n\n"
        "names:\n"
        "  0: plastic-bottle\n"
    )

    print(f"\nDone. Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
