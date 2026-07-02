"""
Convert 3 COCO datasets to flat YOLO format (images/ labels/ dataset.yaml).
Each dataset maps only its relevant category IDs to class 0.
All train/valid/test splits are merged — no split distinction in output.
"""

import json
import shutil
from pathlib import Path

BASE  = Path(__file__).parent
SPLITS = ["train", "valid", "test"]

# dataset_folder -> (output_folder, class_id, class_name, {coco_category_ids to keep})
DATASETS = {
    "glass_bottles.coco": ("dataset_glass_bottles",  1, "glass-bottle",  {1}),
    "cups.coco":          ("dataset_cups",            2, "cup",           {3, 4}),
    "aluminum cans.coco": ("dataset_aluminum_cans",  3, "aluminum-can",  {1, 2, 3, 4, 5}),
}


def coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def convert(ds_folder, out_folder, class_id, class_name, keep_ids):
    src  = BASE / ds_folder
    out  = BASE / out_folder
    imgs = out / "images"
    lbls = out / "labels"
    imgs.mkdir(parents=True, exist_ok=True)
    lbls.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_anns   = 0

    for split in SPLITS:
        ann_file = src / split / "_annotations.coco.json"
        if not ann_file.exists():
            continue

        data     = json.loads(ann_file.read_text())
        img_meta = {img["id"]: img for img in data["images"]}

        # only keep annotations whose category_id is in keep_ids
        anns_by_img = {img["id"]: [] for img in data["images"]}
        for ann in data["annotations"]:
            if ann["category_id"] not in keep_ids:
                continue
            iid = ann["image_id"]
            img = img_meta[iid]
            yolo = coco_bbox_to_yolo(ann["bbox"], img["width"], img["height"])
            anns_by_img[iid].append(yolo)  # class_id written at output time

        for iid, img in img_meta.items():
            boxes = anns_by_img[iid]
            if not boxes:          # skip images with no relevant annotations
                continue

            src_img = src / split / img["file_name"]
            if not src_img.exists():
                continue

            shutil.copy(src_img, imgs / img["file_name"])

            lbl_path = lbls / (Path(img["file_name"]).stem + ".txt")
            with open(lbl_path, "w") as f:
                for (cx, cy, w, h) in boxes:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            total_images += 1
            total_anns   += len(boxes)

    # dataset.yaml
    (out / "dataset.yaml").write_text(
        f"path: {out.resolve()}\n\n"
        "train: images\n"
        "val: images\n\n"
        "names:\n"
        f"  {class_id}: {class_name}\n"
    )

    print(f"{ds_folder:30s} -> {out_folder}  (class {class_id}: {class_name}, {total_images} images, {total_anns} annotations)")


if __name__ == "__main__":
    for ds_folder, (out_folder, class_id, class_name, keep_ids) in DATASETS.items():
        convert(ds_folder, out_folder, class_id, class_name, keep_ids)
    print("\nDone.")
