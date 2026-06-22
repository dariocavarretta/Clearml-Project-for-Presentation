from clearml import Dataset
import random
import shutil
from pathlib import Path
from datetime import datetime
import yaml


def main(val_ratio, restructured_path):

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    source_path = Path(restructured_path)

    split_path = (
        source_path.parent
        / f"split_at_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    shutil.copytree(source_path, split_path)

    images_root = split_path / "images"
    labels_root = split_path / "labels"

    images_train = images_root / "train"
    images_val = images_root / "val"

    labels_train = labels_root / "train"
    labels_val = labels_root / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    image_files = [
        p for p in images_root.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    random.shuffle(image_files)

    n_total = len(image_files)
    n_val = int(n_total * val_ratio)

    val_files = set(image_files[:n_val])

    for img_path in image_files:

        label_path = labels_root / f"{img_path.stem}.txt"

        if img_path in val_files:
            dst_img = images_val
            dst_lbl = labels_val
        else:
            dst_img = images_train
            dst_lbl = labels_train

        shutil.move(
            str(img_path),
            str(dst_img / img_path.name)
        )

        if label_path.exists():
            shutil.move(
                str(label_path),
                str(dst_lbl / label_path.name)
            )

    # Update YAML with absolute path for ClearML compatibility
    yaml_path = split_path / "meta.yaml"

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = str(split_path.absolute())
    data["train"] = "images/train"
    data["val"] = "images/val"

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(
        f"Total: {n_total}\n"
        f"Train: {n_total - n_val}\n"
        f"Val: {n_val}"
    )

    split_dataset = Dataset.create(
        dataset_project="Yolo-Ready",
        dataset_name="Split-Taco",
        output_uri=(
            "s3://8b56346322f98ed029a3c888fba38a69.r2.cloudflarestorage.com:443/"
            "ml-storage/processed-datasets/"
        )
    )

    split_dataset.add_files(str(split_path))
    split_dataset.upload()
    split_dataset.finalize()

    return split_dataset.id, split_path