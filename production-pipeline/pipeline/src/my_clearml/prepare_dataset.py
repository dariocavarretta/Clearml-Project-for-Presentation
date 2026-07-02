from clearml import Dataset
from pathlib import Path
import yaml
import shutil
import random


def _flatten_dir(directory: Path) -> None:
    """
    Move all files from immediate subdirectories up into directory, then remove subdirs.
    Handles mixed layouts (some files already flat + some inside batch_N/).
    Resolves filename collisions by appending the subdir name as a prefix.
    """
    for sub in list(directory.iterdir()):
        if sub.is_dir():
            for f in sub.iterdir():
                if f.is_file():
                    dest = directory / f.name
                    if dest.exists():
                        # avoid collision: prefix with the subdir name
                        dest = directory / f"{sub.name}__{f.name}"
                    shutil.move(str(f), str(dest))
            sub.rmdir()


def _merge_yamls(path: Path) -> dict:
    """
    Find all *.yaml files at top-level of path, merge their 'names' dicts
    (union of all class_id -> class_name entries), return a single merged dict.
    """
    yamls = list(path.glob("*.yaml"))
    merged_names = {}
    for y in yamls:
        data = yaml.safe_load(y.read_text())
        names = data.get("names", {})
        # normalise: YOLO yaml names can be a list or a dict
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        merged_names.update(names)
    return merged_names


def main(dataset_id: str, val_ratio: float) -> str:
    dataset = Dataset.get(dataset_id=dataset_id)
    path = Path(dataset.get_local_copy())
    images_dir = path / "images"
    labels_dir = path / "labels"

    # ── Step 1: flatten any batch_N subdirectories ─────────────────────────
    _flatten_dir(images_dir)
    _flatten_dir(labels_dir)

    # ── Step 2: merge all top-level yamls into one name map ─────────────────
    merged_names = _merge_yamls(path)

    # remove all existing yaml files — we'll write one clean one at the end
    for y in path.glob("*.yaml"):
        y.unlink()

    # ── Step 3: train / val split ───────────────────────────────────────────
    for split in ["train", "val"]:
        (images_dir / split).mkdir(exist_ok=True)
        (labels_dir / split).mkdir(exist_ok=True)

    image_files = [
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    random.seed(42)
    random.shuffle(image_files)

    n_val = int(len(image_files) * val_ratio)
    val_images = set(image_files[:n_val])

    for img_path in image_files:
        split = "val" if img_path in val_images else "train"

        shutil.move(str(img_path), str(images_dir / split / img_path.name))

        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.move(str(label_path), str(labels_dir / split / label_path.name))

    # ── Step 4: write merged dataset.yaml ───────────────────────────────────
    final_yaml = {
        "train": "images/train",
        "val":   "images/val",
        "names": dict(sorted(merged_names.items())),
    }

    yaml_path = path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(final_yaml, f, sort_keys=False)

    print("[prepare_dataset] Final dataset.yaml:")
    print(yaml.safe_dump(final_yaml, sort_keys=False))

    # ── Step 5: upload prepared dataset ─────────────────────────────────────
    prepared_dataset = Dataset.create(
        dataset_project="Project - Litter",
        dataset_name=f"[prepared] - {dataset_id}",
        output_uri="s3://8b56346322f98ed029a3c888fba38a69.r2.cloudflarestorage.com:443/ml-storage/processed-datasets/"
    )

    prepared_dataset.add_files(path)
    prepared_dataset.upload()
    prepared_dataset.finalize()

    return prepared_dataset.id
