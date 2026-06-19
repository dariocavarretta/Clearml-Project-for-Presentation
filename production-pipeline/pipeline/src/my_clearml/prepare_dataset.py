from clearml import Dataset
from pathlib import Path
import yaml
import shutil
import random

def main(dataset_id: str, val_ratio: float) -> str:
    dataset = Dataset.get(dataset_id=dataset_id)
    path = Path(dataset.get_local_copy())
    images_dir = path / "images"
    labels_dir = path / "labels"


    for series_dir in images_dir.iterdir():
        if series_dir.is_dir():
            for file in series_dir.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(images_dir / file.name))

            series_dir.rmdir()
    
    for series_dir in labels_dir.iterdir():
        if series_dir.is_dir():
            for file in series_dir.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(labels_dir / file.name))

            series_dir.rmdir()
    
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

        label_path = labels_dir / f"{img_path.stem}.txt"

        shutil.move(
            str(img_path),
            str(images_dir / split / img_path.name)
        )

        if label_path.exists():
            shutil.move(
                str(label_path),
                str(labels_dir / split / label_path.name)
            )
    
    prepared_dataset= Dataset.create(
        dataset_project="Project - Litter",
        dataset_name = f"[prepared] - {dataset_id}",
        output_uri="s3://8b56346322f98ed029a3c888fba38a69.r2.cloudflarestorage.com:443/ml-storage/processed-datasets/"
    )

    yaml_path = Path(path) / "dataset.yaml"

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    data.pop("path", None)
    data["train"] = "images/train"
    data["val"] = "images/val"

    print(yaml.safe_dump(data, sort_keys=False))

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    prepared_dataset.add_files(path)
    prepared_dataset.upload()
    prepared_dataset.finalize()
    
    return prepared_dataset.id