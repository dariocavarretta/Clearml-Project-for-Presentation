from clearml import PipelineDecorator, Dataset, Task, OutputModel, Model
from pathlib import Path
import shutil
import random
from ultralytics import YOLO
import yaml

@PipelineDecorator.component(execution_queue = "my_laptop-cpu-tasks",  return_values=["passed"], cache=True)
def health_check(dataset_id: str):
    passed = False
    dataset = Dataset.get(dataset_id = dataset_id)
    files = dataset.list_files()
    
    images = set()
    labels = set()

    for f in dataset.list_files():
        p = Path(f)

        if p.suffix.lower() in {".jpg"}:
            images.add(Path(*p.parts[1:]).with_suffix(""))

        elif p.suffix.lower() == ".txt":
            labels.add(Path(*p.parts[1:]).with_suffix(""))

    missing_labels = images - labels
    missing_image = labels - images
    actual_mismatch = ((len(missing_image) + len(missing_labels)) / len(files))*100

    print("[info] Analyzed dataset's health. Here is the Summary:")
    print(f"Missing Labels: {len(missing_labels)}")
    print(f"Missing Images: {len(missing_image)}")
    print("Mismatch ratio threshold set at 0.05")
    print(f"Mismatch ratio found in Dataset: {actual_mismatch}")
    
    if (len(missing_image) + len(missing_labels)) / (len(files) - 1) > 0.05:
        raise ValueError ("[error] too many mismatches. Aborting ...")
    else:
        print("[success] Health check passed succesfully")
        passed = True
    return passed


@PipelineDecorator.component(execution_queue = "my_laptop-cpu-tasks", return_values=["prepared_dataset_id"], cache=True)
def prepare_dataset(dataset_id: str, passed: bool, val_ratio: float):
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
    
    return str(prepared_dataset.id)

@PipelineDecorator.component(execution_queue = "my_laptop-gpu-tasks", return_values=["model_id"], cache=False)
def train_model(prepared_dataset_id: str, 
    epochs,
    imgsz,
    optimizer,
    lr0,
    mosaic,
    close_mosaic,
    scale,
    translate,
    workers,
    device,
    weights):

    prepared_dataset = Dataset.get(dataset_id = prepared_dataset_id)
    path = prepared_dataset.get_local_copy()

    data_yaml = Path(path) / "dataset.yaml"
    model = YOLO(weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        lr0=lr0,
        mosaic=mosaic,
        close_mosaic=close_mosaic,
        scale=scale,
        translate=translate,
        optimizer=optimizer,
        device=device,
        workers = workers
    )
    best_model = Path(results.save_dir) / "weights" / "best.pt"

    task = Task.current_task()

    output_model = OutputModel(
        task=task,
        name=f"YOLOv11-{weights}"
    )

    output_model.update_weights(
        weights_filename=str(best_model)
    )

    return str(output_model.id)

@PipelineDecorator.component(execution_queue = "my_laptop-cpu-tasks", return_values=["onnx_file"], cache=False)
def export_model(output_model_id: str):

    pt_model = Model(model_id=output_model_id)

    pt_path = pt_model.get_local_copy()

    model = YOLO(pt_path)

    onnx_path = model.export(format="onnx")

    task = Task.current_task()

    onnx_model = OutputModel(
        task=task,
        name=f"{pt_model.name}-onnx"
    )

    onnx_model.update_weights(
        weights_filename=str(onnx_path)
    )

    return onnx_model.id


def execute_pipeline(
    dataset_id,
    val_ratio,
    epochs,
    imgsz,
    optimizer,
    lr0,
    mosaic,
    close_mosaic,
    scale,
    translate,
    workers,
    device,
    weights,
):
    passed = health_check(
        dataset_id=dataset_id
    )

    prepared_dataset_id = prepare_dataset(
        dataset_id=dataset_id,
        passed=passed,
        val_ratio=val_ratio,
    )

    model_id = train_model(
        prepared_dataset_id=prepared_dataset_id,
        epochs=epochs,
        imgsz=imgsz,
        optimizer=optimizer,
        lr0=lr0,
        mosaic=mosaic,
        close_mosaic=close_mosaic,
        scale=scale,
        translate=translate,
        workers=workers,
        device=device,
        weights=weights,
    )

    onnx_model_id = export_model(
        output_model_id=model_id
    )

    return onnx_model_id
