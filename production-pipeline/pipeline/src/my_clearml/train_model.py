from clearml import Dataset, OutputModel, Task
from ultralytics import YOLO
from pathlib import Path

def main(prepared_dataset_id: str, 
    epochs: int,
    imgsz: int,
    optimizer: str,
    lr0: float,
    mosaic: float,
    close_mosaic: int,
    scale: float,
    translate: float,
    workers: int,
    device,
    weights: str):

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
        workers = workers,

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