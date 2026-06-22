from ultralytics import YOLO
from clearml import Dataset, Task, OutputModel
from pathlib import Path

def main(
    split_dataset_id: str,
    epochs, 
    optimizer, 
    weights, 
    workers, 
    device
):
    """
    Train a YOLOv8 model.
    """
    prepared_dataset = Dataset.get(dataset_id = split_dataset_id)
    path = prepared_dataset.get_local_copy()

    data_yaml = Path(path) / "meta.yaml"
    model = YOLO(weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        optimizer=optimizer,
        device=device,
        workers = workers
    )
    best_model = Path(results.save_dir) / "weights" / "best.pt"

    task = Task.current_task()

    output_model = OutputModel(
        task=task,
        name=f"YOLOv8-{weights}"
    )

    output_model.update_weights(
        weights_filename=str(best_model)
    )

    return output_model.id