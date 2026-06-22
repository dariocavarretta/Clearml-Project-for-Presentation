"""
Pipeline components for ClearML training pipeline.

"""
from ultralytics import YOLO
from clearml import PipelineDecorator, Dataset, Task, OutputModel, Model
from pathlib import Path
from datetime import datetime


# Initial check: first step

@PipelineDecorator.component(return_values=["task_id"], cache=True)
def health_check(dataset_id: str, mismatch_threshold: float) -> str:
    """
    Check dataset format to abort pipeline immediately in case there are issues.

    Args
        :dataset_id: ClearML training dataset id
        :mismatch_threshold: Maximum mismatch ratio allowed between labels and images
    """
    from steps.health_check import main as check_main

    task_id = check_main(dataset_id=dataset_id, mismatch_threshold=mismatch_threshold)

    return str(task_id)

@PipelineDecorator.component(return_values=["dataset_path", "restructured_path"], cache=True)
def download_dataset(health_check_task_id: str, dataset_id: str, project_name: str):

    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_path = dataset.get_local_copy()

    print(f"[info] Retrieved existing pre-processed dataset: {dataset.name}")
    print(f"[info] Retrieved path at which files have been temporarily saved: {dataset_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = Path(dataset_path)
    restructured_path = dataset_path.parent / f"restructured_{timestamp}"
    restructured_path.mkdir(parents=True, exist_ok=True)
    return str(dataset_path), str(restructured_path)


@PipelineDecorator.component(return_values=["task_id"], cache=True)
def custom_mapping(dataset_path: str, restructured_path: str):
    
    from steps.reduce_classes import main as reduce_main
    
    task_id = reduce_main(
        dataset_path = dataset_path,
        restructured_path= restructured_path,
    )
    
    return str(task_id)

@PipelineDecorator.component(return_values=["restructured_path","task_id"], cache=True)
def simplify_folders(restructured_path, dataset_path: str):
    
    from steps.restructure_folders import main as folder_main
    
    task_id = folder_main(
        restructured_path=restructured_path,
        dataset_path = dataset_path,
    )
    
    return str(task_id)

@PipelineDecorator.component(return_values=["task_id"], cache=True)
def convert_to_yolo(restructure_task_id: str, reduce_task_id: str, restructured_path, dataset_path: str) -> str:
    """
    Convert JSON annotations to YOLO format text files with class mapping.
    
    Args:
        restructure_task_id: Task ID from restructure_folders step (for dependency)
        reduce_task_id: Task ID from reduce_classes step (to get class mapping)
        dataset_path: Path to the dataset
        new_dataset_id: ID of the new dataset to update
    
    Returns:
        task_id: Current task ID
    """
    from steps.label_transform import main as transform_main
    
    task_id = transform_main(
        restructure_task_id=restructure_task_id,
        reduce_task_id=reduce_task_id,
        restructured_path=restructured_path,
        dataset_path=dataset_path,
    )
    
    return str(task_id)


@PipelineDecorator.component(return_values=["split_dataset_id", "split_path"], cache=True)
def split_dataset(
    restructure_task_id: str,
    reduce_task_id: str,
    convert_task_id:str,
    restructured_path: str,
    val_ratio:float,
    ):
    from steps.split import main as split_main

    split_dataset_id, split_path = split_main(val_ratio, restructured_path)
    
    return str(split_dataset_id), str(split_path)

@PipelineDecorator.component(return_values=["task_id"], cache=False)
def train_yolov8(
    split_dataset_id : str, 
    epochs: int,
    optimizer: str,
    weights: str,
    workers: int,
    device: str
    ):
    from steps.train_model import main as train_main
    
    model_pt = train_main(split_dataset_id,epochs, optimizer, weights, workers, device)

    return str(model_pt)

@PipelineDecorator.component(return_values=["onnx_model_id"], cache=False)
def export_model(model_pt: str):

    pt_model = Model(model_id=model_pt)

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

@PipelineDecorator.component(return_values=["cleanup_status"], cache=False)
def cleanup_temp_files(export_model_id: str, dataset_path: str, restructured_path: str, split_path: str):
    """
    Clean up temporary files created during pipeline execution.
    
    Args:
        export_model_id: Model ID from export step (ensures cleanup happens after export)
        dataset_path: Path to downloaded raw dataset
        restructured_path: Path to restructured dataset
        split_path: Path to split dataset
    
    Returns:
        cleanup_status: Status message
    """
    import shutil
    from pathlib import Path
    
    paths_to_delete = [
        Path(dataset_path),
        Path(restructured_path),
        Path(split_path)
    ]
    
    deleted_count = 0
    total_size = 0
    
    for path in paths_to_delete:
        if path.exists():
            # Calculate size before deletion
            if path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                total_size += size
                shutil.rmtree(path)
                print(f"[cleanup] Deleted directory: {path} ({size / (1024**3):.2f} GB)")
                deleted_count += 1
            elif path.is_file():
                size = path.stat().st_size
                total_size += size
                path.unlink()
                print(f"[cleanup] Deleted file: {path} ({size / (1024**2):.2f} MB)")
                deleted_count += 1
    
    status = f"Cleaned up {deleted_count} items, freed {total_size / (1024**3):.2f} GB"
    print(f"[success] {status}")
    
    return status

def execute_pipeline(params):
    """
    Execute the standard training pipeline with the given parameters.

    This function contains the common pipeline logic used by all model types.
    It should be called from within a @PipelineDecorator.pipeline decorated function.

    Args:
        params: PipelineParams object containing all pipeline configuration
    """
    health_check_task_id = health_check(
        dataset_id=params.dataset_id, mismatch_threshold=params.mismatch_threshold
    )

    dataset_path, restructured_path = download_dataset(
        project_name = params.dataset_project_name,
        health_check_task_id= health_check_task_id,
        dataset_id=params.dataset_id
    )

    reduce_task_id = custom_mapping(
        dataset_path = dataset_path,
        restructured_path=restructured_path
    )

    restructure_task_id = simplify_folders(
        restructured_path=restructured_path,
        dataset_path = dataset_path,
    )

    convert_task_id = convert_to_yolo(
        restructure_task_id=restructure_task_id,
        reduce_task_id=reduce_task_id,
        restructured_path=restructured_path,
        dataset_path=dataset_path,
    )

    split_dataset_id, split_path = split_dataset(
        convert_task_id = convert_task_id,
        restructure_task_id = restructure_task_id,
        reduce_task_id = reduce_task_id,
        restructured_path= restructured_path,
        val_ratio = params.val_ratio,
    )

    model_pt = train_yolov8(
        split_dataset_id = split_dataset_id,
        epochs= params.epochs,
        optimizer=params.optimizer,
        weights=params.weights,
        workers=params.workers,
        device=params.device 
    )

    onnx_model_id = export_model(
        model_pt = model_pt
    )
    
    cleanup_temp_files(
        export_model_id=onnx_model_id,
        dataset_path=dataset_path,
        restructured_path=restructured_path,
        split_path=split_path
    )

