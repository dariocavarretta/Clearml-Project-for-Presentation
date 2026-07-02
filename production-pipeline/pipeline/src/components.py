from clearml import PipelineDecorator
from my_clearml.config import repo, repo_branch, working_dir, cpu_queue, gpu_queue

@PipelineDecorator.component(
    execution_queue = cpu_queue,
    return_values=["passed"],
    repo = repo,
    repo_branch = repo_branch,
    working_dir = working_dir,
    packages=["boto3==1.43.29"],
    cache=True)
def health_check(dataset_id: str)-> bool:

    from my_clearml.health_check import main as check_main
    passed = check_main(dataset_id)

    return passed


@PipelineDecorator.component(
    execution_queue = cpu_queue,
    return_values=["prepared_dataset_id"],
    repo = repo,
    repo_branch = repo_branch,
    working_dir = working_dir,
    packages=["boto3==1.43.29"],
    cache=True)
def prepare_dataset(dataset_id: str, passed: bool, val_ratio: float) -> str:

    from my_clearml.prepare_dataset import main as prepare_main
    prepared_dataset_id = prepare_main(dataset_id, val_ratio)

    return str(prepared_dataset_id)

@PipelineDecorator.component(
    execution_queue = gpu_queue,
    return_values=["model_id"],
    repo = repo,
    repo_branch = repo_branch,
    working_dir = working_dir,
    packages=["boto3==1.43.29", "ultralytics"],
    cache=False)
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
    weights)-> str:

    from my_clearml.train_model import main as train_main

    pt_model_id = train_main(
        prepared_dataset_id,
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
        weights
    )
    return str(pt_model_id)


@PipelineDecorator.component(
    execution_queue = cpu_queue,
    return_values=["onnx_file"],
    repo = repo,
    repo_branch = repo_branch,
    working_dir = working_dir,
    packages=["boto3==1.43.29", "ultralytics"],
    cache=False)
def export_model(output_model_id: str)-> str:
    from my_clearml.export_onnx import main as export_main

    onnx_model_id = export_main(output_model_id)

    return str(onnx_model_id)


@PipelineDecorator.component(
    execution_queue = cpu_queue,
    return_values=["cleaned"],
    repo = repo,
    repo_branch = repo_branch,
    working_dir = working_dir,
    packages=["boto3==1.43.29"],
    cache=False)
def cleanup_dataset(prepared_dataset_id: str, model_id: str) -> bool:
    # model_id is passed only to declare the dependency on train_model finishing first;
    # its value is not used here.
    from my_clearml.cleanup_dataset import main as cleanup_main

    cleaned = cleanup_main(prepared_dataset_id)

    return cleaned


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

    # export and cleanup run in parallel — both depend only on train finishing
    onnx_model_id = export_model(
        output_model_id=model_id
    )

    cleanup_dataset(
        prepared_dataset_id=prepared_dataset_id,
        model_id=model_id,       # signals dependency on train_model
    )

    return onnx_model_id
