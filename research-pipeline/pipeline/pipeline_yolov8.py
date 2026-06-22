from clearml import Dataset, Task, PipelineDecorator 

from config import PipelineParams
from steps.logic import execute_pipeline

@PipelineDecorator.pipeline(
    name="Full Yolov8 Pipeline",
    project="Yolov8-Training",
    version="0.1",
)
def pipeline(
    dataset_id,
    dataset_project_name,
    mismatch_threshold,
    seed,
    val_ratio,
    epochs,
    optimizer,
    workers,
    device,
    weights
):
    params = PipelineParams(**locals())
    execute_pipeline(params)

    
if __name__ == "__main__":
    PipelineDecorator.run_locally()
    pipeline(
        dataset_id="403c688f61c944e18b76fd5b0077b8ef",
        dataset_project_name="Taco-Training",
        mismatch_threshold=0.05,
        seed=42,
        val_ratio=0.2,
        epochs = 80,
        optimizer = "SGD",
        workers = 4,
        device = None,
        weights = "yolov8s.pt"
    )