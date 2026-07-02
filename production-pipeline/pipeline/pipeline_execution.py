from clearml import PipelineDecorator
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from components import execute_pipeline

@PipelineDecorator.pipeline(
    name="YOLO-Training-Pipeline",
    project="Production-Pipeline",
    version="0.1",
    pipeline_execution_queue="pipeline-controller"
)
def pipeline(
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
    weights
):

    execute_pipeline(**locals())

if __name__ == "__main__":
    
    pipeline(
        dataset_id="fd0284bf6a2c4ac38194ad0468b64751",
        val_ratio=0.2,
        epochs = 1,
        imgsz=1024,
        optimizer = "SGD",
        lr0 = 0.01,
        mosaic = 1.0,
        close_mosaic = 1,
        scale = 0.5,
        translate = 0.1,
        workers = 3,
        device = None,
        weights = "yolo26n.pt"
    )