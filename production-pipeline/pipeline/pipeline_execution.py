from clearml import PipelineDecorator 

from components import execute_pipeline

@PipelineDecorator.pipeline(
    name="Yolov11-Bottles Detection",
    project="Bottles-Detection",
    version="0.1",
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
    PipelineDecorator.set_default_execution_queue("my_laptop-cpu-tasks")
    pipeline(
        dataset_id="45db8725246e417d8714622b54edd57a",
        val_ratio=0.2,
        epochs = 8,
        imgsz=1024,
        optimizer = "AdamW",
        lr0 = 0.01,
        mosaic = 1.0,
        close_mosaic = 1,
        scale = 0.5,
        translate = 0.1,
        workers = 3,
        device = None,
        weights = "yolo11n.pt"
    )