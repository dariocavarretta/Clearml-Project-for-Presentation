from clearml import Task, OutputModel

task = Task.init(
    project_name="Serving",
    task_name="Upload ONNX"
)

model = OutputModel(
    task=task,
    name="YOLOv8-yolov8s-onnx"
)

model.update_weights(
    weights_filename="runs/detect/train-4/weights/best.onnx",
    upload_uri=(
        "s3://8b56346322f98ed029a3c888fba38a69.r2.cloudflarestorage.com:443/"
        "ml-storage/models/"
    )
)

print(model.id)