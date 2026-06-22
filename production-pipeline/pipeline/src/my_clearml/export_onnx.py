from clearml import Model, Task, OutputModel
from ultralytics import YOLO

def main(output_model_id: str):

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
        weights_filename=str(onnx_path),
        upload_uri="s3://8b56346322f98ed029a3c888fba38a69.r2.cloudflarestorage.com:443/ml-storage/models/"
    )

    return onnx_model.id