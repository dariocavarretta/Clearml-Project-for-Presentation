from clearml import Dataset

dataset = Dataset.create(
    dataset_project="Raw-Taco",
    dataset_name="Taco-for-Yolo",
    output_uri="s3://8b56346322f98ed029a3c888fba38a69.r2.cloudflarestorage.com:443/ml-storage/datasets/"
)

dataset.add_files("downloaded_data")
dataset.upload()
dataset.finalize()