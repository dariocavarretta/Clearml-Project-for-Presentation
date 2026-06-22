from clearml import Dataset


dataset = Dataset.create(
    dataset_project="Production-Example/Pre-Processing",
    dataset_name="Plastic-Bottles",
    dataset_tags = ["Batch 2026-05", "pre-processing"],
    parent_datasets = ["13a160f2d63c44fbb7c952b9ac9ed78d"],
    output_uri="s3://8b56346322f98ed029a3c888fba38a69.r2.cloudflarestorage.com:443/ml-storage/datasets"
)

dataset.add_files("batch_2026-05")
dataset.upload()
dataset.finalize()