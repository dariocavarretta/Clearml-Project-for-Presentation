from clearml import Dataset


dataset = Dataset.create(
    dataset_project="Project-Litter",
    dataset_name="Plastic-Bottles",
    dataset_tags = ["Batch 2025-09", "pre-processed"],
    output_uri="s3://8b56346322f98ed029a3c888fba38a69.r2.cloudflarestorage.com:443/ml-storage/datasets/"
)

dataset.add_files("batch_2025-09")
dataset.upload()
dataset.finalize()