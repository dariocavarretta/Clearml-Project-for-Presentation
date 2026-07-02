from clearml import Dataset


dataset = Dataset.create(
    dataset_project="Production-Example/Pre-Processing",
    dataset_name="Training-Ready-Dataset",
    dataset_tags = ["all_classes", "2026-07"],
    parent_datasets = ["e1e9a5a7874c48a6b655ca15b0a971a4", "015bee009853495db508cfd4399b5a05", "c231432f8f5f431f8f0f9299cca988ba", "9aa5a093b061422fa29a60c0ca41b974"],
    output_uri="s3://8b56346322f98ed029a3c888fba38a69.r2.cloudflarestorage.com:443/ml-storage/datasets"
)

dataset.finalize()