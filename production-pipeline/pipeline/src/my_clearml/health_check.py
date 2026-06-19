from clearml import Dataset
from pathlib import Path

def main (dataset_id):
    passed = False
    dataset = Dataset.get(dataset_id = dataset_id)
    files = dataset.list_files()
    
    images = set()
    labels = set()

    for f in dataset.list_files():
        p = Path(f)

        if p.suffix.lower() in {".jpg"}:
            images.add(Path(*p.parts[1:]).with_suffix(""))

        elif p.suffix.lower() == ".txt":
            labels.add(Path(*p.parts[1:]).with_suffix(""))

    missing_labels = images - labels
    missing_image = labels - images
    actual_mismatch = ((len(missing_image) + len(missing_labels)) / len(files))*100

    print("[info] Analyzed dataset's health. Here is the Summary:")
    print(f"Missing Labels: {len(missing_labels)}")
    print(f"Missing Images: {len(missing_image)}")
    print("Mismatch ratio threshold set at 0.05")
    print(f"Mismatch ratio found in Dataset: {actual_mismatch}")
    
    if (len(missing_image) + len(missing_labels)) / (len(files) - 1) > 0.05:
        raise ValueError ("[error] too many mismatches. Aborting ...")
    else:
        print("[success] Health check passed succesfully")
        passed = True
    return passed