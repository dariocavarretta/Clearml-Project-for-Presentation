from clearml import Dataset, Task 
from pathlib import Path

def main(dataset_id: str, mismatch_threshold: float):
    dataset = Dataset.get(dataset_id = dataset_id)
    task = Task.current_task()
    
    # check mismatches images / json 
    files = dataset.list_files()
    
    images = set()
    jsons = set()

    for f in dataset.list_files():
        p = Path(f)

        if p.suffix.lower() in {".jpg"}:
            images.add(Path(*p.parts[1:]).with_suffix(""))

        elif p.suffix.lower() == ".json":
            jsons.add(Path(*p.parts[1:]).with_suffix(""))

    missing_json = images - jsons
    missing_image = jsons - images
    actual_mismatch = ((len(missing_image) + len(missing_json)) / len(files))*100

    print("[info] Analyzed dataset's health. Here is the Summary:")
    print(f"Missing Labels: {len(missing_json)}")
    print(f"Missing Images: {len(missing_image)}")
    print(f"Mismatch ratio threshold set: {mismatch_threshold}")
    print(f"Mismatch ratio found in Dataset: {actual_mismatch}")
    
    if len(missing_image) + len(missing_json) > mismatch_threshold:
        raise ValueError ("[error] too many mismatches. Aborting ...")
    else:
        print("[success] Health check passed succesfully")
    return task.id