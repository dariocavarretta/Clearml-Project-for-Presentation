import json
import yaml

with open("TACO-raw-annotations/data/annotations.json", "r") as f:
    coco = json.load(f)

categories = sorted(coco["categories"], key=lambda x: x["id"])

data = {
    "path": "downloaded_data",
    "train": "images/train",
    "val": "images/val",
    "names": {i: cat["name"] for i, cat in enumerate(categories)}
}

with open("downloaded_data/data.yaml", "w") as f:
    yaml.dump(data, f, sort_keys=False)

print("Saved data.yaml")