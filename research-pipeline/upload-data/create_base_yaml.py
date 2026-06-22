import json
import yaml

with open("TACO-raw-annotations/data/annotations.json", "r") as f:
    coco = json.load(f)

categories = sorted(coco["categories"], key=lambda x: x["id"])

# Use original TACO class IDs, not re-indexed
data = {
    "path": "downloaded_data",
    "train": "images/train",
    "val": "images/val",
    "names": {cat["id"]: cat["name"] for cat in categories}
}

with open("downloaded_data/data.yaml", "w") as f:
    yaml.dump(data, f, sort_keys=False)

print(f"Saved data.yaml with {len(data['names'])} classes using original TACO IDs")
print(f"Class ID range: {min(data['names'].keys())} to {max(data['names'].keys())}")