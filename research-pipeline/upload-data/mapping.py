import json

with open("TACO-raw-annotations/data/annotations.json") as f:
    data = json.load(f)

mapping = {
    c["id"]: {
        "name": c["name"],
        "supercategory": c["supercategory"]
    }
    for c in data["categories"]
}

for k, v in mapping.items():
    print(k, v["name"], "->", v["supercategory"])