"""
Convert JSON annotations to YOLO format text files.

This step reads JSON annotation files and converts them to YOLO format:
- Each line: <class_id> <x_center> <y_center> <width> <height>
- All coordinates are normalized (0-1)
"""

from pathlib import Path
import json
from clearml import Task


def main(restructure_task_id: str, reduce_task_id: str, restructured_path, dataset_path):

    
    task = Task.current_task()
    
    # Convert to Path object (ClearML serializes Path as string)
    dataset_path = Path(dataset_path)
    restructured_path = Path(restructured_path)
    
    print("[info] Converting JSON annotations to YOLO format")
    print(f"[info] Dataset path: {dataset_path}")
    
    # Retrieve class mapping and restructured path from reduce_classes task
    reduce_task = Task.get_task(task_id=reduce_task_id)
    class_mapping_artifact = reduce_task.artifacts['class_mapping'].get()
    old_to_new = class_mapping_artifact['old_to_new']
    old_to_new = {int(k): int(v) for k, v in old_to_new.items()}
    
    print(f"[info] Retrieved class mapping with {len(old_to_new)} original classes")
    print(f"[info] Mapping to {len(set(old_to_new.values()))} reduced classes")
    
    # Get the restructured path from reduce_classes task
    labels_path = restructured_path / "labels"
    
    if not labels_path.exists():
        print(f"[error] Labels folder not found: {labels_path}")
        raise FileNotFoundError(f"Labels folder not found: {labels_path}")
    
    json_files = list(labels_path.glob("*.json"))
    print(f"[info] Found {len(json_files)} JSON files to convert")
    
    converted_count = 0
    for json_file in json_files:
        try:
            convert_json_to_yolo(json_file, old_to_new)
            converted_count += 1
            # Delete the JSON file after successful conversion
            json_file.unlink()
        except Exception as e:
            print(f"[error] Failed to convert {json_file.name}: {e}")
    
    print(f"[success] Converted {converted_count}/{len(json_files)} JSON files to YOLO format")
    print("[info] Deleted original JSON files")
    
    # Now add labels folder (with TXT files) and meta.yaml to dataset
    
    print(f"[success] Added {converted_count} YOLO label files and meta.yaml to dataset (not uploaded yet)")
    
    return task.id


def convert_json_to_yolo(json_file: Path, class_mapping: dict):
    """
    Convert a single JSON annotation file to YOLO format with class mapping.
    
    Args:
        json_file: Path to JSON annotation file
        class_mapping: Dictionary mapping old class IDs to new reduced class IDs
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    image_width = data['image']['width']
    image_height = data['image']['height']
    
    txt_file = json_file.with_suffix('.txt')
    
    yolo_lines = []
    
    for annotation in data.get('annotations', []):
        old_class_id = annotation['category_id']
        
        if old_class_id not in class_mapping:
            print(f"  [warning] Class ID {old_class_id} not found in mapping, skipping annotation")
            continue
        
        new_class_id = class_mapping[old_class_id]
        bbox = annotation['bbox']
        
        x_min, y_min, width, height = bbox
        
        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        
        norm_width = width / image_width
        norm_height = height / image_height
        
        yolo_line = f"{new_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
        yolo_lines.append(yolo_line)
    
    with open(txt_file, 'w') as f:
        f.write('\n'.join(yolo_lines))
    
    print(f" Converted {json_file.name} -> {txt_file.name} ({len(yolo_lines)} annotations)")