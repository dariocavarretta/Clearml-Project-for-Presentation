from clearml import Dataset, Task
from pathlib import Path
import yaml
from collections import Counter
import json
import matplotlib.pyplot as plt


def main(dataset_path, restructured_path):
    """
    Reduce classes using material-based grouping.
    
    Groups classes by material type:
    - Plastic, Paper, Metal, Glass, Fabric, Food, Other
    """

    task = Task.current_task()
    dataset_path = Path(dataset_path)
    restructured_path = Path(restructured_path)
    
    # Convert to Path object (ClearML serializes Path as string)
    dataset_path = Path(dataset_path)
    
    yaml_files = list(dataset_path.rglob("*.yaml")) + list(dataset_path.rglob("*.yml"))
    
    if not yaml_files:
        raise FileNotFoundError("No YAML file found in the dataset")
    
    yaml_file = yaml_files[0] 
    print(f"[info] Found YAML file: {yaml_file}")
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    original_names = data.get('names', {})
    print(f"[info] Original classes: {len(original_names)}")
    print(f"[info] Classes: {list(original_names.values())}")
    
    original_frequencies = count_class_frequencies(dataset_path, original_names)
    
    # Load material-based mapping from YAML (maps old_id -> material_name)
    id_to_material = load_material_mapping()
    
    # Create mapping from old class IDs to new class IDs
    old_to_new = {}
    new_names = {}
    material_groups = {}
    
    print("\n[info] Class reduction mapping (material-based):")
    
    # Build the mapping ONLY for classes that exist in the dataset
    for old_id, class_name in original_names.items():
        if old_id in id_to_material:
            material = id_to_material[old_id]
        else:
            print(f"[warning] Class ID {old_id} ('{class_name}') not in mapping, assigning to 'Other'")
            material = 'Other'
        
        if material not in material_groups:
            new_id = len(material_groups)
            material_groups[material] = new_id
            new_names[new_id] = material
        
        old_to_new[old_id] = material_groups[material]
    
    # Print mapping
    for material, new_id in sorted(material_groups.items(), key=lambda x: x[1]):
        old_classes = [class_name for old_id, class_name in original_names.items() if old_to_new.get(old_id) == new_id]
        print(f"  {material} <- {old_classes}")
    
    print(f"\n[info] Reduced from {len(original_names)} to {len(new_names)} classes")
    
    # Calculate new frequencies
    new_frequencies = {}
    skipped_classes = []
    for old_id, count in original_frequencies.items():
        if old_id not in old_to_new:
            skipped_classes.append(old_id)
            continue
        new_id = old_to_new[old_id]
        new_frequencies[new_id] = new_frequencies.get(new_id, 0) + count
    
    if skipped_classes:
        print(f"[warning] Skipped {len(skipped_classes)} annotations with unknown class IDs: {skipped_classes}")
    
    # Create frequency charts
    create_frequency_charts(original_names, original_frequencies, new_names, new_frequencies, task)
    
    # Update YAML with new class names
    # Note: train/val paths will be set during split step
    data['names'] = new_names
    data['nc'] = len(new_names)
    
    # Create unique restructured folder with timestamp to avoid accumulation on reruns
   
    output_yaml = restructured_path / "meta.yaml"
    
    
    with open(output_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"[success] Created reduced YAML: {output_yaml}")
    
    # No need to add files here - restructure_folders will add the entire folder
    # The meta.yaml is in restructured_path which will be added by restructure_folders
    
    # Log the mapping as artifact
    task.upload_artifact('class_mapping', artifact_object={
        'old_to_new': old_to_new,
        'original_classes': original_names,
        'reduced_classes': new_names,
        'original_frequencies': original_frequencies,
        'reduced_frequencies': new_frequencies
    })
    
    print("[success] Added YAML file to dataset (not uploaded yet)")
    
    return task.id


def load_material_mapping():
    """Load material-based mapping from YAML file."""
    mapping_file = Path(__file__).parent.parent / "taco_material_mapping.yaml"
    
    with open(mapping_file, 'r') as f:
        # Load the mapping: old_class_id -> material_group
        id_to_material = yaml.safe_load(f)
    
    return id_to_material


def count_class_frequencies(dataset_path, class_names):
    """Count how many annotations exist for each class."""
    frequencies = Counter()
    
    # Find all JSON annotation files
    labels_path = dataset_path / "labels"
    if not labels_path.exists():
        print("[warning] Labels folder not found, skipping frequency count")
        return {class_id: 0 for class_id in class_names.keys()}
    
    json_files = list(labels_path.rglob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            for annotation in data.get('annotations', []):
                class_id = annotation.get('category_id')
                if class_id is not None:
                    frequencies[class_id] += 1
        except Exception as e:
            print(f"[warning] Failed to read {json_file.name}: {e}")
    
    return dict(frequencies)


def create_frequency_charts(original_names, original_freq, new_names, new_freq, task):
    """Create and upload frequency charts as artifacts."""
    
    # Chart 1: Original class frequencies
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    sorted_orig = sorted(original_freq.items(), key=lambda x: x[1], reverse=True)
    class_labels = [original_names.get(cid, f"Class {cid}") for cid, _ in sorted_orig]
    counts = [count for _, count in sorted_orig]
    
    ax1.bar(range(len(class_labels)), counts)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Annotations')
    ax1.set_title(f'Original Class Distribution ({len(original_names)} classes)')
    ax1.set_xticks(range(len(class_labels)))
    ax1.set_xticklabels(class_labels, rotation=90, ha='right')
    plt.tight_layout()
    
    task.get_logger().report_matplotlib_figure(
        title="Original Class Frequencies",
        series="Before Reduction",
        figure=fig1,
        iteration=0
    )
    plt.close(fig1)
    
    # Chart 2: Reduced class frequencies
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    sorted_new = sorted(new_freq.items(), key=lambda x: x[1], reverse=True)
    new_class_labels = [new_names.get(cid, f"Class {cid}") for cid, _ in sorted_new]
    new_counts = [count for _, count in sorted_new]
    
    ax2.bar(range(len(new_class_labels)), new_counts, color='green')
    ax2.set_xlabel('Material Group')
    ax2.set_ylabel('Number of Annotations')
    ax2.set_title(f'Reduced Class Distribution ({len(new_names)} classes)')
    ax2.set_xticks(range(len(new_class_labels)))
    ax2.set_xticklabels(new_class_labels, rotation=45, ha='right')
    plt.tight_layout()
    
    task.get_logger().report_matplotlib_figure(
        title="Post-Processing Class Frequencies",
        series="After Reduction",
        figure=fig2,
        iteration=0
    )
    plt.close(fig2)
    
    print("[success] Uploaded frequency charts to ClearML")