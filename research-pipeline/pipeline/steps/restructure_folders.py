"""
Restructure dataset from batch folders to flat structure with renamed files.

Pipeline step to flatten batch subfolders into a single images/ and labels/ directory.
"""

from pathlib import Path
import shutil
from clearml import Task


def main(dataset_path, restructured_path):
    
    task = Task.current_task()
    
    # Convert to Path object (ClearML serializes Path as string)
    dataset_path = Path(dataset_path)
    restructured_path = Path(restructured_path)
    
    print(f"[info] Source dataset path: {dataset_path}")
    
    # Get the restructured path from reduce_classes task
    output_path = Path(restructured_path)
    output_images = output_path / "images"
    output_labels = output_path / "labels"
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # Process images
    images_source = dataset_path / "images"
    if images_source.exists():
        print("[info] Processing images...")
        file_count = process_folder(images_source, output_images)
        print(f"[info] Processed {file_count} images")
    else:
        print(f"[warning] Images folder not found: {images_source}")
    
    # Process labels
    labels_source = dataset_path / "labels"
    if labels_source.exists():
        print("[info] Processing labels...")
        file_count = process_folder(labels_source, output_labels)
        print(f"[info] Processed {file_count} labels")
    else:
        print(f"[warning] Labels folder not found: {labels_source}")
        
    
    print("[success] Added images folder to dataset (not uploaded yet)")
    
    return restructured_path, task.id


def process_folder(source_folder: Path, output_folder: Path) -> int:
    """
    Process a folder by flattening batch subfolders.
    
    Args:
        source_folder: Source folder containing batch subfolders
        output_folder: Output folder for flattened structure
    
    Returns:
        Number of files processed
    """
    file_count = 0
    
    batch_folders = [d for d in source_folder.iterdir() if d.is_dir()]
    
    if not batch_folders:
        print(f"  [warning] No batch folders found in {source_folder}")
        return 0
    
    print(f"  Found {len(batch_folders)} batch folders")
    
    for batch_folder in sorted(batch_folders):
        batch_name = batch_folder.name
        
        for file_path in batch_folder.iterdir():
            if file_path.is_file():
                new_filename = f"{batch_name}_{file_path.name}"
                output_file = output_folder / new_filename
                
                if output_file.exists():
                    counter = 1
                    stem = file_path.stem
                    suffix = file_path.suffix
                    while output_file.exists():
                        new_filename = f"{batch_name}_{stem}_{counter}{suffix}"
                        output_file = output_folder / new_filename
                        counter += 1
                
                shutil.copy2(file_path, output_file)
                file_count += 1
    
    return file_count
