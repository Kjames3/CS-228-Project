"""
Train Teacher Models (YOLOv8, YOLOv11, YOLOv26) for Soda Can Detection

This script trains multiple YOLO models to serve as "teachers" for knowledge distillation.
It uses a combined dataset of COCO (80 classes) + Custom Cans (1 class) to ensure
the models are robust and generalizable.

Models trained:
1. YOLOv8n (Standard)
2. YOLOv11n (State-of-the-art)
3. YOLOv26n (Custom Edge-optimized)

Usage:
    python training/train_teachers.py [--epochs 150] [--batch 16]
"""

import os
import shutil
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch


def get_project_root():
    """Get the project root directory (viam_projects, not training/)."""
    return Path(__file__).parent.parent  # Go up from training/ to viam_projects/


def prepare_combined_dataset(project_root: Path, force_rebuild: bool = False, include_coco: bool = True):
    """
    Prepare a combined dataset by merging all can datasets.
    
    If include_coco is True (Default for Teachers):
    - Remaps all can labels to class 80 (COCO has 0-79).
    - creates a data.yaml that includes COCO dataset paths + combined can paths.
    
    Supports dynamic discovery of datasets in datasets/ folder.
    Excludes: 'combined_cans_blurred', 'model_training_data' (output dir)
    """
    # New output directory to avoid overwriting source datasets
    combined_dir = project_root / "datasets" / "model_training_data"
    
    # Target class ID for cans
    # If using COCO (80 classes, 0-79), cans should be class 80
    target_class_id = 80 if include_coco else 0
    
    # Check if already prepared (and check if config matches desired mode)
    yaml_path = combined_dir / "data.yaml"
    if combined_dir.exists() and yaml_path.exists() and not force_rebuild:
        print(f"✓ Combined dataset already exists at {combined_dir}")
        return combined_dir
    
    # Clean up if rebuilding
    if combined_dir.exists():
        print("Removing existing combined dataset...")
        try:
            shutil.rmtree(combined_dir)
        except Exception as e:
            print(f"Error removing {combined_dir}: {e}")
            # Try to continue despite error (windows lock issues sometimes)
    
    # Create directory structure
    for split in ["train", "valid", "test"]:
        (combined_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (combined_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    print("Preparing combined dataset from all available datasets...")
    import random
    random.seed(42)  # For reproducibility
    
    stats = {"train": 0, "valid": 0, "test": 0}
    
    # Dynamic dataset discovery
    datasets_root = project_root / "datasets"
    all_items = list(datasets_root.iterdir())
    
    # Filter for directories that look like datasets
    datasets_to_process = []
    
    # Exclude these directories
    excluded_dirs = {
        "combined_cans",
        "combined_cans_blurred", 
        "model_training_data", 
        "__pycache__"
    }
    
    for item in all_items:
        if not item.is_dir():
            continue
        
        if item.name in excluded_dirs:
            continue
            
        # Check if it looks like a dataset (has data.yaml or train/images or just images)
        has_yaml = (item / "data.yaml").exists()
        has_train = (item / "train").exists()
        has_images = (item / "images").exists() or (item / "train" / "images").exists()
        
        if has_yaml or has_train or has_images:
            # Determine structure
            has_splits = (item / "train").exists() and (item / "valid").exists()
            datasets_to_process.append({
                "name": item.name,
                "path": item,
                "has_splits": has_splits,
                "valid_classes": None # Assuming keep all classes for now, or map all to 'can'
            })
            
    print(f"Found {len(datasets_to_process)} datasets to merge: {[d['name'] for d in datasets_to_process]}")
    
    for dataset_info in datasets_to_process:
        dataset_name = dataset_info["name"]
        dataset_dir = dataset_info["path"]
        
        print(f"\n  Processing {dataset_name}...")
        
        if dataset_info["has_splits"]:
            # Dataset has train/valid/test splits
            for split in ["train", "valid", "test"]:
                count = process_dataset_split(
                    dataset_dir, combined_dir, dataset_name, split,
                    valid_classes=dataset_info.get("valid_classes"),
                    target_class_id=target_class_id
                )
                stats[split] += count
                if count > 0:
                    print(f"    ✓ {split}: {count} images")
        else:
            # Dataset might be flat or only have train
            # Try to find images
            src_images = dataset_dir / "train" / "images"
            src_labels = dataset_dir / "train" / "labels"
            
            if not src_images.exists():
                # Try root/images
                src_images = dataset_dir / "images"
                src_labels = dataset_dir / "labels"
            
            if not src_images.exists():
                print(f"    ⚠ No images found in standard paths, skipping...")
                continue
            
            image_files = list(src_images.glob("*"))
            image_files = [f for f in image_files 
                          if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
            
            random.shuffle(image_files)
            
            n_files = len(image_files)
            n_train = int(n_files * 0.8)
            n_valid = int(n_files * 0.1)
            
            splits = {
                "train": image_files[:n_train],
                "valid": image_files[n_train:n_train + n_valid],
                "test": image_files[n_train + n_valid:]
            }
            
            for split, files in splits.items():
                dst_images = combined_dir / split / "images"
                dst_labels = combined_dir / split / "labels"
                
                for img_file in files:
                    new_name = f"{dataset_name}_{img_file.name}"
                    shutil.copy2(img_file, dst_images / new_name)
                    
                    # Handle labels
                    # Check for label in src_labels
                    label_file = src_labels / f"{img_file.stem}.txt"
                    
                    dst_label_path = dst_labels / f"{dataset_name}_{img_file.stem}.txt"
                    
                    if label_file.exists():
                        remap_label_file(
                            label_file, 
                            dst_label_path,
                            valid_classes=dataset_info.get("valid_classes"),
                            target_class_id=target_class_id
                        )
                    else:
                        # Create empty label file if no annotations
                        dst_label_path.touch()
                    
                    stats[split] += 1
                
                print(f"    ✓ {split}: {len(files)} images (auto-split)")
    
    # Check if COCO is requested and available
    if include_coco:
        from ultralytics import settings
        datasets_dir = Path(settings['datasets_dir'])
        coco_dir_v1 = datasets_dir / "coco"
        coco_dir_v2 = datasets_dir / "coco2017" # Sometimes downloaded as coco2017
        
        # Check standard locations
        coco_exists = (coco_dir_v1 / "train2017.txt").exists() or \
                      (coco_dir_v2 / "train2017.txt").exists() or \
                      (coco_dir_v1 / "images" / "train2017").exists()
                      
        if not coco_exists:
            print(f"\n{'!'*60}")
            print("WARNING: COCO dataset not found in:")
            print(f"  - {coco_dir_v1}")
            print(f"  - {coco_dir_v2}")
            print("Teacher training will proceed with CANS ONLY (1 class).")
            print("To train with COCO, please download it first or use a 'coco.yaml' run.")
            print(f"{'!'*60}\n")
            include_coco = False
            target_class_id = 0 # Revert to class 0
            
    # Create data.yaml configuration
    if include_coco:
        # Standard COCO classes (80)
        coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        
        names = coco_names + ['can'] # Class 80 is 'can'
        
        from ultralytics import settings
        
        # Try to find valid coco paths
        datasets_dir = Path(settings['datasets_dir'])
        train_path = "coco/train2017.txt"
        val_path = "coco/val2017.txt"
        
        # Fallback if txt not found but images are
        if not (datasets_dir / train_path).exists():
             if (datasets_dir / "coco" / "images" / "train2017").exists():
                 train_path = "coco/images/train2017"
                 val_path = "coco/images/val2017"

        data_yaml = {
            "path": str(settings['datasets_dir']), # Base path
            "train": [
                train_path, 
                str(combined_dir / "train" / "images")
            ],
            "val": [
                val_path, 
                str(combined_dir / "valid" / "images")
            ],
            "test": str(combined_dir / "test" / "images"), 
            
            "nc": 81,
            "names": names
        }
        
    else:
        # Standard Cans Only (fallback)
        data_yaml = {
            "path": str(combined_dir.absolute()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 1,
            "names": ["can"]
        }
    
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n{'='*50}")
    print(f"✓ Combined dataset created at {combined_dir}")
    print(f"  - Mode: {'COCO + Cans (81 classes)' if include_coco else 'Cans Only (1 class)'}")
    print(f"  - Cans Train: {stats['train']}")
    print(f"  - Cans Valid: {stats['valid']}")
    print(f"  - Cans Test: {stats['test']}")
    print(f"  - Config: {yaml_path}")
    print(f"{'='*50}")
    
    return combined_dir


def process_dataset_split(dataset_dir: Path, combined_dir: Path, 
                          dataset_name: str, split: str,
                          valid_classes: list = None,
                          target_class_id: int = 0) -> int:
    """Process a single split (train/valid/test) from a dataset."""
    src_images = dataset_dir / split / "images"
    src_labels = dataset_dir / split / "labels"
    
    if not src_images.exists():
        return 0
    
    dst_images = combined_dir / split / "images"
    dst_labels = combined_dir / split / "labels"
    
    count = 0
    for img_file in src_images.glob("*"):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            # Copy image with prefix to avoid name conflicts
            new_name = f"{dataset_name}_{img_file.name}"
            shutil.copy2(img_file, dst_images / new_name)
            
            # Copy and remap label file
            label_file = src_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                remap_label_file(
                    label_file, 
                    dst_labels / f"{dataset_name}_{img_file.stem}.txt",
                    valid_classes=valid_classes,
                    target_class_id=target_class_id
                )
            else:
                # Create empty label file if no annotations
                (dst_labels / f"{dataset_name}_{img_file.stem}.txt").touch()
            
            count += 1
    
    return count


def remap_label_file(src_path: Path, dst_path: Path, valid_classes: list = None, target_class_id: int = 0):
    """
    Remap all class indices in a YOLO label file to class target_class_id.
    """
    with open(src_path, "r") as f:
        lines = f.readlines()
    
    remapped_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            
            if valid_classes is not None and class_id not in valid_classes:
                continue
                
            parts[0] = str(target_class_id)
            remapped_lines.append(" ".join(parts) + "\n")
    
    with open(dst_path, "w") as f:
        f.writelines(remapped_lines)


def train_single_model(
    model_config: dict,
    data_yaml: Path,
    epochs: int,
    batch_size: int,
    img_size: int,
    device: str,
    project_root: Path
):
    """Train a single YOLO model."""
    print(f"\n{'='*60}")
    print(f"Training Teacher Model: {model_config['name']}")
    print(f"Architecture: {model_config['weights']}")
    print(f"Description: {model_config['desc']}")
    print(f"{'='*60}")
    
    # Resolve weights path
    weights_name = model_config['weights']
    
    # Check local training folder first for custom weights like yolo26n.pt
    training_weights = project_root / "training" / weights_name
    if training_weights.exists():
        weights_path = str(training_weights)
        print(f"Loading local weights: {weights_path}")
    else:
        # Fallback to standard ultralytics usage (will auto-download for v8/v11)
        weights_path = weights_name
        print(f"Using standard weights (or download): {weights_path}")

    model = YOLO(weights_path)
    
    # Define save directory
    project_dir = project_root / "runs" / "teachers"
    run_name = model_config['name']
    
    # Training
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,
        single_cls=False, # 81 classes
        rect=True,
        device=device,
        patience=50,
        save=True,
        plots=True
    )
    
    # Save final teacher model to models/teachers/ for easy access
    teachers_dir = project_root / "models" / "teachers"
    teachers_dir.mkdir(parents=True, exist_ok=True)
    
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    final_dest = teachers_dir / f"{model_config['name']}.pt"
    
    if best_weights.exists():
        shutil.copy2(best_weights, final_dest)
        print(f"✓ Saved teacher model to: {final_dest}")
    else:
        print(f"⚠ Could not find best.pt at {best_weights}")
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Teacher Models")
    parser.add_argument("--epochs", type=int, default=150, help="Epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device (0 or cpu)")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild dataset")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch to verify setup")
    
    args = parser.parse_args()
    project_root = get_project_root()
    
    # Always include COCO for teachers
    combined_dir = prepare_combined_dataset(
        project_root, 
        force_rebuild=args.rebuild_dataset,
        include_coco=True 
    )
    data_yaml = combined_dir / "data.yaml"
    
    # Define models to train
    models_to_train = [
        {
            "name": "yolov8_teacher", 
            "weights": "yolov8n.pt", 
            "desc": "Standard YOLOv8n Teacher"
        },
        {
            "name": "yolov11_teacher", 
            "weights": "yolo11n.pt", 
            "desc": "YOLOv11n Teacher (SOTA)"
        },
        {
            "name": "yolov26_teacher", 
            "weights": "yolo26n.pt", 
            "desc": "Custom Edge-Optimized YOLOv26n Teacher"
        }
    ]
    
    if args.dry_run:
        print("\n⚠ DRY RUN MODE: Training for 1 epoch only")
        args.epochs = 1
    
    for config in models_to_train:
        try:
            train_single_model(
                config, 
                data_yaml, 
                epochs=args.epochs,
                batch_size=args.batch,
                img_size=args.imgsz,
                device=args.device,
                project_root=project_root
            )
        except Exception as e:
            print(f"❌ Failed to train {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll teacher training completed.")

if __name__ == "__main__":
    main()
