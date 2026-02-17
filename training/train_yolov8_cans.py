"""
YOLOv26n Fine-tuning Script for Soda Can Detection

This script fine-tunes the YOLOv26n model (optimized for edge devices like
Raspberry Pi 5) to detect soda cans using Roboflow datasets. All classes 
are merged into a single "can" class.

After training, the model is exported to NCNN format for fast inference
on ARM devices.

Datasets:
- can1_dataset: 783 train + 88 valid + 44 test images (4 classes -> 1 class)
- can2_dataset: 288 train images (1 class)
- ...

Total: ~1,071 training images

Usage:
    python train_yolov8_cans.py [--epochs 100] [--batch 16] [--imgsz 640]
"""

import os
import shutil
import argparse
from pathlib import Path
import yaml


def get_project_root():
    """Get the project root directory (viam_projects, not training/)."""
    return Path(__file__).parent.parent  # Go up from training/ to viam_projects/


def prepare_combined_dataset(project_root: Path, force_rebuild: bool = False, include_coco: bool = False):
    """
    Prepare a combined dataset by merging all can datasets.
    
    If include_coco is True:
    - Remaps all can labels to class 80 (COCO has 0-79).
    - creates a data.yaml that includes COCO dataset paths + combined can paths.
    
    If include_coco is False:
    - Remaps all can labels to class 0.
    - Creates a data.yaml for just the cans.
    
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
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
            # Simple check: if we want coco, check if nc is 81
            is_coco_config = (cfg.get("nc") == 81)
            if is_coco_config == include_coco:
                print(f"✓ Combined dataset already exists at {combined_dir}")
                return combined_dir
            else:
                print(f"ℹ Existing dataset config mismatch (COCO={is_coco_config}, requested={include_coco}). Rebuilding...")
    
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
    
    # Create data.yaml configuration
    if include_coco:
        # COCO 80 classes + 1 can class
        # We need to define the dataset such that it inherits COCO or we define all 81 classes
        # Ultralytics supports training on multiple datasets by passing a list in the 'data' arg, 
        # CAUTION: 'data' arg in model.train() can take a yaml. If that yaml has paths...
        # Standard approach for mixing: Create a YAML that points to both.
        # But for COCO, we don't want to re-download/manage it manually if possible.
        # However, to mix, we usually need to download COCO.
        
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
        
        # In this custom yaml, we should allow Ultralytics to handle 'coco.yaml' if we can, 
        # or we just point to our cans and we add 'coco.yaml' to the training list?
        # Ultralytics doesn't easily support "mix coco + custom" without a single unified YAML.
        # So we will define a YAML that *includes* the COCO paths if we know them, 
        # or we will rely on the fact that if we use "coco.yaml" it downloads it.
        # BUT we want to ADD 'can'.
        # Best approach: Define our YAML with 81 classes. 
        # Point 'train' to a LIST of dataset paths: [coco_train_path, my_cans_train_path].
        # But we don't know where COCO is on the user's machine unless we check 'settings'.
        
        # Let's try to find where ultralytics puts datasets
        from ultralytics import settings
        datasets_dir = Path(settings['datasets_dir'])
        coco_dir = datasets_dir / "coco"
        
        if not coco_dir.exists():
            print("⚠ COCO dataset not found in standard path. It is required for combined training.")
            print("  Run 'yolo train data=coco.yaml epochs=1' to trigger download, or ensure it's in datasets_dir.")
            # We will fallback to a config that assumes standard coco folders relative to datasets_dir
            # or we might download it? No, that's 20GB.
        
        # We will assume COCO is or will be at ../coco relative to this yaml, OR absolute paths.
        # Ultralytics data loaders look for data relative to the yaml file, or in settings['datasets_dir'].
        
        data_yaml = {
            "path": str(settings['datasets_dir']), # Base path
            "train": [
                "coco/train2017.txt", 
                str(combined_dir / "train" / "images")
            ],
            "val": [
                "coco/val2017.txt", 
                str(combined_dir / "valid" / "images")
            ],
            # Test on cans only? Or both? Let's verify on cans.
            "test": str(combined_dir / "test" / "images"), 
            
            "nc": 81,
            "names": names
        }
        
    else:
        # Standard Cans Only
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
    """
    Process a single split (train/valid/test) from a dataset.
    
    Returns:
        Number of images processed
    """
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
    If valid_classes is provided, only keep lines with those class indices.
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    """
    with open(src_path, "r") as f:
        lines = f.readlines()
    
    remapped_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            
            # Filter classes if valid_classes is specified
            if valid_classes is not None and class_id not in valid_classes:
                continue
                
            # Replace class index with target_class_id, keep bounding box coordinates
            parts[0] = str(target_class_id)
            remapped_lines.append(" ".join(parts) + "\n")
    
    with open(dst_path, "w") as f:
        f.writelines(remapped_lines)


def train_model(
    data_yaml: Path,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    pretrained_weights: str = "yolo26n.pt",  # YOLOv26n - optimized for edge devices
    project_name: str = "can_detection",
    run_name: str = "yolo26n_cans",
    resume: bool = False,
    device: str = None
):
    """
    Train YOLOv26n model for can detection.
    
    Args:
        data_yaml: Path to dataset configuration file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        pretrained_weights: Path to pretrained weights or model name
        project_name: Project name for organizing runs
        run_name: Name for this training run
        resume: Whether to resume from last checkpoint
        device: Device to use (None for auto-detect, '0' for GPU 0, 'cpu' for CPU)
    """
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("Error: ultralytics package not found!")
        print("Install it with: pip install ultralytics")
        return None
    
    # Auto-detect CUDA and set device
    if device is None:
        if torch.cuda.is_available():
            device = "0"  # Use first GPU
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 CUDA GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            device = "cpu"
            print("⚠️  No CUDA GPU detected, using CPU (training will be slower)")
    
    print(f"\n{'='*60}")
    print("YOLOv26n Can Detection Training")
    print(f"{'='*60}")
    print(f"Dataset config: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    
    # If training with COCO, we generally want to start with standard yolov8n/s/m/l as base
    # But if we have specific edge optimized weights, we use those.
    print(f"Pretrained weights: {pretrained_weights}")
    print(f"{'='*60}\n")
    
    # Load pretrained model
    project_root = get_project_root()
    
    # Try finding weights in likely locations
    candidate_paths = [
        project_root / pretrained_weights,
        project_root / "training" / pretrained_weights,
        Path(pretrained_weights)
    ]
    
    weights_path = None
    for p in candidate_paths:
        if p.exists():
            weights_path = p
            break
    
    if weights_path and weights_path.exists():
        print(f"Loading pretrained weights from {weights_path}")
        model = YOLO(str(weights_path))
    else:
        print(f"Downloading {pretrained_weights} from Ultralytics hub...")
        model = YOLO(pretrained_weights)
    
    # Training arguments
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "project": str(project_root / "runs" / project_name),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "SGD",   # Explicitly use SGD for stability
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,  # Allow Multi-class if COCO is involved (81 classes) or Single class if just cans
        "rect": True,         # Rectangular training - 30-50% faster with minimal accuracy loss
        "cos_lr": True,
        "close_mosaic": 20,   # Disable mosaic earlier for better fine-tuning
        "resume": resume,
        "amp": True,          # Automatic mixed precision
        "patience": 50,       # Early stopping - increased for stability
        "save": True,
        "save_period": 10,    # Save checkpoint every 10 epochs (reduces I/O)
        "cache": True,        # Cache images in RAM for faster training
        "workers": 8,         # Parallel data loading (Linux)
        "freeze": 10,         # Freeze more layers (backbone) to prevent overshooting
        "plots": True,        # Generate training plots
        "device": device,     # Explicitly set device
        "lr0": 0.001,         # Lower initial learning rate for fine-tuning
        "lrf": 0.01,          # Final learning rate fraction
    }

    
    # Data augmentation settings (optimized for can detection)
    train_args.update({
        "hsv_h": 0.015,       # Hue augmentation
        "hsv_s": 0.7,         # Saturation augmentation
        "hsv_v": 0.4,         # Value augmentation
        "degrees": 10.0,      # Rotation - cans can be tilted
        "translate": 0.1,     # Translation
        "scale": 0.5,         # Scale
        "shear": 0.0,         # Shear
        "perspective": 0.0,   # Perspective
        "flipud": 0.0,        # Vertical flip (cans don't flip upside down)
        "fliplr": 0.5,        # Horizontal flip
        "mosaic": 1.0,        # Mosaic augmentation
        "mixup": 0.1,         # Mixup - helps generalization
        "copy_paste": 0.1,    # Copy-paste - excellent for object detection
    })
    
    # Start training
    print("Starting training...")
    results = model.train(**train_args)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    print(f"Last model saved to: {results.save_dir}/weights/last.pt")
    
    return model, results


def evaluate_model(model_path: Path, data_yaml: Path):
    """
    Evaluate the trained model on the test set.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found!")
        return None
    
    print(f"\n{'='*60}")
    print("Model Evaluation")
    print(f"{'='*60}")
    
    model = YOLO(str(model_path))
    
    # Validate on test set
    results = model.val(
        data=str(data_yaml),
        split="test",
        verbose=True,
        plots=True
    )
    
    print(f"\nTest Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results


def export_model(model_path: Path, formats: list = None):
    """
    Export the trained model to various formats.
    
    After export, copies the .pt model and NCNN model folder to the
    models/ directory for easy deployment to Raspberry Pi.
    
    Args:
        model_path: Path to the trained .pt model
        formats: List of export formats (default: ['ncnn', 'onnx'])
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found!")
        return None
    
    if formats is None:
        formats = ["ncnn", "onnx"]  # NCNN first for Pi, ONNX as backup
    
    print(f"\n{'='*60}")
    print("Model Export")
    print(f"{'='*60}")
    
    model = YOLO(str(model_path))
    project_root = get_project_root()
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Copy .pt model to models/ directory
    pt_dest = models_dir / "yolo26n_cans.pt"
    shutil.copy2(model_path, pt_dest)
    print(f"\n✓ Copied .pt model to: {pt_dest}")
    
    exported_paths = []
    for fmt in formats:
        print(f"\nExporting to {fmt.upper()}...")
        try:
            path = model.export(format=fmt)
            exported_paths.append(path)
            print(f"  ✓ Exported to: {path}")
            
            # Copy NCNN model folder to models/ directory
            if fmt == "ncnn":
                ncnn_src = Path(path)
                ncnn_dest = models_dir / "yolo26n_cans_ncnn_model"
                if ncnn_dest.exists():
                    shutil.rmtree(ncnn_dest)
                shutil.copytree(ncnn_src, ncnn_dest)
                print(f"  ✓ Copied NCNN model to: {ncnn_dest}")
                
        except Exception as e:
            print(f"  ✗ Failed to export to {fmt}: {e}")
    
    return exported_paths


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv26n for soda can detection"
    )
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: '0' for GPU, 'cpu' for CPU (default: auto)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    parser.add_argument(
        "--rebuild-dataset", action="store_true",
        help="Force rebuild of combined dataset"
    )
    parser.add_argument(
        "--eval-only", type=str, default=None,
        help="Only evaluate a trained model (provide path to .pt file)"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export model after training"
    )
    parser.add_argument(
        "--export-formats", type=str, nargs="+", default=["ncnn", "onnx"],
        help="Export formats (default: ncnn, onnx)"
    )
    parser.add_argument(
        "--include-coco", action="store_true",
        help="Include COCO dataset in training (80 classes + can)"
    )
    
    args = parser.parse_args()
    
    project_root = get_project_root()
    
    # Prepare combined dataset
    combined_dir = prepare_combined_dataset(
        project_root, 
        force_rebuild=args.rebuild_dataset,
        include_coco=args.include_coco
    )
    data_yaml = combined_dir / "data.yaml"
    
    # Evaluation only mode
    if args.eval_only:
        model_path = Path(args.eval_only)
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            return
        evaluate_model(model_path, data_yaml)
        return
    
    # Train the model
    model, results = train_model(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        resume=args.resume
    )
    
    if model is None:
        return
    
    # Get best model path
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    
    # Evaluate on test set
    evaluate_model(best_model_path, data_yaml)
    
    # Export if requested
    if args.export:
        export_model(best_model_path, args.export_formats)
    
    print(f"\n{'='*60}")
    print("All Done!")
    print(f"{'='*60}")
    print(f"\nTo use your trained model:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{best_model_path}')")
    print(f"  results = model.predict('image.jpg')")


if __name__ == "__main__":
    main()
