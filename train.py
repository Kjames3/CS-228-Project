import argparse
import sys
import os

# Add project root to path so we can import project.distillation_trainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add training directory to path to import script
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training'))

from project.distillation_trainer import DistillationTrainer
from train_yolov8_cans import prepare_combined_dataset, get_project_root
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 with Self-Guided Knowledge Distillation (Cho et al. 2022)')
    
    # Dataset arguments
    parser.add_argument('--rebuild-dataset', action='store_true', help='Force rebuild of combined dataset')
    parser.add_argument('--data', type=str, default=None, help='Path to data.yaml (optional, defaults to auto-combined)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Pre-trained model to distill from (Teacher)')
    parser.add_argument('--name', type=str, default='student', help='Name for this training run')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (Recommended: 50-100)')
    parser.add_argument('--batch', type=int, default=8, help='Batch size (Recommended: 8 for 8GB VRAM, 16 for 12GB+)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Auto-prepare combined dataset if no specific data provided OR if we want to default to the robust one
    if args.data is None or "combined_cans" in str(args.data):
        print("Preparing combined dataset from all sources (can1-5 + custom)...")
        project_root = get_project_root()
        combined_dir = prepare_combined_dataset(project_root, force_rebuild=args.rebuild_dataset)
        data_cfg = str(combined_dir / "data.yaml")
    else:
        data_cfg = args.data

    print(f"Starting Blur-Robust Training:")
    print(f"  - Dataset: {data_cfg}")
    print(f"  - Teacher Model: {args.model}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch}")
    
    try:
        trainer = DistillationTrainer(
            model_name=args.model,
            data_cfg=data_cfg,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            run_name=args.name
        )
        trainer.train() 
        print("\n✅ Training Complete!")
        
    except Exception as e:
        print(f"\n❌ Training Failed: {e}")
        print("Tip: Ensure your dataset path is correct relative to the project root.")

if __name__ == "__main__":
    main()
