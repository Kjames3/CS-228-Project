import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fixed_distillation_trainer import FixedDistillationTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Teacher model path/name')
    parser.add_argument('--name', type=str, default='student_fixed', help='Output run name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu, cuda)')
    parser.add_argument('--rebuild-dataset', action='store_true', help='Force rebuild of merged dataset (placeholder)')

    args = parser.parse_args()

    print(f"Starting FIXED distillation training for {args.name}...")
    
    trainer = FixedDistillationTrainer(
        model_name=args.model,
        data_cfg=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        run_name=args.name,
        device=args.device
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
