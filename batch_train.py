import subprocess
import time
import sys
import os

# Define the models to train
# We prioritize using the 'cans' specific models if they exist, otherwise fallback to standard.
# Based on user request order: v8 -> v11 -> v26
models_to_train = [
    {
        "name": "yolov8n_cans",
        "file": "models/yolov8n_cans.pt",
        "fallback": "yolov8n.pt",
        "force_rebuild": True  # Force merge of all datasets before first run
    },
    {
        "name": "yolo11n_cans",
        "file": "models/yolo11n_cans.pt",
        "fallback": "yolo11n.pt"
    },
    {
        "name": "yolo26n_cans",
        "file": "models/yolo26n_cans.pt",
        "fallback": "models/yolo26n_cans.pt" # 26n is likely custom, no standard fallback
    }
]

# Training configuration
EPOCHS = 150
BATCH_SIZE = 8
DATA_CONFIG = "training/datasets/combined_cans/data.yaml"

def run_training_step(model_info):
    model_name = model_info["name"]
    model_file = model_info["file"]
    
    # Check if file exists, else use fallback
    if not os.path.exists(model_file):
        print(f"Warning: {model_file} not found. Checking fallback {model_info.get('fallback')}...")
        if model_info.get("fallback") and (os.path.exists(model_info["fallback"]) or "models/" not in model_info["fallback"]):
             # Fallback exists OR it's a standard ultralytics name (like yolov8n.pt) which auto-downloads
             model_file = model_info["fallback"]
        else:
             print(f"Error: No valid model file found for {model_name}. Skipping.")
             return False

    student_name = f"student_{model_name}"
    print(f"\n{'='*60}")
    print(f"Starting Training for: {model_name}")
    print(f"Teacher: {model_file}")
    print(f"Output Name: {student_name}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, "project/train.py",
        "--model", model_file,
        "--name", student_name,
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH_SIZE),
        "--data", DATA_CONFIG
    ]

    # Force dataset rebuild for the first model to ensure 'combined_cans' 
    # definitely contains all source data (can1-5, custom).
    if model_info.get("force_rebuild", False):
        cmd.append("--rebuild-dataset")
    
    start_t = time.time()
    result = subprocess.run(cmd)
    end_t = time.time()
    
    duration = (end_t - start_t) / 60
    if result.returncode == 0:
        print(f"\n✅ {model_name} training completed successfully in {duration:.1f} minutes.")
        print(f"   Best model saved to: project/{student_name}_best.pt")
        return True
    else:
        print(f"\n❌ {model_name} training failed.")
        return False

def main():
    print("🚀 Starting Batch Training Sequence: v8 -> v11 -> v26")
    results = {}
    
    for model_info in models_to_train:
        success = run_training_step(model_info)
        results[model_info["name"]] = "Success" if success else "Failed"
        
        # Optional: Sleep briefly between runs to let GPU cool?
        if success:
            time.sleep(10)

    print("\n" + "="*50)
    print("BATCH TRAINING SUMMARY")
    print("="*50)
    for name, status in results.items():
        print(f"{name}: {status}")

if __name__ == "__main__":
    main()
