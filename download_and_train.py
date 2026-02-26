import os
from ultralytics.data.utils import check_det_dataset
from ultralytics import settings
import subprocess
import shutil

# 1. Download COCO
print(f"Ultralytics dataset dir: {settings['datasets_dir']}")
print('Starting COCO download. This may take 20+ minutes...')
check_det_dataset('coco.yaml')
print('COCO Download Complete!')

# 2. Delete corrupt model training folder
print("Deleting existing model_training_data to force rebuild...")
dir_path = os.path.join(os.getcwd(), 'datasets', 'model_training_data')
if os.path.exists(dir_path):
    shutil.rmtree(dir_path)

# 3. Start Training using GPU natively
print("Launching training script...")
subprocess.run(["python", "training/train_teachers.py", "--epochs", "150", "--batch", "16", "--device", "0"])
