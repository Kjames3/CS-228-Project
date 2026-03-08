from ultralytics.data.utils import check_det_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data import build_dataloader
import torch

dataset_info = check_det_dataset("datasets/model_training_data/data.yaml")
dataset = YOLODataset(dataset_info["train"], imgsz=640, batch_size=8, augment=False, data=dataset_info, task="detect", classes=dataset_info["names"])
dataloader = build_dataloader(dataset, batch=8, workers=0)

print("Starting loader...")
for i, batch in enumerate(dataloader):
    print(f"Batch {i}: bboxes size: {batch['bboxes'].shape}, cls size: {batch['cls'].shape}, batch_idx size: {batch['batch_idx'].shape}")
    if i > 2:
        break
