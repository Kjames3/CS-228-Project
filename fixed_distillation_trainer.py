import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ultralytics import YOLO
from ultralytics.utils import colorstr
import os
import copy
from torch.utils.data import DataLoader
from ultralytics.data import build_dataloader, build_yolo_dataset

class SpatialAttentionLoss(nn.Module):
    """
    Feature-Level De-blurring via Spatial Attention Transfer
    Computes spatial attention map sum(|F_c|^2) across channels and applies MSE loss with temperature scaling.
    """
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
        
    def forward(self, student_feat, teacher_feat, temperature=4.0):
        if student_feat.shape != teacher_feat.shape:
            target_size = student_feat.shape[2:]
            teacher_feat = F.adaptive_avg_pool2d(teacher_feat, target_size)
        
        # Compute spatial attention map: sum of squared activations across channel dim
        student_attention = torch.sum(student_feat.pow(2), dim=1, keepdim=True)
        teacher_attention = torch.sum(teacher_feat.pow(2), dim=1, keepdim=True)
        
        # Apply temperature scaling
        student_attention = student_attention / temperature
        teacher_attention = teacher_attention / temperature
        
        if self.normalize:
            # Flatten spatial dims to normalize attention map
            B, C, H, W = student_attention.shape
            s_flat = student_attention.view(B, -1)
            t_flat = teacher_attention.view(B, -1)
            
            s_flat = F.normalize(s_flat, p=2, dim=1)
            t_flat = F.normalize(t_flat, p=2, dim=1)
            
            student_attention = s_flat.view(B, C, H, W)
            teacher_attention = t_flat.view(B, C, H, W)
            
        return F.mse_loss(student_attention, teacher_attention.detach())

class FixedDistillationTrainer:
    def __init__(self, model_name='yolov8n.pt', data_cfg='coco8.yaml', 
                 epochs=10, batch_size=4, lr=0.001, run_name='student',
                 device=None, fraction=0.1, cache=False):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fraction = fraction
        self.cache = cache
        self.run_name = run_name
        self.best_map = 0.0
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_cfg = data_cfg
        
        print(colorstr(f"Initializing Fixed Distillation Trainer on {self.device}..."))
        
        # Logging
        os.makedirs("project", exist_ok=True)
        self.log_file = f"project/distillation_stats_{self.run_name}.csv"
        with open(self.log_file, "w") as f:
            f.write("epoch,det_loss,distill_loss,total_loss,val_map50\n")
        
        # Initialize models
        self._init_models(model_name)
        
        # Loss functions - Issue C Fix (Spatial Attention Transfer)
        self.distill_loss_fn = SpatialAttentionLoss(normalize=True)
        
        # Blur augmentation
        from blur_augment import BatchedBlurAugment
        self.blur_augment = BatchedBlurAugment(blur_prob=0.8, device=self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(self.student_model.parameters(), lr=lr, weight_decay=0.001)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Data loader initialization
        self._init_dataloader()

    def _init_models(self, model_name):
        # Teacher (frozen)
        self.teacher = YOLO(model_name)
        self.teacher_model = self.teacher.model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Student (trainable)
        self.student = YOLO(model_name)
        self.student_model = self.student.model.to(self.device)
        self.student_model.train()
        
        # Issue A Fix: Dynamic Feature Layer Detection
        self._setup_feature_extraction()
        
        # Critical Fix: Initialize Detection Loss properly
        self._setup_detection_loss()
        
    def _init_dataloader(self):
        """Initialize dataloader using Ultralytics utilities"""
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data import build_dataloader
        
        dataset_info = check_det_dataset(self.data_cfg)
        dataset = YOLODataset(
            dataset_info['train'], 
            imgsz=640, 
            batch_size=self.batch_size, 
            augment=True, 
            data=dataset_info, 
            classes=dataset_info['names'],
            fraction=self.fraction,
            cache=self.cache
        )
        self.dataloader = build_dataloader(dataset, batch=self.batch_size, workers=0)
    def _setup_feature_extraction(self):
        self.feature_layer_idx = self._find_feature_layer()
        print(f"Feature extraction layer index: {self.feature_layer_idx}")
        
        self.feature_storage = {'teacher': None, 'student': None}
        
        def make_hook(key):
            def hook(module, input, output):
                # Issue B Fix: Handle Tuple Outputs
                if isinstance(output, tuple):
                    feat = output[0]
                else:
                    feat = output
                # Fix: Memory leak prevention by overriding instead of appending
                self.feature_storage[key] = feat
            return hook
        
        t_children = list(self.teacher_model.model.children())
        s_children = list(self.student_model.model.children())
        
        t_layer = t_children[self.feature_layer_idx]
        s_layer = s_children[self.feature_layer_idx]
        
        t_layer.register_forward_hook(make_hook('teacher'))
        s_layer.register_forward_hook(make_hook('student'))

    def _find_feature_layer(self):
        """
        Dynamically locate a suitable feature extraction layer by checking module signatures.
        Prefers Spatial Pyramid Pooling (SPP) variants which typically reside near the end 
        of the backbone before the path aggregation neck.
        """
        children = list(self.teacher_model.model.children())
        
        # Look for SPPF or SPP functionally rather than just name
        for i, module in enumerate(children):
            if hasattr(module, 'cv1') and hasattr(module, 'cv2') and hasattr(module, 'm'):
                # Many SPP/SPPF modules have internal maxpool ('m') and conv ('cv1', 'cv2')
                name = type(module).__name__
                if 'SPP' in name:
                    return i
                    
        # Fallback: Find the transition point between downsampling (stride 2) and upsampling
        for i, module in enumerate(children):
            name = type(module).__name__
            if 'Upsample' in name or 'Concat' in name:
                return max(0, i - 1)
                
        # Last resort fallback
        return len(children) // 2

    def _setup_detection_loss(self):
        from ultralytics.utils.loss import v8DetectionLoss
        self.student.overrides['task'] = 'detect'
        self.criterion = v8DetectionLoss(self.student_model)

    def _apply_blur(self, images):
        """Batched GPU blur augmentation"""
        return self.blur_augment(images)

    def train(self):
        print(colorstr("blue", "bold", "Starting Fixed Distillation Training..."))
        
        for epoch in range(self.epochs):
            self.student_model.train()
            epoch_det_loss = 0
            epoch_distill_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.dataloader):
                # Move batch to device for v8DetectionLoss
                batch = {k: v.to(self.device).float() if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                self.feature_storage['teacher'] = None
                self.feature_storage['student'] = None
                
                # Forward pass
                with torch.no_grad():
                    _ = self.teacher_model(batch['img'].float() / 255.0)
                teacher_feat = self.feature_storage['teacher']
                
                # Student forward (with blur)
                blurred_images = self._apply_blur(batch['img'].float() / 255.0)
                preds = self.student_model(blurred_images)
                student_feat = self.feature_storage['student']
                
                # Compute detection loss
                det_loss, loss_items = self.criterion(preds, batch)
                
                # Dynamic Temperature Scaling: decrease temperature as epochs progress
                # Starts at 4.0, ends at 1.0
                current_temp = 4.0 - 3.0 * (epoch / max(1, self.epochs - 1))
                
                # Compute distillation loss
                distill_loss = self.distill_loss_fn(student_feat, teacher_feat, temperature=current_temp)
                
                # Total loss
                loss = distill_loss + det_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_distill_loss += distill_loss.item()
                epoch_det_loss += det_loss.item()
                num_batches += 1
            
            print(f"Epoch {epoch+1}/{self.epochs} | Det Loss: {epoch_det_loss/num_batches:.4f} | Distill Loss: {epoch_distill_loss/num_batches:.4f}")
            self.scheduler.step()

