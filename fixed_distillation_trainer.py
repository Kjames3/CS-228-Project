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
from tqdm import tqdm

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
    def __init__(self, teacher_name='models/yolov8n_cans.pt', student_name='yolov8n.pt', data_cfg='coco8.yaml', 
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
        self._init_models(teacher_name, student_name)
        
        # Loss functions - Issue C Fix (Spatial Attention Transfer)
        self.distill_loss_fn = SpatialAttentionLoss(normalize=True)
        
        # Blur augmentation
        from blur_augment import BatchedBlurAugment
        self.blur_augment = BatchedBlurAugment(blur_prob=0.8, device=self.device)
        
        # Optimizer - use lower LR to prevent weight collapse within first epoch
        # A fresh student starting from yolov8n.pt weights can overfit in one pass at lr=0.001
        self.optimizer = optim.AdamW(self.student_model.parameters(), lr=lr, weight_decay=0.005)
        # OneCycleLR: starts at lr/25, ramps up to lr over first 30% epochs, then cosine decays
        # This prevents the catastrophic det_loss→0 collapse we observed.
        # We estimate steps_per_epoch from fraction of COCO (~11829 images)
        # actual steps will be computed lazily once dataloader is ready
        self._lr_max = lr
        self.scheduler = None  # Will be initialized lazily in first training step
        
        # Data loader initialization
        self._init_dataloader()

    def _init_models(self, teacher_name, student_name):
        # Teacher (frozen) -> E.g., 'models/teachers/yolov11_teacher.pt'
        self.teacher = YOLO(teacher_name)
        self.teacher_model = self.teacher.model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Student (trainable) -> E.g., 'yolo11n.pt'
        self.student = YOLO(student_name)
        self.student_model = self.student.model.to(self.device)
        self.student_model.train()
        for param in self.student_model.parameters():
            param.requires_grad = True
        
        # Issue A Fix: Dynamic Feature Layer Detection
        self._setup_feature_extraction()
        
        # Critical Fix: Initialize Detection Loss properly
        # The Ultralytics DetectionModel needs args injected since we bypassed standard trainer
        from copy import deepcopy
        
        # Populate model.args so loss compute can read hyp configurations dynamically
        args_dict = getattr(self.student_model, 'args', None)
        if args_dict is None or isinstance(args_dict, dict):
            class ArgsWrapper:
                def __init__(self, d, device_str):
                    self.box = 7.5
                    self.cls = 0.5
                    self.dfl = 1.5
                    self.fl_gamma = 0.0
                    self.label_smoothing = 0.0
                    self.device = device_str
                    if d:
                        for k, v in d.items():
                            setattr(self, k, v)
                            
            self.student_model.args = ArgsWrapper(args_dict if isinstance(args_dict, dict) else {}, str(self.device))
        
    def _init_dataloader(self):
        """Initialize dataloader using Ultralytics utilities"""
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data import build_dataloader
        
        dataset_info = check_det_dataset(self.data_cfg)
        
        # Add cache to dataset info if enabled
        if self.cache:
            dataset_info['cache'] = 'ram' # User confirmed 60GB RAM availability on Colab Pro
            
        dataset = YOLODataset(
            dataset_info['train'], 
            imgsz=640, 
            batch_size=self.batch_size, 
            augment=True, 
            data=dataset_info, 
            classes=dataset_info['names'],
            fraction=self.fraction
        )
        # Attempt to inject cache behavior if it exists
        if self.cache:
            dataset.cache = 'ram'
            
        self.dataloader = build_dataloader(dataset, batch=self.batch_size, workers=4)
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
        # We no longer manually instantiate v8DetectionLoss
        # Ultralytics DetectionModel calculates it automatically
        # when a dictionary is passed to its forward pass!
        pass

    def _apply_blur(self, images):
        """Batched GPU blur augmentation"""
        return self.blur_augment(images)

    def train(self):
        print(colorstr("blue", "bold", "Starting Fixed Distillation Training..."))
        
        # Initialize OneCycleLR lazily now that we know steps_per_epoch
        steps_per_epoch = len(self.dataloader)
        total_steps = self.epochs * steps_per_epoch
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self._lr_max,
            total_steps=total_steps,
            pct_start=0.1,        # 10% of training is warmup
            anneal_strategy='cos',
            div_factor=25.0,      # start_lr = max_lr / 25
            final_div_factor=1e4  # end_lr = max_lr / (25 * 1e4)
        )
        
        for epoch in range(self.epochs):
            self.student_model.train()
            epoch_det_loss = 0
            epoch_distill_loss = 0
            num_batches = 0
            
            pbar = tqdm(enumerate(self.dataloader), total=steps_per_epoch,
                        desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch",
                        bar_format='{l_bar}{bar:30}{r_bar}')
            
            for batch_idx, batch in pbar:
                # Move batch to device for v8DetectionLoss
                batch = {k: v.to(self.device).float() if isinstance(v, torch.Tensor) and v.dtype == torch.float64 else v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                self.feature_storage['teacher'] = None
                self.feature_storage['student'] = None
                
                # Teacher forward (frozen)
                with torch.no_grad():
                    _ = self.teacher_model(batch['img'].float() / 255.0)
                teacher_feat = self.feature_storage['teacher']
                
                # Student forward (with blur)
                blurred_images = self._apply_blur(batch['img'].float() / 255.0)
                
                # Pass full batch dict to get native Ultralytics loss
                batch['img'] = blurred_images
                
                with torch.set_grad_enabled(True):
                    det_loss, loss_items = self.student_model(batch)
                
                student_feat = self.feature_storage['student']
                
                # Reduce to scalar, normalize by batch size
                if isinstance(det_loss, (tuple, list)):
                    det_loss = sum(det_loss)
                det_loss = det_loss.sum() / batch['img'].shape[0]
                
                # Dynamic Temperature Scaling
                current_temp = 4.0 - 3.0 * (epoch / max(1, self.epochs - 1))
                
                # Distillation loss
                distill_loss = self.distill_loss_fn(student_feat, teacher_feat, temperature=current_temp)
                distill_loss = distill_loss.sum()
                
                # Total loss
                loss = distill_loss + det_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent explosions
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=10.0)
                
                self.optimizer.step()
                self.scheduler.step()  # Per-batch stepping for OneCycleLR
                
                epoch_distill_loss += distill_loss.item()
                epoch_det_loss += det_loss.item()
                num_batches += 1
                
                # Update progress bar with running averages
                avg_det = epoch_det_loss / num_batches
                avg_dist = epoch_distill_loss / num_batches
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'det': f'{avg_det:.4f}',
                    'dist': f'{avg_dist:.4f}',
                    'lr': f'{current_lr:.6f}'
                })
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.epochs} | Det Loss: {epoch_det_loss/num_batches:.4f} | Distill Loss: {epoch_distill_loss/num_batches:.4f} | LR: {current_lr:.6f}")
