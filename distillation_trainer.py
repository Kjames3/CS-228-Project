import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import colorstr
from project.blur_augment import BlurAugment
import copy
import os

class FeatureLoss(nn.Module):
    """
    Computes the Mean Squared Error between Teacher and Student feature maps.
    Used for Feature-Level De-blurring.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_features, teacher_features):
        loss = 0
        # Debugging structure
        print(f"DEBUG: Student Type: {type(student_features)} Len: {len(student_features)}")
        print(f"DEBUG: Teacher Type: {type(teacher_features)} Len: {len(teacher_features)}")
        
        # Iterate over all hooked layers
        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            print(f"DEBUG: Layer {i} | Student: {s_feat.shape} | Teacher: {t_feat.shape}")
            print(f"DEBUG: Student Grad: {s_feat.requires_grad} | Teacher Grad: {t_feat.requires_grad}")
            loss += self.mse(s_feat, t_feat)
        return loss

class DistillationTrainer:
    def __init__(self, model_name='yolov8n.pt', data_cfg='coco8.yaml', epochs=10, batch_size=4, lr=0.001, run_name='student'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.run_name = run_name
        self.best_loss = float('inf')
        print(colorstr(f"Initializing Distillation Trainer on {self.device} for {self.run_name}..."), flush=True)
        
        # Logging setup
        self.log_file = f"project/distillation_stats_{self.run_name}.csv"
        os.makedirs("project", exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write("epoch,total_loss\n")

        # 1. Initialize Teacher (Frozen)
        print(f"DEBUG: Loading Teacher {model_name}...", flush=True)
        self.teacher = YOLO(model_name)
        print("DEBUG: Teacher loaded. Moving to device...", flush=True)
        self.teacher_model = self.teacher.model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # 2. Initialize Student (Trainable)
        print("DEBUG: Loading Student...", flush=True)
        self.student = YOLO(model_name) 
        print("DEBUG: Student loaded. Moving to device...", flush=True)
        self.student_model = self.student.model.to(self.device)
        self.student_model.train()
        for param in self.student_model.parameters():
            param.requires_grad = True
        print(f"DEBUG: Student Params Require Grad: {next(self.student_model.parameters()).requires_grad}", flush=True)
        
        print("DEBUG: Registering hooks...", flush=True)

        # 3. Setup Hooks for Feature Extraction
        # Target usually the last layer of the backbone (SPPF) and maybe some Neck layers
        # For YOLOv8n, layer 9 is SPPF (Backbone exit).
        self.target_layers = [9] 
        self.teacher_features = []
        self.student_features = []
        self._register_hooks()

        # 4. Setup Components
        print("DEBUG: Setting up BlurAugment...", flush=True)
        self.blur_augment = BlurAugment(blur_prob=0.8) # High prob for training
        print("DEBUG: Setting up DistillLoss...", flush=True)
        self.distill_loss_fn = FeatureLoss()
        print("DEBUG: Setting up Optimizer...", flush=True)
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=lr)
        
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=lr)
        
        # This is a simplified task loss holder. 
        # In a full impl, we'd use ultralytics.utils.loss.v8DetectionLoss
        # For now, we rely on the model's internal loss computation if accessible, 
        # or we might focus purely on the feature alignment for this prototype phase.
        from ultralytics.utils.loss import v8DetectionLoss
        print("DEBUG: Initializing v8DetectionLoss...", flush=True)
        try:
            self.criterion = v8DetectionLoss(self.student_model)
            print("DEBUG: v8DetectionLoss initialized successfully.", flush=True)
        except Exception as e:
            print(f"DEBUG: v8DetectionLoss Init Failed: {e}", flush=True)
            raise e

        # 5. Data Handling
        self.batch_size = batch_size
        try:
            # We rely on specific ultralytics internal methods to build a dataloader easily
            print(f"Building dataloader for {data_cfg}...")
            
            from ultralytics.cfg import get_cfg
            
            # FIX: Ensure imgsz is present and access is via attributes (SimpleNamespace)
            overrides = self.student.overrides.copy()
            overrides['imgsz'] = 640
            overrides['mode'] = 'train'
            # Load default config with overrides
            cfg = get_cfg(overrides=overrides)
            
            # Note: This is an internal API usage, might vary by version. 
            # We use a standard construction for now.
            dataset = build_yolo_dataset(cfg, data_cfg, batch_size, data=overrides, mode='train', rect=False, stride=32)
            self.dataloader = build_dataloader(dataset, batch=batch_size, workers=0, shuffle=True, rank=-1)
        except Exception as e:
            print(colorstr("yellow", f"WARNING: Failed to build real dataloader: {e}"))
            print(colorstr("yellow", "Using DUMMY dataloader for verification/testing purposes."))
            self.dataloader = self._get_dummy_dataloader()
            
        self.epochs = epochs

    def _get_dummy_dataloader(self):
        """Generates fake data for testing the training loop."""
        import torch
        dummy_data = []
        for _ in range(5): # 5 batches
            batch = {
                'img': torch.rand(self.batch_size, 3, 640, 640) * 255,
                'cls': torch.zeros(self.batch_size, 5), # dummy classes
                'bboxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]] * self.batch_size * 5).view(self.batch_size, 5, 4),
                'batch_idx': torch.arange(self.batch_size).repeat_interleave(5)
            }
            dummy_data.append(batch)
        return dummy_data

    def set_epochs(self, epochs):
        self.epochs = epochs

    def _register_hooks(self):
        """Register forward hooks to extract features."""
        def get_teacher_hook(module, input, output):
            print(f"DEBUG: Teacher Hook Output Type: {type(output)}")
            if isinstance(output, tuple):
                 print(f"DEBUG: Teacher Tuple Len: {len(output)}")
                 self.teacher_features.append(output[0]) # usually first element is features
            else:
                 print(f"DEBUG: Teacher Tensor Shape: {output.shape}")
                 self.teacher_features.append(output)

        def get_student_hook(module, input, output):
            print(f"DEBUG: Student Hook Output Type: {type(output)}")
            if isinstance(output, tuple):
                 print(f"DEBUG: Student Tuple Len: {len(output)}")
                 self.student_features.append(output[0])
            else:
                 print(f"DEBUG: Student Tensor Shape: {output.shape}")
                 self.student_features.append(output)

        # Clear existing hooks if any logic was added effectively
        # Register on specific layers by index
        # YOLOv8 model.model is a nn.Sequential-like list of modules
        for layer_idx in self.target_layers:
            t_layer = self.teacher_model.model[layer_idx]
            s_layer = self.student_model.model[layer_idx]
            
            t_layer.register_forward_hook(get_teacher_hook)
            s_layer.register_forward_hook(get_student_hook)
            print(f"DEBUG: Hooked Teacher Layer {layer_idx}: {t_layer}")
            print(f"DEBUG: Hooked Student Layer {layer_idx}: {s_layer}")
        
        print(f"Registered hooks on layers: {self.target_layers}")

    def train(self):
        print("DEBUG: Entering train() method...", flush=True)
        print(colorstr("blue", "bold", "Starting Distillation Training..."))
        
        for epoch in range(self.epochs):
            total_loss = 0
            steps = 0
            
            self.student_model.train()
            
            for batch in self.dataloader:
                # Batch structure from build_dataloader:
                # {'img': tensor, 'cls': tensor, 'bboxes': tensor, ...}
                
                clean_imgs = batch['img'].to(self.device).float() / 255.0
                print(f"DEBUG: Image Stats: Min={clean_imgs.min().item():.6f}, Max={clean_imgs.max().item():.6f}")
                
                # Reset feature containers
                self.teacher_features = []
                self.student_features = []
                
                # 1. Teacher Forward (Clean)
                with torch.no_grad():
                    # triggers hooks -> populates self.teacher_features
                    _ = self.teacher_model(clean_imgs) 
                
                # 2. Student Forward (Blurred)
                # Apply blur augmentation on CPU numpy then back to Tensor
                # (For efficiency in real training, do this in Dataset __getitem__)
                # Here we do on-the-fly for the prototype
                blurred_imgs_list = []
                for i in range(clean_imgs.shape[0]):
                    img_np = clean_imgs[i].cpu().permute(1, 2, 0).numpy() # CHW -> HWC
                    img_np = (img_np * 255).astype('uint8')
                    blurred_np = self.blur_augment(img_np)
                    blurred_t = torch.from_numpy(blurred_np).permute(2, 0, 1).float() / 255.0
                    blurred_imgs_list.append(blurred_t)
                
                blurred_imgs = torch.stack(blurred_imgs_list).to(self.device)

                # Forward pass student
                # We need the full loss from the model.
                # Use the custom loss function or the internal one.
                # Ultralytics model forward returns prediction. Loss is computed separately usually.
                # For this prototype, we'll focus on the Distillation Loss + Standard Loop
                
                preds = self.student_model(blurred_imgs)
                
                # Note: To compute the detection loss properly without the Trainer wrapper is complex 
                # because inputs need formatting (Hyperparameters, anchors).
                # To simplify: We will assume we can compute loss using the v8DetectionLoss utility
                # batch needs to be preprocessed for the loss function usually.
                
                # DEBUG: Check feature shapes
                print(f"DEBUG: Teacher Features Len: {len(self.teacher_features)}")
                if self.teacher_features:
                    print(f"DEBUG: T-Feat[0] Shape: {self.teacher_features[0].shape}")
                print(f"DEBUG: Student Features Len: {len(self.student_features)}")
                if self.student_features:
                    print(f"DEBUG: S-Feat[0] Shape: {self.student_features[0].shape}")

                loss_distill = self.distill_loss_fn(self.student_features, self.teacher_features)
                print(f"DEBUG: Distillation Loss: {loss_distill.item()}")
                
                # Setup targets for loss
                # print(f"DEBUG: Preds Shape: {preds.shape}")
                
                # DEBUG: Check Preds structure for Task Loss (UNBUFFERED)
                print(f"DEBUG: Preds Type: {type(preds)}", flush=True)
                if isinstance(preds, (list, tuple)):
                     print(f"DEBUG: Preds Length: {len(preds)}", flush=True)
                     if len(preds) > 0:
                         if isinstance(preds[0], torch.Tensor):
                             print(f"DEBUG: Preds[0] Shape: {preds[0].shape}", flush=True)
                         else:
                             print(f"DEBUG: Preds[0] Type: {type(preds[0])}", flush=True)
                elif isinstance(preds, torch.Tensor):
                     print(f"DEBUG: Preds Tensor Shape: {preds.shape}", flush=True)
                
                # DIAGNOSIS: Removed try-except to expose errors
                # print(f"DEBUG: Preds info: Type={type(preds)}")
                loss_det, loss_items = self.criterion(preds, batch)
                print(f"DEBUG: Task Loss calculated: {loss_det.item()}")
                
                # Combined Loss
                # lambda weight could be tuned. 
                loss = loss_det + (0.5 * loss_distill) 
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                steps += 1
                
                if steps % 10 == 0:
                     print(f"Epoch {epoch+1}/{self.epochs} | Step {steps} | Loss: {loss.item():.4f} (Distill: {loss_distill.item():.4f})")
            
            avg_loss = total_loss / steps if steps > 0 else 0
            print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")
            
            # Log to CSV
            if self.log_file:
                 with open(self.log_file, "a") as f:
                     f.write(f"{epoch+1},{avg_loss:.6f}\n")

            # Simple Checkpoint
            if (epoch + 1) % 5 == 0:
                save_path = f"project/{self.run_name}_epoch_{epoch+1}.pt"
                torch.save(self.student_model.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")

            # Save Best Model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                best_save_path = f"project/{self.run_name}_best.pt"
                torch.save(self.student_model.state_dict(), best_save_path)
                print(colorstr("green", f"New Best Model found (Loss: {avg_loss:.4f}) -> Saved to {best_save_path}"))

if __name__ == "__main__":
    # Example usage
    # Example usage
    # try:
    trainer = DistillationTrainer(epochs=1, batch_size=2)
    print("DEBUG: Trainer initialized. Calling train()...", flush=True)
    trainer.train() 
    print("Trainer execution: Ready.")
    # except Exception as e:
    #     print(f"Setup failed: {e}")
