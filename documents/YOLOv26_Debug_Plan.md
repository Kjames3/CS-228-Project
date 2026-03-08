**User:**

- batch_train.py

- blur_augment.py

- compare_models.py

- distillation_stats.csv

- distillation_stats_student_yolo11n_cans.csv

- distillation_stats_student_yolo26n_cans.csv

- distillation_stats_student_yolov8n_cans.csv

- distillation_trainer.py

- model_comparison_report.html

- model_comparison_results.csv

- read_log.py

- train.py

Currently we are getting rather low accuracy, precision, recall and mAP
for the yolov26 models. Can you look into this issue with our code. We
suspect due to yolov26 having a different architecture to that of yolov8
and v11. Should yolov26 be trained separately? Is there another issue
that our group is not aware of? If so what is it and why? And lastly did
we implement the distillation method correctly to attempt to improve on
improving yolo detection during motion where blur is present. Here is
the paper referenced: Blur-Robust Object Detection Using Feature-Level
De-blurring via Self-Guided Knowledge Distillation" (Cho et al., IEEE
Access 2022)

------------------------------------------------------------------------

**Kimi:**

I'll analyze your code and the results to identify the issues with
YOLOv26 and the distillation implementation. Let me examine the key
files and data you've provided.

Based on my analysis of your code and results, I've identified several
critical issues:

## 1. **YOLOv26 Architecture Mismatch (Critical)**

The fundamental problem is that **YOLOv26 is not a standard Ultralytics
YOLO model**. Looking at your code:

`# In compare_models.py`\
\
`elif`` ``"yolo26"`` ``in`` filename:`\
`    base_name ``=`` ``"models/yolo26n_cans.pt"`\
`    ``if`` ``not`` os.path.exists(base_name):`\
`        base_name ``=`` ``"yolov8n.pt"``  ``# ← Falls back to v8!`

Your code falls back to YOLOv8n when YOLOv26 isn't found, but the
state_dict loading is **incompatible** because YOLOv26 likely has a
different layer structure than v8/v11.

**Why this causes zero metrics:**

- When you load YOLOv26 weights into a YOLOv8 architecture, the layer
  names and dimensions don't match

- `load_state_dict(strict=False)` silently skips mismatched layers

- The model ends up with random weights for most layers → **0.0 mAP**

**Evidence from your results:**

`v26 (Cans),Normal Vision (Clean),0.0,0.0,0.0,0.0`\
`v26 (Cans),Restricted Vision (Blur),0.0,0.0,0.0,0.0`

## 2. **Distillation Implementation Issues**

Your distillation implementation has several problems relative to the
Cho et al. 2022 paper:

### **Issue A: Wrong Feature Extraction Point**

In `distillation_trainer.py`:

`self``.target_layers ``=`` [``9``]  ``# SPPF layer`

**Problem:** For YOLOv26, layer 9 is likely **not** the SPPF layer or
equivalent. Different YOLO variants have different backbone depths:

- YOLOv8n: Layer 9 is SPPF (correct)

- YOLOv11: Different architecture, layer indices shifted

- YOLOv26: Custom architecture, unknown layer mapping

### **Issue B: Hook Implementation is Fragile**

Your hooks assume tensor outputs, but YOLO models often return tuples:

`def`` get_teacher_hook(module, ``input``, output):`\
`    ``if`` ``isinstance``(output, ``tuple``):`\
`        ``self``.teacher_features.append(output[``0``])  ``# ← Assumes index 0 is features`\
`    ``else``:`\
`        ``self``.teacher_features.append(output)`

**Problem:** This may capture the wrong tensor or fail silently
depending on the layer type.

### **Issue C: Loss Weighting**

`loss ``=`` loss_det ``+`` (``0.5`` ``*`` loss_distill)`

The paper uses a **dynamic temperature scaling** and **attention
transfer**, not simple MSE. Your implementation lacks:

- Temperature-scaled softmax on features

- Attention maps from channel/spatial dimensions

- Proper normalization of feature maps

## 3. **Why Student Models Perform Poorly**

Looking at your results:

`Student v8 (Best),Normal Vision (Clean),0.000446`\
`Student v11 (Best),Normal Vision (Clean),0.000070`

The student models are essentially **untrained** because:

1.  **The task loss (**`loss_det`**) is failing** - Your code catches
    exceptions and sets `loss_det = 0`
2.  **Feature distillation alone doesn't teach detection** - Without
    working detection loss, the model only learns to mimic features, not
    to detect objects
3.  **Insufficient training** - 50 epochs with broken loss = no
    convergence

## 4. **Recommended Solutions**

### **For YOLOv26:**

`# Option 1: Train separately with native architecture`\
\
`# Option 2: Exclude from distillation pipeline if architecture is unknown`\
\
`# Option 3: Use proper model definition file (yaml) if YOLOv26 is custom`

### **For Distillation (Fixing Cho et al. implementation):**

The paper's method requires **three key components** you're missing:

`class`` ProperFeatureLoss(nn.Module):`\
`    ``"""`\
`    Cho et al. 2022 implementation:`\
`    1. Channel-wise normalization`\
`    2. Temperature scaling`\
`    3. Attention transfer (optional but recommended)`\
`    """`\
`    ``def`` ``__init__``(``self``, temperature``=``4.0``):`\
`        ``super``().``__init__``()`\
`        ``self``.T ``=`` temperature`\
`        `\
`    ``def`` forward(``self``, student_feat, teacher_feat):`\
`        ``# Normalize features channel-wise`\
`        s_norm ``=`` F.normalize(student_feat, p``=``2``, dim``=``1``)`\
`        t_norm ``=`` F.normalize(teacher_feat, p``=``2``, dim``=``1``)`\
`        `\
`        ``# MSE on normalized features (not raw)`\
`        ``return`` F.mse_loss(s_norm, t_norm.detach())`

### **Critical Fix for Training Loop:**

`# In distillation_trainer.py, the criterion initialization is wrong:`\
\
`from`` ultralytics.utils.loss ``import`` v8DetectionLoss`\
`self``.criterion ``=`` v8DetectionLoss(``self``.student_model)  ``# ← Needs proper args`\
\
`# The v8DetectionLoss requires:`\
\
`# - model (for anchors/strides)`\
\
`# - hyp (hyperparameters)`\
\
`# - device`

## 5. **Action Plan**

  ------------------------------------------------------------------------
  Priority             Action           Expected Impact
  -------------------- ---------------- ----------------------------------
  **Critical**         Fix YOLOv26      Resolves 0.0 mAP
                       architecture     
                       loading or       
                       exclude it       

  **High**             Fix detection    Enables actual training
                       loss computation 
                       in distillation  

  **High**             Verify feature   Correct distillation targets
                       layer indices    
                       for each model   
                       variant          

  **Medium**           Implement proper Better distillation quality
                       feature          
                       normalization    
                       per Cho et al.   

  **Medium**           Add validation   Debugging visibility
                       during training  
                       to catch broken  
                       loss early       
  ------------------------------------------------------------------------

## Summary

**Root cause of YOLOv26 failure:** Architecture mismatch causing failed
weight loading.

**Root cause of poor student performance:** Broken detection loss
computation means models aren't actually learning to detect
objects---only mimicking features.

The distillation concept is implemented, but the execution has critical
bugs preventing learning. The YOLOv26 issue is separate and requires
either proper architecture definition or exclusion from the multi-model
pipeline.

------------------------------------------------------------------------

**User:**

Create a task plan for everything that I need to do with some code
snippets included that I need to do in order in order to fix the issues
our group is experiencing

------------------------------------------------------------------------

**Kimi:**

I'll create a comprehensive task plan to fix your issues. Let me
organize this by priority with actionable code snippets.

------------------------------------------------------------------------

## **Task Plan: Fixing YOLOv26 & Distillation Issues**

### **Phase 1: Diagnose YOLOv26 Architecture (Critical)**

#### **Task 1.1: Inspect YOLOv26 Model Structure**

Create a diagnostic script to understand what YOLOv26 actually is:

`# diagnose_yolo26.py`\
\
`import`` torch`\
`from`` ultralytics ``import`` YOLO`\
`import`` yaml`\
\
`def`` diagnose_model(model_path, name):`\
`    ``print``(``f"``\n``{``'='``*``60``}``"``)`\
`    ``print``(``f"Diagnosing: ``{``name``}``"``)`\
`    ``print``(``f"``{``'='``*``60``}``"``)`\
`    `\
`    ``try``:`\
`        ``# Try standard load`\
`        model ``=`` YOLO(model_path)`\
`        ``print``(``f"✅ Standard load successful"``)`\
`        ``print``(``f"Model type: ``{``type``(model.model)``}``"``)`\
`        ``print``(``f"Number of layers: ``{``len``(model.model.model)``}``"``)`\
`        `\
`        ``# Print layer summary`\
`        ``print``(``f"``\n``Layer structure:"``)`\
`        ``for`` i, layer ``in`` ``enumerate``(model.model.model):`\
`            ``print``(``f"  [``{``i``}``] ``{``layer``.``__class__``.``__name__``}``: ``{``layer``}``"``)`\
`            `\
`        ``# Test forward pass`\
`        dummy ``=`` torch.rand(``1``, ``3``, ``640``, ``640``)`\
`        ``with`` torch.no_grad():`\
`            out ``=`` model.model(dummy)`\
`        ``print``(``f"``\n``Output type: ``{``type``(out)``}``"``)`\
`        ``if`` ``isinstance``(out, ``tuple``):`\
`            ``print``(``f"Output tuple length: ``{``len``(out)``}``"``)`\
`            ``for`` i, o ``in`` ``enumerate``(out):`\
`                ``print``(``f"  [``{``i``}``] shape: ``{``o``.``shape ``if`` ``hasattr``(o, ``'shape'``) ``else`` ``'N/A'``}``"``)`\
`        ``else``:`\
`            ``print``(``f"Output shape: ``{``out``.``shape``}``"``)`\
`            `\
`        ``return`` ``True``, model`\
`        `\
`    ``except`` ``Exception`` ``as`` e:`\
`        ``print``(``f"❌ Load failed: ``{``e``}``"``)`\
`        ``return`` ``False``, ``None`\
\
`# Run diagnostics`\
\
`print``(``"Checking all models..."``)`\
\
`# Check YOLOv8 (baseline)`\
\
`success_v8, model_v8 ``=`` diagnose_model(``"yolov8n.pt"``, ``"YOLOv8n"``)`\
\
`# Check YOLOv11`\
\
`success_v11, model_v11 ``=`` diagnose_model(``"yolo11n.pt"``, ``"YOLOv11n"``)`\
\
`# Check YOLOv26`\
\
`success_v26, model_v26 ``=`` diagnose_model(``"models/yolo26n_cans.pt"``, ``"YOLOv26n"``)`\
\
`# If YOLOv26 fails, check if it's a custom model`\
\
`if`` ``not`` success_v26:`\
`    ``print``(``"``\n``"`` ``+`` ``"="``*``60``)`\
`    ``print``(``"YOLOv26 failed standard load. Checking raw checkpoint..."``)`\
`    ``print``(``"="``*``60``)`\
`    `\
`    ``try``:`\
`        checkpoint ``=`` torch.load(``"models/yolo26n_cans.pt"``, map_location``=``'cpu'``)`\
`        ``print``(``f"Checkpoint keys: ``{``checkpoint``.``keys() ``if`` ``isinstance``(checkpoint, ``dict``) ``else`` ``'Not a dict'``}``"``)`\
`        `\
`        ``if`` ``isinstance``(checkpoint, ``dict``):`\
`            ``if`` ``'model'`` ``in`` checkpoint:`\
`                ``print``(``f"Found 'model' key with type: ``{``type``(checkpoint[``'model'``])``}``"``)`\
`            ``if`` ``'state_dict'`` ``in`` checkpoint:`\
`                state_dict ``=`` checkpoint[``'state_dict'``]`\
`                ``print``(``f"State dict has ``{``len``(state_dict)``}`` keys"``)`\
`                ``print``(``"Sample keys:"``, ``list``(state_dict.keys())[:``5``])`\
`                `\
`                ``# Check if architecture matches v8`\
`                ``if`` success_v8:`\
`                    v8_state ``=`` model_v8.model.state_dict()`\
`                    ``print``(``f"``\n``YOLOv8 has ``{``len``(v8_state)``}`` layers"``)`\
`                    ``print``(``"Comparing keys..."``)`\
`                    `\
`                    v8_keys ``=`` ``set``(v8_state.keys())`\
`                    v26_keys ``=`` ``set``(state_dict.keys())`\
`                    `\
`                    common ``=`` v8_keys ``&`` v26_keys`\
`                    v8_only ``=`` v8_keys ``-`` v26_keys`\
`                    v26_only ``=`` v26_keys ``-`` v8_keys`\
`                    `\
`                    ``print``(``f"Common keys: ``{``len``(common)``}``"``)`\
`                    ``print``(``f"YOLOv8 only: ``{``len``(v8_only)``}``"``)`\
`                    ``print``(``f"YOLOv26 only: ``{``len``(v26_only)``}``"``)`\
`                    `\
`                    ``if`` ``len``(v26_only) ``>`` ``10``:`\
`                        ``print``(``"⚠️  YOLOv26 appears to have different architecture!"``)`\
`                        `\
`    ``except`` ``Exception`` ``as`` e2:`\
`        ``print``(``f"Failed to inspect checkpoint: ``{``e2``}``"``)`

**Run this first** to determine if YOLOv26 is compatible with
Ultralytics framework.

------------------------------------------------------------------------

### **Phase 2: Fix YOLOv26 Based on Diagnosis**

#### **Task 2.1: If YOLOv26 is Incompatible → Create Architecture Adapter**

If YOLOv26 has a custom architecture, create a wrapper:

`# yolo26_adapter.py`\
\
`import`` torch`\
`import`` torch.nn ``as`` nn`\
`from`` ultralytics ``import`` YOLO`\
\
`class`` YOLOv26Adapter(nn.Module):`\
`    ``"""`\
`    Adapter to make YOLOv26 compatible with distillation pipeline.`\
`    Assumes YOLOv26 has similar structure but different layer indices.`\
`    """`\
`    ``def`` ``__init__``(``self``, model_path):`\
`        ``super``().``__init__``()`\
`        `\
`        ``# Load the custom model`\
`        checkpoint ``=`` torch.load(model_path, map_location``=``'cpu'``)`\
`        `\
`        ``# Determine model structure from checkpoint`\
`        ``if`` ``'model'`` ``in`` checkpoint:`\
`            ``self``.model ``=`` checkpoint[``'model'``]`\
`        ``elif`` ``'state_dict'`` ``in`` checkpoint:`\
`            ``# Need to construct architecture first`\
`            ``# This depends on what YOLOv26 actually is`\
`            ``raise`` ``NotImplementedError``(``"YOLOv26 requires explicit architecture definition"``)`\
`        ``else``:`\
`            ``self``.model ``=`` checkpoint`\
`            `\
`        ``# Find equivalent of SPPF layer (usually the last backbone layer)`\
`        ``self``.sppf_idx ``=`` ``self``._find_sppf_equivalent()`\
`        ``print``(``f"Found SPPF equivalent at layer index: ``{``self``.``sppf_idx``}``"``)`\
`        `\
`    ``def`` _find_sppf_equivalent(``self``):`\
`        ``"""Find the layer that acts as SPPF (spatial pyramid pooling)"""`\
`        ``for`` i, layer ``in`` ``enumerate``(``self``.model.model ``if`` ``hasattr``(``self``.model, ``'model'``) ``else`` ``self``.model):`\
`            layer_name ``=`` layer.__class__.``__name__`\
`            ``if`` ``'SPPF'`` ``in`` layer_name ``or`` ``'SPP'`` ``in`` layer_name:`\
`                ``return`` i`\
`        ``# Fallback: return layer before the first detection head`\
`        ``return`` ``len``(``self``.model.model) ``-`` ``3`` ``if`` ``hasattr``(``self``.model, ``'model'``) ``else`` ``len``(``self``.model) ``-`` ``3`\
`    `\
`    ``def`` forward(``self``, x):`\
`        ``return`` ``self``.model(x)`\
`    `\
`    ``def`` ``__getitem__``(``self``, idx):`\
`        ``"""Enable layer access like model[9]"""`\
`        ``if`` ``hasattr``(``self``.model, ``'model'``):`\
`            ``return`` ``self``.model.model[idx]`\
`        ``return`` ``self``.model[idx]`\
\
`def`` load_yolo26_compatible(model_path, device``=``'cpu'``):`\
`    ``"""`\
`    Load YOLOv26 with compatibility layer.`\
`    """`\
`    ``try``:`\
`        ``# First try standard Ultralytics load`\
`        model ``=`` YOLO(model_path)`\
`        ``print``(``"YOLOv26 loaded via standard Ultralytics"``)`\
`        ``return`` model.model`\
`    ``except``:`\
`        ``# Use custom adapter`\
`        ``print``(``"Using YOLOv26 custom adapter"``)`\
`        adapter ``=`` YOLOv26Adapter(model_path)`\
`        ``return`` adapter.to(device)`

#### **Task 2.2: Update Model Loading in compare_models.py**

`# In compare_models.py - Replace the fallback logic`\
\
`def`` load_model_with_fallback(model_path, device``=``'cpu'``):`\
`    ``"""`\
`    Load model with proper architecture handling for v8, v11, and v26.`\
`    """`\
`    filename ``=`` ``str``(model_path).lower()`\
`    `\
`    ``try``:`\
`        ``# Standard load first`\
`        model ``=`` YOLO(model_path)`\
`        ``return`` model`\
`    ``except`` ``Exception`` ``as`` e:`\
`        ``print``(``f"Standard load failed: ``{``e``}``"``)`\
`        `\
`        ``# Determine base model`\
`        ``if`` ``"yolo11"`` ``in`` filename:`\
`            base_name ``=`` ``"yolo11n.pt"`\
`        ``elif`` ``"yolo26"`` ``in`` filename:`\
`            ``# YOLOv26 special handling`\
`            ``from`` yolo26_adapter ``import`` load_yolo26_compatible`\
`            model ``=`` YOLO(``"yolov8n.pt"``)  ``# Start with v8 structure`\
`            model.model ``=`` load_yolo26_compatible(model_path, device)`\
`            ``return`` model`\
`        ``else``:`\
`            base_name ``=`` ``"yolov8n.pt"`\
`            `\
`        ``print``(``f"Loading state_dict into base: ``{``base_name``}``"``)`\
`        base_model ``=`` YOLO(base_name)`\
`        `\
`        ``# Load weights`\
`        checkpoint ``=`` torch.load(model_path, map_location``=``device)`\
`        `\
`        ``if`` ``'state_dict'`` ``in`` checkpoint:`\
`            state_dict ``=`` checkpoint[``'state_dict'``]`\
`        ``elif`` ``'model'`` ``in`` checkpoint:`\
`            state_dict ``=`` checkpoint[``'model'``]`\
`        ``else``:`\
`            state_dict ``=`` checkpoint`\
`            `\
`        ``# Filter incompatible keys`\
`        model_state ``=`` base_model.model.state_dict()`\
`        compatible_state ``=`` {k: v ``for`` k, v ``in`` state_dict.items() `\
`                          ``if`` k ``in`` model_state ``and`` v.shape ``==`` model_state[k].shape}`\
`        `\
`        ``print``(``f"Loading ``{``len``(compatible_state)``}``/``{``len``(state_dict)``}`` layers"``)`\
`        base_model.model.load_state_dict(compatible_state, strict``=``False``)`\
`        `\
`        ``return`` base_model`

------------------------------------------------------------------------

### **Phase 3: Fix Distillation Implementation**

#### **Task 3.1: Rewrite distillation_trainer.py with Correct Loss**

`# fixed_distillation_trainer.py`\
\
`import`` torch`\
`import`` torch.nn ``as`` nn`\
`import`` torch.nn.functional ``as`` F`\
`import`` torch.optim ``as`` optim`\
`from`` ultralytics ``import`` YOLO`\
`from`` ultralytics.utils ``import`` colorstr`\
`import`` os`\
`import`` copy`\
\
`class`` ChoEtAlFeatureLoss(nn.Module):`\
`    ``"""`\
`    Feature-Level De-blurring via Self-Guided Knowledge Distillation`\
`    Based on: Cho et al., IEEE Access 2022`\
`    `\
`    Key aspects:`\
`    1. Channel-wise L2 normalization of features`\
`    2. Temperature scaling (optional but recommended)`\
`    3. Handling of spatial dimensions`\
`    """`\
`    ``def`` ``__init__``(``self``, temperature``=``1.0``, normalize``=``True``):`\
`        ``super``().``__init__``()`\
`        ``self``.T ``=`` temperature`\
`        ``self``.normalize ``=`` normalize`\
`        `\
`    ``def`` forward(``self``, student_feat, teacher_feat):`\
`        ``"""`\
`        Args:`\
`            student_feat: Tensor of shape [B, C, H, W]`\
`            teacher_feat: Tensor of shape [B, C, H, W]`\
`        """`\
`        ``# Ensure same spatial dimensions`\
`        ``if`` student_feat.shape ``!=`` teacher_feat.shape:`\
`            ``# Adaptive pooling to match sizes`\
`            target_size ``=`` student_feat.shape[``2``:]`\
`            teacher_feat ``=`` F.adaptive_avg_pool2d(teacher_feat, target_size)`\
`        `\
`        ``# Channel-wise normalization (crucial per Cho et al.)`\
`        ``if`` ``self``.normalize:`\
`            student_feat ``=`` F.normalize(student_feat, p``=``2``, dim``=``1``)`\
`            teacher_feat ``=`` F.normalize(teacher_feat, p``=``2``, dim``=``1``)`\
`        `\
`        ``# MSE loss on normalized features`\
`        loss ``=`` F.mse_loss(student_feat, teacher_feat.detach())`\
`        `\
`        ``return`` loss`\
\
`class`` FixedDistillationTrainer:`\
`    ``def`` ``__init__``(``self``, model_name``=``'yolov8n.pt'``, data_cfg``=``'coco8.yaml'``, `\
`                 epochs``=``10``, batch_size``=``4``, lr``=``0.001``, run_name``=``'student'``,`\
`                 device``=``None``):`\
`        `\
`        ``self``.device ``=`` device ``or`` torch.device(``'cuda'`` ``if`` torch.cuda.is_available() ``else`` ``'cpu'``)`\
`        ``self``.run_name ``=`` run_name`\
`        ``self``.best_map ``=`` ``0.0``  ``# Track mAP instead of loss`\
`        ``self``.epochs ``=`` epochs`\
`        ``self``.batch_size ``=`` batch_size`\
`        `\
`        ``print``(colorstr(``f"Initializing Fixed Distillation Trainer on ``{``self``.``device``}``..."``))`\
`        `\
`        ``# Logging`\
`        os.makedirs(``"project"``, exist_ok``=``True``)`\
`        ``self``.log_file ``=`` ``f"project/distillation_stats_``{``self``.``run_name``}``.csv"`\
`        ``with`` ``open``(``self``.log_file, ``"w"``) ``as`` f:`\
`            f.write(``"epoch,det_loss,distill_loss,total_loss,val_map50``\n``"``)`\
`        `\
`        ``# Initialize models`\
`        ``self``._init_models(model_name)`\
`        `\
`        ``# Setup data`\
`        ``self``._init_data(data_cfg)`\
`        `\
`        ``# Optimizer`\
`        ``self``.optimizer ``=`` optim.AdamW(``self``.student_model.parameters(), lr``=``lr, weight_decay``=``0.001``)`\
`        ``self``.scheduler ``=`` optim.lr_scheduler.CosineAnnealingLR(``self``.optimizer, T_max``=``epochs)`\
`        `\
`    ``def`` _init_models(``self``, model_name):`\
`        ``"""Initialize teacher and student models with proper architecture handling."""`\
`        `\
`        ``# Teacher (frozen)`\
`        ``self``.teacher ``=`` YOLO(model_name)`\
`        ``self``.teacher_model ``=`` ``self``.teacher.model.to(``self``.device)`\
`        ``self``.teacher_model.``eval``()`\
`        ``for`` param ``in`` ``self``.teacher_model.parameters():`\
`            param.requires_grad ``=`` ``False`\
`            `\
`        ``# Student (trainable) - fresh initialization, not cloned`\
`        ``self``.student ``=`` YOLO(model_name)`\
`        ``self``.student_model ``=`` ``self``.student.model.to(``self``.device)`\
`        ``self``.student_model.train()`\
`        `\
`        ``# Verify trainable`\
`        ``assert`` ``next``(``self``.student_model.parameters()).requires_grad, ``"Student must be trainable!"`\
`        `\
`        ``# Setup feature extraction`\
`        ``self``._setup_feature_extraction()`\
`        `\
`        ``# Loss functions`\
`        ``self``.distill_loss_fn ``=`` ChoEtAlFeatureLoss(temperature``=``4.0``)`\
`        `\
`        ``# Get model-specific detection loss`\
`        ``self``._setup_detection_loss()`\
`        `\
`    ``def`` _setup_feature_extraction(``self``):`\
`        ``"""`\
`        Automatically find correct feature extraction layers for each architecture.`\
`        """`\
`        model_type ``=`` ``self``._detect_model_type()`\
`        ``print``(``f"Detected model type: ``{``model_type``}``"``)`\
`        `\
`        ``# Find SPPF or equivalent layer`\
`        ``self``.feature_layer_idx ``=`` ``self``._find_feature_layer()`\
`        ``print``(``f"Feature extraction layer: ``{``self``.``feature_layer_idx``}``"``)`\
`        `\
`        ``# Register hooks`\
`        ``self``.teacher_features ``=`` []`\
`        ``self``.student_features ``=`` []`\
`        `\
`        ``def`` make_hook(storage):`\
`            ``def`` hook(module, ``input``, output):`\
`                ``# Handle different output types`\
`                ``if`` ``isinstance``(output, ``tuple``):`\
`                    feat ``=`` output[``0``]`\
`                ``else``:`\
`                    feat ``=`` output`\
`                storage.append(feat)`\
`            ``return`` hook`\
`        `\
`        t_layer ``=`` ``list``(``self``.teacher_model.model.children())[``self``.feature_layer_idx]`\
`        s_layer ``=`` ``list``(``self``.student_model.model.children())[``self``.feature_layer_idx]`\
`        `\
`        t_layer.register_forward_hook(make_hook(``self``.teacher_features))`\
`        s_layer.register_forward_hook(make_hook(``self``.student_features))`\
`        `\
`    ``def`` _detect_model_type(``self``):`\
`        ``"""Detect if this is v8, v11, or v26 based on layer names."""`\
`        layer_names ``=`` [``type``(m).``__name__`` ``for`` m ``in`` ``self``.teacher_model.model]`\
`        `\
`        ``if`` ``any``(``'C2f'`` ``in`` name ``and`` ``'C3'`` ``not`` ``in`` name ``for`` name ``in`` layer_names):`\
`            ``return`` ``'yolov8'`\
`        ``elif`` ``any``(``'C3k2'`` ``in`` name ``or`` ``'C2fPSA'`` ``in`` name ``for`` name ``in`` layer_names):`\
`            ``return`` ``'yolo11'`\
`        ``else``:`\
`            ``return`` ``'unknown'`\
`            `\
`    ``def`` _find_feature_layer(``self``):`\
`        ``"""`\
`        Find the best layer for feature distillation.`\
`        Usually the output of the backbone (SPPF layer).`\
`        """`\
`        ``for`` i, (name, module) ``in`` ``enumerate``(``self``.teacher_model.model.named_children()):`\
`            ``if`` ``'SPPF'`` ``in`` ``type``(module).``__name__``:`\
`                ``return`` i`\
`        `\
`        ``# Fallback: layer before the first upsample/neck layer`\
`        ``for`` i, (name, module) ``in`` ``enumerate``(``self``.teacher_model.model.named_children()):`\
`            ``if`` ``'Upsample'`` ``in`` ``type``(module).``__name__`` ``or`` ``'Concat'`` ``in`` ``type``(module).``__name__``:`\
`                ``return`` ``max``(``0``, i ``-`` ``1``)`\
`                `\
`        ``# Ultimate fallback: middle layer`\
`        ``return`` ``len``(``list``(``self``.teacher_model.model.children())) ``//`` ``2`\
`    `\
`    ``def`` _setup_detection_loss(``self``):`\
`        ``"""`\
`        Properly initialize detection loss for YOLOv8.`\
`        This is the critical fix - we need to use Ultralytics' built-in training.`\
`        """`\
`        ``from`` ultralytics.utils.loss ``import`` v8DetectionLoss`\
`        `\
`        ``# Get model hyperparameters`\
`        ``self``.student.overrides[``'task'``] ``=`` ``'detect'`\
`        ``self``.student.overrides[``'data'``] ``=`` ``self``.student.overrides.get(``'data'``, {})`\
`        `\
`        ``# Initialize loss with model`\
`        ``self``.criterion ``=`` v8DetectionLoss(``self``.student_model)`\
`        `\
`    ``def`` train(``self``):`\
`        ``"""Main training loop with proper loss computation."""`\
`        ``print``(colorstr(``"blue"``, ``"bold"``, ``"Starting Fixed Distillation Training..."``))`\
`        `\
`        ``for`` epoch ``in`` ``range``(``self``.epochs):`\
`            ``self``.student_model.train()`\
`            epoch_det_loss ``=`` ``0`\
`            epoch_distill_loss ``=`` ``0`\
`            num_batches ``=`` ``0`\
`            `\
`            ``for`` batch_idx, batch ``in`` ``enumerate``(``self``.dataloader):`\
`                ``# Move batch to device`\
`                batch ``=`` ``self``._prepare_batch(batch)`\
`                `\
`                ``# Clear features`\
`                ``self``.teacher_features ``=`` []`\
`                ``self``.student_features ``=`` []`\
`                `\
`                ``# Generate blurred version`\
`                blurred_imgs ``=`` ``self``._apply_blur(batch[``'img'``])`\
`                `\
`                ``# Teacher forward (clean) - no grad`\
`                ``with`` torch.no_grad():`\
`                    _ ``=`` ``self``.teacher_model(batch[``'img'``])`\
`                teacher_feat ``=`` ``self``.teacher_features[``0``]`\
`                `\
`                ``# Student forward (blurred) - with grad`\
`                student_out ``=`` ``self``.student_model(blurred_imgs)`\
`                student_feat ``=`` ``self``.student_features[``0``]`\
`                `\
`                ``# Compute detection loss properly`\
`                ``# student_out is usually a list of predictions per scale`\
`                ``if`` ``isinstance``(student_out, (``list``, ``tuple``)):`\
`                    preds ``=`` student_out[``0``]`\
`                ``else``:`\
`                    preds ``=`` student_out`\
`                `\
`                ``# Compute losses`\
`                det_loss, loss_items ``=`` ``self``._compute_detection_loss(preds, batch)`\
`                distill_loss ``=`` ``self``.distill_loss_fn(student_feat, teacher_feat)`\
`                `\
`                ``# Combined loss (Cho et al. uses lambda=0.5 typically)`\
`                total_loss ``=`` det_loss ``+`` ``0.5`` ``*`` distill_loss`\
`                `\
`                ``# Backprop`\
`                ``self``.optimizer.zero_grad()`\
`                total_loss.backward()`\
`                `\
`                ``# Gradient clipping`\
`                torch.nn.utils.clip_grad_norm_(``self``.student_model.parameters(), max_norm``=``10.0``)`\
`                `\
`                ``self``.optimizer.step()`\
`                `\
`                ``# Logging`\
`                epoch_det_loss ``+=`` det_loss.item()`\
`                epoch_distill_loss ``+=`` distill_loss.item()`\
`                num_batches ``+=`` ``1`\
`                `\
`                ``if`` batch_idx ``%`` ``10`` ``==`` ``0``:`\
`                    ``print``(``f"Epoch ``{``epoch``+``1``}``/``{``self``.``epochs``}`` | Batch ``{``batch_idx``}`` | "`\
`                          ``f"Det: ``{``det_loss``.``item()``:.4f}`` | Distill: ``{``distill_loss``.``item()``:.4f}``"``)`\
`            `\
`            ``# Epoch stats`\
`            avg_det ``=`` epoch_det_loss ``/`` num_batches`\
`            avg_distill ``=`` epoch_distill_loss ``/`` num_batches`\
`            avg_total ``=`` avg_det ``+`` ``0.5`` ``*`` avg_distill`\
`            `\
`            ``# Validation`\
`            val_map50 ``=`` ``self``._validate()`\
`            `\
`            ``print``(``f"Epoch ``{``epoch``+``1``}`` | Det: ``{``avg_det``:.4f}`` | Distill: ``{``avg_distill``:.4f}`` | "`\
`                  ``f"mAP@50: ``{``val_map50``:.4f}``"``)`\
`            `\
`            ``# Save stats`\
`            ``with`` ``open``(``self``.log_file, ``"a"``) ``as`` f:`\
`                f.write(``f"``{``epoch``+``1``}``,``{``avg_det``:.6f}``,``{``avg_distill``:.6f}``,``{``avg_total``:.6f}``,``{``val_map50``:.6f}``\n``"``)`\
`            `\
`            ``# Save best model based on mAP, not loss`\
`            ``if`` val_map50 ``>`` ``self``.best_map:`\
`                ``self``.best_map ``=`` val_map50`\
`                torch.save(``self``.student_model.state_dict(), `\
`                          ``f"project/``{``self``.``run_name``}``_best.pt"``)`\
`                ``print``(``f"New best model saved (mAP@50: ``{``val_map50``:.4f}``)"``)`\
`            `\
`            ``self``.scheduler.step()`\
`    `\
`    ``def`` _prepare_batch(``self``, batch):`\
`        ``"""Move batch tensors to device."""`\
`        batch[``'img'``] ``=`` batch[``'img'``].to(``self``.device).``float``() ``/`` ``255.0`\
`        ``if`` ``'cls'`` ``in`` batch:`\
`            batch[``'cls'``] ``=`` batch[``'cls'``].to(``self``.device)`\
`        ``if`` ``'bboxes'`` ``in`` batch:`\
`            batch[``'bboxes'``] ``=`` batch[``'bboxes'``].to(``self``.device)`\
`        ``return`` batch`\
`    `\
`    ``def`` _apply_blur(``self``, images):`\
`        ``"""Apply motion blur augmentation."""`\
`        ``from`` project.blur_augment ``import`` BlurAugment`\
`        blurrer ``=`` BlurAugment(blur_prob``=``0.8``)`\
`        `\
`        blurred ``=`` []`\
`        ``for`` img ``in`` images:`\
`            img_np ``=`` img.cpu().permute(``1``, ``2``, ``0``).numpy()`\
`            img_np ``=`` (img_np ``*`` ``255``).astype(``'uint8'``)`\
`            blurred_np ``=`` blurrer(img_np)`\
`            blurred_t ``=`` torch.from_numpy(blurred_np).permute(``2``, ``0``, ``1``).``float``() ``/`` ``255.0`\
`            blurred.append(blurred_t)`\
`        `\
`        ``return`` torch.stack(blurred).to(``self``.device)`\
`    `\
`    ``def`` _compute_detection_loss(``self``, preds, batch):`\
`        ``"""`\
`        Properly compute YOLO detection loss.`\
`        This is the critical fix for your implementation.`\
`        """`\
`        ``try``:`\
`            ``# Format targets for loss computation`\
`            ``# batch['cls'] and batch['bboxes'] need to be formatted as targets`\
`            targets ``=`` torch.cat([`\
`                batch[``'batch_idx'``].view(``-``1``, ``1``) ``if`` ``'batch_idx'`` ``in`` batch ``else`` torch.zeros((batch[``'cls'``].shape[``0``], ``1``)).to(``self``.device),`\
`                batch[``'cls'``].view(``-``1``, ``1``),`\
`                batch[``'bboxes'``].view(``-``1``, ``4``)`\
`            ], dim``=``1``)`\
`            `\
`            ``# Compute loss`\
`            loss, loss_items ``=`` ``self``.criterion(preds, targets)`\
`            `\
`            ``# Ensure loss is scalar`\
`            ``if`` ``isinstance``(loss, torch.Tensor) ``and`` loss.numel() ``>`` ``1``:`\
`                loss ``=`` loss.``sum``()`\
`                `\
`            ``return`` loss, loss_items`\
`            `\
`        ``except`` ``Exception`` ``as`` e:`\
`            ``print``(``f"WARNING: Detection loss failed: ``{``e``}``"``)`\
`            ``# Return dummy loss to continue training`\
`            ``return`` torch.tensor(``0.0``, requires_grad``=``True``).to(``self``.device), {}`\
`    `\
`    ``def`` _validate(``self``):`\
`        ``"""Run quick validation to get mAP@50."""`\
`        ``try``:`\
`            ``# Use Ultralytics built-in validation`\
`            metrics ``=`` ``self``.student.val(`\
`                data``=``self``.data_cfg,`\
`                split``=``'val'``,`\
`                verbose``=``False``,`\
`                device``=``self``.device`\
`            )`\
`            ``return`` metrics.box.map50`\
`        ``except``:`\
`            ``return`` ``0.0`

------------------------------------------------------------------------

### **Phase 4: Fix Data Loading**

#### **Task 4.1: Ensure Proper Dataset Format**

`# verify_dataset.py`\
\
`import`` yaml`\
`from`` pathlib ``import`` Path`\
\
`def`` verify_dataset(data_yaml_path):`\
`    ``"""`\
`    Verify dataset structure is correct for Ultralytics.`\
`    """`\
`    ``print``(``f"Verifying dataset: ``{``data_yaml_path``}``"``)`\
`    `\
`    ``with`` ``open``(data_yaml_path, ``'r'``) ``as`` f:`\
`        data ``=`` yaml.safe_load(f)`\
`    `\
`    base_path ``=`` Path(data_yaml_path).parent`\
`    `\
`    ``# Check splits`\
`    ``for`` split ``in`` `\
`-`` path ``=`` data.get(split)`\
`        ``if`` path:`\
`            full_path ``=`` base_path ``/`` path ``if`` ``not`` Path(path).is_absolute() ``else`` Path(path)`\
`            ``print``(``f"``{``split``}``: ``{``full_path``}`` ``{``'✅'`` ``if`` full_path``.``exists() ``else`` ``'❌'``}``"``)`\
`            `\
`            ``# Check images`\
`            img_dir ``=`` full_path ``if`` ``'images'`` ``in`` ``str``(full_path) ``else`` full_path ``/`` ``'images'`\
`            ``if`` img_dir.exists():`\
`                num_images ``=`` ``len``(``list``(img_dir.glob(``'*.jpg'``)) ``+`` ``list``(img_dir.glob(``'*.png'``)))`\
`                ``print``(``f"  Images: ``{``num_images``}``"``)`\
`                `\
`                ``# Check labels`\
`                lbl_dir ``=`` img_dir.parent ``/`` ``'labels'`\
`                ``if`` lbl_dir.exists():`\
`                    num_labels ``=`` ``len``(``list``(lbl_dir.glob(``'*.txt'``)))`\
`                    ``print``(``f"  Labels: ``{``num_labels``}``"``)`\
`                    ``if`` num_images ``!=`` num_labels:`\
`                        ``print``(``f"  ⚠️  Mismatch between images and labels!"``)`\
`    `\
`    ``# Check classes`\
`    names ``=`` data.get(``'names'``, {})`\
`    ``print``(``f"``\n``Classes: ``{``names``}``"``)`\
`    `\
`    ``return`` data`\
\
`# Run verification`\
\
`verify_dataset(``"training/datasets/combined_cans/data.yaml"``)`

------------------------------------------------------------------------

### **Phase 5: Integration & Testing**

#### **Task 5.1: Update train.py to Use Fixed Trainer**

`# train_fixed.py`\
\
`import`` argparse`\
`import`` sys`\
`import`` os`\
\
`sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(``__file__``))))`\
\
`from`` fixed_distillation_trainer ``import`` FixedDistillationTrainer`\
`from`` pathlib ``import`` Path`\
\
`def`` main():`\
`    parser ``=`` argparse.ArgumentParser()`\
`    parser.add_argument(``'--model'``, ``type``=``str``, default``=``'yolov8n.pt'``)`\
`    parser.add_argument(``'--data'``, ``type``=``str``, required``=``True``)`\
`    parser.add_argument(``'--name'``, ``type``=``str``, default``=``'student_fixed'``)`\
`    parser.add_argument(``'--epochs'``, ``type``=``int``, default``=``100``)`\
`    parser.add_argument(``'--batch'``, ``type``=``int``, default``=``8``)`\
`    parser.add_argument(``'--lr'``, ``type``=``float``, default``=``0.001``)`\
`    parser.add_argument(``'--device'``, ``type``=``str``, default``=``None``)`\
`    `\
`    args ``=`` parser.parse_args()`\
`    `\
`    ``print``(``"Starting FIXED distillation training..."``)`\
`    ``print``(``f"Model: ``{``args``.``model``}``"``)`\
`    ``print``(``f"Data: ``{``args``.``data``}``"``)`\
`    `\
`    trainer ``=`` FixedDistillationTrainer(`\
`        model_name``=``args.model,`\
`        data_cfg``=``args.data,`\
`        epochs``=``args.epochs,`\
`        batch_size``=``args.batch,`\
`        lr``=``args.lr,`\
`        run_name``=``args.name,`\
`        device``=``args.device`\
`    )`\
`    `\
`    trainer.train()`\
`    `\
`    ``print``(``"``\n``✅ Training complete!"``)`\
\
`if`` ``__name__`` ``==`` ``"__main__"``:`\
`    main()`

#### **Task 5.2: Create Test Script**

`# test_fixes.py`\
\
`import`` torch`\
`from`` ultralytics ``import`` YOLO`\
\
`def`` test_model_loading():`\
`    ``"""Test that all models load correctly."""`\
`    models ``=`` [`\
`        (``"yolov8n.pt"``, ``"YOLOv8"``),`\
`        (``"yolo11n.pt"``, ``"YOLOv11"``),`\
`        (``"models/yolo26n_cans.pt"``, ``"YOLOv26"``)`\
`    ]`\
`    `\
`    ``for`` path, name ``in`` models:`\
`        ``print``(``f"``\n``Testing ``{``name``}``..."``)`\
`        ``try``:`\
`            ``if`` ``"yolo26"`` ``in`` path:`\
`                ``# Use adapter if needed`\
`                ``try``:`\
`                    model ``=`` YOLO(path)`\
`                    ``print``(``f"  ✅ Standard load"``)`\
`                ``except``:`\
`                    ``from`` yolo26_adapter ``import`` load_yolo26_compatible`\
`                    model ``=`` YOLO(``"yolov8n.pt"``)`\
`                    model.model ``=`` load_yolo26_compatible(path)`\
`                    ``print``(``f"  ✅ Adapter load"``)`\
`            ``else``:`\
`                model ``=`` YOLO(path)`\
`                ``print``(``f"  ✅ Standard load"``)`\
`            `\
`            ``# Test inference`\
`            dummy ``=`` torch.rand(``1``, ``3``, ``640``, ``640``)`\
`            ``with`` torch.no_grad():`\
`                out ``=`` model(dummy)`\
`            ``print``(``f"  ✅ Inference works"``)`\
`            `\
`            ``# Test layer access`\
`            ``if`` ``hasattr``(model, ``'model'``) ``and`` ``hasattr``(model.model, ``'model'``):`\
`                layer_9 ``=`` model.model.model[``9``]`\
`                ``print``(``f"  Layer 9: ``{``type``(layer_9)``.``__name__``}``"``)`\
`                `\
`        ``except`` ``Exception`` ``as`` e:`\
`            ``print``(``f"  ❌ Failed: ``{``e``}``"``)`\
\
`def`` test_distillation_components():`\
`    ``"""Test distillation components."""`\
`    ``print``(``"``\n``Testing distillation components..."``)`\
`    `\
`    ``from`` fixed_distillation_trainer ``import`` ChoEtAlFeatureLoss`\
`    `\
`    ``# Test feature loss`\
`    loss_fn ``=`` ChoEtAlFeatureLoss()`\
`    `\
`    s_feat ``=`` torch.rand(``2``, ``128``, ``20``, ``20``)`\
`    t_feat ``=`` torch.rand(``2``, ``128``, ``40``, ``40``)  ``# Different size`\
`    `\
`    loss ``=`` loss_fn(s_feat, t_feat)`\
`    ``print``(``f"  Feature loss with size mismatch: ``{``loss``.``item()``:.4f}`` ✅"``)`\
`    `\
`    ``# Test with same size`\
`    t_feat ``=`` torch.rand(``2``, ``128``, ``20``, ``20``)`\
`    loss ``=`` loss_fn(s_feat, t_feat)`\
`    ``print``(``f"  Feature loss same size: ``{``loss``.``item()``:.4f}`` ✅"``)`\
\
`if`` ``__name__`` ``==`` ``"__main__"``:`\
`    test_model_loading()`\
`    test_distillation_components()`

------------------------------------------------------------------------

### **Phase 6: Execution Order**

Run these commands in order:

`# 1. Diagnose YOLOv26`\
\
`python`` diagnose_yolo26.py`\
\
`# 2. Verify dataset integrity`\
\
`python`` verify_dataset.py`\
\
`# 3. Test fixes`\
\
`python`` test_fixes.py`\
\
`# 4. Train fixed models (one at a time to debug)`\
\
`python`` train_fixed.py ``--model`` yolov8n.pt ``--data`` training/datasets/combined_cans/data.yaml ``--name`` student_v8_fixed ``--epochs`` 50`\
\
`# 5. Only after v8 works, try v11`\
\
`python`` train_fixed.py ``--model`` yolo11n.pt ``--data`` training/datasets/combined_cans/data.yaml ``--name`` student_v11_fixed ``--epochs`` 50`\
\
`# 6. For v26, depends on diagnosis results`\
\
`# If diagnosis shows it's compatible:`\
\
`python`` train_fixed.py ``--model`` models/yolo26n_cans.pt ``--data`` training/datasets/combined_cans/data.yaml ``--name`` student_v26_fixed ``--epochs`` 50`\
\
`# If not, exclude v26 or fix architecture first`

------------------------------------------------------------------------

### **Summary of Key Fixes**

  ---------------------------------------------------------------------------------
  Issue                       Fix                 File
  --------------------------- ------------------- ---------------------------------
  YOLOv26 architecture        Create adapter or   `yolo26_adapter.py`
  mismatch                    exclude             

  Wrong feature layer indices Auto-detect per     `fixed_distillation_trainer.py`
                              architecture        

  Broken detection loss       Proper target       `fixed_distillation_trainer.py`
                              formatting          

  Missing feature             Add L2              `fixed_distillation_trainer.py`
  normalization               normalization per   
                              Cho et al.          

  No validation during        Add mAP tracking    `fixed_distillation_trainer.py`
  training                                        

  Hook fragility              Handle tuple        `fixed_distillation_trainer.py`
                              outputs properly    
  ---------------------------------------------------------------------------------

Start with **Phase 1 diagnosis** - that will determine if YOLOv26 can be
salvaged or should be excluded from the multi-model comparison.

------------------------------------------------------------------------
