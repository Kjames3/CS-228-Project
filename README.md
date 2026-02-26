# Cross-Architecture Motion Blur Robustness via Feature-Level Knowledge Distillation

**Team Members:**
- Kamren James (M.S. Robotics, UCR) - kjame001@ucr.edu
- Thi Thi Khine (M.S. Robotics, UCR) - tkhin002@ucr.edu
- Sharmeen Kazi (M.S. Computational Data Science, UCR) - skazi013@ucr.edu

## Abstract

Mobile robots often suffer from detection degradation due to motion blur and rolling shutter artifacts, particularly when using lightweight edge processors. This project investigates whether **Feature-Level Knowledge Distillation (KD)** can consistently recover lost accuracy across different generations of object detectors. We propose a pipeline where a "Teacher" model (trained on sharp images) supervises "Student" models (exposed to synthetic motion blur) via dynamic temperature scaling and attention transfer.

We will benchmark this approach across YOLOv8n, YOLOv11n, and YOLOv26n to determine if the distillation benefits are architecture-agnostic, ultimately deploying the most robust model to a Viam Rover 2 (Raspberry Pi 5).

## 1. Problem Statement & Motivation

Autonomous robots operate in dynamic environments where rapid movement induces severe motion blur due to vibration. Standard detectors, trained on clean internet datasets (COCO), fail catastrophically under these conditions. While heavier models can handle blur, they are too slow for our 50ms latency target. We aim to transfer the "blur-invariance" of a heavy teacher to a nano-scale student without adding inference latency.

## 2. Methodology

We have implemented a custom Distillation Trainer in PyTorch that extends the standard Ultralytics training loop. Our approach consists of three key components:

1.  **Synthetic Blur Augmentation:** A custom pipeline (`BlurAugment`) applies randomized directional motion blur and Gaussian noise to training batches on-the-fly, creating a domain gap between the Teacher (who sees sharp images) and the Student (who sees blurred images).
2.  **Feature-Level Loss:** Unlike standard Logit Distillation, we calculate the MSE loss between the intermediate feature maps of the Student and Teacher. This forces the Student to learn geometric features (edges/shapes) that align with the Teacher’s "clean" perception, effectively ignoring the blur artifacts.
    *   **Reference Paper:** "Blur-Robust Object Detection Using Feature-Level Deblurring via Self-Guided Knowledge Distillation" (Cho et al., IEEE Access 2022).
3.  **Cross-Architecture Validation:** We will train and compare three distinct student architectures—YOLOv8-Nano, YOLOv11-Nano, and YOLOv26-Nano—to evaluate if newer architectures respond better to distillation.

## 3. Dataset and Evaluation

*   **Training Data:** 2017 COCO train/val annotation dataset, augmented with synthetic blur.
*   **Test Data:** A "Golden Test Set" of 600 real-world blurry images captured from the moving robot.
*   **Metrics:** mAP@50 (Accuracy) and Inference Latency (ms) on Raspberry Pi 5 / Jetson Orin Nano.

## 4. How to Run

### Requirements
Ensure you have the necessary dependencies installed:
```bash
pip install torch ultralytics opencv-python numpy pyyaml
```

### Step 1: Train Teacher Models
First, train the standard "teacher" models (YOLOv8n, YOLOv11n, YOLOv26n) on the clean, sharp datasets.
```bash
python training/train_teachers.py --epochs 150 --batch 16
```
This script automatically combines the COCO and Custom Cans datasets and saves the output models to the `models/teachers/` directory.

### Step 2: Run Distillation Training
Once the teacher models are ready, run the batch distillation training. This applies the batched GPU blur augmentation and the Spatial Attention Transfer distillation loss to train the student models.
```bash
python batch_train.py
```
This script sequentially trains the student models, evaluates their performance (mAP@50) on the Golden Test Set, and logs average inference latency metrics.

### Step 3: Fast Training via Google Colab (Optional but Recommended)
If local laptop GPUs (e.g. RTX 4060 8GB) are bottlenecking your training speed due to batch-size limits, you can easily migrate the pipeline to Google Colab to access up to 80GB VRAM (H100/A100) and dramatically increase `--batch 128` throughput:

1. **Prepare Datasets**: Run `python prep_colab_upload.py` to securely package your local `Cans` datasets without copying the massive 19GB COCO dataset.
2. **Move to Cloud**: Create a folder named `CS-228-Project` in Google Drive. Upload `custom_datasets_only.zip` and all python scripts (`training/`, `project/`, `*.py`) to this folder.
3. **Execute**: Open `Colab_Training_Pipeline.ipynb` using Google Colaboratory, switch the runtime to **A100** or **L4** (or T4), and run all cells. It will bypass Drive I/O bottlenecks by utilizing local NVMe storage and rapidly download the remaining COCO datasets natively over Google's datacenter backbone.