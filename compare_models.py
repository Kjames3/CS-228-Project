import argparse
import torch
import time
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import sys
import os
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project.blur_augment import BlurAugment
from training.train_yolov8_cans import prepare_combined_dataset, get_project_root

def benchmark_model(model_path, data_yaml, device='cpu', apply_blur=False):
    """
    Evaluates a model on the test set defined in data_yaml.
    Calculates mAP, Precision, Recall, and Inference Latency.
    """
    print(f"\nBenchmarking: {model_path} | Blur: {apply_blur}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Warning: Standard load failed ({e}). Attempting state_dict load...")
        try:
            # Fallback: Load as state_dict into appropriate base model
            filename = str(model_path).lower()
            
            # Determine base model based on filename
            # Determine base model based on filename
            if "yolo11" in filename:
                base_name = "models/yolo11n_cans.pt"
                if not os.path.exists(base_name):
                     base_name = "yolo11n.pt"
            elif "yolo26" in filename:
                base_name = "models/yolo26n_cans.pt"
                if not os.path.exists(base_name):
                    base_name = "yolov8n.pt"
            else:
                base_name = "models/yolov8n_cans.pt"
                if not os.path.exists(base_name):
                    base_name = "yolov8n.pt"
                
            print(f"  -> Loading state_dict into base: {base_name}")
            base_model = YOLO(base_name) 
            state_dict = torch.load(model_path, map_location=device)
            base_model.model.load_state_dict(state_dict, strict=False)
            model = base_model
            print("  -> Log: Loaded state_dict successfully.")
        except Exception as e2:
            print(f"Error loading model {model_path}: {e2}")
            return None

    # Measure Latency (on dummy input)
    dummy_input = torch.rand(1, 3, 640, 640).to(device)
    model.to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input, verbose=False)
        
    # Timing
    start_time = time.time()
    for _ in range(50):
        _ = model(dummy_input, verbose=False)
    end_time = time.time()
    latency_ms = ((end_time - start_time) / 50) * 1000

    # Validation Arguments
    val_args = {
        'data': data_yaml,
        'split': 'test',
        'verbose': False,
        'device': device,
        'plots': False
    }

    # If blur is requested, we need a way to inject it. 
    # Since Ultralytics .val() uses internal dataloaders, injecting blur ON THE FLY is hard without modifying source.
    # WORKAROUND: We will assume the unseen 'test' set is clean for the "Clean" benchmark.
    # For "Blurred" benchmark, we will use the 'project/blur_augment.py' logic but practically,
    # re-generating a "blurred_test" folder is the most reliable way compatible with .val().
    
    if apply_blur:
        print("Creating temporary blurred test set...")
        # (Logic to be implemented in main to avoid re-doing it per model)
        # Assuming args.data now points to the BLURRED yaml
        pass

    results = model.val(**val_args)
    
    model_name_clean = Path(model_path).stem
    
    # Improve readability
    if "student" in model_name_clean:
         # e.g. student_yolov8n_cans_best -> Student v8 (Best)
         if "yolov8" in model_name_clean:
             model_name_clean = "Student v8 (Best)"
         elif "yolo11" in model_name_clean:
             model_name_clean = "Student v11 (Best)"
         elif "yolo26" in model_name_clean:
             model_name_clean = "Student v26 (Best)"
    elif "cans" in model_name_clean:
         # e.g. yolov8n_cans -> Teacher v8 (Distilled) or just v8 (Fine-tuned)
         model_name_clean = model_name_clean.replace("yolov8n", "v8").replace("yolo11n", "v11").replace("yolo26n", "v26").replace("_cans", " (Cans)")

    metrics = {
        'Model': model_name_clean,
        'Condition': 'Restricted Vision (Blur)' if apply_blur else 'Normal Vision (Clean)',
        'mAP@50': results.box.map50,
        'mAP@50-95': results.box.map,
        'Precision': results.box.mp,
        'Recall': results.box.mr,
        'Latency (ms)': latency_ms
    }
    
    print(f"  -> mAP@50: {metrics['mAP@50']:.4f}")
    print(f"  -> Latency: {metrics['Latency (ms)']:.2f} ms")
    return metrics

def create_blurred_dataset(clean_yaml_path, output_dir_name="combined_cans_blurred"):
    """
    Creates a temporary dataset where all test images are blurred.
    """
    import yaml
    
    with open(clean_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    root_path = Path(clean_yaml_path).parent.parent # Assuming data.yaml is in a subfolder or we use relative paths correctly
    # Fix for our specific directory structure
    if 'path' in data:
         p = Path(data['path'])
         if p.is_absolute():
             base_path = p
         else:
             # Resolve relative to the yaml file location
             base_path = (Path(clean_yaml_path).parent / p).resolve()
    else:
         base_path = Path(clean_yaml_path).parent.resolve()

    test_images_path = base_path / data.get('test', 'test/images')
    
    if not test_images_path.exists():
        # Fallback for combined_cans structure
        test_images_path = base_path / "test" / "images"

    print(f"Generating Blurred Dataset from: {test_images_path}")
    
    new_root = base_path.parent / output_dir_name
    new_test_imgs = new_root / "test" / "images"
    new_test_lbls = new_root / "test" / "labels"
    
    new_test_imgs.mkdir(parents=True, exist_ok=True)
    new_test_lbls.mkdir(parents=True, exist_ok=True)
    
    # Copy labels explicitly
    src_labels = test_images_path.parent / "labels"
    if src_labels.exists():
        import shutil
        for label_file in src_labels.glob("*.txt"):
             shutil.copy(label_file, new_test_lbls / label_file.name)
        
    blurrer = BlurAugment()
    
    for img_path in test_images_path.glob("*"):
        if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']: continue
        
        # Read
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # Blur
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blurred_rgb = blurrer(img_rgb)
        blurred_bgr = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR)
        
        # Save
        cv2.imwrite(str(new_test_imgs / img_path.name), blurred_bgr)
        
    # Create new YAML
    new_yaml = data.copy()
    new_yaml['path'] = str(new_root.absolute())
    new_yaml['test'] = "test/images"
    new_yaml['train'] = "test/images" # Dummy
    new_yaml['val'] = "test/images" # Dummy
    
    new_yaml_path = new_root / "data_blurred.yaml"
    with open(new_yaml_path, 'w') as f:
        yaml.dump(new_yaml, f)
        
    return str(new_yaml_path)

def generate_report(df, output_path):
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        print("\nGenerating Interactive Dashboard...")
        
        # 1. Scatter Plot (Efficiency Frontier)
        fig_scatter = px.scatter(
            df, 
            x="Latency (ms)", 
            y="mAP@50", 
            color="Model", 
            symbol="Condition",
            hover_data=["Precision", "Recall", "mAP@50-95"],
            title="Efficiency Frontier: Accuracy vs. Latency",
            text="Model"
        )
        fig_scatter.update_traces(textposition='top center')
        
        # 2. Bar Chart (Accuracy Comparison)
        fig_bar = px.bar(
            df, 
            x="Model", 
            y="mAP@50", 
            color="Condition", 
            barmode="group",
            title="Model Accuracy by Condition",
            text="mAP@50"
        )
        fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        
        # Create HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("<html><head><title>Model Comparison Dashboard</title></head><body>")
            f.write("<h1>🚀 Model Comparison Dashboard</h1>")
            f.write(fig_scatter.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig_bar.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("<h3>Raw Data</h3>")
            f.write(df.to_html(index=False, classes='data-table'))
            f.write("</body></html>")
            
        print(f"✅ Dashboard saved to: {output_path}")
        return True
        
    except ImportError:
        print("Warning: 'plotly' not installed. Skipping dashboard generation.")
        print("Run: pip install plotly")
        return False
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        return False

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--models', nargs='+', required=True, help='Paths to .pt models to compare')
        parser.add_argument('--data', type=str, required=True, help='Path to clean data.yaml')
        parser.add_argument('--device', type=str, default='cpu')
        args = parser.parse_args()
        # 1. Expand Directory and Glob Arguments
        model_paths = []
        for path_str in args.models:
            # Handle glob patterns (e.g. project/*_best.pt)
            if "*" in path_str:
                import glob
                matched = glob.glob(path_str)
                if not matched:
                    print(f"Warning: No files matched pattern '{path_str}'")
                else:
                    sorted_matched = sorted(matched)
                    model_paths.extend(sorted_matched)
                continue

            p = Path(path_str)
            if p.is_dir():
                # Find all .pt files in the directory
                found_models = list(p.glob("*.pt"))
                if not found_models:
                    print(f"Warning: No .pt files found in directory '{path_str}'")
                else:
                    # Sort for consistency
                    found_models.sort()
                    model_paths.extend([str(m) for m in found_models])
            else:
                model_paths.append(str(p))
        
        args.models = model_paths
        
        if not args.models:
            print("Error: No models found to benchmark.")
            return

        print(f"\nModels selected for benchmarking ({len(args.models)}):")
        for m in args.models:
            print(f" - {m}")

        # 2. Setup Data
        print("Step 1: Preparing Datasets...", flush=True)
        clean_yaml = args.data
        blurred_yaml = create_blurred_dataset(clean_yaml)
        
        results_list = []

        # 3. Benchmark Loop
        print("\nStep 2: Running Benchmarks...", flush=True)
        for model_path in args.models:
            # Test on Clean
            res_clean = benchmark_model(model_path, clean_yaml, args.device, apply_blur=False)
            if res_clean: results_list.append(res_clean)
            
            # Test on Blur
            res_blur = benchmark_model(model_path, blurred_yaml, args.device, apply_blur=True)
            if res_blur: results_list.append(res_blur)

        # 4. Report
        print("\n" + "="*50, flush=True)
        print("FINAL COMPARISON REPORT", flush=True)
        print("="*50, flush=True)
        if not results_list:
            print("No results to report.")
            return
            
        df = pd.DataFrame(results_list)
        print(df.to_markdown(index=False), flush=True)
        
        csv_path = "project/model_comparison_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved detailed results to {csv_path}", flush=True)
        
        # 5. Generate Dashboard
        report_path = "project/model_comparison_report.html"
        generate_report(df, report_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL FAILURE: {e}", flush=True)

if __name__ == "__main__":
    main()
