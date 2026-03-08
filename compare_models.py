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
from blur_augment import BlurAugment
from training.train_yolov8_cans import prepare_combined_dataset, get_project_root

def _remap_test_labels_for_model(src_yaml: str, model_nc: int) -> tuple:
    """
    Creates a temporary evaluation dataset whose ground-truth class IDs match
    the class schema used when the model was trained.

    Returns: (eval_yaml_path: str, extra_val_kwargs: dict)

    Supported cases
    ---------------
    nc == 3   -> test set already uses 3-class labels; no remapping needed.
    nc == 81  -> teacher trained with COCO(80) + can(80). Testset classes
                 0=bottle→39, 1=can→80, 2=cup→41.
                 val kwargs: classes=[39,80,41] (only those 3 are evaluated).
    nc == 1   -> single-class model; remap all testset classes to 0.
    other     -> fall back to single_cls=True so at least bbox overlap is scored.
    """
    import yaml, shutil, tempfile

    src_yaml = Path(src_yaml)
    with open(src_yaml, 'r') as f:
        data = yaml.safe_load(f)

    # Resolve test images path — always relative to yaml file, handles '../test/images' correctly
    rel_test = data.get('test', 'test/images')
    rel_test_p = Path(rel_test)
    if rel_test_p.is_absolute():
        test_imgs = rel_test_p.resolve()
    else:
        test_imgs = (src_yaml.parent / rel_test).resolve()

    # Labels: Ultralytics convention = replace 'images' segment with 'labels'
    test_lbls = Path(str(test_imgs).replace(os.sep + 'images', os.sep + 'labels'))
    if not test_lbls.exists():
        test_lbls = test_imgs.parent / 'labels'

    if not test_imgs.exists():
        print(f"  -> Warning: test images not found at {test_imgs}, skipping remap")
        return str(src_yaml), {}

    # ---- Passthrough for matching nc ----
    if model_nc == 3:
        return str(src_yaml), {}

    # ---- Build class remapping table ----
    if model_nc == 81:
        # testset: 0=bottle, 1=can, 2=cup
        # COCO+custom: bottle=39, can=80, cup=41
        class_map  = {0: 39, 1: 80, 2: 41}
        nc_out     = 81
        names_out  = [str(i) for i in range(80)] + ['can']  # minimal names list
        extra_val  = {'classes': [39, 80, 41]}
    elif model_nc == 1:
        class_map  = {i: 0 for i in range(10)}
        nc_out     = 1
        names_out  = ['object']
        extra_val  = {}
    else:
        # Unknown schema – evaluate purely on box overlap
        return str(src_yaml), {'single_cls': True}

    # ---- Copy images + remapped labels to a temp dir ----
    tmp = Path(tempfile.mkdtemp(prefix=f'yolo_remap_nc{model_nc}_'))
    tmp_imgs = tmp / 'test' / 'images'
    tmp_lbls = tmp / 'test' / 'labels'
    tmp_imgs.mkdir(parents=True)
    tmp_lbls.mkdir(parents=True)

    # Copy images
    for img in test_imgs.glob('*'):
        if img.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            shutil.copy2(img, tmp_imgs / img.name)

    # Write remapped labels
    if test_lbls.exists():
        for lbl in test_lbls.glob('*.txt'):
            out_lines = []
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig_cls = int(parts[0])
                        new_cls  = class_map.get(orig_cls, orig_cls)
                        out_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")
            with open(tmp_lbls / lbl.name, 'w') as f:
                f.writelines(out_lines)

    # Write new data.yaml
    # Use forward slashes (as_posix) — Windows backslashes break YAML parsing in Ultralytics
    new_yaml_data = {
        'path': tmp.as_posix(),
        'train': tmp_imgs.as_posix(),   # absolute path avoids joining issues on Windows
        'val':   tmp_imgs.as_posix(),   # dummy
        'test':  tmp_imgs.as_posix(),   # absolute path to copied test images
        'nc':    nc_out,
        'names': names_out,
    }
    new_yaml_path = tmp / 'data_remapped.yaml'
    with open(new_yaml_path, 'w') as f:
        yaml.dump(new_yaml_data, f)

    print(f"  -> Created remapped eval dataset for nc={model_nc} at {tmp}")
    return str(new_yaml_path), extra_val


def benchmark_model(model_path, data_yaml, device='cpu', apply_blur=False, label_override=None):
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
                    base_name = "yolov26n.pt"
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
        
    if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(50):
        _ = model(dummy_input, verbose=False)
        
    if device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'):
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    latency_ms = ((end_time - start_time) / 50) * 1000

    # ---- Detect model class count and build the right eval yaml ----
    model_nc = None
    try:
        model_nc = int(model.model.nc)
        print(f"  -> Detected model nc={model_nc}")
    except Exception:
        print("  -> Could not detect model nc; using single_cls fallback")

    eval_yaml, extra_val_kwargs = _remap_test_labels_for_model(data_yaml, model_nc)

    # Validation Arguments
    val_args = {
        'data': eval_yaml,
        'split': 'test',
        'verbose': False,
        'device': device,
        'plots': False,
        **extra_val_kwargs,
    }

    results = model.val(**val_args)
    
    # Use the caller-supplied label if available, otherwise derive from filename
    if label_override:
        model_name_clean = label_override
    else:
        model_name_clean = Path(model_path).stem
        import re
        if "student" in model_name_clean:
             match = re.search(r'yolo[v]?(\d+)', model_name_clean.lower())
             if match:
                 model_name_clean = f"Student v{match.group(1)} (Best)"
             else:
                 model_name_clean = "Student (Best)"
        elif "cans" in model_name_clean:
             match = re.search(r'yolo[v]?(\d+)', model_name_clean.lower())
             if match:
                 model_name_clean = f"v{match.group(1)} (Cans)"
             else:
                 model_name_clean = "Teacher (Cans)"

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

def create_blurred_dataset(clean_yaml_path, output_dir_name="CS228_Testset_blurred"):
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

    # Determine the test images path
    rel_test = data.get('test', 'test/images')
    test_images_path = base_path / rel_test
    
    if not test_images_path.exists() and (base_path / "test" / "images").exists():
        test_images_path = base_path / "test" / "images"
    
    if not test_images_path.exists():
        print(f"Error: test dataset not found at {test_images_path}. Skipping blur dataset generation.")
        return None

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
        
        # Sanitize path to prevent writes outside project directory
        out_p = Path(output_path).resolve()
        proj_dir = Path("project").resolve()
        if proj_dir not in out_p.parents:
            print(f"Warning: Output path {output_path} is outside project directory. Redirecting to project/.")
            out_p = proj_dir / out_p.name
            
        with open(out_p, 'w', encoding='utf-8') as f:
            f.write("<html><head><title>Model Comparison Dashboard</title></head><body>")
            f.write("<h1>🚀 Model Comparison Dashboard</h1>")
            f.write(fig_scatter.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write(fig_bar.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("<h3>Raw Data</h3>")
            f.write(df.to_html(index=False, classes='data-table'))
            f.write("</body></html>")
            
        print(f"✅ Dashboard saved to: {out_p}")
        return True
        
    except ImportError:
        print("Warning: 'plotly' not installed. Skipping dashboard generation.")
        print("Run: pip install plotly")
        return False
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        return False

def _derive_label(pt_path: Path, search_root: Path) -> str:
    """
    Derives a human-readable label from a best.pt path inside colab_results.
    e.g.  colab_results/detect/yolov11/weights/best.pt  ->  yolov11_student
          colab_results/teacher/yolov26/weights/best.pt  ->  yolov26_teacher
    Falls back to the stem of the filename if the pattern is unrecognised.
    """
    import re
    try:
        rel = pt_path.relative_to(search_root)          # detect/yolov11/weights/best.pt
        parts = rel.parts                                # ('detect','yolov11','weights','best.pt')
        role_part  = parts[0].lower() if len(parts) > 0 else ""
        ver_part   = parts[1].lower() if len(parts) > 1 else ""

        # Determine role
        if "teacher" in role_part:
            role = "teacher"
        elif role_part in ("detect", "student"):
            role = "student"
        else:
            role = role_part

        # Normalise version string: yolov11 / yolo26 / yolov8 etc.
        m = re.search(r'v?(\d+)', ver_part)
        version = f"v{m.group(1)}" if m else ver_part

        return f"yolo{version}_{role}"
    except ValueError:
        return pt_path.stem


def main():
    try:
        # Default to the new CS228 Testset if it exists
        default_data_path = "datasets/CS228_Testset.v1i.yolo/data.yaml"
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--models', nargs='+', required=True, help='Paths to .pt models / dirs to compare')
        
        if os.path.exists(default_data_path):
            parser.add_argument('--data', type=str, default=default_data_path, help='Path to clean data.yaml')
        else:
            parser.add_argument('--data', type=str, required=True, help='Path to clean data.yaml')
            
        parser.add_argument('--device', type=str, default='cpu')
        args = parser.parse_args()

        # 1. Expand Directory and Glob Arguments
        #    model_entries: list of (absolute_path_str, label_str)
        model_entries = []   # [(path, label), ...]

        import glob as _glob
        for path_str in args.models:
            # Handle glob patterns (e.g. project/*_best.pt)
            if "*" in path_str:
                matched = sorted(_glob.glob(path_str))
                if not matched:
                    print(f"Warning: No files matched pattern '{path_str}'")
                else:
                    for m in matched:
                        model_entries.append((m, Path(m).stem))
                continue

            p = Path(path_str)
            if p.is_dir():
                # Recursively find every best.pt (preferred) then any .pt
                best_pts = sorted(p.rglob("best.pt"))
                if best_pts:
                    for bp in best_pts:
                        label = _derive_label(bp, p)
                        model_entries.append((str(bp), label))
                else:
                    # Fall back to any .pt in immediate dir
                    found = sorted(p.glob("*.pt"))
                    if not found:
                        print(f"Warning: No .pt files found in directory '{path_str}'")
                    else:
                        for f in found:
                            model_entries.append((str(f), f.stem))
            else:
                model_entries.append((str(p), p.stem))

        if not model_entries:
            print("Error: No models found to benchmark.")
            return

        # Keep backward-compat: args.models = list of paths
        args.models = [e[0] for e in model_entries]
        label_map   = {e[0]: e[1] for e in model_entries}

        print(f"\nModels selected for benchmarking ({len(args.models)}):")
        for m, lbl in model_entries:
            print(f" - {lbl}  ({m})")

        # 2. Setup Data
        print("Step 1: Preparing Datasets...", flush=True)
        clean_yaml = args.data
        print(f"Note: Blur dataset generation currently writes duplicated blurred images to disk via create_blurred_dataset().")
        print(f"      This disk I/O occurs offline prior to evaluation and does not impact model latency benchmarks.")
        blurred_yaml = create_blurred_dataset(clean_yaml)
        
        results_list = []
        
        if not blurred_yaml:
            print("Failed to generate blurred dataset. Cannot run benchmarks.")
            return

        # 3. Benchmark Loop
        print("\nStep 2: Running Benchmarks...", flush=True)
        for model_path in args.models:
            lbl = label_map[model_path]
            # Test on Clean
            res_clean = benchmark_model(model_path, clean_yaml, args.device, apply_blur=False, label_override=lbl)
            if res_clean: results_list.append(res_clean)
            
            # Test on Blur
            res_blur = benchmark_model(model_path, blurred_yaml, args.device, apply_blur=True, label_override=lbl)
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
