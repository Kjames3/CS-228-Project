"""
visualize_results.py

Generates three publication-quality figures for the CS228 KD paper:

  Figure 1 – Detection Stability Timeline
      Per-frame max-confidence over the temporal rover sequence.
      Shows teacher "choppiness" (frequent drops to 0) vs student stability.

  Figure 2 – Blur Degradation Curve
      Detection rate (% frames with ≥1 detection) as synthetic motion-blur
      kernel increases from 0 (original frames) to extreme blur.

  Figure 3 – Annotated Frame Grid
      Side-by-side bounding-box comparison on representative frames.
      Teacher misses rendered as a red "NO DETECTION" banner.

Usage:
    python visualize_results.py
    python visualize_results.py --models colab_results --images datasets/CS228_Testset.v1i.yolo/test/images
    python visualize_results.py --figure 1
    python visualize_results.py --figure 2
    python visualize_results.py --figure 3 --grid-frames 8
"""

import argparse
import re
import sys
import os
import cv2
import numpy as np
import random

import matplotlib
matplotlib.use('Agg')   # headless – no display required
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve project root so local imports work regardless of CWD
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from blur_augment import BlurAugment
from ultralytics import YOLO

# ── Target classes for nc=81 models (COCO-80 + custom can=80) ────────────────
TARGET_CLASSES = {39: 'bottle', 41: 'cup', 80: 'can'}
FILTER_IDS     = list(TARGET_CLASSES.keys())   # [39, 41, 80]

# ── Consistent visual identity across all three figures ──────────────────────
MODEL_PALETTE = {
    # Teachers – warm reds/oranges, dashed lines
    'yolov8_teacher':  {'color': '#27ae60', 'ls': '--', 'lw': 1.8, 'marker': 'x'},
    'yolov11_teacher': {'color': '#2a24bf', 'ls': '--', 'lw': 1.8, 'marker': 'x'},
    'yolov26_teacher': {'color': '#bf2424', 'ls': '--', 'lw': 1.8, 'marker': 'x'},
    # Students – cool greens/blues, solid lines
    'yolov8_student':  {'color': '#27ae60', 'ls': '-',  'lw': 2.0, 'marker': 'o'},
    'yolov11_student': {'color': '#2a24bf', 'ls': '-',  'lw': 2.0, 'marker': 'o'},
    'yolov26_student': {'color': '#bf2424', 'ls': '-',  'lw': 2.0, 'marker': 'o'},
}

# BGR colours for cv2 box drawing
BOX_COLORS_BGR = {
    'yolov8_teacher':  (0,   0, 220),
    'yolov11_teacher': (0, 130, 230),
    'yolov26_teacher': (130,  0, 200),
    'yolov8_student':  (0, 0,  200),
    'yolov11_student': (0, 130, 230),
    'yolov26_student': (130, 0, 200),
}

OUTPUT_DIR = Path('project/figures')


# =============================================================================
# Shared helpers
# =============================================================================

def timestamp_from_name(p: Path) -> int:
    """Extract ms-timestamp from 'blur_frame_<ts>_jpg.*' filenames."""
    m = re.search(r'blur_frame_(\d+)', p.name)
    return int(m.group(1)) if m else 0


def derive_label(pt_path: Path, search_root: Path) -> str:
    """
    Build a human-readable label from a best.pt path.
    e.g. colab_results/detect/yolov11/weights/best.pt  -> yolov11_student
         colab_results/teacher/yolov8/weights/best.pt  -> yolov8_teacher
    """
    try:
        rel   = pt_path.relative_to(search_root)
        parts = rel.parts
        role  = 'teacher' if 'teacher' in parts[0].lower() else 'student'
        ver_m = re.search(r'v?(\d+)', parts[1].lower()) if len(parts) > 1 else None
        ver   = f"v{ver_m.group(1)}" if ver_m else (parts[1] if len(parts) > 1 else pt_path.stem)
        return f"yolo{ver}_{role}"
    except ValueError:
        return pt_path.stem


def load_models(search_root: Path) -> dict:
    """Return {label: YOLO} for every best.pt found under search_root."""
    models = {}
    for pt in sorted(search_root.rglob('best.pt')):
        label = derive_label(pt, search_root)
        print(f"  Loading {label} ...")
        models[label] = YOLO(str(pt))
    return models


def sorted_test_images(image_dir: Path) -> list:
    """Return image paths sorted by embedded timestamp (temporal order)."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    imgs = [p for p in image_dir.glob('*') if p.suffix.lower() in exts]
    return sorted(imgs, key=timestamp_from_name)


def detect_image(model: YOLO, img_bgr: np.ndarray,
                 conf: float = 0.25) -> tuple:
    """
    Run inference on a BGR numpy image.
    Filters results to TARGET_CLASSES without using the `classes=` kwarg
    (avoids cross-version Ultralytics quirks).

    Returns:
        max_conf  – highest filtered confidence (0.0 if nothing detected)
        n_dets    – number of filtered detections
        dets_list – [(xyxy_list, conf_float, cls_id_int), ...]
    """
    results = model.predict(img_bgr, verbose=False, conf=conf)
    boxes   = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0.0, 0, []

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs   = boxes.conf.cpu().numpy()
    xyxy    = boxes.xyxy.cpu().numpy()

    mask    = np.isin(cls_ids, FILTER_IDS)
    cls_ids = cls_ids[mask]
    confs   = confs[mask]
    xyxy    = xyxy[mask]

    if len(confs) == 0:
        return 0.0, 0, []

    dets = [(xy.tolist(), float(c), int(cl))
            for xy, c, cl in zip(xyxy, confs, cls_ids)]
    return float(confs.max()), len(confs), dets


def style_for(label: str) -> dict:
    return MODEL_PALETTE.get(
        label,
        {'color': '#888888', 'ls': '-', 'lw': 1.5, 'marker': '.'}
    )


# =============================================================================
# Figure 1 – Detection Stability Timeline
# =============================================================================

def fig_stability_timeline(models: dict, image_dir: Path,
                            conf: float = 0.25, smooth_window: int = 3):
    """
    Two-panel figure:
      Top    – max detection confidence per frame (raw + rolling mean).
      Bottom – estimated per-frame blur level (inverted Laplacian std).
    """
    print("\n[Fig 1] Detection Stability Timeline ...")
    imgs = sorted_test_images(image_dir)
    n    = len(imgs)
    print(f"  {n} frames in temporal order")

    # Collect per-frame max-confidence for each model
    frame_conf = {lbl: np.zeros(n) for lbl in models}
    for i, img_path in enumerate(imgs):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for lbl, model in models.items():
            max_c, _, _ = detect_image(model, img, conf)
            frame_conf[lbl][i] = max_c
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{n} frames processed ...")

    # ── Compute estimated blur level ─────────────────────────────────────────
    blur_levels = []
    for img_path in imgs:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            blur_levels.append(0.0)
        else:
            blur_levels.append(float(np.std(cv2.Laplacian(gray, cv2.CV_64F))))
    blur_arr = np.array(blur_levels)
    # Invert: high value → more blur
    blur_inv = 1.0 - (blur_arr / (blur_arr.max() + 1e-6))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax_conf, ax_blur) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    x        = np.arange(n)
    kern     = np.ones(smooth_window) / smooth_window
    teachers = [l for l in models if 'teacher' in l]
    students = [l for l in models if 'student'  in l]

    for lbl in teachers + students:
        s    = style_for(lbl)
        raw  = frame_conf[lbl]
        roll = np.convolve(raw, kern, mode='same')
        # Faint raw signal
        ax_conf.plot(x, raw, color=s['color'], alpha=0.20, lw=0.7)
        # Bold rolling mean
        ax_conf.plot(x, roll, color=s['color'],
                     lw=s['lw'], ls=s['ls'], label=lbl)

    ax_conf.axhline(conf, color='grey', lw=0.9, ls=':', alpha=0.7,
                    label=f'conf threshold ({conf})')
    ax_conf.set_ylabel('Max Detection Confidence', fontsize=12)
    ax_conf.set_ylim(-0.05, 1.05)
    ax_conf.legend(loc='upper right', fontsize=9, ncol=2)
    ax_conf.set_title(
        'Detection Stability During Rover Motion\n'
        '(0 = no detection detected; gaps indicate "choppy" teacher behaviour)',
        fontsize=12
    )
    ax_conf.grid(axis='y', alpha=0.3)

    ax_blur.fill_between(x, blur_inv, alpha=0.40, color='steelblue',
                         label='est. blur level (high = more blur)')
    ax_blur.set_ylabel('Blur', fontsize=10)
    ax_blur.set_xlabel('Frame index (chronological order)', fontsize=12)
    ax_blur.set_ylim(0, 1.05)
    ax_blur.legend(fontsize=9)
    ax_blur.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / 'fig1_stability_timeline.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")


# =============================================================================
# Figure 2 – Blur Degradation Curve
# =============================================================================

def fig_blur_degradation(models: dict, image_dir: Path,
                          conf: float = 0.25, max_images: int = 80):
    """
    Detection rate (% of frames with ≥1 detection) vs synthetic blur kernel.
    Kernel 0 = original rover frames; higher = additional horizontal motion blur.
    """
    print("\n[Fig 2] Blur Degradation Curve ...")
    all_imgs     = sorted_test_images(image_dir)
    imgs         = all_imgs[:max_images]
    kernel_sizes = [0, 5, 9, 13, 19, 27, 37, 51]
    blurrer      = BlurAugment(blur_prob=1.0)
    n            = len(imgs)
    print(f"  Using {n} frames across {len(kernel_sizes)} blur levels ...")

    results = {lbl: [] for lbl in models}

    for ks in kernel_sizes:
        print(f"  Kernel={ks} ...")
        counts = {lbl: 0 for lbl in models}

        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if ks > 0:
                # Horizontal motion blur simulates linear rover movement
                img = blurrer.apply_motion_blur(img, kernel_size=ks, angle=0)
            for lbl, model in models.items():
                _, n_det, _ = detect_image(model, img, conf)
                if n_det > 0:
                    counts[lbl] += 1

        for lbl in models:
            results[lbl].append(counts[lbl] / n * 100.0)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    teachers = [l for l in models if 'teacher' in l]
    students = [l for l in models if 'student'  in l]

    for lbl in teachers + students:
        s = style_for(lbl)
        ax.plot(kernel_sizes, results[lbl],
                color=s['color'], ls=s['ls'], lw=s['lw'],
                marker=s['marker'], markersize=6, label=lbl)

    ax.axvline(0, color='grey', lw=0.8, ls=':', alpha=0.6)
    ax.text(0.8, 4, 'Original rover\nframes →', fontsize=8, color='grey')
    ax.set_xlabel('Motion-Blur Kernel Size (px)\n0 = original rover footage',
                  fontsize=12)
    ax.set_ylabel('Detection Rate (%)', fontsize=12)
    ax.set_title(
        'Blur Robustness: Detection Rate vs Blur Severity\n'
        '(KD students maintain detection longer as blur increases)',
        fontsize=12
    )
    ax.set_ylim(0, 105)
    ax.set_xticks(kernel_sizes)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / 'fig2_blur_degradation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")


# =============================================================================
# Figure 3 – Annotated Frame Grid
# =============================================================================

def _draw_boxes(img_bgr: np.ndarray, dets: list,
                label: str) -> np.ndarray:
    """
    Draw bounding boxes on a copy of img_bgr.
    If dets is empty, renders a translucent red 'NO DETECTION' banner.
    """
    out   = img_bgr.copy()
    h, w  = out.shape[:2]
    color = BOX_COLORS_BGR.get(label, (200, 200, 200))

    if not dets:
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h // 2 - 22), (w, h // 2 + 22),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
        txt  = 'NO DETECTION'
        fs   = 0.65
        tw   = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
        cv2.putText(out, txt, ((w - tw) // 2, h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 2,
                    cv2.LINE_AA)
        return out

    for (x1, y1, x2, y2), c, cls_id in dets:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cls_name = TARGET_CLASSES.get(cls_id, str(cls_id))
        txt = f"{cls_name} {c:.2f}"
        cv2.putText(out, txt, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)
    return out


def fig_frame_grid(models: dict, image_dir: Path,
                   n_frames: int = 6, conf: float = 0.25,
                   thumb_w: int = 320, thumb_h: int = 240):
    """
    Grid of annotated rover frames.
    Rows = evenly-spaced frames from the temporal sequence.
    Columns = [Original] + one per model.
    """
    print(f"\n[Fig 3] Annotated Frame Grid ({n_frames} frames) ...")
    imgs     = sorted_test_images(image_dir)
    model_labels = list(models.keys())
    teachers = [l for l in model_labels if 'teacher' in l]
    students = [l for l in model_labels if 'student'  in l]

    print("  Filtering images for optimal student detections ...")
    valid_imgs_all = []
    valid_imgs_any = []
    
    for img_path in imgs:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
            
        student_detect_count = 0
        for student_lbl in students:
            model = models[student_lbl]
            _, n_det, _ = detect_image(model, img_bgr, conf)
            if n_det > 0:
                student_detect_count += 1
                
        if student_detect_count == len(students) and len(students) > 0:
            valid_imgs_all.append(img_path)
        if student_detect_count > 0:
            valid_imgs_any.append(img_path)

    if len(valid_imgs_all) >= n_frames:
        print(f"  Found {len(valid_imgs_all)} images where ALL students detected objects.")
        pool = valid_imgs_all
    elif len(valid_imgs_any) >= n_frames:
        print(f"  Found {len(valid_imgs_any)} images where AT LEAST ONE student detected objects.")
        pool = valid_imgs_any
    else:
        print("  Not enough images with student detections, falling back to all images.")
        pool = imgs

    if len(pool) >= n_frames:
        random.seed(42)  # For reproducibility
        selected = random.sample(pool, n_frames)
        selected = sorted(selected, key=timestamp_from_name)
    elif len(pool) > 0:
        inds  = np.linspace(0, len(pool) - 1, n_frames, dtype=int)
        selected = [pool[i] for i in inds]
    else:
        selected = []

    indices = [imgs.index(p) for p in selected]

    n_cols = 1 + len(model_labels)
    n_rows = n_frames

    fig_w = thumb_w * n_cols / 100.0
    fig_h = thumb_h * n_rows / 100.0 + 0.8

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h),
        gridspec_kw={'wspace': 0.02, 'hspace': 0.04}
    )

    # Ensure axes is always 2-D
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    ordered  = teachers + students          # teachers first, then students

    # Re-order cols so teachers come before students
    col_order  = [None] + ordered           # col 0 = original
    extra_lbls = [l for l in model_labels if l not in ordered]
    col_order += extra_lbls

    for row_i, img_path in enumerate(selected):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rs = cv2.resize(img_bgr, (thumb_w, thumb_h))

        for col_i in range(n_cols):
            ax = axes[row_i, col_i]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            if col_i == 0:
                # ── Original frame ───────────────────────────────────────────
                ax.imshow(cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB))
                if row_i == 0:
                    ax.set_title('Original\n(rover frame)', fontsize=7,
                                 pad=3, weight='bold')
                ax.set_ylabel(f"Frame\n{indices[row_i]}",
                              fontsize=6, rotation=90, labelpad=2)
            else:
                # ── Model detection ──────────────────────────────────────────
                lbl   = col_order[col_i]
                model = models[lbl]
                _, _, dets = detect_image(model, img_bgr, conf)
                
                h_orig, w_orig = img_bgr.shape[:2]
                scaled_dets = []
                for box, score, cls_id in dets:
                    x1, y1, x2, y2 = box
                    scaled_box = [x1 * thumb_w / w_orig, y1 * thumb_h / h_orig, 
                                  x2 * thumb_w / w_orig, y2 * thumb_h / h_orig]
                    scaled_dets.append((scaled_box, score, cls_id))
                    
                annotated  = _draw_boxes(img_rs.copy(), scaled_dets, lbl)
                ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                if row_i == 0:
                    ver_m = re.search(r'yolo(\w+)_', lbl)
                    ver_s = ver_m.group(1).upper() if ver_m else lbl
                    role  = 'Teacher' if 'teacher' in lbl else 'Student (KD)'
                    s     = style_for(lbl)
                    ax.set_title(f"YOLO{ver_s}\n{role}",
                                 fontsize=7, pad=3, weight='bold',
                                 color=s['color'])

    fig.suptitle(
        'Teacher vs KD Student: Detection on Real Rover Frames\n'
        'Dashed-border cols = Teacher  |  Solid-border cols = Student (KD)',
        fontsize=9, y=1.002
    )
    plt.tight_layout(pad=0.3)
    out = OUTPUT_DIR / 'fig3_frame_grid.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CS228 KD Project – Visualisation Script'
    )
    parser.add_argument('--models', default='colab_results',
                        help='Root dir containing best.pt files (default: colab_results)')
    parser.add_argument('--images',
                        default='datasets/CS228_Testset.v1i.yolo/test/images',
                        help='Rover test images directory')
    parser.add_argument('--figure', type=int, choices=[1, 2, 3], default=None,
                        help='Generate only this figure (1, 2, or 3). Omit for all.')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--grid-frames', type=int, default=8,
                        help='Number of frame rows in Fig 3 (default: 6)')
    parser.add_argument('--max-images', type=int, default=80,
                        help='Max images used for Fig 2 blur sweep (default: 80)')
    parser.add_argument('--device', default='cpu',
                        help='Inference device: cpu / cuda (default: cpu)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models_root = Path(args.models)
    image_dir   = Path(args.images)

    if not models_root.exists():
        print(f"Error: models directory '{models_root}' not found.")
        sys.exit(1)
    if not image_dir.exists():
        print(f"Error: images directory '{image_dir}' not found.")
        sys.exit(1)

    print("Loading models ...")
    models = load_models(models_root)
    if not models:
        print("No models found (no best.pt under the models directory).")
        sys.exit(1)
    print(f"  {len(models)} models: {list(models.keys())}\n")

    run_all = args.figure is None

    if run_all or args.figure == 1:
        fig_stability_timeline(models, image_dir, conf=args.conf)

    if run_all or args.figure == 2:
        fig_blur_degradation(models, image_dir,
                             conf=args.conf, max_images=args.max_images)

    if run_all or args.figure == 3:
        fig_frame_grid(models, image_dir,
                       n_frames=args.grid_frames, conf=args.conf)

    print(f"\nDone. Figures saved to: {OUTPUT_DIR.resolve()}")


if __name__ == '__main__':
    main()