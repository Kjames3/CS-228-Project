# Plan: `visualize_results.py`

**Purpose:** Generate three publication-quality figures for the CS228 KD paper that
demonstrate teacher "choppiness" vs student detection stability during rover motion.

---

## Context & Key Constraints

- All models (`colab_results/**/best.pt`) have **nc=81** (COCO-80 + custom `can` = class 80).
- Target classes to filter predictions to: `39=bottle`, `41=cup`, `80=can`.
- Test images are real rover frames named `blur_frame_<timestamp>_jpg.rf.<hash>.jpg`.
  Sorting by the embedded timestamp gives **temporal (chronological) order**.
- `BlurAugment` (from `blur_augment.py`) already exists and can be reused for Fig 2.
- Output directory: `project/figures/` (create if missing).
- Dependencies: `ultralytics`, `cv2`, `numpy`, `matplotlib`, `blur_augment` (local).
- Use `matplotlib.use('Agg')` so the script runs headless (no display needed).

---

## Script Interface

```
python visualize_results.py
python visualize_results.py --models colab_results --images datasets/CS228_Testset.v1i.yolo/test/images
python visualize_results.py --figure 1          # only Fig 1
python visualize_results.py --figure 2          # only Fig 2
python visualize_results.py --figure 3          # only Fig 3
python visualize_results.py --figure 3 --grid-frames 8
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | `colab_results` | Root dir containing `best.pt` files |
| `--images` | `datasets/CS228_Testset.v1i.yolo/test/images` | Rover test images dir |
| `--figure` | `None` (all) | Which figure to generate: 1, 2, or 3 |
| `--conf` | `0.25` | Detection confidence threshold |
| `--grid-frames` | `6` | Number of frame rows in Fig 3 |
| `--device` | `cpu` | Inference device (`cpu` / `cuda`) |

---

## Shared Helpers (used by all three figures)

### `timestamp_from_name(path) -> int`
Extract the integer ms-timestamp from `blur_frame_<ts>_jpg.*` filenames using regex
`r'blur_frame_(\d+)'`. Returns 0 if no match.

### `derive_label(pt_path, search_root) -> str`
Replicates the logic from `compare_models.py::_derive_label`.
- Reads relative path parts: `parts[0]` = role dir, `parts[1]` = version dir.
- Role: `'teacher'` if `'teacher'` in `parts[0]`, else `'student'`.
- Version: regex `r'v?(\d+)'` on `parts[1]`, produces e.g. `'v8'`, `'v11'`, `'v26'`.
- Returns: `f"yolo{ver}_{role}"` → e.g. `"yolov8_teacher"`, `"yolov11_student"`.

### `load_models(search_root) -> dict[str, YOLO]`
`rglob('best.pt')` under `search_root`, call `derive_label` for each, load with `YOLO(str(pt))`.
Returns `{label: model}`.

### `sorted_test_images(image_dir) -> list[Path]`
Glob all `.jpg/.jpeg/.png/.bmp` files in `image_dir`, sort by `timestamp_from_name`.

### `detect_image(model, img_bgr, conf=0.25) -> (max_conf, n_dets, dets_list)`
Run `model.predict(img_bgr, verbose=False, conf=conf)`.
Filter `results[0].boxes` to `cls` in `[39, 41, 80]`.
Returns:
- `max_conf`: highest confidence among filtered detections (0.0 if none)
- `n_dets`: count of filtered detections
- `dets_list`: `[(xyxy, conf, cls_id), ...]`

---

## Visual Identity

Consistent colours/styles across all figures so the paper reads clearly.

```python
MODEL_PALETTE = {
    # Teachers – warm reds/oranges, dashed lines
    'yolov8_teacher':  {'color': '#e74c3c', 'ls': '--', 'lw': 1.8, 'marker': 'x'},
    'yolov11_teacher': {'color': '#e67e22', 'ls': '--', 'lw': 1.8, 'marker': 'x'},
    'yolov26_teacher': {'color': '#9b59b6', 'ls': '--', 'lw': 1.8, 'marker': 'x'},
    # Students – cool greens/blues, solid lines
    'yolov8_student':  {'color': '#27ae60', 'ls': '-',  'lw': 2.0, 'marker': 'o'},
    'yolov11_student': {'color': '#2980b9', 'ls': '-',  'lw': 2.0, 'marker': 'o'},
    'yolov26_student': {'color': '#16a085', 'ls': '-',  'lw': 2.0, 'marker': 'o'},
}

# For cv2 box drawing (BGR)
BOX_COLORS_BGR = {
    'yolov8_teacher':  (0,   0, 220),
    'yolov11_teacher': (0, 130, 230),
    'yolov26_teacher': (130,  0, 200),
    'yolov8_student':  (50, 200,  50),
    'yolov11_student': (200, 120,  50),
    'yolov26_student': (200, 180,   0),
}
```

---

## Figure 1 – Detection Stability Timeline

**Output:** `project/figures/fig1_stability_timeline.png`

**Goal:** Show that teacher confidence drops to 0 frequently ("choppy") while
student confidence stays elevated throughout the rover motion sequence.

### Layout
Two vertically stacked subplots sharing the x-axis:
- **Top (3:1 ratio):** Confidence vs frame index.
- **Bottom (1:1 ratio):** Estimated blur level per frame.

### Top subplot
1. Sort test images by timestamp → `x = [0, 1, …, N-1]`.
2. For each frame, run `detect_image` for every model → record `max_conf`.
3. For each model:
   - Plot raw `max_conf` values at 25% opacity (thin, shows raw noise).
   - Plot a rolling mean (window=3) at full opacity with model's `ls` and `color`.
4. Draw horizontal dashed line at `y=0.25` labelled "conf threshold".
5. Y-axis: 0–1.05, label "Max Detection Confidence".
6. Title: `"Detection Stability During Rover Motion\n(0 = no detection; gaps = 'choppy' behaviour)"`.
7. Legend upper-right, 2 columns.

### Bottom subplot (blur indicator)
- For each frame: compute `std(Laplacian(gray_image))`.
  High Laplacian std = sharp; low = blurry.
- Invert: `blur_inv = 1 - (lap_std / max_lap_std)` so the chart reads high = more blur.
- `fill_between(x, blur_inv)` in `steelblue` at 40% alpha.
- Label: "est. blur level", y-axis 0–1.

---

## Figure 2 – Blur Degradation Curve

**Output:** `project/figures/fig2_blur_degradation.png`

**Goal:** Quantitatively show that KD students degrade more gracefully as blur
severity increases, while teachers lose detection sooner.

### Method
```
kernel_sizes = [0, 5, 9, 13, 19, 27, 37, 51]
```
- Kernel 0 = original rover frames (already blurry), no additional processing.
- Kernel > 0 = apply `BlurAugment.apply_motion_blur(img, kernel_size=k, angle=0)`
  (horizontal = simulates linear rover motion).
- Use `max_images=80` frames (subset for speed; randomly sample or take first 80).
- For each kernel size and each frame, run `detect_image` → count frames with `n_dets > 0`.
- **Detection rate** = `(frames_with_detection / total_frames) * 100`.

### Plot
- X-axis: kernel size (int), labelled `"Motion-Blur Kernel Size (pixels)\n0 = original rover footage"`.
- Y-axis: detection rate (%), 0–105.
- One line per model using `MODEL_PALETTE` styles, with markers.
- Vertical grey dotted line at x=0 with text annotation "Original rover frames →".
- Legend 2 columns.
- Title: `"Blur Robustness: Detection Rate vs Blur Severity"`.

---

## Figure 3 – Annotated Frame Grid

**Output:** `project/figures/fig3_frame_grid.png`

**Goal:** Qualitative side-by-side visual. Show teacher missing detections (red "NO
DETECTION" banner) while student maintains bounding boxes on the same frame.

### Layout
```
Grid: n_rows × n_cols
  n_rows = n_frames  (default 6, set by --grid-frames)
  n_cols = 1 + len(models)   (Original frame + one column per model)
```
Figure size calculated as `(thumb_w * n_cols / 100, thumb_h * n_rows / 100)` inches.
Default `thumb_w=320, thumb_h=240` px per cell.

### Frame selection
Use `np.linspace(0, N-1, n_frames, dtype=int)` to pick evenly-spaced indices from
the sorted temporal sequence. This captures frames across the full motion period.

### `_draw_boxes(img_bgr, dets, label) -> img_bgr`
Helper that draws boxes on a copy of the image:
- If `dets` is empty: draw a semi-transparent black bar at vertical centre and write
  `"NO DETECTION"` in red text.
- Otherwise: for each `(xyxy, conf, cls_id)` draw a coloured rectangle (`BOX_COLORS_BGR[label]`)
  and label `"{class_name} {conf:.2f}"`.

### Column headers (row 0 only)
- Col 0: `"Original (rover frame)"` in bold.
- Other cols: `"YOLO{VER}\n{Teacher/Student (KD)}"` in the model's palette colour.

### Row labels
Left y-axis label for each row: `"Frame {index}"`.

### Figure title
`"Teacher vs KD Student: Detection on Real Rover Frames"`

---

## `main()` Function Structure

```python
def main():
    # 1. Parse args
    # 2. Create OUTPUT_DIR
    # 3. load_models(models_root)  → dict
    # 4. if figure in (None, 1): fig_stability_timeline(...)
    # 5. if figure in (None, 2): fig_blur_degradation(...)
    # 6. if figure in (None, 3): fig_frame_grid(...)
    # 7. Print final output path
```

---

## File Structure

```
visualize_results.py          ← single self-contained script
project/
  figures/
    fig1_stability_timeline.png
    fig2_blur_degradation.png
    fig3_frame_grid.png
```

---

## Dependencies

```
ultralytics   # YOLO inference
opencv-python # image I/O and blur
numpy
matplotlib
# blur_augment (local module, already exists at project root)
```

---

## Notes & Gotchas

1. **nc=81 filtering**: All models have nc=81. Always filter predictions to `[39, 41, 80]`
   in `detect_image` — do NOT pass `classes=` to `model.predict()` to avoid version issues;
   filter the returned boxes tensor directly by `.cls`.

2. **Windows paths**: Use `Path` objects throughout; avoid hardcoded backslashes.

3. **Temporal ordering**: Sort images by `timestamp_from_name`, not alphabetically —
   alphabetical order happens to match here but is coincidental.

4. **Fig 2 speed**: Running all 6 models × 8 kernel sizes × 80 images = 3,840 inference
   calls on CPU. Expect ~10–20 min. Can reduce `max_images` to 40 to halve this.

5. **Fig 3 thumb size**: `thumb_w=320, thumb_h=240` balances readability vs file size.
   Increase to `640×480` for a higher-res version suitable for printing.

6. **`blur_augment.py` import**: The script uses `sys.path.append(project_root)` to
   ensure `from blur_augment import BlurAugment` works regardless of CWD.
