# Tumor Segmentation Using Deep Learning

## Primary Production Pipeline: YOLO-Based Two-Stage Inference

This repository deploys a **YOLO-first production pipeline** with automatic fallback:

1. **Stage-1 (YOLO Detector)**: Finds candidate mass regions (`models/best_yolo.pt`)
2. **Stage-2 (UNet++ Classifier)**: Refines segmentation and classifies benign vs malignant (`models/best_classifier.pth`)
3. **Desktop App**: Shows segmentation overlays and Grad-CAM-style heatmaps

### Model Priority

- **Primary**: YOLO two-stage pipeline (active when `models/best_yolo.pt` and `models/best_classifier.pth` exist)
- **Fallback**: Legacy single-model pipeline (`models/best_model.pth`) — used only when YOLO models are missing
- **No silent fallback**: App explicitly logs which pipeline is active on startup

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run desktop app:

```bash
python app/main.py
```

The app will print logs indicating which pipeline loaded:

```
[INFERENCE] Loading YOLO two-stage pipeline...
[INFERENCE]   Stage-1 (YOLO detector): models/best_yolo.pt
[INFERENCE]   Stage-2 (UNet++ classifier): models/best_classifier.pth
[INFERENCE] ✅ YOLO two-stage pipeline ready
```

Or if YOLO models are missing:

```
[INFERENCE] ⚠️  YOLO weights missing:
[INFERENCE]   - Stage-1 detector not found: models/best_yolo.pt
[INFERENCE] Falling back to legacy pipeline...
[INFERENCE] Loading legacy single-model pipeline...
[INFERENCE]   Model: models/best_model.pth
[INFERENCE] ✅ Legacy pipeline ready (fallback only)
```

```bash
python app/main.py
```

## Inference Artifacts

The app reads model paths from [configs/config.yaml](configs/config.yaml) and follows this priority:

### YOLO Two-Stage Pipeline (PRIMARY)

When these files exist, the app uses the YOLO-based pipeline:

- `models/best_yolo.pt` — YOLO detector (Stage-1)
- `models/best_classifier.pth` — UNet++ classifier (Stage-2)

### Legacy Single-Model Pipeline (FALLBACK ONLY)

If YOLO models are missing, the app falls back to:

- `models/best_model.pth` — Legacy single-model backup

The app never silently uses the legacy model when YOLO models exist. Terminal output clearly indicates which pipeline is active.

## Prepare CBIS-DDSM Dataset for YOLO Training

Before training YOLO, convert CBIS-DDSM images and ROI masks to YOLOv8 segmentation format.

Preparation script: [scripts/prepare_yolo_dataset.py](scripts/prepare_yolo_dataset.py)

### Quick start (full dataset):

```bash
python scripts/prepare_yolo_dataset.py
```

- Reads CSV: `data/CBIS_Master_Index.csv`
- Dataset root: `data/CBIS-DDSM-512-FULL1/CBIS-DDSM-512-FULL1`
- Output: `data/yolo_run85/`
- Auto-generates train/val split with class stratification (BENIGN=0, MALIGNANT=1)

### Quick test (200 samples):

```bash
python scripts/prepare_yolo_dataset.py --max-samples 200
```

### Custom parameters:

```bash
python scripts/prepare_yolo_dataset.py \
  --csv data/CBIS_Master_Index.csv \
  --data-root data/CBIS-DDSM-512-FULL1/CBIS-DDSM-512-FULL1 \
  --output-dir data/yolo_run85 \
  --val-size 0.2 \
  --max-samples 1000
```

The script:
- Extracts polygon contours from ROI masks
- Generates YOLO segmentation labels (normalized polygon coordinates)
- Creates `data.yaml` for training
- Logs class distribution and sample counts

Output structure:

```
data/yolo_run85/
├── images/
│   ├── train/    (training images)
│   └── val/      (validation images)
├── labels/
│   ├── train/    (YOLO polygon labels)
│   └── val/
└── data.yaml     (dataset metadata)
```

## Train YOLO Detector (Notebook-Aligned)

Training script: [scripts/train_yolo.py](scripts/train_yolo.py)

### Basic command (uses prepared dataset):

```bash
python scripts/train_yolo.py
```

- Uses prepared dataset: `data/yolo_run85`
- Trains on GPU 0 for 50 epochs with batch size 8
- Image size: 1024

### Custom parameters:

```bash
python scripts/train_yolo.py \
  --dataset-dir data/yolo_run85 \
  --work-dir outputs \
  --device 0 \
  --epochs 50 \
  --imgsz 1024 \
  --batch 8 \
  --model yolov8s.pt
```

All hyperparameters match the notebook (AdamW optimizer, lr0=1e-3, weight_decay=5e-4, iou=0.8, augmentation settings).

### Trained model location:

After training completes, the best model is saved to:

```
outputs/yolo_run85/weights/best.pt
```

### Copy to app models directory:

```bash
copy outputs\yolo_run85\weights\best.pt models\best_yolo.pt
```

(On Linux/Mac, use `cp` instead of `copy`)

## Evaluate Production Pipeline

Run evaluation through the deployed inference path:

```bash
python -m src.evaluate
```

Artifacts are written to:

- `outputs/metrics/yolo_pipeline_metrics.json`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/eval_overlay_*.png`

## Project Structure

```text
Tumor-Segmentation-Using-Deep-Learning/
|- app/
|  |- main.py
|- configs/
|  |- config.yaml
|- data/
|- docs/
|- models/                  ← deployed inference weights (do not delete)
|  |- best_yolo.pt          ← Stage-1 YOLO detector          (PRIMARY)
|  |- best_classifier.pth   ← Stage-2 UNet++ classifier       (PRIMARY)
|  |- best_model.pth        ← Legacy single-model backup      (FALLBACK ONLY)
|- notebooks/
|  |- final_experiment.ipynb
|  |- archive/
|- outputs/
|  |- figures/
|  |- metrics/
|- scripts/
|  |- train_yolo.py
|- src/
|- tests/
|- requirements.txt
|- README.md
```
