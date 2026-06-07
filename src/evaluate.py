from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.inference import TumorInferenceEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_CSV = PROJECT_ROOT / "data" / "CBIS_Master_Index.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def _resolve_image_path(row: pd.Series, dataset_root: Path | None = None) -> Path | None:
    candidate_columns = [
        "Full_Path",
        "full_path",
        "full image file path",
        "cropped image file path",
        "cropped_path",
        "Crop_Path",
        "crop_path",
        "image_path",
        "path",
    ]

    for col in candidate_columns:
        if col in row and isinstance(row[col], str) and row[col].strip():
            raw = row[col].strip().replace('\\', '/')
            p = Path(raw)
            if p.exists():
                return p
            if dataset_root is not None:
                q = dataset_root / raw
                if q.exists():
                    return q
    return None


def _resolve_label(row: pd.Series) -> int | None:
    if "Label" in row and pd.notna(row["Label"]):
        return int(row["Label"])

    if "Pathology" in row and pd.notna(row["Pathology"]):
        val = str(row["Pathology"]).upper()
        if "MALIGNANT" in val:
            return 1
        if "BENIGN" in val:
            return 0
    return None


def _dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)

    inter = float(np.logical_and(pred_bin, gt_bin).sum())
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    union = float(np.logical_or(pred_bin, gt_bin).sum())

    eps = 1e-6
    dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
    iou = (inter + eps) / (union + eps)
    return dice, iou


def evaluate(csv_path: Path = DEFAULT_CSV, max_samples: int | None = None) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    engine = TumorInferenceEngine()
    df = pd.read_csv(csv_path)

    dataset_root = None
    if CONFIG_PATH.exists():
        import yaml

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ds = cfg.get("paths", {}).get("dataset_root")
        if ds:
            p = Path(ds)
            dataset_root = p if p.is_absolute() else (PROJECT_ROOT / p)

    y_true = []
    y_prob = []
    y_pred = []

    dice_scores = []
    iou_scores = []

    vis_saved = 0

    for _, row in df.iterrows():
        label = _resolve_label(row)
        if label is None:
            continue

        image_path = _resolve_image_path(row, dataset_root=dataset_root)
        if image_path is None:
            continue

        try:
            out = engine.predict_numpy(str(image_path))
        except Exception:
            continue

        p_malig = max((d.get("p_malig", 0.0) for d in out["dets"]), default=0.0)
        pred_label = int(p_malig >= 0.5)

        y_true.append(label)
        y_prob.append(float(p_malig))
        y_pred.append(pred_label)

        roi_col = "ROI_Path" if "ROI_Path" in row else "roi_path"
        if roi_col in row and isinstance(row[roi_col], str) and row[roi_col].strip():
            gt_path = Path(row[roi_col].strip())
            if not gt_path.exists() and dataset_root is not None:
                gt_path = dataset_root / row[roi_col].strip().replace('\\', '/')
            if gt_path.exists():
                gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                if gt is not None:
                    pmask = (out["mask"] > engine.config.mask_threshold).astype(np.uint8)
                    gt = cv2.resize(gt, (pmask.shape[1], pmask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    gt = (gt > 0).astype(np.uint8)
                    d, j = _dice_iou(pmask, gt)
                    dice_scores.append(d)
                    iou_scores.append(j)

        if vis_saved < 12:
            rgb = out["rgb"]
            pmask = (out["mask"] > engine.config.mask_threshold).astype(np.uint8)
            overlay = rgb.copy()
            overlay[pmask > 0] = [255, 0, 0]
            blended = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)

            fig_path = FIGURES_DIR / f"eval_overlay_{vis_saved:03d}.png"
            cv2.imwrite(str(fig_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            vis_saved += 1

        if max_samples is not None and len(y_true) >= max_samples:
            break

    if not y_true:
        raise RuntimeError("No valid samples were evaluated. Check CSV paths/columns.")

    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / max(1, (tn + fp))

    metrics = {
        "samples": int(len(y_true)),
        "auc_roc": float(auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "sensitivity": float(rec),
        "specificity": float(specificity),
        "f1": float(f1),
        "dice": float(np.mean(dice_scores)) if dice_scores else None,
        "iou": float(np.mean(iou_scores)) if iou_scores else None,
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = METRICS_DIR / "yolo_pipeline_metrics.json"
    pd.Series(metrics).to_json(metrics_path, indent=2)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print("\n===== YOLO PIPELINE EVALUATION =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"metrics file: {metrics_path}")
    print(f"figures dir: {FIGURES_DIR}")

    return metrics


if __name__ == "__main__":
    evaluate()
