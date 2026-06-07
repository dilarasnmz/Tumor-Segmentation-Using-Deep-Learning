#!/usr/bin/env python
"""
Convert CBIS-DDSM dataset to YOLOv8 segmentation format.

Reads images and ROI masks from CBIS-DDSM, generates YOLO-format
polygon labels with class information (BENIGN=0, MALIGNANT=1),
and creates train/val splits with stratification.

Output structure:
  data/yolo_run85/
    images/train/
    images/val/
    labels/train/
    labels/val/
    data.yaml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_cbis_csv(csv_path: Path) -> pd.DataFrame:
    """Read CBIS Master Index CSV."""
    df = pd.read_csv(csv_path)
    return df


def resolve_image_path(row: pd.Series, dataset_root: Path) -> Optional[Path]:
    """Resolve full image path from CSV row with fallback column names."""
    candidates = [
        "Full_Path",
        "full_path",
        "full image file path",
        "cropped image file path",
        "Crop_Path",
        "crop_path",
        "cropped_path",
        "image_path",
        "path",
    ]

    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            path_str = str(row[col]).strip()
            if not path_str:
                continue

            path = Path(path_str)
            if path.is_absolute() and path.exists():
                return path

            rel_path = dataset_root / path_str
            if rel_path.exists():
                return rel_path

    return None


def resolve_roi_path(row: pd.Series, dataset_root: Path) -> Optional[Path]:
    """Resolve ROI mask path from CSV row with fallback column names."""
    candidates = [
        "ROI_Path",
        "roi_path",
        "ROI file path",
        "Roi_Path",
    ]

    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            path_str = str(row[col]).strip()
            if not path_str:
                continue

            path = Path(path_str)
            if path.is_absolute() and path.exists():
                return path

            rel_path = dataset_root / path_str
            if rel_path.exists():
                return rel_path

    return None


def resolve_pathology(row: pd.Series) -> Optional[int]:
    """Extract class from Pathology column. BENIGN=0, MALIGNANT=1."""
    candidates = ["Pathology", "pathology", "Label", "label", "class"]

    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).strip().upper()
            if "BENIGN" in val:
                return 0
            elif "MALIGNANT" in val:
                return 1

    return None


def extract_polygon_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract polygon coordinates from binary mask.

    Returns normalized polygon coordinates as array of shape (N, 2)
    with values in [0, 1], or None if no contour found.
    """
    if mask is None or mask.size == 0:
        return None

    mask_bin = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10:
        return None

    polygon = cnt.squeeze().astype(np.float32)
    if polygon.ndim == 1:
        polygon = polygon.reshape(-1, 2)

    h, w = mask.shape[:2]
    polygon[:, 0] /= w
    polygon[:, 1] /= h

    polygon = np.clip(polygon, 0, 1)

    return polygon


def polygon_to_yolo_str(class_id: int, polygon: np.ndarray) -> str:
    """Convert polygon to YOLO label string format."""
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
    return f"{class_id} {coords}"


def prepare_yolo_dataset(
    csv_path: Path,
    dataset_root: Path,
    output_dir: Path,
    val_size: float = 0.2,
    max_samples: Optional[int] = None,
) -> dict:
    """
    Prepare YOLO dataset from CBIS-DDSM.

    Returns statistics dict with counts of processed samples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale output folders so leftover files from previous runs cannot
    # pollute the new dataset (this is what caused the overwrite/count mismatch).
    for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
        subdir_path = output_dir / subdir
        if subdir_path.exists():
            shutil.rmtree(subdir_path)
            print(f"[PREPARE] Cleared: {subdir_path}")
        subdir_path.mkdir(parents=True, exist_ok=True)

    print(f"[PREPARE] Reading CSV: {csv_path}")
    df = read_cbis_csv(csv_path)

    if max_samples:
        df = df.head(max_samples)
        print(f"[PREPARE] Limited to {max_samples} samples for testing")

    print(f"[PREPARE] Total samples in CSV: {len(df)}")

    valid_samples = []

    for idx, row in df.iterrows():
        img_path = resolve_image_path(row, dataset_root)
        if img_path is None:
            continue

        roi_path = resolve_roi_path(row, dataset_root)
        if roi_path is None:
            continue

        class_id = resolve_pathology(row)
        if class_id is None:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            continue

        mask = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.size == 0:
            continue

        polygon = extract_polygon_from_mask(mask)
        if polygon is None or len(polygon) < 3:
            continue

        valid_samples.append((idx, img_path, mask, class_id, polygon))

    print(f"[PREPARE] Valid samples with images and masks: {len(valid_samples)}")

    if len(valid_samples) == 0:
        raise ValueError("No valid samples found. Check CSV paths and data.")

    indices = list(range(len(valid_samples)))
    class_labels = [valid_samples[i][3] for i in indices]

    # Handle stratified split: ensure enough samples per class
    if len(valid_samples) >= 10:
        # Use stratification for larger datasets
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            stratify=class_labels,
            random_state=42,
        )
    else:
        # Simple split for small datasets (can't stratify with so few samples)
        print(f"[PREPARE] ⚠️  Small dataset (<10 samples), using simple random split")
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=42,
        )

    print(f"[PREPARE] Train/val split: {len(train_idx)} train, {len(val_idx)} val")

    stats = {
        "total_samples": len(valid_samples),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "benign": 0,
        "malignant": 0,
    }

    # Use a running sequential counter *per split* so that samples which share
    # the same source image stem (common in CBIS-DDSM where multiple ROI rows
    # refer to the same mammogram) get distinct output filenames.
    for split_name, split_indices in [("train", train_idx), ("val", val_idx)]:
        for seq, sample_idx in enumerate(split_indices, start=1):
            row_idx, img_path, mask, class_id, polygon = valid_samples[sample_idx]

            # Truncate stem to 80 chars so paths stay filesystem-safe.
            stem = img_path.stem[:80]
            unique_stem = f"{split_name}_{seq:06d}_{stem}"

            # --- write image ---
            dst_img = output_dir / "images" / split_name / f"{unique_stem}.png"
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[PREPARE] ⚠️  Could not re-read image: {img_path}  — skipping")
                continue
            cv2.imwrite(str(dst_img), img)
            if not dst_img.exists():
                raise RuntimeError(f"Image write failed: {dst_img}")

            # --- write label ---
            label_str = polygon_to_yolo_str(class_id, polygon)
            dst_lbl = output_dir / "labels" / split_name / f"{unique_stem}.txt"
            with open(dst_lbl, "w") as f:
                f.write(label_str + "\n")
            if not dst_lbl.exists():
                raise RuntimeError(f"Label write failed: {dst_lbl}")

            if class_id == 0:
                stats["benign"] += 1
            else:
                stats["malignant"] += 1

    # ---- physical file counts ----
    train_images = sorted((output_dir / "images" / "train").glob("*.png"))
    train_labels = sorted((output_dir / "labels" / "train").glob("*.txt"))
    val_images   = sorted((output_dir / "images" / "val").glob("*.png"))
    val_labels   = sorted((output_dir / "labels" / "val").glob("*.txt"))

    stats["physical_train_images"] = len(train_images)
    stats["physical_train_labels"] = len(train_labels)
    stats["physical_val_images"]   = len(val_images)
    stats["physical_val_labels"]   = len(val_labels)

    print(f"[PREPARE] ✅ Images written to {output_dir}/images/")
    print(f"[PREPARE] ✅ Labels written to {output_dir}/labels/")
    print(f"[PREPARE] Class distribution:")
    print(f"[PREPARE]   BENIGN:     {stats['benign']}")
    print(f"[PREPARE]   MALIGNANT:  {stats['malignant']}")
    print()
    print(f"[PREPARE] Physical file counts:")
    print(f"[PREPARE]   train images : {len(train_images)}")
    print(f"[PREPARE]   train labels : {len(train_labels)}")
    print(f"[PREPARE]   val   images : {len(val_images)}")
    print(f"[PREPARE]   val   labels : {len(val_labels)}")

    # Integrity check: every image must have a matching label stem.
    train_img_stems = {p.stem for p in train_images}
    train_lbl_stems = {p.stem for p in train_labels}
    val_img_stems   = {p.stem for p in val_images}
    val_lbl_stems   = {p.stem for p in val_labels}

    train_mismatch = train_img_stems.symmetric_difference(train_lbl_stems)
    val_mismatch   = val_img_stems.symmetric_difference(val_lbl_stems)
    if train_mismatch:
        print(f"[PREPARE] ⚠️  Train image/label mismatch ({len(train_mismatch)} stems): {sorted(train_mismatch)[:5]}")
    if val_mismatch:
        print(f"[PREPARE] ⚠️  Val image/label mismatch ({len(val_mismatch)} stems): {sorted(val_mismatch)[:5]}")
    if not train_mismatch and not val_mismatch:
        print(f"[PREPARE] ✅ All image/label pairs match.")

    return stats


def create_data_yaml(output_dir: Path):
    """Create data.yaml for YOLO training."""
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {0: "BENIGN", 1: "MALIGNANT"},
    }

    import yaml

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"[PREPARE] ✅ data.yaml created: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CBIS-DDSM dataset for YOLO training."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "CBIS_Master_Index.csv",
        help="Path to CBIS Master Index CSV (default: data/CBIS_Master_Index.csv)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "CBIS-DDSM-512-FULL1" / "CBIS-DDSM-512-FULL1",
        help="Root directory of CBIS-DDSM dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "yolo_run85",
        help="Output directory for YOLO dataset (default: data/yolo_run85)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation set size as fraction (default: 0.2)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (useful for testing)",
    )

    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    if not args.data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.data_root}")

    print("=" * 80)
    print("CBIS-DDSM to YOLO Dataset Preparation")
    print("=" * 80)
    print(f"CSV:         {args.csv}")
    print(f"Data root:   {args.data_root}")
    print(f"Output:      {args.output_dir}")
    print(f"Val size:    {args.val_size}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("=" * 80)
    print()

    stats = prepare_yolo_dataset(
        args.csv,
        args.data_root,
        args.output_dir,
        val_size=args.val_size,
        max_samples=args.max_samples,
    )

    create_data_yaml(args.output_dir)

    print()
    print("=" * 80)
    print("✅ Dataset preparation complete!")
    print("=" * 80)
    print()
    print(f"Next: Train YOLO with the prepared dataset:")
    print()
    print(
        f"  python scripts/train_yolo.py --work-dir outputs --epochs 50 --imgsz 1024 --batch 8 --device 0"
    )
    print()


if __name__ == "__main__":
    main()
