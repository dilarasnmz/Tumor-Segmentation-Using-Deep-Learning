from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "yolo_run85"
DEFAULT_WORK_DIR = PROJECT_ROOT / "outputs"


def build_data_yaml(dataset_dir: Path, output_yaml: Path) -> Path:
    train_images = dataset_dir / "images" / "train"
    val_images = dataset_dir / "images" / "val"

    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(
            "Expected YOLO dataset folders: "
            f"{train_images} and {val_images}"
        )

    # Check if data.yaml already exists in the dataset directory
    # If so, use its class definitions
    existing_yaml = dataset_dir / "data.yaml"
    if existing_yaml.exists():
        import yaml as yaml_lib
        with open(existing_yaml, "r") as f:
            existing = yaml_lib.safe_load(f)
        nc = existing.get("nc", 1)
        names = existing.get("names", ["class"])
    else:
        # Default to single-class detection (backward compatible)
        nc = 1
        names = ["mass"]

    payload = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": names,
    }

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return output_yaml


def train_detector(
    data_yaml: Path,
    work_dir: Path,
    device: str,
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = 8,
    model: str = "yolov8s.pt",
):
    detector = YOLO(model)

    return detector.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        cls=2.0,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        close_mosaic=10,
        scale=0.8,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        iou=0.8,
        project=str(work_dir),
        name="yolo_run85",
        device=device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO detector using final notebook hyperparameters."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="YOLO dataset root containing images/train, images/val, labels/train, labels/val",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_yolo_run85.yaml",
        help="Path for generated data.yaml",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Directory where YOLO training outputs are saved (default: outputs)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device value (default: 0; use 'cpu' for CPU-only)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Image size for training (default: 1024)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="YOLO base model to use (default: yolov8s.pt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_yaml = build_data_yaml(args.dataset_dir, args.data_yaml)
    print(f"data.yaml generated: {data_yaml}")

    results = train_detector(
        data_yaml,
        args.work_dir,
        args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model=args.model,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining run directory: {results.save_dir}")
    print(f"Best detector checkpoint: {best}")
    print("\n✅ Training complete!")
    print(f"\nTo use the trained model in the app, copy it:")
    print(f"  cp {best} models/best_yolo.pt")


if __name__ == "__main__":
    main()
