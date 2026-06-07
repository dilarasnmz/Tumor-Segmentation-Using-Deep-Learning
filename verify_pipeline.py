#!/usr/bin/env python
"""
Verify YOLO-first pipeline logic and fallback behavior.

This script tests:
1. Which pipeline is selected when models exist
2. Logging output showing active pipeline
3. Fallback behavior
"""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import TumorInferenceEngine


def main():
    print("=" * 80)
    print("TUMOR SEGMENTATION - PIPELINE VERIFICATION")
    print("=" * 80)
    print()

    # Check model file existence
    yolo_detector = PROJECT_ROOT / "models" / "best_yolo.pt"
    stage2_classifier = PROJECT_ROOT / "models" / "best_classifier.pth"
    legacy_model = PROJECT_ROOT / "models" / "best_model.pth"

    print("Model File Status:")
    print(f"  YOLO Stage-1 (best_yolo.pt):     {'✅ EXISTS' if yolo_detector.exists() else '❌ MISSING'}")
    print(f"  Stage-2 Classifier (best_classifier.pth): {'✅ EXISTS' if stage2_classifier.exists() else '❌ MISSING'}")
    print(f"  Legacy Fallback (best_model.pth):        {'✅ EXISTS' if legacy_model.exists() else '❌ MISSING'}")
    print()

    print("Expected Pipeline Selection:")
    if yolo_detector.exists() and stage2_classifier.exists():
        print("  → YOLO two-stage pipeline (YOLO-first strategy active)")
    else:
        print("  → Legacy single-model pipeline (fallback)")
    print()

    print("Initializing Inference Engine...")
    print("-" * 80)
    try:
        engine = TumorInferenceEngine()
        engine._lazy_load()

        print("-" * 80)
        print()
        print("Pipeline Initialization Result:")
        print(f"  Active Mode:  {engine._mode}")
        if engine._mode == "run85":
            print("  ✅ YOLO two-stage pipeline is ACTIVE")
            print(f"     - Detector loaded: {engine.detector is not None}")
            print(f"     - Classifier loaded: {engine.classifier is not None}")
            print(f"     - Grad-CAM ready: {engine.gradcam is not None}")
        elif engine._mode == "legacy":
            print("  ⚠️  LEGACY single-model pipeline is ACTIVE (YOLO models missing)")
            print(f"     - Legacy model loaded: {engine.legacy_model is not None}")
            print(f"     - Grad-CAM ready: {engine.legacy_gradcam is not None}")
        else:
            print(f"  ❓ Unknown mode: {engine._mode}")

    except Exception as e:
        print("-" * 80)
        print()
        print("❌ Inference Engine Initialization Failed!")
        print(f"Error: {e}")
        return 1

    print()
    print("=" * 80)
    print("✅ Pipeline verification complete")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
