from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PySide6.QtGui import QImage, QPixmap

from src.model import MTL_EfficientUNetPlusPlus, Run85UNetPlusPlus
from src.run831_pipeline import Run831Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
LEGACY_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
DEBUG_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "debug"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False


def _format_float_list(values) -> str:
    return "[" + ", ".join(f"{float(v):.4f}" for v in values) + "]"


@dataclass
class InferenceConfig:
    segmentation_model_path: Path
    classifier_model_path: Path
    image_size: int
    confidence_threshold: float
    mask_threshold: float
    yolo_iou: float
    yolo_topk: int
    second_nms_iou: float
    physical_window_px: int


class ModelWithTemperature(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        seg, cls = self.model(x)
        t = self.temperature.unsqueeze(1).expand(cls.size(0), cls.size(1))
        return seg, cls / t


class TumorInferenceEngine:
    """
    Final notebook-aligned deployment pipeline:
    YOLO detector -> fixed physical crop -> filter fusion -> stage-2 seg/classifier.

    If final artifacts are missing, falls back to legacy `best_model.pth`
    to keep the desktop app operational.
    """

    def __init__(self):
        self.config = self._load_config()
        self._loaded = False
        self._mode = "run85"

        self.detector = None
        self.classifier = None
        self.gradcam = None

        self.legacy_model = None
        self.legacy_gradcam = None
        self._last_debug_dir: Path | None = None

        # Run 8.31 clean pipeline (replaces the old _predict_run85_numpy path)
        self._run831: Run831Pipeline | None = None

    @staticmethod
    def _safe_stem(path: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(path).stem) or "image"

    def _create_debug_dir(self, image_path: str) -> Path:
        DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = DEBUG_OUTPUT_DIR / f"{stamp}_{self._safe_stem(image_path)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._last_debug_dir = run_dir
        return run_dir

    @staticmethod
    def _write_debug_image(path: Path, image: np.ndarray) -> None:
        if image.ndim == 2:
            cv2.imwrite(str(path), image)
            return
        if image.ndim == 3 and image.shape[2] == 3:
            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return
        raise ValueError(f"Unsupported image for debug dump: shape={image.shape}")

    def _save_debug_bundle(
        self,
        image_path: str,
        yolo_input_rgb: np.ndarray,
        raw_detections: list[dict],
        low_conf_probe_detections: list[dict],
        segmentation_overlay_rgb: np.ndarray,
        gradcam_overlay_rgb: np.ndarray,
        out: dict,
        label: str,
    ) -> None:
        run_dir = self._create_debug_dir(image_path)

        self._write_debug_image(run_dir / "yolo_input_rgb.png", yolo_input_rgb)
        self._write_debug_image(run_dir / "segmentation_overlay.png", segmentation_overlay_rgb)
        self._write_debug_image(run_dir / "gradcam_overlay.png", gradcam_overlay_rgb)

        # Save raw model maps as 8-bit visualizations to inspect whether heatmap/mask are non-empty.
        mask = np.clip(out.get("mask", np.zeros(yolo_input_rgb.shape[:2], np.float32)), 0.0, 1.0)
        cam = np.clip(out.get("cam", np.zeros(yolo_input_rgb.shape[:2], np.float32)), 0.0, 1.0)
        self._write_debug_image(run_dir / "raw_mask_u8.png", np.uint8(mask * 255.0))
        self._write_debug_image(run_dir / "raw_cam_u8.png", np.uint8(cam * 255.0))

        payload = {
            "image_path": str(image_path),
            "mode": self._mode,
            "label": label,
            "detection_count": int(len(raw_detections)),
            "raw_detections": raw_detections,
            "low_conf_probe_detection_count": int(len(low_conf_probe_detections)),
            "low_conf_probe_detections": low_conf_probe_detections,
            "stage2_detections": out.get("dets", []),
            "mask_max": float(mask.max()) if mask.size else 0.0,
            "cam_max": float(cam.max()) if cam.size else 0.0,
        }
        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[INFERENCE DEBUG] Saved debug artifacts to: {run_dir}")

    def _load_config(self) -> InferenceConfig:
        defaults = {
            "segmentation_model_path": "models/detector_yolov8s.pt",
            "classifier_model_path": "models/segmenter_bias_fixed_2cls.pth",
            "image_size": 512,
            "confidence_threshold": 0.25,
            "mask_threshold": 0.5,
            "yolo_iou": 0.8,
            "yolo_topk": 3,
            "second_nms_iou": 0.3,
            "physical_window_px": 320,
        }

        cfg = dict(defaults)
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            cfg.update(raw.get("model", {}))

        def to_abs(p: str) -> Path:
            path = Path(p)
            return path if path.is_absolute() else PROJECT_ROOT / path

        return InferenceConfig(
            segmentation_model_path=to_abs(str(cfg["segmentation_model_path"])),
            classifier_model_path=to_abs(str(cfg["classifier_model_path"])),
            image_size=int(cfg["image_size"]),
            confidence_threshold=float(cfg["confidence_threshold"]),
            mask_threshold=float(cfg["mask_threshold"]),
            yolo_iou=float(cfg["yolo_iou"]),
            yolo_topk=int(cfg["yolo_topk"]),
            second_nms_iou=float(cfg["second_nms_iou"]),
            physical_window_px=int(cfg["physical_window_px"]),
        )

    @staticmethod
    def _torch_load_weights(path: Path):
        # Prefer the safer loading mode when available (PyTorch >=2.1).
        try:
            return torch.load(path, map_location=DEVICE, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=DEVICE)

    def _lazy_load(self):
        if self._loaded:
            return

        yolo_available = self.config.segmentation_model_path.exists()
        classifier_available = self.config.classifier_model_path.exists()

        if yolo_available and classifier_available:
            try:
                from ultralytics import YOLO

                print(f"[INFERENCE] Loading YOLO two-stage pipeline...")
                print(f"[INFERENCE]   Stage-1 (YOLO detector): {self.config.segmentation_model_path}")
                print(f"[INFERENCE]   Stage-2 (UNet++ classifier): {self.config.classifier_model_path}")

                self.detector = YOLO(str(self.config.segmentation_model_path))

                base = Run85UNetPlusPlus(encoder_weights=None).to(DEVICE)
                state = self._torch_load_weights(self.config.classifier_model_path)

                if isinstance(state, dict) and "temperature" in state:
                    wrapped = ModelWithTemperature(base).to(DEVICE)
                    fixed_state = {}
                    for k, v in state.items():
                        if k.startswith("model.") and not k.startswith("model.model."):
                            fixed_state[k.replace("model.", "model.model.", 1)] = v
                        else:
                            fixed_state[k] = v
                    wrapped.load_state_dict(fixed_state, strict=False)
                    self.classifier = wrapped
                else:
                    fixed_state = {}
                    for k, v in state.items():
                        if k == "temperature":
                            continue
                        # If state dict keys don't start with model. but base expects model.
                        if not k.startswith("model."):
                            fixed_state["model." + k] = v
                        else:
                            fixed_state[k] = v
                    base.load_state_dict(fixed_state, strict=False)
                    self.classifier = base

                self.classifier.eval()
                self._mode = "run85"

                self.gradcam = None
                try:
                    from pytorch_grad_cam import GradCAM as TorchGradCAM

                    class ClsWrapper(nn.Module):
                        def __init__(self, m):
                            super().__init__()
                            self.m = m

                        def forward(self, x):
                            return self.m(x)[1]

                    wrapped_cls = ClsWrapper(self.classifier).to(DEVICE).eval()
                    base_model = self.classifier.model.model if hasattr(self.classifier, "temperature") else self.classifier.model

                    if hasattr(base_model.encoder, "_blocks"):
                        target_layers = [base_model.encoder._blocks[-1]]
                    else:
                        convs = [m for m in base_model.encoder.modules() if isinstance(m, nn.Conv2d)]
                        target_layers = [convs[-1]]

                    self.gradcam = TorchGradCAM(model=wrapped_cls, target_layers=target_layers)
                except Exception:
                    self.gradcam = None

                # ── Instantiate the clean Run 8.31 pipeline ──────────────
                self._run831 = Run831Pipeline(
                    yolo_path=self.config.segmentation_model_path,
                    classifier_path=self.config.classifier_model_path,
                    device=DEVICE,
                )
                self._run831.load()

                print(f"[INFERENCE] \u2705 YOLO two-stage pipeline ready (Run831Pipeline)")
                self._loaded = True
                return

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[INFERENCE] \u26a0\ufe0f  YOLO pipeline failed: {e}")
                print(f"[INFERENCE] Falling back to legacy pipeline...")

        else:
            print(f"[INFERENCE] ⚠️  YOLO weights missing:")
            if not yolo_available:
                print(f"[INFERENCE]   - Stage-1 detector not found: {self.config.segmentation_model_path}")
            if not classifier_available:
                print(f"[INFERENCE]   - Stage-2 classifier not found: {self.config.classifier_model_path}")
            print(f"[INFERENCE] Falling back to legacy pipeline...")

        self._load_legacy_pipeline()
        self._loaded = True

    def _load_legacy_pipeline(self):
        from src.gradcam import GradCAM

        print(f"[INFERENCE] Loading legacy single-model pipeline...")
        print(f"[INFERENCE]   Model: {LEGACY_MODEL_PATH}")

        if not LEGACY_MODEL_PATH.exists():
            raise FileNotFoundError(
                "No runnable model found. Expected YOLO pipeline artifacts or legacy backup at "
                f"{LEGACY_MODEL_PATH}."
            )

        self.legacy_model = MTL_EfficientUNetPlusPlus().to(DEVICE)
        self.legacy_model.load_state_dict(self._torch_load_weights(LEGACY_MODEL_PATH))
        self.legacy_model.eval()
        self.legacy_gradcam = GradCAM(
            self.legacy_model,
            target_layer=self.legacy_model.smp_base.encoder.model.blocks[-2][-1],
        )
        self._mode = "legacy"
        print(f"[INFERENCE] ✅ Legacy pipeline ready (fallback only)")

    @staticmethod
    def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image

        arr = image.astype(np.float32)
        if arr.size == 0:
            return arr.astype(np.uint8)

        arr_min = float(np.nanmin(arr))
        arr_max = float(np.nanmax(arr))
        if not np.isfinite(arr_min) or not np.isfinite(arr_max):
            return np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)

        # Typical normalized floating tensors in [0, 1].
        if np.issubdtype(image.dtype, np.floating) and arr_min >= 0.0 and arr_max <= 1.0:
            return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        # Preserve contrast for 12/16-bit medical images via min-max scaling.
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_gray8(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None.")

        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            channels = image.shape[2]
            if channels == 1:
                gray = image[:, :, 0]
            elif channels == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif channels == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                raise ValueError(f"Unsupported channel count for grayscale conversion: {channels}.")
        else:
            raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}.")

        return TumorInferenceEngine._normalize_to_uint8(gray)

    @staticmethod
    def profile_window_and_clahe(gray8: np.ndarray) -> np.ndarray:
        gray8 = TumorInferenceEngine._to_gray8(gray8)
        img_f = gray8.astype(np.float32)
        profile = img_f > (img_f.max() * 0.05)
        if profile.any():
            lo, hi = img_f[profile].min(), img_f[profile].max()
        else:
            lo, hi = img_f.min(), img_f.max()

        img_w = np.clip((img_f - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
        img_u8 = img_w.astype(np.uint8)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_u8)

    @staticmethod
    def load_rgb8(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image '{path}'. The file may be unreadable or unsupported.")

        logger.debug(
            "[INFERENCE DEBUG] loaded image shape=%s dtype=%s min=%.3f max=%.3f",
            img.shape,
            img.dtype,
            float(np.min(img)),
            float(np.max(img)),
        )

        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3:
            channels = img.shape[2]
            if channels == 1:
                rgb = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
            elif channels == 3:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError(f"Unsupported image channel count {channels} for '{path}'.")
        else:
            raise ValueError(f"Unsupported image shape {img.shape} for '{path}'.")

        rgb = TumorInferenceEngine._normalize_to_uint8(rgb)

        gray = TumorInferenceEngine._to_gray8(rgb)
        prepped = TumorInferenceEngine.profile_window_and_clahe(gray)
        rgb_out = cv2.cvtColor(prepped, cv2.COLOR_GRAY2RGB)

        logger.debug(
            "[INFERENCE DEBUG] final preprocessing image shape=%s dtype=%s min=%d max=%d",
            rgb_out.shape,
            rgb_out.dtype,
            int(np.min(rgb_out)),
            int(np.max(rgb_out)),
        )
        return rgb_out

    def preprocess(self, image_path: str):
        rgb = self.load_rgb8(image_path)
        gray = self._to_gray8(rgb)
        resized = cv2.resize(gray, (self.config.image_size, self.config.image_size), interpolation=cv2.INTER_LINEAR)
        logger.debug("[INFERENCE DEBUG] preprocess resized image shape=%s dtype=%s", resized.shape, resized.dtype)
        tensor = torch.tensor(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        return gray, tensor

    @staticmethod
    def iou_xyxy(a, b):
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    @staticmethod
    def second_nms(boxes, confs, iou_thresh=0.3):
        if len(boxes) == 0:
            return boxes, confs
        order = confs.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            ious = np.array([TumorInferenceEngine.iou_xyxy(boxes[i], boxes[j]) for j in order[1:]])
            order = order[1:][ious < iou_thresh]
        return boxes[keep], confs[keep]

    @staticmethod
    def make_fused_3ch(rgb_crop: np.ndarray) -> np.ndarray:
        gray = TumorInferenceEngine._to_gray8(rgb_crop)
        ch0 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
        ch1 = np.clip(gray.astype(np.float32) * 1.5 - blurred.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        ch2 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        return cv2.merge([ch0, ch1, ch2])

    @staticmethod
    def class_probabilities_from_logits(cls_logits: torch.Tensor) -> tuple[list[float], float, float]:
        logits = cls_logits.detach().float().cpu().view(-1)
        raw_logits = [float(v) for v in logits.tolist()]

        if logits.numel() == 1:
            p_malig = float(torch.sigmoid(logits[0]).item())
            return raw_logits, 1.0 - p_malig, p_malig

        probabilities = torch.softmax(logits, dim=0)
        return raw_logits, float(probabilities[0].item()), float(probabilities[1].item())

    def fixed_physical_crop(self, rgb: np.ndarray, cx: int, cy: int) -> np.ndarray:
        win_px = self.config.physical_window_px
        h, w = rgb.shape[:2]
        half = win_px // 2
        x0, y0 = cx - half, cy - half
        x1, y1 = cx + half, cy + half

        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - w)
        pad_bottom = max(0, y1 - h)

        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(w, x1), min(h, y1)
        crop = rgb[y0c:y1c, x0c:x1c].copy()

        if any([pad_left, pad_top, pad_right, pad_bottom]):
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        return crop

    def _predict_run85_numpy(self, image_path: str):
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        rgb = self.load_rgb8(image_path)
        logger.debug("[INFERENCE DEBUG] YOLO input image shape=%s dtype=%s", rgb.shape, rgb.dtype)
        h, w = rgb.shape[:2]

        res = self.detector.predict(
            rgb[..., ::-1],
            conf=self.config.confidence_threshold,
            iou=self.config.yolo_iou,
            verbose=False,
        )[0]

        full_mask = np.zeros((h, w), np.float32)
        full_cam = np.zeros((h, w), np.float32)
        dets = []
        raw_detections: list[dict] = []
        low_conf_probe_detections: list[dict] = []
        raw_detection_count = int(len(res.boxes)) if res.boxes is not None else 0
        logger.debug("[INFERENCE DEBUG] YOLO detections count: %d", raw_detection_count)

        if res.boxes is not None and len(res.boxes) > 0:
            sx, sy = w / res.orig_shape[1], h / res.orig_shape[0]
            boxes_raw = res.boxes.xyxy.cpu().numpy().copy()
            boxes_raw[:, [0, 2]] *= sx
            boxes_raw[:, [1, 3]] *= sy
            confs_raw = res.boxes.conf.cpu().numpy()
            cls_raw = res.boxes.cls.cpu().numpy() if getattr(res.boxes, "cls", None) is not None else np.zeros_like(confs_raw)
            logger.debug("[INFERENCE DEBUG] YOLO confidence raw: %s", _format_float_list(confs_raw))

            for i in range(len(confs_raw)):
                raw_detections.append(
                    {
                        "box_xyxy": [float(v) for v in boxes_raw[i].tolist()],
                        "confidence": float(confs_raw[i]),
                        "class": float(cls_raw[i]),
                    }
                )

            boxes_nms, confs_nms = self.second_nms(boxes_raw, confs_raw, iou_thresh=self.config.second_nms_iou)
            order = confs_nms.argsort()[::-1][: self.config.yolo_topk]
            logger.debug("[INFERENCE DEBUG] YOLO detections after NMS/top-k: %d", int(len(order)))
            logger.debug("[INFERENCE DEBUG] YOLO confidence after NMS: %s", _format_float_list(confs_nms))

            for k in order:
                b = boxes_nms[k]
                cx_b = int((b[0] + b[2]) / 2)
                cy_b = int((b[1] + b[3]) / 2)

                crop = self.fixed_physical_crop(rgb, cx_b, cy_b)
                fused = self.make_fused_3ch(crop)
                fused_resized = cv2.resize(
                    fused,
                    (self.config.image_size, self.config.image_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                logger.debug(
                    "[INFERENCE DEBUG] Stage-2 fused image shape=%s dtype=%s",
                    fused_resized.shape,
                    fused_resized.dtype,
                )

                x = ((torch.from_numpy(fused_resized).float().permute(2, 0, 1) / 255.0 - 0.5) / 0.5).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    seg, cls = self.classifier(x)

                win_px = self.config.physical_window_px
                half = win_px // 2

                x0_b = max(0, cx_b - half)
                y0_b = max(0, cy_b - half)
                x1_b = min(w, cx_b + half)
                y1_b = min(h, cy_b + half)

                pad_left = max(0, -(cx_b - half))
                pad_top = max(0, -(cy_b - half))
                valid_w = x1_b - x0_b
                valid_h = y1_b - y0_b

                pm_full = cv2.resize(
                    torch.sigmoid(seg)[0, 0].cpu().numpy(),
                    (win_px, win_px),
                    interpolation=cv2.INTER_LINEAR,
                )
                pm_valid = pm_full[pad_top : pad_top + valid_h, pad_left : pad_left + valid_w]
                full_mask[y0_b:y1_b, x0_b:x1_b] = np.maximum(full_mask[y0_b:y1_b, x0_b:x1_b], pm_valid)

                if self.gradcam is not None:
                    cam = self.gradcam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
                    cam_full = cv2.resize(cam, (win_px, win_px), interpolation=cv2.INTER_LINEAR)
                    cam_valid = cam_full[pad_top : pad_top + valid_h, pad_left : pad_left + valid_w]
                    full_cam[y0_b:y1_b, x0_b:x1_b] = np.maximum(full_cam[y0_b:y1_b, x0_b:x1_b], cam_valid)

                raw_logits, p_benign, p_malig = self.class_probabilities_from_logits(cls)
                logger.debug(
                    "[INFERENCE DEBUG] Detection %d: YOLO confidence=%.4f, "
                    "Stage-2 raw classification logit(s)=%s, "
                    "Stage-2 probability benign=%.4f malignant=%.4f",
                    int(k),
                    float(confs_nms[k]),
                    _format_float_list(raw_logits),
                    p_benign,
                    p_malig,
                )
                dets.append(
                    {
                        "box": (x0_b, y0_b, x1_b, y1_b),
                        "conf": float(confs_nms[k]),
                        "stage2_logits": raw_logits,
                        "p_benign": p_benign,
                        "p_malig": p_malig,
                    }
                )
        else:
            logger.debug("[INFERENCE DEBUG] YOLO confidence raw: []")

            # Debug-only probe: inspect whether detections exist at very low confidence.
            try:
                probe_res = self.detector.predict(
                    rgb[..., ::-1],
                    conf=0.01,
                    iou=0.95,
                    verbose=False,
                )[0]
                if probe_res.boxes is not None and len(probe_res.boxes) > 0:
                    sx, sy = w / probe_res.orig_shape[1], h / probe_res.orig_shape[0]
                    probe_boxes = probe_res.boxes.xyxy.cpu().numpy().copy()
                    probe_boxes[:, [0, 2]] *= sx
                    probe_boxes[:, [1, 3]] *= sy
                    probe_confs = probe_res.boxes.conf.cpu().numpy()
                    probe_cls = probe_res.boxes.cls.cpu().numpy() if getattr(probe_res.boxes, "cls", None) is not None else np.zeros_like(probe_confs)

                    for i in range(len(probe_confs)):
                        low_conf_probe_detections.append(
                            {
                                "box_xyxy": [float(v) for v in probe_boxes[i].tolist()],
                                "confidence": float(probe_confs[i]),
                                "class": float(probe_cls[i]),
                            }
                        )
            except Exception as probe_exc:
                logger.debug("[INFERENCE DEBUG] Low-confidence probe failed: %s", probe_exc)

        if not dets:
            logger.debug("[INFERENCE DEBUG] YOLO produced no usable detections; Stage-2 classifier was not run.")

        if np.max(full_cam) <= 0:
            full_cam = full_mask.copy()

        return {
            "rgb": rgb,
            "mask": full_mask,
            "dets": dets,
            "cam": full_cam,
            "raw_detections": raw_detections,
            "low_conf_probe_detections": low_conf_probe_detections,
        }

    def _predict_legacy_numpy(self, image_path: str):
        from src.gradcam import overlay_gradcam

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image '{image_path}'. The file may be unreadable or unsupported.")
        logger.debug("[INFERENCE DEBUG] loaded image shape=%s dtype=%s", img.shape, img.dtype)
        original = img.copy()

        img = self.profile_window_and_clahe(img)
        img = cv2.resize(img, (self.config.image_size, self.config.image_size), interpolation=cv2.INTER_LINEAR)
        logger.debug("[INFERENCE DEBUG] final preprocessing image shape=%s dtype=%s", img.shape, img.dtype)
        tensor = torch.tensor(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            seg_logits, cls_logits = self.legacy_model(tensor)
            mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()

        cam = self.legacy_gradcam.generate(tensor)
        cam_overlay = overlay_gradcam(original, cam)

        p_malig = float(torch.sigmoid(cls_logits).item())
        raw_logit = float(cls_logits.detach().float().cpu().view(-1)[0].item())
        logger.debug(
            "[INFERENCE DEBUG] Legacy raw classification logit=%.4f, "
            "legacy sigmoid probability malignant=%.4f",
            raw_logit,
            p_malig,
        )
        return {
            "rgb": cv2.cvtColor(original, cv2.COLOR_GRAY2RGB),
            "mask": cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR),
            "dets": [{"p_malig": p_malig, "p_benign": 1.0 - p_malig}],
            "cam": self._to_gray8(cam_overlay).astype(np.float32) / 255.0,
        }

    def predict_numpy(self, image_path: str):
        self._lazy_load()
        if self._mode == "run85" and self._run831 is not None:
            # Delegate entirely to the clean Run 8.31 pipeline module
            return self._run831.predict(image_path)
        return self._predict_legacy_numpy(image_path)

    def predict(self, image_path: str):
        out = self.predict_numpy(image_path)

        dets = out.get("dets", [])
        if dets:
            malignant_dets = [d for d in dets if d.get("p_malig", 0.0) >= 0.5]
            if malignant_dets:
                selected_detection = max(malignant_dets, key=lambda d: d.get("p_malig", 0.0))
            else:
                selected_detection = max(dets, key=lambda d: d.get("p_benign", 0.0))
                
            p_malig = float(selected_detection.get("p_malig", 0.0))
            label = "Malignant" if p_malig >= 0.5 else "Benign"
            confidence = p_malig if label == "Malignant" else 1.0 - p_malig
        else:
            label = "No Detection"
            confidence = 0.0

        mask_bin = (out["mask"] > self.config.mask_threshold).astype(np.uint8) * 255
        if label == "No Detection":
            traces = out.get("low_conf_probe_detections", [])
            if traces:
                segmentation_rgb = self.create_traces_overlay(out["rgb"], traces, "YOLO traces (Below 20% conf)")
                gradcam_rgb = self.create_traces_overlay(out["rgb"], traces, "No Grad-CAM available")
            else:
                segmentation_rgb = self.create_no_detection_placeholder(out["rgb"], "No detection available")
                gradcam_rgb = self.create_no_detection_placeholder(out["rgb"], "No Grad-CAM available")
            segmentation_pixmap = self.cv_to_pixmap(segmentation_rgb)
            gradcam_pixmap = self.cv_to_pixmap(gradcam_rgb)
        else:
            segmentation_rgb = self.create_segmentation_overlay_rgb(out["rgb"], mask_bin)
            gradcam_rgb = self.create_heatmap_overlay_rgb(out["rgb"], out["cam"])
            segmentation_pixmap = self.cv_to_pixmap(segmentation_rgb)
            gradcam_pixmap = self.cv_to_pixmap(gradcam_rgb)

        try:
            self._save_debug_bundle(
                image_path=image_path,
                yolo_input_rgb=out["rgb"],
                raw_detections=out.get("raw_detections", []),
                low_conf_probe_detections=out.get("low_conf_probe_detections", []),
                segmentation_overlay_rgb=segmentation_rgb,
                gradcam_overlay_rgb=gradcam_rgb,
                out=out,
                label=label,
            )
        except Exception as debug_exc:
            logger.debug("[INFERENCE DEBUG] Failed to save debug artifacts: %s", debug_exc)

        logger.debug("[INFERENCE DEBUG] final predicted label: %s", label)
        logger.debug(
            "[INFERENCE DEBUG] final displayed confidence: %.4f (%.1f%%)",
            confidence,
            confidence * 100.0,
        )

        return label, confidence, segmentation_pixmap, gradcam_pixmap

    @staticmethod
    def create_traces_overlay(rgb: np.ndarray, traces: list[dict], message: str) -> np.ndarray:
        overlay = rgb.copy()
        
        # Dim the image slightly so the traces pop out
        overlay = cv2.addWeighted(overlay, 0.6, np.zeros_like(overlay), 0.4, 0)
        
        for t in traces[:3]:
            box = t["box_xyxy"]
            conf = t["confidence"]
            x0, y0, x1, y1 = map(int, box)
            
            # Draw dashed effect by drawing a rectangle, then overlaying it with another
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
            
            # Text background
            label = f"{conf*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x0, y0 - th - 5), (x0 + tw, y0), (0, 255, 255), -1)
            cv2.putText(overlay, label, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
        h, w = rgb.shape[:2]
        cv2.putText(overlay, message, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        return overlay

    @staticmethod
    def create_no_detection_placeholder(rgb: np.ndarray, message: str) -> np.ndarray:
        h, w = rgb.shape[:2]
        placeholder = np.full((h, w, 3), 20, dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, min(w, h) / 900.0)
        thick = 1 if scale < 0.8 else 2
        text_size, _ = cv2.getTextSize(message, font, scale, thick)
        x = max(10, (w - text_size[0]) // 2)
        y = max(30, (h + text_size[1]) // 2)

        cv2.putText(placeholder, message, (x, y), font, scale, (230, 230, 230), thick, cv2.LINE_AA)
        return placeholder

    @staticmethod
    def create_segmentation_overlay_rgb(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = rgb.copy()
        overlay[mask > 0] = [255, 0, 0]
        return cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)

    def create_segmentation_overlay(self, rgb: np.ndarray, mask: np.ndarray):
        return self.cv_to_pixmap(self.create_segmentation_overlay_rgb(rgb, mask))

    @staticmethod
    def create_heatmap_overlay_rgb(rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
        try:
            from pytorch_grad_cam.utils.image import show_cam_on_image
            heat = np.clip(heat, 0.0, 1.0)
            overlay = show_cam_on_image(rgb.astype(np.float32) / 255.0, heat, use_rgb=True)
            return overlay
        except ImportError:
            heat = np.clip(heat, 0.0, 1.0)
            heat_u8 = np.uint8(255 * heat)
            heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            return cv2.addWeighted(rgb, 0.6, heatmap, 0.4, 0)

    def create_heatmap_overlay(self, rgb: np.ndarray, heat: np.ndarray):
        return self.cv_to_pixmap(self.create_heatmap_overlay_rgb(rgb, heat))

    @staticmethod
    def cv_to_pixmap(img_rgb: np.ndarray):
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(
            img_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(qimg)
