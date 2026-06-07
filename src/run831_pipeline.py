"""
Run 8.31 Inference Pipeline
============================
Implements the exact notebook pipeline:

  Stage 1 — detector_yolov8s.pt  (YOLOv8s, imgsz=640, conf=0.20)
  Stage 2 — segmenter_bias_fixed_2cls.pth  (EfficientNet-B0 + UNet++,
             temperature-scaled T baked into weights)

Pipeline per detection:
  CLAHE image → YOLO → bbox → expand by CROP_MARGIN=25%
  → crop → filter fusion (CLAHE / unsharp / top-hat → 3-ch)
  → resize to 256 px → normalise (x-0.5)/0.5
  → EfficientNet UNet++ → temperature-scaled softmax
  → threshold 0.50 → Malignant / Benign (max-voting across crops)

Public API
----------
  pipeline = Run831Pipeline(yolo_path, classifier_path, device)
  pipeline.load()
  result  = pipeline.predict(image_path)

  result keys:
    "rgb"   np.ndarray (H,W,3) uint8  – min-max normalised original
    "mask"  np.ndarray (H,W)   float32 [0,1] – segmentation mask (full image)
    "cam"   np.ndarray (H,W)   float32 [0,1] – GradCAM++ heatmap (full image)
    "dets"  list[dict]  – per detection:
              box (x0,y0,x1,y1), conf, p_benign, p_malig, label
    "raw_detections"             list[dict]
    "low_conf_probe_detections"  list[dict]
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline constants  (exact values from notebook Run 8.31)
# ---------------------------------------------------------------------------
YOLO_IMGSZ        = 640     # baked into detector_yolov8s.pt
YOLO_CONF         = 0.20    # from notebook diagnostic optimisation
YOLO_IOU          = 0.8
YOLO_TOPK         = 3
SECOND_NMS_IOU    = 0.3

CROP_MARGIN       = 0.25    # expand YOLO bbox by 25% on each side
CLASSIFIER_IMGSZ  = 256     # CROP_SIZE — resize every crop to 256x256
NORM_MEAN         = 0.5     # (x/255 - mean) / std
NORM_STD          = 0.5

MALIG_THRESHOLD   = 0.50    # Youden's J optimal threshold (single model)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert any numeric array to uint8 via min-max scaling."""
    if img.dtype == np.uint8:
        return img
    arr = img.astype(np.float32)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan_to_num(arr, nan=0, posinf=255, neginf=0).clip(0, 255).astype(np.uint8)
    if np.issubdtype(img.dtype, np.floating) and lo >= 0.0 and hi <= 1.0:
        return (arr * 255).clip(0, 255).astype(np.uint8)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return (arr * 255).clip(0, 255).astype(np.uint8)


def _load_raw_rgb(path: str) -> np.ndarray:
    """
    Load image (or DICOM) as uint8 RGB with profile-windowing + CLAHE.
    YOLO was trained on CLAHE-enhanced mammograms so this must match.
    """
    if str(path).lower().endswith('.dcm'):
        import pydicom
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        if getattr(dcm, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            img = np.amax(img) - img
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot load image: '{path}'")

    # Normalise to uint8 first
    if img.ndim == 2:
        gray = _normalize_to_uint8(img)
    elif img.ndim == 3:
        ch = img.shape[2]
        if ch == 1:
            gray = _normalize_to_uint8(img[:, :, 0])
        elif ch == 3:
            gray = cv2.cvtColor(_normalize_to_uint8(img), cv2.COLOR_BGR2GRAY)
        elif ch == 4:
            gray = cv2.cvtColor(_normalize_to_uint8(img), cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported channel count {ch}")
    else:
        raise ValueError(f"Unsupported image shape {img.shape}")

    # Profile-window + CLAHE (matches YOLO training preprocessing)
    img_f = gray.astype(np.float32)
    profile = img_f > (img_f.max() * 0.05)
    lo = float(img_f[profile].min()) if profile.any() else float(img_f.min())
    hi = float(img_f[profile].max()) if profile.any() else float(img_f.max())
    img_w = np.clip((img_f - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
    img_u8 = img_w.astype(np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_u8)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def _make_fused_3ch(rgb_crop: np.ndarray) -> np.ndarray:
    """
    Filter fusion — converts a crop to the 3-channel input the classifier
    was trained on:
      ch0 = CLAHE-enhanced grayscale
      ch1 = unsharp-mask sharpened grayscale
      ch2 = top-hat morphological residual
    """
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = _normalize_to_uint8(gray)

    ch0 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
    ch1 = np.clip(
        gray.astype(np.float32) * 1.5 - blurred.astype(np.float32) * 0.5,
        0, 255,
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ch2 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    return cv2.merge([ch0, ch1, ch2])   # (H, W, 3) uint8


def _expand_bbox_crop(rgb: np.ndarray, b: np.ndarray,
                      margin: float = CROP_MARGIN) -> tuple[np.ndarray, int, int, int, int]:
    """
    Expand a YOLO bbox by `margin` fraction on each side, clamp to image,
    and return the crop plus the clamped coordinates (x0, y0, x1, y1).
    The crop is NOT padded — the coordinates already account for image edges.
    Returns (crop_rgb, x0, y0, x1, y1).
    """
    h, w = rgb.shape[:2]
    bw = b[2] - b[0]
    bh = b[3] - b[1]
    x0 = int(max(0,   b[0] - bw * margin))
    y0 = int(max(0,   b[1] - bh * margin))
    x1 = int(min(w,   b[2] + bw * margin))
    y1 = int(min(h,   b[3] + bh * margin))
    return rgb[y0:y1, x0:x1].copy(), x0, y0, x1, y1


def _preprocess_for_classifier(crop_rgb: np.ndarray) -> torch.Tensor:
    """
    Apply filter fusion → resize to CLASSIFIER_IMGSZ → normalise.
    Returns a (1, 3, 256, 256) float32 tensor.
    """
    fused = _make_fused_3ch(crop_rgb)
    resized = cv2.resize(fused, (CLASSIFIER_IMGSZ, CLASSIFIER_IMGSZ),
                         interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    t = (t - NORM_MEAN) / NORM_STD
    return t.unsqueeze(0)   # (1, 3, 256, 256)


def _softmax2_probs(cls_logits: torch.Tensor):
    """Return (p_benign, p_malig) from 2-class or binary logits."""
    logits = cls_logits.detach().float().cpu().view(-1)
    if logits.numel() == 1:
        p_malig = float(torch.sigmoid(logits[0]).item())
        return 1.0 - p_malig, p_malig
    probs = torch.softmax(logits, dim=0)
    return float(probs[0].item()), float(probs[1].item())


def _iou_xyxy(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih   = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter    = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _second_nms(boxes: np.ndarray, confs: np.ndarray,
                iou_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, confs
    order = confs.argsort()[::-1]
    keep  = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious  = np.array([_iou_xyxy(boxes[i], boxes[j]) for j in order[1:]])
        order = order[1:][ious < iou_thresh]
    return boxes[keep], confs[keep]


# ---------------------------------------------------------------------------
# GradCAM++ wrapper
# ---------------------------------------------------------------------------

class _ClsWrapper(nn.Module):
    """Exposes only the classification head output to GradCAM."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model

    def forward(self, x):
        _, cls = self.m(x)
        return cls


# ---------------------------------------------------------------------------
# Public pipeline class
# ---------------------------------------------------------------------------

class Run831Pipeline:
    """
    Notebook-aligned Run 8.31 two-stage pipeline.

    Preprocessing:
      CLAHE image → YOLO (640 px, conf=0.20)
      → bbox expanded by 25% margin → filter fusion (CLAHE/unsharp/tophat)
      → 256 px resize → (x-0.5)/0.5

    Classification:
      temperature-scaled softmax → p_malig
      → threshold 0.50 → Malignant / Benign  (max-voting across crops)
    """

    def __init__(self, yolo_path: str | Path, classifier_path: str | Path,
                 device: torch.device):
        self.yolo_path        = Path(yolo_path)
        self.classifier_path  = Path(classifier_path)
        self.device           = device

        self._detector   = None
        self._classifier = None
        self._gradcam    = None
        self._loaded     = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        if self._loaded:
            return
        self._load_detector()
        self._load_classifier()
        self._setup_gradcam()
        self._loaded = True

    def _load_detector(self):
        from ultralytics import YOLO
        logger.info("[Run831] Loading YOLO detector: %s", self.yolo_path)
        self._detector = YOLO(str(self.yolo_path))
        logger.info("[Run831] YOLO detector loaded (task=%s, names=%s).",
                    self._detector.task, self._detector.names)

    def _load_classifier(self):
        from src.model import Run85UNetPlusPlus
        from src.inference import ModelWithTemperature

        logger.info("[Run831] Loading classifier: %s", self.classifier_path)
        try:
            state = torch.load(self.classifier_path, map_location=self.device,
                               weights_only=True)
        except TypeError:
            state = torch.load(self.classifier_path, map_location=self.device)

        base = Run85UNetPlusPlus(encoder_weights=None).to(self.device)
        has_temperature = isinstance(state, dict) and "temperature" in state

        if has_temperature:
            wrapped = ModelWithTemperature(base).to(self.device)
            # State keys: "temperature", "model.encoder..." → need
            # "temperature", "model.model.encoder..." (ModelWithTemperature.model = base)
            fixed = {}
            for k, v in state.items():
                if k.startswith("model.") and not k.startswith("model.model."):
                    fixed[k.replace("model.", "model.model.", 1)] = v
                else:
                    fixed[k] = v
            wrapped.load_state_dict(fixed, strict=False)
            T = float(wrapped.temperature.item())
            logger.info("[Run831] Temperature-scaled classifier loaded. T=%.4f", T)
            self._classifier = wrapped
        else:
            fixed = {k: v for k, v in state.items() if k != "temperature"}
            base.load_state_dict(fixed, strict=False)
            logger.info("[Run831] Classifier loaded (no temperature scaling).")
            self._classifier = base

        self._classifier.eval()

    def _setup_gradcam(self):
        try:
            from pytorch_grad_cam import GradCAMPlusPlus

            base_module = (self._classifier.model
                           if hasattr(self._classifier, "temperature")
                           else self._classifier)
            enc = base_module.model.encoder

            if hasattr(enc, "_blocks"):
                target_layers = [enc._blocks[-1]]
            else:
                convs = [m for m in enc.modules() if isinstance(m, nn.Conv2d)]
                if not convs:
                    raise RuntimeError("No Conv2d found in encoder.")
                target_layers = [convs[-1]]

            cls_wrapper = _ClsWrapper(self._classifier).to(self.device).eval()
            self._gradcam = GradCAMPlusPlus(model=cls_wrapper,
                                            target_layers=target_layers)
            logger.info("[Run831] GradCAM++ ready on %s.",
                        target_layers[0].__class__.__name__)
        except Exception as exc:
            logger.warning("[Run831] GradCAM++ setup failed: %s. "
                           "Heatmap will mirror the mask.", exc)
            self._gradcam = None

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, image_path: str) -> dict:
        if not self._loaded:
            self.load()

        # ── Load image (raw, no CLAHE — YOLO needs this) ─────────────
        rgb = _load_raw_rgb(image_path)
        h, w = rgb.shape[:2]

        # ── Stage 1: YOLO detection ───────────────────────────────────
        bgr = rgb[..., ::-1]
        res = self._detector.predict(
            bgr,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            verbose=False,
        )[0]

        full_mask = np.zeros((h, w), np.float32)
        full_cam  = np.zeros((h, w), np.float32)
        dets: list[dict]      = []
        raw_detections: list[dict] = []
        low_conf_probe: list[dict] = []

        raw_count = len(res.boxes) if res.boxes is not None else 0
        logger.debug("[Run831] YOLO raw detections: %d", raw_count)

        if res.boxes is not None and raw_count > 0:
            sx = w / res.orig_shape[1]
            sy = h / res.orig_shape[0]
            boxes_raw = res.boxes.xyxy.cpu().numpy().copy()
            boxes_raw[:, [0, 2]] *= sx
            boxes_raw[:, [1, 3]] *= sy
            confs_raw = res.boxes.conf.cpu().numpy()

            for i in range(len(confs_raw)):
                raw_detections.append({
                    "box_xyxy":   boxes_raw[i].tolist(),
                    "confidence": float(confs_raw[i]),
                })

            boxes_nms, confs_nms = _second_nms(boxes_raw, confs_raw, SECOND_NMS_IOU)
            order = confs_nms.argsort()[::-1][:YOLO_TOPK]
            logger.debug("[Run831] After NMS/top-k: %d detections.", len(order))

            # ── Stage 2: bbox+margin crop → filter fusion → classifier ─
            for k in order:
                b = boxes_nms[k]

                # Expand bbox by CROP_MARGIN on each side
                crop, x0_b, y0_b, x1_b, y1_b = _expand_bbox_crop(rgb, b)
                crop_w = x1_b - x0_b
                crop_h = y1_b - y0_b

                if crop_w < 4 or crop_h < 4:
                    logger.debug("[Run831] Skipping degenerate crop for det %d", k)
                    continue

                x_tensor = _preprocess_for_classifier(crop).to(self.device)

                with torch.no_grad():
                    seg_logits, cls_logits = self._classifier(x_tensor)

                # Remap segmentation mask to expanded bbox region on full canvas
                pm_crop = cv2.resize(
                    torch.sigmoid(seg_logits)[0, 0].cpu().numpy(),
                    (crop_w, crop_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                full_mask[y0_b:y1_b, x0_b:x1_b] = np.maximum(
                    full_mask[y0_b:y1_b, x0_b:x1_b], pm_crop
                )

                # GradCAM++ heatmap remapped to same region
                if self._gradcam is not None:
                    try:
                        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                        cam_map = self._gradcam(
                            input_tensor=x_tensor,
                            targets=[ClassifierOutputTarget(1)],
                        )[0]
                        cam_crop = cv2.resize(
                            cam_map,
                            (crop_w, crop_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        full_cam[y0_b:y1_b, x0_b:x1_b] = np.maximum(
                            full_cam[y0_b:y1_b, x0_b:x1_b], cam_crop
                        )
                    except Exception as cam_exc:
                        logger.debug("[Run831] GradCAM++ failed for det %d: %s",
                                     k, cam_exc)

                p_benign, p_malig = _softmax2_probs(cls_logits)
                label = "Malignant" if p_malig >= MALIG_THRESHOLD else "Benign"
                logger.debug(
                    "[Run831] det %d  YOLO_conf=%.3f  p_benign=%.3f  "
                    "p_malig=%.3f  → %s",
                    k, float(confs_nms[k]), p_benign, p_malig, label,
                )
                dets.append({
                    "box":      (x0_b, y0_b, x1_b, y1_b),
                    "conf":     float(confs_nms[k]),
                    "p_benign": p_benign,
                    "p_malig":  p_malig,
                    "label":    label,
                })

        else:
            # Debug probe at very low confidence to help diagnose no-detection cases
            try:
                probe = self._detector.predict(
                    bgr, imgsz=YOLO_IMGSZ, conf=0.01, iou=0.95, verbose=False,
                )[0]
                if probe.boxes is not None and len(probe.boxes) > 0:
                    sx = w / probe.orig_shape[1]
                    sy = h / probe.orig_shape[0]
                    pb = probe.boxes.xyxy.cpu().numpy().copy()
                    pb[:, [0, 2]] *= sx
                    pb[:, [1, 3]] *= sy
                    pc = probe.boxes.conf.cpu().numpy()
                    for i in range(len(pc)):
                        low_conf_probe.append({
                            "box_xyxy":   pb[i].tolist(),
                            "confidence": float(pc[i]),
                        })
            except Exception:
                pass

        # Fall back to mask if GradCAM produced nothing
        if full_cam.max() <= 0:
            full_cam = full_mask.copy()

        return {
            "rgb":                        rgb,
            "mask":                       full_mask,
            "cam":                        full_cam,
            "dets":                       dets,
            "raw_detections":             raw_detections,
            "low_conf_probe_detections":  low_conf_probe,
        }
