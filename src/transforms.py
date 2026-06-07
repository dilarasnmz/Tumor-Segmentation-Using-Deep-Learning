import cv2
import math
import numpy as np

from monai.transforms import (
    MapTransform,
    Compose,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotated,
)


class ConditionalFlipd(MapTransform):
    """
    If the mammogram belongs to the RIGHT breast, flip it horizontally.
    This helps normalize left/right breast orientation.
    """

    def __init__(self, keys, id_key="patient_id"):
        super().__init__(keys)
        self.id_key = id_key

    def __call__(self, data):
        d = dict(data)
        patient_id = str(d[self.id_key])

        if "RIGHT" in patient_id.upper():
            for key in self.keys:
                d[key] = cv2.flip(d[key], 1)

        return d


class PectoralRemovalMLOd(MapTransform):
    """
    Attempts to remove the pectoral muscle region from FULL MLO mammograms.

    It skips:
    - CC views
    - Crop images
    - Small images
    """

    def __init__(self, keys, id_key="patient_id"):
        super().__init__(keys)
        self.id_key = id_key

    def __call__(self, data):
        d = dict(data)
        patient_id = str(d[self.id_key])
        image_type = d.get("image_type", "FULL")

        # Do not apply to CC views or crop images
        if "_CC" in patient_id.upper() or image_type == "CROP":
            return d

        for key in self.keys:
            img = d[key]

            # Skip small images, usually crops
            if img.shape[0] < 1500:
                continue

            img_8u = (img / 256).astype(np.uint8)

            _, binary = cv2.threshold(
                img_8u,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )

            h, w = binary.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)

            cv2.floodFill(binary, flood_mask, (0, 0), 255)

            pectoral_region = flood_mask[1:-1, 1:-1]

            d[key] = np.where(pectoral_region == 1, 0, img)

        return d


class CLAHE16Bitd(MapTransform):
    """
    Applies CLAHE contrast enhancement to 16-bit mammogram images.
    Only apply this to image, not ROI mask.
    """

    def __init__(self, keys, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__(keys)
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size,
        )

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            d[key] = self.clahe.apply(d[key].astype(np.uint16))

        return d


class ForegroundZScored(MapTransform):
    """
    Applies z-score normalization only on the foreground breast region.
    Background pixels remain unchanged.
    """

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            img = d[key].astype(np.float32)

            foreground_mask = img > 0

            if foreground_mask.sum() > 0:
                img[foreground_mask] = (
                    img[foreground_mask] - img[foreground_mask].mean()
                ) / (img[foreground_mask].std() + 1e-8)

            d[key] = img

        return d


def build_train_pipeline():
    """
    Training preprocessing pipeline.

    Applies:
    - right breast flipping
    - pectoral muscle removal for full MLO images
    - CLAHE enhancement
    - foreground z-score normalization
    - channel-first conversion
    - random flip augmentation
    - random rotation augmentation
    """

    return Compose(
        [
            ConditionalFlipd(keys=["image", "roi"]),
            PectoralRemovalMLOd(keys=["image"]),
            CLAHE16Bitd(keys=["image"]),
            ForegroundZScored(keys=["image"]),
            EnsureChannelFirstd(
                keys=["image", "roi"],
                channel_dim="no_channel",
            ),
            RandFlipd(
                keys=["image", "roi"],
                prob=0.5,
                spatial_axis=1,
            ),
            RandRotated(
                keys=["image", "roi"],
                range_x=15 * math.pi / 180,
                prob=0.5,
                mode=["bilinear", "nearest"],
                padding_mode="zeros",
            ),
        ]
    )


def build_val_pipeline():
    """
    Validation/test preprocessing pipeline.

    No random augmentation here.
    Validation and test data must stay stable.
    """

    return Compose(
        [
            ConditionalFlipd(keys=["image", "roi"]),
            PectoralRemovalMLOd(keys=["image"]),
            CLAHE16Bitd(keys=["image"]),
            ForegroundZScored(keys=["image"]),
            EnsureChannelFirstd(
                keys=["image", "roi"],
                channel_dim="no_channel",
            ),
        ]
    )


def profile_window_and_clahe(gray_image: np.ndarray) -> np.ndarray:
    """
    Notebook-aligned grayscale preprocessing:
    profile windowing over breast tissue + CLAHE enhancement.
    """
    img = gray_image.astype(np.float32)
    profile = img > (img.max() * 0.05)

    if profile.any():
        lo = img[profile].min()
        hi = img[profile].max()
    else:
        lo = img.min()
        hi = img.max()

    img = np.clip((img - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def make_fused_3ch(rgb_crop: np.ndarray) -> np.ndarray:
    """
    Notebook final fusion channels:
    Ch0=CLAHE, Ch1=unsharp mask, Ch2=top-hat.
    """
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)

    ch0 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
    ch1 = np.clip(
        gray.astype(np.float32) * 1.5 - blurred.astype(np.float32) * 0.5,
        0,
        255,
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ch2 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    return cv2.merge([ch0, ch1, ch2])


def resize_and_normalize_rgb(image_rgb: np.ndarray, size: int = 512) -> np.ndarray:
    """
    Resize and scale RGB image to [-1, 1], matching notebook model input convention.
    """
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return normalized