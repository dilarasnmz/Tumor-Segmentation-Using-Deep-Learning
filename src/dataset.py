import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


FULL_PATH_CANDIDATES = ["Full_Path", "full_path", "image_path"]
ROI_PATH_CANDIDATES = ["ROI_Path", "roi_path", "mask_path"]
CROP_PATH_CANDIDATES = ["Crop_Path", "crop_path"]
CROP_ROI_CANDIDATES = ["Crop_ROI_Path", "crop_roi_path", "CropROI_Path"]


def resolve_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of {candidates} found in dataframe columns")


def sanitize_path(raw_path):
    if raw_path is None:
        return raw_path
    path = str(raw_path).replace("\\", "/")

    # Notebook exports can include absolute local paths. Keep only project-relative tail.
    markers = [
        "CBIS-DDSM-512-FULL1/CBIS-DDSM-512-FULL1/",
        "CBIS-DDSM-512-FULL1/",
        "CBIS-DDSM-1024fixed2/",
    ]
    for marker in markers:
        if marker in path:
            path = path.split(marker, 1)[-1]
    return path


def join_data_path(data_dir, rel_or_abs_path):
    p = sanitize_path(rel_or_abs_path)
    if p is None:
        raise FileNotFoundError("Empty path value in dataset row")
    if os.path.isabs(p):
        return p
    return os.path.join(data_dir, p)


def load_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    # Match notebook preprocessing: CLAHE before normalization.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0  # normalize
    img = np.expand_dims(img, axis=0)     # -> [1, H, W]
    return img


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {path}")

    mask = (mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=0)  # -> [1, H, W]
    return mask


class CBISDDSMDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.full_col = resolve_col(self.df, FULL_PATH_CANDIDATES)
        self.roi_col = resolve_col(self.df, ROI_PATH_CANDIDATES)

        if "Label" not in self.df.columns and "Pathology" in self.df.columns:
            self.df = self.df.copy()
            self.df["Label"] = self.df["Pathology"].map(
                {"BENIGN": 0.0, "MALIGNANT": 1.0}
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = join_data_path(self.data_dir, row[self.full_col])
        roi_path = join_data_path(self.data_dir, row[self.roi_col])

        image = load_grayscale(img_path)
        roi = load_mask(roi_path)

        data_dict = {
            "image": image,
            "roi": roi,
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return {
            "image": torch.tensor(data_dict["image"], dtype=torch.float32),
            "roi": torch.tensor(data_dict["roi"], dtype=torch.float32),
            "label": torch.tensor([float(row["Label"])], dtype=torch.float32),
        }


class CropCBISDDSMDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.crop_col = resolve_col(self.df, CROP_PATH_CANDIDATES)
        self.crop_roi_col = resolve_col(self.df, CROP_ROI_CANDIDATES)

        if "Label" not in self.df.columns and "Pathology" in self.df.columns:
            self.df = self.df.copy()
            self.df["Label"] = self.df["Pathology"].map(
                {"BENIGN": 0.0, "MALIGNANT": 1.0}
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = join_data_path(self.data_dir, row[self.crop_col])
        roi_path = join_data_path(self.data_dir, row[self.crop_roi_col])

        image = load_grayscale(img_path)
        roi = load_mask(roi_path)

        data_dict = {
            "image": image,
            "roi": roi,
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return {
            "image": torch.tensor(data_dict["image"], dtype=torch.float32),
            "roi": torch.tensor(data_dict["roi"], dtype=torch.float32),
            "label": torch.tensor([float(row["Label"])], dtype=torch.float32),
        }