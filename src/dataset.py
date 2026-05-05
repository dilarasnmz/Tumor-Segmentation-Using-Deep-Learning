import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    # CLAHE burada, LOCAL (pickle-safe)
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.data_dir, row["Full_Path"])
        roi_path = os.path.join(self.data_dir, row["ROI_Path"])

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
            "label": torch.tensor([row["Label"]], dtype=torch.float32),
        }


class CropCBISDDSMDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.data_dir, row["Crop_Path"])
        roi_path = os.path.join(self.data_dir, row["Crop_ROI_Path"])

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
            "label": torch.tensor([row["Label"]], dtype=torch.float32),
        }