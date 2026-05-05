import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, EnsureChannelFirstd
import warnings
warnings.filterwarnings("ignore")


class CBISDDSMDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df       = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        # Strictly 1-to-1 mapping. No dataset doubling.
        return len(self.df)

    def __getitem__(self, idx):
        row          = self.df.iloc[idx]
        patient_id   = row['PatientID']
        
        # Load ONLY the Full image and its corresponding Mask
        img_path   = os.path.join(self.data_dir, row['Full_Path'])
        roi_path   = os.path.join(self.data_dir, row['ROI_Path'])
        
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        roi   = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        if roi is None:
            raise FileNotFoundError(f"Could not load ROI: {roi_path}")

        roi = (roi > 0).astype(np.float32) # Binarize mask

        # Build MONAI Dictionary
        data_dict = {
            "image":      image,
            "roi":        roi,
            "patient_id": patient_id,
            "image_type": "FULL", # Hardcoded to FULL to bypass any crop logic
            "label":      np.array([row['Label']], dtype=np.float32)
        }

        # Apply Pipeline
        if self.transform:
            data_dict = self.transform(data_dict)

        img_tensor   = data_dict["image"].detach().clone().to(torch.float32)
        roi_tensor   = data_dict["roi"].detach().clone().to(torch.float32)
        label_tensor = torch.tensor(data_dict["label"], dtype=torch.float32)

        return {"image": img_tensor, "roi": roi_tensor, "label": label_tensor}

# ==========================================================
# Quick Test to Prove it Works
# ==========================================================
if __name__ == "__main__":
    DATA_DIR = "/kaggle/input/datasets/abdelrahmanelmugh/cbis-ddsm-512-full1-wm/CBIS-DDSM-512-FULL1/CBIS-DDSM-512-FULL1"
    df = pd.read_csv("CBIS_Master_Index_Clean.csv").head(2) # Take 2 patients
    print(f"Original CSV Rows: {len(df)}")
    
    test_transform = Compose([
        EnsureChannelFirstd(keys=["image", "roi"], channel_dim="no_channel")
    ])
    
    standard_dataset = CBISDDSMDataset(df, DATA_DIR, transform=test_transform)
    print(f"Standard Dataset Length: {len(standard_dataset)} (Strict 1-to-1 mapping)\n")
    
    # Iterate through the dataset
    for i in range(len(standard_dataset)):
        sample = standard_dataset[i]
        # We handle 'image_type' safely here in case it gets stripped by transformations
        print(f"Index {i}: {df.iloc[i]['PatientID']} | Image Shape: {sample['image'].shape} | Mask Shape: {sample['roi'].shape}")