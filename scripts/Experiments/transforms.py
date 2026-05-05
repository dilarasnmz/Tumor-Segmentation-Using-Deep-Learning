import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from monai.transforms import MapTransform, Compose, EnsureChannelFirstd, RandFlipd, RandRotated

# --- Configuration ---
DATA_DIR = "/kaggle/input/datasets/abdelrahmanelmugh/cbis-ddsm-512-full1-wm/CBIS-DDSM-512-FULL1/CBIS-DDSM-512-FULL1" 
CSV_PATH = "CBIS_Master_Index_Clean.csv"      
RANDOM_SEED = 42

# ==============================================================================
# 1. Custom MONAI Dictionary Transforms
# ==============================================================================

class ConditionalFlipd(MapTransform):
    def __init__(self, keys, id_key="patient_id"):
        super().__init__(keys)
        self.id_key = id_key

    def __call__(self, data):
        d = dict(data)
        patient_id = d[self.id_key]
        if "_RIGHT_" in patient_id:
            for key in self.keys:
                d[key] = cv2.flip(d[key], 1)
        return d

class PectoralRemovalMLOd(MapTransform):
    def __init__(self, keys, id_key="patient_id"):
        super().__init__(keys)
        self.id_key = id_key

    def __call__(self, data):
        d = dict(data)
        patient_id = d[self.id_key]
        
        for key in self.keys:
            img = d[key]
            
            # AUTOMATIC BYPASS: CC views or images smaller than 1500px (Crops)
            if "_CC_" in patient_id or img.shape[0] < 1500:
                continue
                
            img_8u = (img / 256).astype(np.uint8)
            _, binary = cv2.threshold(img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            h, w = binary.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(binary, flood_mask, (0, 0), 255)
            
            pectoral_region = flood_mask[1:-1, 1:-1]
            d[key] = np.where(pectoral_region == 1, 0, img)
            
        return d

class CLAHE16Bitd(MapTransform):
    def __init__(self, keys, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__(keys)
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.clahe.apply(d[key].astype(np.uint16))
        return d

class ForegroundZScored(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key].astype(np.float32)
            mask = img > 0
            if mask.sum() > 0:
                img[mask] = (img[mask] - img[mask].mean()) / (img[mask].std() + 1e-8)
            d[key] = img
        return d

# ==============================================================================
# 2. Pipeline Definitions (Train vs Val/Test)
# ==============================================================================

def build_train_pipeline():
    return Compose([
        ConditionalFlipd(keys=["image", "roi"]), 
        PectoralRemovalMLOd(keys=["image"]),     
        CLAHE16Bitd(keys=["image"]),             
        ForegroundZScored(keys=["image"]),       
        EnsureChannelFirstd(keys=["image", "roi"], channel_dim="no_channel")
    ])

def build_val_pipeline():
    return Compose([
        ConditionalFlipd(keys=["image", "roi"]), 
        PectoralRemovalMLOd(keys=["image"]),     
        CLAHE16Bitd(keys=["image"]),             
        ForegroundZScored(keys=["image"]), 
        # --- NEW: Augmentations (Training Only) ---
        RandFlipd(keys=["image", "roi"], prob=0.5, spatial_axis=1),
        RandRotated(keys=["image", "roi"], range_x=15 * math.pi / 180, prob=0.5, mode=["bilinear", "nearest"], padding_mode="zeros"),
        # ------------------------------------------
        EnsureChannelFirstd(keys=["image", "roi"], channel_dim="no_channel")
    ])

# ==============================================================================
# 3. Main Execution (Testing Both Full and Crop with Augmentations)
# ==============================================================================

def main():
    df = pd.read_csv(CSV_PATH)
    train_df = df[df['Patient_Split'] == 'Train'].sample(3, random_state=RANDOM_SEED)

    # We use the train_pipeline here to visualize the augmentations
    pipeline = build_train_pipeline()

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for idx, (_, row) in enumerate(train_df.iterrows()):
        patient_id = row['PatientID']
        
        # Load BOTH images and masks
        full_path = os.path.join(DATA_DIR, row['Full_Path'])
        crop_path = os.path.join(DATA_DIR, row['Crop_Path'])
        roi_path  = os.path.join(DATA_DIR, row['ROI_Path'])
        
        raw_full = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        raw_crop = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
        raw_roi  = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)
        raw_roi  = (raw_roi > 0).astype(np.float32)

        # Process FULL image
        full_dict = pipeline({"image": raw_full, "roi": raw_roi, "patient_id": patient_id})
        proc_full = full_dict["image"].squeeze()
        proc_full_roi = full_dict["roi"].squeeze()

        # Process CROP image
        crop_dict = pipeline({"image": raw_crop, "roi": raw_roi, "patient_id": patient_id, "image_type": "CROP"})
        proc_crop = crop_dict["image"].squeeze()
        proc_crop_roi = crop_dict["roi"].squeeze()

        # Visualization
        raw_full_display = raw_full.astype(np.float32) / 65535.0

        axes[idx, 0].imshow(raw_full_display, cmap='gray')
        axes[idx, 0].set_title(f"Original FULL\n{patient_id}")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(proc_full, cmap='gray', vmin=-3, vmax=3)
        axes[idx, 1].imshow(np.ma.masked_where(proc_full_roi == 0, proc_full_roi), cmap='autumn', alpha=0.4)
        axes[idx, 1].set_title("Augmented FULL + Mask\n(Rotation/Flip Applied)")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(proc_crop, cmap='gray', vmin=-3, vmax=3)
        axes[idx, 2].imshow(np.ma.masked_where(proc_crop_roi == 0, proc_crop_roi), cmap='autumn', alpha=0.4)
        axes[idx, 2].set_title("Augmented CROP + Mask\n(Muscle Bypassed)")
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()