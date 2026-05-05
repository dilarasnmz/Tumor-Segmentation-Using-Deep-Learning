# ==============================================================================
# EXP-01 — Data Loading, Mass Filter & Crop ROI Matching
# ==============================================================================

import os
import re
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_DIR = "/kaggle/input/datasets/abdelrahmanelmugh/cbis-ddsm-512-full1-wm/CBIS-DDSM-512-FULL1/CBIS-DDSM-512-FULL1"
KAGGLE_CSV_DIR = "/kaggle/input/datasets/abdelrahmanelmugh/metdatakagglegrad/Kaggle_CSVs"
MASTER_CSV_PATH = "/kaggle/input/datasets/abdelrahmanelmugh/cbis-ddsm-512-full1-wm/CBIS_Master_Index.csv"
RANDOM_SEED = 42

def load_and_extract_birads():
    csv_files = [
        "mass_case_description_train_set.csv", "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv", "calc_case_description_test_set.csv"
    ]
    combined_df = pd.DataFrame()
    for file in csv_files:
        path = os.path.join(KAGGLE_CSV_DIR, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    def get_case_id(path): return str(path).split('/')[0].strip()
    combined_df['Case_ID'] = combined_df['image file path'].apply(get_case_id)
    return dict(zip(combined_df['Case_ID'], combined_df['assessment']))

def match_birads(patient_id, birads_map):
    if patient_id in birads_map: return birads_map[patient_id]
    base = re.sub(r'_\d+$', '', patient_id)
    if base in birads_map: return birads_map[base]
    return None

def match_crop_to_roi(df, data_dir):
    print("\n--- Matching Crop Files to Crop ROI Masks ---")
    
    # 1. Gather all files in the roi/ folder
    roi_dir = os.path.join(data_dir, "roi")
    all_roi_files = []
    if os.path.exists(roi_dir):
        for root, _, files in os.walk(roi_dir):
            for file in files:
                all_roi_files.append(os.path.relpath(os.path.join(root, file), data_dir))
    
    # 2. Build a mapping of Base Key -> ROI File Path
    roi_map = {}
    for roi_path in all_roi_files:
        filename = os.path.basename(roi_path)
        # Extract base key: strip everything from _ROI_ onward
        if "_ROI_" in filename:
            base_key = filename.split("_ROI_")[0]
            roi_map[base_key] = roi_path

    # 3. Match crops to ROIs
    matched_paths = []
    success_count = 0
    
    for crop_path in df['Crop_Path']:
        if pd.isna(crop_path):
            matched_paths.append(None)
            continue
            
        crop_filename = os.path.basename(crop_path)
        # Extract base key: strip everything from _CROP_ onward
        if "_CROP_" in crop_filename:
            base_key = crop_filename.split("_CROP_")[0]
            
            if base_key in roi_map:
                matched_paths.append(roi_map[base_key])
                success_count += 1
            else:
                matched_paths.append(None)
        else:
            matched_paths.append(None)
            
    df['Crop_ROI_Path'] = matched_paths
    print(f"Successfully matched {success_count} crops to aligned ROI masks out of {len(df)} total rows.")
    
    # Drop rows that failed to find a crop mask match
    df = df.dropna(subset=['Crop_ROI_Path']).copy()
    return df

def main():
    print("--- 1. Loading and Filtering Data ---")
    df = pd.read_csv(MASTER_CSV_PATH)
    initial_count = len(df)
    
    birads_map = load_and_extract_birads()
    df['BI_RADS'] = df['PatientID'].apply(lambda pid: match_birads(pid, birads_map))
    df = df[df['BI_RADS'].notna() & (df['BI_RADS'] != 3)].copy()
    print(f"Dropped {initial_count - len(df)} rows (BI-RADS 3 or missing). Remaining: {len(df)}")
    
    # Mass-only filter
    df = df[df['PatientID'].str.contains("Mass", case=False, na=False)].copy()
    print(f"Filtered for Mass cases only. Remaining: {len(df)}")
    
    # Match Crop ROIs
    df = match_crop_to_roi(df, DATA_DIR)
    
    df['Label'] = df['Pathology'].map({'MALIGNANT': 1, 'BENIGN': 0})
    df['True_Patient_ID'] = df['PatientID'].apply(lambda x: re.search(r'(P_\d+)', x).group(1))

    print("\n--- 2. Patient-Level Stratified Splitting ---")
    patient_df = df.groupby('True_Patient_ID')['Label'].max().reset_index()
    patient_df.columns = ['True_Patient_ID', 'Patient_Label']
    
    train_patients, temp_patients, _, temp_labels = train_test_split(
        patient_df['True_Patient_ID'], patient_df['Patient_Label'],
        test_size=0.30, random_state=RANDOM_SEED, stratify=patient_df['Patient_Label']
    )
    
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=(2/3), random_state=RANDOM_SEED, stratify=temp_labels
    )
    
    train_set, val_set, test_set = set(train_patients), set(val_patients), set(test_patients)

    def assign_split(pid):
        if pid in train_set: return 'Train'
        if pid in val_set: return 'Val'
        return 'Test'

    df['Patient_Split'] = df['True_Patient_ID'].apply(assign_split)

    print("\n--- 3. Visualizing Alignment (FULL | CROP | CROP-ALIGNED ROI) ---")
    sample_df = df.sample(3, random_state=RANDOM_SEED)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15)) 

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        full_path = os.path.join(DATA_DIR, row['Full_Path'])
        crop_path = os.path.join(DATA_DIR, row['Crop_Path'])
        crop_roi_path = os.path.join(DATA_DIR, row['Crop_ROI_Path'])
        
        full_img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        crop_img = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
        crop_roi_img = cv2.imread(crop_roi_path, cv2.IMREAD_UNCHANGED)
        
        full_disp = full_img.astype(np.float32) / 65535.0
        crop_disp = crop_img.astype(np.float32) / 65535.0
        crop_roi_disp = crop_roi_img.astype(np.float32) / np.max(crop_roi_img) if np.max(crop_roi_img) > 0 else crop_roi_img

        axes[idx, 0].imshow(full_disp, cmap='gray')
        axes[idx, 0].set_title(f"FULL: {row['PatientID']}")
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(crop_disp, cmap='gray')
        axes[idx, 1].set_title(f"CROP IMAGE")
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(crop_roi_disp, cmap='gray')
        axes[idx, 2].set_title(f"CROP-ALIGNED ROI MASK")
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.show()

    out_csv = "CBIS_Master_Index_Clean.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved clean dataset to: {out_csv}")

if __name__ == "__main__":
    main()