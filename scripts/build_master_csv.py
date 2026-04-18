import os
import re
import pandas as pd
from pathlib import Path

# --- Configuration ---
KAGGLE_CSV_DIR = r"/path/to/Kaggle_CSVs"
IMAGE_DIR = r"/path/to/CBIS-DDSM-512"


def load_and_combine_kaggle_csvs():
    csv_files = [
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv"
    ]
    combined_df = pd.DataFrame()
    for file in csv_files:
        path = os.path.join(KAGGLE_CSV_DIR, file)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['pathology'] = df['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df


def get_ids_from_mask(filename):
    """
    Parses 'Calc-Test_P_00485_LEFT_CC_4_ROI_123.png'
    Returns:
      base_id: 'Calc-Test_P_00485_LEFT_CC' (Used to find the FULL image & Label)
      lesion_prefix: 'Calc-Test_P_00485_LEFT_CC_4' (Used to find the specific CROP)
    """
    lesion_prefix = filename.split('_ROI_')[0]
    # Regex to find the base ID ending in CC or MLO, ignoring the lesion number
    match = re.search(r'(.*_(?:CC|MLO))(?:_\d+)?', lesion_prefix)
    if match:
        base_id = match.group(1)
        return base_id, lesion_prefix
    return lesion_prefix, lesion_prefix  # Fallback


def main():
    print("Loading official Kaggle CSVs...")
    kaggle_df = load_and_combine_kaggle_csvs()

    full_dir = Path(os.path.join(IMAGE_DIR, "full"))
    roi_dir = Path(os.path.join(IMAGE_DIR, "roi"))
    crop_dir = Path(os.path.join(IMAGE_DIR, "cropped"))

    # FIX: Loop through ROI masks instead of FULL images. 1 Mask = 1 Row.
    roi_images = list(roi_dir.glob("*.png"))

    master_data = []
    success_count = 0
    missing_label_count = 0

    print("Building flattened dataset (One row per lesion)...")
    for roi_path in roi_images:
        filename = roi_path.name
        base_id, lesion_prefix = get_ids_from_mask(filename)

        # 1. Get Clinical Label (Using base_id)
        match = kaggle_df[kaggle_df['image file path'].str.contains(base_id, na=False, case=False)]
        if match.empty:
            missing_label_count += 1
            continue
        row = match.iloc[0]

        # 2. Find matching FULL image (Using base_id)
        full_match = list(full_dir.glob(f"{base_id}_FULL_*.png"))
        full_rel_path = f"full/{full_match[0].name}" if full_match else None

        # 3. Find specific matching CROP (Using lesion_prefix)
        crop_match = list(crop_dir.glob(f"{lesion_prefix}_CROP_*.png"))
        crop_rel_path = f"cropped/{crop_match[0].name}" if crop_match else None

        # If we are missing the full image for this mask, skip it
        if not full_rel_path:
            continue

        # 4. Build Row
        master_data.append({
            "PatientID": lesion_prefix,  # Use specific lesion ID as the row identifier
            "Abnormality_Type": row.get('abnormality type', 'UNKNOWN'),
            "Split": "Train" if "train" in base_id.lower() else "Test",
            "Pathology": row['pathology'],
            "Full_Path": full_rel_path,
            "ROI_Path": f"roi/{filename}",
            "Crop_Path": crop_rel_path
        })
        success_count += 1

    # Export
    output_csv = os.path.join(IMAGE_DIR, "CBIS_Master_Index.csv")
    pd.DataFrame(master_data).to_csv(output_csv, index=False)

    print("\n--- Summary ---")
    print(f"Total Lesions Mapped: {success_count}")
    print(f"Missing Labels: {missing_label_count}")
    print(f"Saved to: {output_csv}")


if __name__ == "__main__":
    main()