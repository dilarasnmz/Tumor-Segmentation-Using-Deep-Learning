import os
import re
import pandas as pd
from pathlib import Path

KAGGLE_CSV_DIR = r"C:\Users\mashe\Desktop\GradDatasets\metdataKaggleGrad\archive (3)\Kaggle_CSVs"
IMAGE_DIR      = r"C:\Users\mashe\Desktop\CBIS-DDSM-1536fixed2"


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
            df['pathology'] = df['pathology'].replace(
                'BENIGN_WITHOUT_CALLBACK', 'BENIGN'
            )
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df


def get_ids_from_mask(filename):
    lesion_prefix = filename.split('_ROI_')[0]
    match = re.search(r'(.*_(?:CC|MLO))(?:_\d+)?', lesion_prefix)
    if match:
        base_id = match.group(1)
        return base_id, lesion_prefix
    return lesion_prefix, lesion_prefix


def main():
    print("Loading official Kaggle CSVs...")
    kaggle_df = load_and_combine_kaggle_csvs()

    full_dir = Path(os.path.join(IMAGE_DIR, "full"))
    roi_dir  = Path(os.path.join(IMAGE_DIR, "roi"))
    crop_dir = Path(os.path.join(IMAGE_DIR, "cropped"))

    roi_images = list(roi_dir.glob("*.png"))
    print(f"Found {len(roi_images)} ROI files.")

    master_data         = []
    success_count       = 0
    missing_label_count = 0
    missing_full_count  = 0
    missing_crop_count  = 0

    for roi_path in roi_images:
        filename = roi_path.name
        base_id, lesion_prefix = get_ids_from_mask(filename)

        try:
            abnormality_id = int(lesion_prefix.split('_')[-1])
        except ValueError:
            missing_label_count += 1
            continue

        match = kaggle_df[
            (kaggle_df['image file path'].str.contains(
                base_id, na=False, case=False)) &
            (kaggle_df['abnormality id'] == abnormality_id)
        ]

        if match.empty:
            missing_label_count += 1
            continue

        row = match.iloc[0]

        patient_id_match = re.search(r'(P_\d+)', base_id)
        true_patient_id  = patient_id_match.group(1) if patient_id_match else "UNKNOWN"

        full_match = list(full_dir.glob(f"{base_id}_FULL_*.png"))
        if not full_match:
            missing_full_count += 1
            continue

        crop_match = list(crop_dir.glob(f"{lesion_prefix}_CROP_*.png"))
        if not crop_match:
            missing_crop_count += 1

        master_data.append({
            "PatientID":        true_patient_id,
            "CropID":           lesion_prefix,
            "Abnormality_Type": row.get('abnormality type', 'UNKNOWN'),
            "Assessment":       row.get('assessment', -1),
            "Split":            "Train" if "train" in base_id.lower() else "Test",
            "Pathology":        row['pathology'],
            "Full_Path":        f"full/{full_match[0].name}",
            "ROI_Path":         f"roi/{roi_path.name}",
            "Crop_Path":        f"cropped/{crop_match[0].name}" if crop_match else None,
        })
        success_count += 1

    output_csv = os.path.join(IMAGE_DIR, "CBIS_Master_Index.csv")
    pd.DataFrame(master_data).to_csv(output_csv, index=False)

    print(f"\n--- Summary ---")
    print(f"Lesions mapped:  {success_count}")
    print(f"Missing labels:  {missing_label_count}")
    print(f"Missing FULL:    {missing_full_count}")
    print(f"Missing CROP:    {missing_crop_count}")
    print(f"Saved to:        {output_csv}")


if __name__ == "__main__":
    main()