import os
import cv2
import pydicom
import numpy as np
from pathlib import Path

# --- Configuration ---
INPUT_DIR = r"/path/to/CBIS-DDSM"
OUTPUT_DIR = r"/path/to/output"
TARGET_SIZE = 512

# Output subdirectories
FULL_DIR = os.path.join(OUTPUT_DIR, "full")
ROI_DIR  = os.path.join(OUTPUT_DIR, "roi")
CROP_DIR = os.path.join(OUTPUT_DIR, "cropped")

# SeriesDescription tags in CBIS-DDSM DICOM headers
FULL_MAMMO_TAG = "full mammogram images"
ROI_MASK_TAG   = "roi mask images"
CROPPED_TAG    = "cropped images"


def get_series_type(ds):
    """Identify image type from DICOM SeriesDescription header — never from file path."""
    series_desc = str(getattr(ds, 'SeriesDescription', '')).lower().strip()
    if FULL_MAMMO_TAG in series_desc:
        return "FULL"
    elif ROI_MASK_TAG in series_desc:
        return "ROI"
    elif CROPPED_TAG in series_desc:
        return "CROP"
    else:
        return "UNKNOWN"


def process_dicom(dicom_path):
    try:
        # 1. Load DICOM
        ds = pydicom.dcmread(dicom_path)

        # 2. Identify type via DICOM header
        series_type = get_series_type(ds)
        if series_type == "UNKNOWN":
            return "skipped", f"Skipped (UNKNOWN SeriesDescription): {dicom_path.name}"

        # 3. Load pixel array
        img = ds.pixel_array.astype(np.float32)
        native_h, native_w = img.shape

        # 4. MONOCHROME1 inversion — full mammograms and crops only
        if series_type in ("FULL", "CROP"):
            if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == 'MONOCHROME1':
                img = np.max(img) - img

        # 5. Normalize based on type
        if series_type in ("FULL", "CROP"):
            img_min, img_max = np.min(img), np.max(img)
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min) * 65535.0
            img = img.astype(np.uint16)
        else:
            # ROI mask — binary 0/255, no normalization
            img = (img > 0).astype(np.uint8) * 255

        # 6. Aspect-ratio preserving resize
        h, w = img.shape
        scale   = TARGET_SIZE / max(h, w)
        new_w   = int(w * scale)
        new_h   = int(h * scale)

        interp      = cv2.INTER_NEAREST if series_type == "ROI" else cv2.INTER_AREA
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # 7. Pad to square canvas
        dtype  = np.uint8 if series_type == "ROI" else np.uint16
        canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=dtype)
        y_offset = (TARGET_SIZE - new_h) // 2
        x_offset = (TARGET_SIZE - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

        # 8. Build filename — PatientID + type + SOPInstanceUID (no collisions)
        clean_patient_id = str(ds.PatientID).replace(" ", "_").replace("/", "-")
        instance_uid     = str(ds.SOPInstanceUID).split(".")[-1]

        if series_type == "FULL":
            filename    = f"{clean_patient_id}_FULL_{instance_uid}.png"
            output_path = os.path.join(FULL_DIR, filename)
        elif series_type == "ROI":
            filename    = f"{clean_patient_id}_ROI_{instance_uid}.png"
            output_path = os.path.join(ROI_DIR, filename)
        else:  # CROP
            filename    = f"{clean_patient_id}_CROP_{native_h}x{native_w}_{instance_uid}.png"
            output_path = os.path.join(CROP_DIR, filename)

        cv2.imwrite(output_path, canvas)
        return "success", filename

    except Exception as e:
        return "error", f"Failed {dicom_path.name}: {str(e)}"


def main():
    print("Starting CBIS-DDSM Decompression and Standardization...")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    for d in [FULL_DIR, ROI_DIR, CROP_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    dicom_files = list(Path(INPUT_DIR).rglob("*.dcm"))
    total_files = len(dicom_files)
    print(f"Found {total_files} DICOM files. Beginning processing...\n")

    counts   = {"success": 0, "skipped": 0, "error": 0}
    fail_log = []

    for i, file_path in enumerate(dicom_files, 1):
        status, msg = process_dicom(file_path)
        counts[status] += 1

        if status == "error":
            fail_log.append(msg)
            print(f"ERROR [{i}/{total_files}]: {msg}")
        elif i % 100 == 0 or i == total_files:
            print(f"Progress: [{i}/{total_files}] | "
                  f"Saved: {counts['success']} | "
                  f"Skipped: {counts['skipped']} | "
                  f"Errors: {counts['error']}")

    print(f"\nDone!")
    print(f"  Saved:   {counts['success']}")
    print(f"  Skipped: {counts['skipped']} (unknown SeriesDescription)")
    print(f"  Errors:  {counts['error']}")

    if fail_log:
        fail_path = os.path.join(OUTPUT_DIR, "failed_files.txt")
        with open(fail_path, "w") as f:
            f.write("\n".join(fail_log))
        print(f"  Error log: {fail_path}")


if __name__ == "__main__":
    main()
