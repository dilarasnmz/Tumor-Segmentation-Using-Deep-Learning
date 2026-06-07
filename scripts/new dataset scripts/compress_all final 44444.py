import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
INPUT_DIR   = r"E:\grad proj dataset\Curated Breast Imaging Subset of Digital Database for Screening Mammography\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM"

OUTPUT_DIR  = r"C:\Users\mashe\Desktop\CBIS-DDSM-1536fixed2"
CSV_PATH    = os.path.join(INPUT_DIR, "CBIS_Master_Index.csv")
WINDOW_SIZE = 640  # from 1536px full images

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def extract_fixed_window(full_img, roi_mask, window_size=640):
    h, w = full_img.shape[:2]
    coords = np.argwhere(roi_mask > 0)
    if len(coords) == 0:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = coords.mean(axis=0).astype(int)

    half = window_size // 2
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    pad_top    = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    pad_left   = max(0, -x1)
    pad_right  = max(0, x2 - w)

    crop      = full_img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    mask_crop = roi_mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0  # black padding, no mirror
        )
        mask_crop = cv2.copyMakeBorder(
            mask_crop, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )

    return crop, mask_crop


def process_row(args):
    idx, row = args
    try:
        full_img = cv2.imread(row['Full_Path'], cv2.IMREAD_UNCHANGED)
        roi_mask = cv2.imread(row['ROI_Path'],  cv2.IMREAD_UNCHANGED)

        if full_img is None or roi_mask is None:
            return idx, None, None, "load_failed"

        crop, mask_crop = extract_fixed_window(full_img, roi_mask, WINDOW_SIZE)

        stem           = Path(row['Full_Path']).stem
        crop_path      = os.path.join(OUTPUT_DIR, f"{stem}_FCROP.png")
        crop_mask_path = os.path.join(OUTPUT_DIR, f"{stem}_FCROP_MASK.png")

        cv2.imwrite(crop_path,      crop)
        cv2.imwrite(crop_mask_path, mask_crop)

        h, w   = full_img.shape[:2]
        coords = np.argwhere(roi_mask > 0)
        cy, cx = (coords.mean(axis=0).astype(int) if len(coords) > 0
                  else [h//2, w//2])
        boundary = any([
            cy - WINDOW_SIZE//2 < 0,
            cy + WINDOW_SIZE//2 > h,
            cx - WINDOW_SIZE//2 < 0,
            cx + WINDOW_SIZE//2 > w,
        ])

        return idx, crop_path, crop_mask_path, "boundary" if boundary else "ok"

    except Exception as e:
        return idx, None, None, str(e)


def main():
    print(f"Input:       {INPUT_DIR}")
    print(f"Output:      {OUTPUT_DIR}")
    print(f"Window size: {WINDOW_SIZE}px\n")

    df = pd.read_csv(CSV_PATH)
    print(f"Processing {len(df)} rows...\n")

    results = {}
    with ProcessPoolExecutor(max_workers=4) as exe:
        futures = {exe.submit(process_row, (i, row)): i
                   for i, row in df.iterrows()}
        for f in as_completed(futures):
            idx, crop_path, mask_path, status = f.result()
            results[idx] = (crop_path, mask_path, status)
            if len(results) % 200 == 0:
                print(f"  {len(results)}/{len(df)}")

    df['Fixed_Crop_Path']      = [results[i][0] for i in range(len(df))]
    df['Fixed_Crop_Mask_Path'] = [results[i][1] for i in range(len(df))]
    df['Crop_Status']          = [results[i][2] for i in range(len(df))]

    df.to_csv(CSV_PATH, index=False)

    print(f"\nDone.")
    print(df['Crop_Status'].value_counts())
    print(f"CSV updated: {CSV_PATH}")


if __name__ == "__main__":
    main()