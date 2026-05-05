from pathlib import Path
import cv2
import numpy as np
import torch
from PySide6.QtGui import QImage, QPixmap

from src.model import MTL_EfficientUNetPlusPlus
from src.gradcam import GradCAM, overlay_gradcam


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TumorInferenceEngine:
    def __init__(self):
        self.model = MTL_EfficientUNetPlusPlus().to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()

        self.gradcam = GradCAM(
            self.model,
            target_layer=self.model.smp_base.encoder.model.blocks[-2][-1],
        )

    def preprocess(self, image_path: str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Image could not be loaded.")

        original = img.copy()

        img = cv2.resize(img, (512, 512))
        img = img.astype(np.float32) / 255.0

        tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        return original, tensor

    def predict(self, image_path: str):
        original, tensor = self.preprocess(image_path)

        with torch.no_grad():
            seg_logits, cls_logits = self.model(tensor)

            prob = torch.sigmoid(cls_logits).item()
            label = "Malignant" if prob >= 0.5 else "Benign"
            confidence = prob if label == "Malignant" else 1 - prob

            mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            mask = (mask > 0.1).astype(np.uint8) * 255
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

        segmentation_pixmap = self.create_segmentation_overlay(original, mask)

        cam = self.gradcam.generate(tensor)
        gradcam_img = overlay_gradcam(original, cam)
        gradcam_pixmap = self.cv_to_pixmap(gradcam_img)

        return label, confidence, segmentation_pixmap, gradcam_pixmap

    def create_segmentation_overlay(self, original, mask):
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

        overlay = original_rgb.copy()
        overlay[mask > 0] = [255, 0, 0]

        blended = cv2.addWeighted(original_rgb, 0.7, overlay, 0.3, 0)

        return self.cv_to_pixmap(blended)

    def cv_to_pixmap(self, img_rgb):
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(
            img_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

        return QPixmap.fromImage(qimg)