# ==============================================================================
# EXP-05 — Staged Training (Phase 1: Seg Only -> Phase 2: MTL)
# ==============================================================================

import os
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import monai
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from monai.transforms import MapTransform, Compose, EnsureChannelFirstd, RandFlipd, RandRotated
from monai.losses import TverskyLoss
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tqdm import tqdm

# --- Configuration ---
DATA_DIR      = "/kaggle/input/datasets/abdelrahmanelmugh/cbis-ddsm-512-full1-wm/CBIS-DDSM-512-FULL1/CBIS-DDSM-512-FULL1"
CSV_PATH      = "/kaggle/working/CBIS_Master_Index_Clean.csv"
BATCH_SIZE    = 4
MAX_EPOCHS    = 30
LEARNING_RATE = 1e-4
PATIENCE      = 10
RANDOM_SEED   = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
# 1. Transforms
# ==============================================================================

class ConditionalFlipd(MapTransform):
    def __init__(self, keys, id_key="patient_id"):
        super().__init__(keys)
        self.id_key = id_key

    def __call__(self, data):
        d = dict(data)
        if "RIGHT" in d[self.id_key].upper():
            for key in self.keys:
                d[key] = cv2.flip(d[key], 1)
        return d

class PectoralRemovalMLOd(MapTransform):
    def __init__(self, keys, id_key="patient_id"):
        super().__init__(keys)
        self.id_key = id_key

    def __call__(self, data):
        d = dict(data)
        if "_CC" in d[self.id_key].upper() or d.get("image_type") == "CROP":
            return d
        for key in self.keys:
            img    = d[key]
            img_8u = (img / 256).astype(np.uint8)
            _, binary = cv2.threshold(img_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            h, w   = binary.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(binary, flood_mask, (0, 0), 255)
            pectoral = flood_mask[1:-1, 1:-1]
            d[key]  = np.where(pectoral == 1, 0, img)
        return d

class CLAHE16Bitd(MapTransform):
    def __init__(self, keys, clip_limit=4.0, tile_grid_size=(8, 8)):
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
            img  = d[key].astype(np.float32)
            mask = img > 0
            if mask.sum() > 0:
                img[mask] = (img[mask] - img[mask].mean()) / (img[mask].std() + 1e-8)
            d[key] = img
        return d

def build_val_pipeline():
    return Compose([
        ConditionalFlipd(keys=["image", "roi"]),
        PectoralRemovalMLOd(keys=["image"]),
        CLAHE16Bitd(keys=["image"]),
        ForegroundZScored(keys=["image"]),
        EnsureChannelFirstd(keys=["image", "roi"], channel_dim="no_channel"),
    ])

def build_train_pipeline():
    return Compose([
        ConditionalFlipd(keys=["image", "roi"]),
        PectoralRemovalMLOd(keys=["image"]),
        CLAHE16Bitd(keys=["image"]),
        ForegroundZScored(keys=["image"]),
        EnsureChannelFirstd(keys=["image", "roi"], channel_dim="no_channel"),
        RandFlipd(keys=["image", "roi"], prob=0.5, spatial_axis=1),
        RandRotated(keys=["image", "roi"], range_x=15 * math.pi / 180, prob=0.5, mode=["bilinear", "nearest"], padding_mode="zeros"),
    ])

train_pipeline = build_train_pipeline()
val_pipeline   = build_val_pipeline()

# ==============================================================================
# 2. Dataset (Strictly 1-to-1, No Crops)
# ==============================================================================

class CBISDDSMDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df       = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row          = self.df.iloc[idx]
        patient_id   = row['PatientID']

        img_path   = os.path.join(self.data_dir, row['Full_Path'])
        roi_path   = os.path.join(self.data_dir, row['ROI_Path'])

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        roi = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        if roi is None:
            raise FileNotFoundError(f"Could not load ROI: {roi_path}")

        roi = (roi > 0).astype(np.float32)

        data_dict = {
            "image":      img,
            "roi":        roi,
            "patient_id": patient_id,
            "image_type": "FULL",
            "label":      np.array([row['Label']], dtype=np.float32),
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        img_tensor   = data_dict["image"].detach().clone().to(torch.float32)
        roi_tensor   = data_dict["roi"].detach().clone().to(torch.float32)
        label_tensor = torch.tensor(data_dict["label"], dtype=torch.float32)

        return {"image": img_tensor, "roi": roi_tensor, "label": label_tensor}

# ==============================================================================
# 3. Model & Custom Loss
# ==============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1      = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu     = nn.ReLU()
        self.fc2      = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        mx  = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg + mx)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.max(x,  dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class MTL_EfficientUNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.smp_base = smp.UnetPlusPlus(
            encoder_name="tu-tf_efficientnet_b0",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation=None,
        )
        bottleneck_ch = self.smp_base.encoder.out_channels[-1]
        self.classification_branch = nn.Sequential(
            CBAM(in_planes=bottleneck_ch),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features    = self.smp_base.encoder(x)
        seg_mask    = self.smp_base.segmentation_head(self.smp_base.decoder(features))
        class_score = self.classification_branch(features[-1])
        return seg_mask, class_score

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ==============================================================================
# 4. Helpers
# ==============================================================================

def calc_dice(pred_logits, target, smooth=1e-5):
    pred         = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union        = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2. * intersection + smooth) / (union + smooth)).mean().item()

# ==============================================================================
# 5. Main
# ==============================================================================

def run_exp_05():
    set_seed(RANDOM_SEED)
    print(f"--- EXP-05: Staged Training on {DEVICE} ---")

    # Splits
    df       = pd.read_csv(CSV_PATH)
    train_df = df[df['Patient_Split'] == 'Train'].reset_index(drop=True)
    val_df   = df[df['Patient_Split'] == 'Val'].reset_index(drop=True)
    test_df  = df[df['Patient_Split'] == 'Test'].reset_index(drop=True)

    # Datasets & Loaders
    train_dataset = CBISDDSMDataset(train_df, DATA_DIR, train_pipeline)
    val_dataset   = CBISDDSMDataset(val_df,   DATA_DIR, val_pipeline)
    test_dataset  = CBISDDSMDataset(test_df,  DATA_DIR, val_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model         = MTL_EfficientUNetPlusPlus().to(DEVICE)
    
    # --- CHANGED: Using Tversky for Segmentation ---
    criterion_seg = TverskyLoss(alpha=0.3, beta=0.7, sigmoid=True)
    criterion_cls = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    
    optimizer     = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_dice     = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_auc': []}

    # INITIAL FREEZE: Phase 1 setup
    for param in model.classification_branch.parameters():
        param.requires_grad = False

    # Training
    for epoch in range(MAX_EPOCHS):
        
        # --- PHASE CONTROL ---
        if epoch == 0:
            print("\n--- PHASE 1: Segmentation Only (Epochs 1-10) ---")
            print("Classification branch is FROZEN.")
        elif epoch == 10:
            print("\n--- PHASE 2: Combined MTL (Epochs 11-30) ---")
            print("Unfreezing classification branch...")
            for param in model.classification_branch.parameters():
                param.requires_grad = True
            epochs_no_improve = 0  # reset patience for Phase 2
        model.train()
        train_loss = train_dice_loss = train_bce_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        for batch in loop:
            images = batch["image"].to(DEVICE)
            masks  = batch["roi"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            pred_masks, pred_logits = model(images)

            l_seg = criterion_seg(pred_masks, masks)
            l_cls = criterion_cls(pred_logits, labels)
            
            # --- PHASE LOSS LOGIC ---
            if epoch < 10:
                loss = l_seg
            else:
                loss = 0.7 * l_seg + 0.3 * l_cls

            loss.backward()
            optimizer.step()

            train_loss      += loss.item()
            train_dice_loss += l_seg.item()
            train_bce_loss  += l_cls.item()
            
            if epoch < 10:
                loop.set_postfix(Loss=f"{loss.item():.4f}", Tversky=f"{l_seg.item():.4f}", Focal="N/A")
            else:
                loop.set_postfix(Loss=f"{loss.item():.4f}", Tversky=f"{l_seg.item():.4f}", Focal=f"{l_cls.item():.4f}")

        # Validation
        model.eval()
        val_loss = val_dice_sum = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks  = batch["roi"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                pred_masks, pred_logits = model(images)
                l_seg = criterion_seg(pred_masks, masks)
                l_cls = criterion_cls(pred_logits, labels)

                if epoch < 10:
                    val_loss += l_seg.item()
                else:
                    val_loss += (0.7 * l_seg + 0.3 * l_cls).item()
                    
                val_dice_sum += calc_dice(pred_masks, masks)
                all_labels.extend(labels.cpu().numpy().flatten())
                all_preds.extend(torch.sigmoid(pred_logits).cpu().numpy().flatten())

        t_len      = len(train_loader)
        v_len      = len(val_loader)
        avg_t_loss = train_loss  / t_len
        avg_v_loss = val_loss    / v_len
        avg_v_dice = val_dice_sum / v_len

        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            val_auc = 0.5

        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)
        history['val_dice'].append(avg_v_dice)
        history['val_auc'].append(val_auc)

        phase_lbl = "[P1]" if epoch < 10 else "[P2]"
        print(
            f"Ep {epoch+1:02d}/{MAX_EPOCHS} {phase_lbl} | "
            f"T_Loss: {avg_t_loss:.4f} | V_Loss: {avg_v_loss:.4f} | V_Dice: {avg_v_dice:.4f} | V_AUC: {val_auc:.4f}"
        )

        scheduler.step(avg_v_dice)

        if avg_v_dice > best_val_dice:
            best_val_dice     = avg_v_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model_exp05.pth")
            print("  [*] Best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n--- Early stopping at epoch {epoch+1} ---")
                break

    # Test Evaluation
    print("\n--- Final Test Evaluation ---")
    model.load_state_dict(torch.load("/kaggle/working/best_model_exp05.pth", map_location=DEVICE))
    model.eval()

    test_dice_sum = 0.0
    t_labels, t_preds, t_binary = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            masks  = batch["roi"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            pred_masks, pred_logits = model(images)
            test_dice_sum += calc_dice(pred_masks, masks)

            probs = torch.sigmoid(pred_logits).cpu().numpy().flatten()
            t_preds.extend(probs)
            t_binary.extend((probs > 0.5).astype(int))
            t_labels.extend(labels.cpu().numpy().flatten())

    final_dice = test_dice_sum / len(test_loader)
    final_auc  = roc_auc_score(t_labels, t_preds)
    final_acc  = accuracy_score(t_labels, t_binary)
    cm         = confusion_matrix(t_labels, t_binary)

    print(f"\nTest Dice:     {final_dice:.4f}  (Target: >0.75)")
    print(f"Test AUC:      {final_auc:.4f}  (Target: >0.85)")
    print(f"Test Accuracy: {final_acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

run_exp_05()