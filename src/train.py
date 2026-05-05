import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import CBISDDSMDataset
from model import MTL_EfficientUNetPlusPlus, BinaryFocalLoss, calc_dice

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# CSV FIX
# -------------------------------------------------
def fix_csv(df):
    df["Label"] = df["Pathology"].map(
        {"BENIGN": 0.0, "MALIGNANT": 1.0}
    )
    return df


# -------------------------------------------------
# DATALOADERS  (WINDOWS SAFE)
# -------------------------------------------------
def get_loaders(csv_path, data_dir, batch_size=8):
    df = pd.read_csv(csv_path)
    df = fix_csv(df)

    train_df, test_df = train_test_split(
        df, test_size=0.12, random_state=42, stratify=df["Label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=42, stratify=train_df["Label"]
    )

    train_ds = CBISDDSMDataset(train_df, data_dir)
    val_ds   = CBISDDSMDataset(val_df, data_dir)
    test_ds  = CBISDDSMDataset(test_df, data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,      # 🔥 Windows fix
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# -------------------------------------------------
# TRAIN ONE EPOCH
# -------------------------------------------------
def train_one_epoch(model, loader, optimizer, seg_loss_fn, cls_loss_fn):
    model.train()
    total_loss = 0

    loop = tqdm(loader)
    for batch in loop:
        images = batch["image"].to(DEVICE)
        masks  = batch["roi"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        seg_out, cls_out = model(images)

        seg_loss = seg_loss_fn(seg_out, masks)
        cls_loss = cls_loss_fn(cls_out, labels)

        loss = seg_loss + cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


# -------------------------------------------------
# VALIDATION
# -------------------------------------------------
@torch.no_grad()
def validate(model, loader, seg_loss_fn, cls_loss_fn):
    model.eval()
    total_dice = 0

    for batch in loader:
        images = batch["image"].to(DEVICE)
        masks  = batch["roi"].to(DEVICE)

        seg_out, _ = model(images)
        dice = calc_dice(seg_out, masks)
        total_dice += dice

    return total_dice / len(loader)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    DATA_DIR = "data/CBIS-DDSM-512-FULL1/CBIS-DDSM-512-FULL1"
    CSV_PATH = "data/CBIS_Master_Index.csv"
    SAVE_PATH = "models/best_model.pth"

    train_loader, val_loader, _ = get_loaders(CSV_PATH, DATA_DIR)

    model = MTL_EfficientUNetPlusPlus().to(DEVICE)

    seg_loss_fn = torch.nn.BCEWithLogitsLoss()
    cls_loss_fn = BinaryFocalLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_dice = 0

    for epoch in range(10):
        print(f"\nEpoch {epoch+1}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, seg_loss_fn, cls_loss_fn
        )

        val_dice = validate(model, val_loader, seg_loss_fn, cls_loss_fn)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice:   {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print("Best model saved!")

    print("Training complete.")


if __name__ == "__main__":
    main()