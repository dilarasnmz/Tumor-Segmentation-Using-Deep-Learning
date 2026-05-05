import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tqdm import tqdm

from src.dataset import CBISDDSMDataset
from src.model import MTL_EfficientUNetPlusPlus, calc_dice


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/CBIS-DDSM-512-FULL1/CBIS-DDSM-512-FULL1"
CSV_PATH = "data/CBIS_Master_Index.csv"
MODEL_PATH = "models/best_model.pth"

BATCH_SIZE = 4


def fix_csv(df):
    df["Label"] = df["Pathology"].map({
        "BENIGN": 0.0,
        "MALIGNANT": 1.0,
    })
    df = df.dropna(subset=["Label"]).copy()
    return df


def get_test_loader():
    df = pd.read_csv(CSV_PATH)
    df = fix_csv(df)

    train_df, test_df = train_test_split(
        df,
        test_size=0.12,
        random_state=42,
        stratify=df["Label"],
    )

    test_dataset = CBISDDSMDataset(test_df, DATA_DIR)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return test_loader


@torch.no_grad()
def evaluate():
    print(f"Using device: {DEVICE}")

    test_loader = get_test_loader()

    model = MTL_EfficientUNetPlusPlus().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dice_scores = []
    all_labels = []
    all_probs = []
    all_preds = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch["image"].to(DEVICE)
        masks = batch["roi"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        seg_out, cls_out = model(images)

        dice = calc_dice(seg_out, masks)
        dice_scores.append(dice)

        probs = torch.sigmoid(cls_out).detach().cpu().numpy().flatten()
        labels_np = labels.detach().cpu().numpy().flatten()

        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_labels.extend(labels_np)
        all_preds.extend(preds)

    avg_dice = sum(dice_scores) / len(dice_scores)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n===== TEST RESULTS =====")
    print(f"Test Dice:     {avg_dice:.4f}")
    print(f"Test AUC:      {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    evaluate()