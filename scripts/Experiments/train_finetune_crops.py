# ==============================================================================
# EXP-07 — Fine-Tune on Crops (Staged) [FIXED DATASET BUG]
# ==============================================================================

class CropCBISDDSMDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df       = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['PatientID']

        img_path = os.path.join(self.data_dir, row['Crop_Path'])
        roi_path = os.path.join(self.data_dir, row['Crop_ROI_Path'])

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        roi = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)

        if img is None: raise FileNotFoundError(f"Could not load image: {img_path}")
        if roi is None: raise FileNotFoundError(f"Could not load ROI: {roi_path}")

        roi = (roi > 0).astype(np.float32)

        data_dict = {
            "image":      img,
            "roi":        roi,
            "patient_id": patient_id,
            "image_type": "CROP",  # <--- CRITICAL FIX: Prevents pectoral removal from erasing the tumor
            "label":      np.array([row['Label']], dtype=np.float32),
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        img_tensor   = torch.tensor(data_dict["image"], dtype=torch.float32)
        roi_tensor   = torch.tensor(data_dict["roi"],   dtype=torch.float32)
        label_tensor = torch.tensor(data_dict["label"], dtype=torch.float32)

        return {"image": img_tensor, "roi": roi_tensor, "label": label_tensor}

def run_exp_07_finetune():
    set_seed(RANDOM_SEED)
    print(f"--- EXP-07: Fine-Tuning on Crops ({DEVICE}) ---")

    df = pd.read_csv(CSV_PATH)
    crop_df = df.dropna(subset=['Crop_Path', 'Crop_ROI_Path']).copy()

    train_df = crop_df[crop_df['Patient_Split'] == 'Train'].reset_index(drop=True)
    val_df   = crop_df[crop_df['Patient_Split'] == 'Val'].reset_index(drop=True)
    test_df  = crop_df[crop_df['Patient_Split'] == 'Test'].reset_index(drop=True)

    # USING THE FIXED CROP DATASET
    train_dataset = CropCBISDDSMDataset(train_df, DATA_DIR, val_pipeline)
    val_dataset   = CropCBISDDSMDataset(val_df,   DATA_DIR, val_pipeline)
    test_dataset  = CropCBISDDSMDataset(test_df,  DATA_DIR, val_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = MTL_EfficientUNetPlusPlus().to(DEVICE)
    try:
        model.load_state_dict(torch.load("/kaggle/working/best_model_exp05.pth", map_location=DEVICE))
        print("Successfully loaded 'best_model_exp05.pth' from 4th run.")
    except Exception as e:
        print(f"Error loading base weights: {e}. Aborting.")
        return

    criterion_seg = TverskyLoss(alpha=0.3, beta=0.7, sigmoid=True)
    criterion_cls = BinaryFocalLoss(alpha=0.25, gamma=2.0)
    optimizer     = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_dice     = 0.0
    epochs_no_improve = 0

    for param in model.smp_base.encoder.parameters():
        param.requires_grad = False

    for epoch in range(MAX_EPOCHS):
        if epoch == 0:
            print("\n--- PHASE 1: Decoder/Head Only on Crops (Epochs 1-7) ---")
        elif epoch == 7:
            print("\n--- PHASE 2: Full Network Fine-tuning (Epochs 8-30) ---")
            for param in model.smp_base.encoder.parameters(): param.requires_grad = True
            for g in optimizer.param_groups: g['lr'] = 1e-5
            epochs_no_improve = 0 

        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        
        for batch in loop:
            images, masks, labels = batch["image"].to(DEVICE), batch["roi"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            pred_masks, pred_logits = model(images)
            l_seg = criterion_seg(pred_masks, masks)
            l_cls = criterion_cls(pred_logits, labels)
            loss  = 0.7 * l_seg + 0.3 * l_cls
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(Loss=f"{loss.item():.4f}")

        model.eval()
        val_dice_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, masks = batch["image"].to(DEVICE), batch["roi"].to(DEVICE)
                pred_masks, _ = model(images)
                val_dice_sum += calc_dice(pred_masks, masks)

        avg_v_dice = val_dice_sum / len(val_loader)
        phase_lbl = "[P1]" if epoch < 7 else "[P2]"
        print(f"Ep {epoch+1:02d}/{MAX_EPOCHS} {phase_lbl} | V_Dice: {avg_v_dice:.4f}")

        if avg_v_dice > best_val_dice:
            best_val_dice = avg_v_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model_finetuned.pth")
            print("  [*] Best finetuned model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n--- Early stopping at epoch {epoch+1} ---")
                break

run_exp_07_finetune()