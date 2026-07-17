import os
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import CFG
from model import BrainTumorCNN
from preprocessing import normalize_slice


# ================================================================
# 10. CREATE A BALANCED EXPERIMENTAL SUBSET
# ================================================================
# This keeps training manageable while preserving the patient-level
# separation already established between train, validation, and test.

TRAIN_SAMPLES_PER_CLASS = 5000
VAL_SAMPLES_PER_CLASS = 1000
TEST_SAMPLES_PER_CLASS = 1000


def create_balanced_subset(dataframe, samples_per_class, seed=42):
    """
    Create a balanced subset containing equal numbers of tumor and
    non-tumor MRI slices.
    """

    negative_df = dataframe[dataframe["label"] == 0]
    positive_df = dataframe[dataframe["label"] == 1]

    number_to_sample = min(
        samples_per_class,
        len(negative_df),
        len(positive_df)
    )

    negative_sample = negative_df.sample(
        n=number_to_sample,
        random_state=seed
    )

    positive_sample = positive_df.sample(
        n=number_to_sample,
        random_state=seed
    )

    balanced_df = pd.concat(
        [negative_sample, positive_sample],
        ignore_index=True
    )

    balanced_df = balanced_df.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(drop=True)

    return balanced_df


balanced_train_df = create_balanced_subset(
    train_df,
    TRAIN_SAMPLES_PER_CLASS,
    CFG.seed
)

balanced_val_df = create_balanced_subset(
    val_df,
    VAL_SAMPLES_PER_CLASS,
    CFG.seed
)

balanced_test_df = create_balanced_subset(
    test_df,
    TEST_SAMPLES_PER_CLASS,
    CFG.seed
)


# ------------------------------------------------
# Recreate datasets and dataloaders
# ------------------------------------------------

train_dataset = BraTSClassificationDataset(balanced_train_df)
val_dataset = BraTSClassificationDataset(balanced_val_df)
test_dataset = BraTSClassificationDataset(balanced_test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True
)


print("Balanced training slices:", len(balanced_train_df))
print(balanced_train_df["label"].value_counts().sort_index())

print("\nBalanced validation slices:", len(balanced_val_df))
print(balanced_val_df["label"].value_counts().sort_index())

print("\nBalanced test slices:", len(balanced_test_df))
print(balanced_test_df["label"].value_counts().sort_index())


# ================================================================
# 11. OPTIMIZED DATASET AND MIXED-PRECISION MODEL TRAINING
# ================================================================


# ------------------------------------------------
# Optimized dataset
# ------------------------------------------------

class OptimizedBraTSClassificationDataset(Dataset):
    """
    Binary classification dataset that reads only the requested
    FLAIR slice from each NIfTI volume.

    Label 0: non-tumor
    Label 1: tumor
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        patient_folder = row["patient_folder"]
        patient_id = row["patient_id"]
        slice_index = int(row["slice_index"])
        label = float(row["label"])

        image_path = os.path.join(
            patient_folder,
            f"{patient_id}_{CFG.modality}.nii.gz"
        )

        # Memory-mapped loading avoids reading the complete volume
        nii_image = nib.load(image_path, mmap=True)
        image = np.asarray(
            nii_image.dataobj[:, :, slice_index],
            dtype=np.float32
        )

        image = normalize_slice(image)

        # [H, W] -> [1, H, W]
        image = np.expand_dims(image, axis=0)

        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image_tensor, label_tensor


# ------------------------------------------------
# Recreate datasets and dataloaders
# ------------------------------------------------

train_dataset = OptimizedBraTSClassificationDataset(balanced_train_df)
val_dataset = OptimizedBraTSClassificationDataset(balanced_val_df)
test_dataset = OptimizedBraTSClassificationDataset(balanced_test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=CFG.num_workers > 0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=CFG.num_workers > 0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=CFG.num_workers > 0
)


# ------------------------------------------------
# Training configuration
# ------------------------------------------------

CFG.epochs = 5
PATIENCE = 2

model = BrainTumorCNN().to(CFG.device)

# The pilot subset is balanced, so class weighting is unnecessary
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=CFG.learning_rate
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)

use_amp = CFG.device == "cuda"
scaler = GradScaler(enabled=use_amp)

best_val_auc = 0.0
epochs_without_improvement = 0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_precision": [],
    "val_recall": [],
    "val_f1": [],
    "val_auc": []
}


# ------------------------------------------------
# Training and validation
# ------------------------------------------------

for epoch in range(CFG.epochs):

    epoch_start = time.time()

    print(f"\nEpoch {epoch + 1}/{CFG.epochs}")
    print("-" * 45)

    # ============================================================
    # Training phase
    # ============================================================

    model.train()

    running_train_loss = 0.0
    train_labels_all = []
    train_predictions_all = []

    train_progress = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for images_batch, labels_batch in train_progress:

        images_batch = images_batch.to(
            CFG.device,
            non_blocking=True
        )

        labels_batch = labels_batch.to(
            CFG.device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images_batch)
            loss = criterion(logits, labels_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_train_loss += loss.item()

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()

        train_labels_all.extend(
            labels_batch.detach().cpu().numpy().astype(int)
        )

        train_predictions_all.extend(
            predictions.detach().cpu().numpy().astype(int)
        )

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_train_loss = (
        running_train_loss / len(train_loader)
    )

    train_accuracy = accuracy_score(
        train_labels_all,
        train_predictions_all
    )


    # ============================================================
    # Validation phase
    # ============================================================

    model.eval()

    running_val_loss = 0.0

    val_labels_all = []
    val_predictions_all = []
    val_probabilities_all = []

    with torch.no_grad():

        val_progress = tqdm(
            val_loader,
            desc="Validation",
            leave=False
        )

        for images_batch, labels_batch in val_progress:

            images_batch = images_batch.to(
                CFG.device,
                non_blocking=True
            )

            labels_batch = labels_batch.to(
                CFG.device,
                non_blocking=True
            )

            with autocast(enabled=use_amp):
                logits = model(images_batch)
                loss = criterion(logits, labels_batch)

            running_val_loss += loss.item()

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).long()

            val_labels_all.extend(
                labels_batch.cpu().numpy().astype(int)
            )

            val_predictions_all.extend(
                predictions.cpu().numpy().astype(int)
            )

            val_probabilities_all.extend(
                probabilities.cpu().numpy()
            )

    average_val_loss = (
        running_val_loss / len(val_loader)
    )

    val_accuracy = accuracy_score(
        val_labels_all,
        val_predictions_all
    )

    val_precision = precision_score(
        val_labels_all,
        val_predictions_all,
        zero_division=0
    )

    val_recall = recall_score(
        val_labels_all,
        val_predictions_all,
        zero_division=0
    )

    val_f1 = f1_score(
        val_labels_all,
        val_predictions_all,
        zero_division=0
    )

    val_auc = roc_auc_score(
        val_labels_all,
        val_probabilities_all
    )


    # ------------------------------------------------------------
    # Save history
    # ------------------------------------------------------------

    history["train_loss"].append(average_train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["val_loss"].append(average_val_loss)
    history["val_accuracy"].append(val_accuracy)
    history["val_precision"].append(val_precision)
    history["val_recall"].append(val_recall)
    history["val_f1"].append(val_f1)
    history["val_auc"].append(val_auc)


    # ------------------------------------------------------------
    # Print epoch summary
    # ------------------------------------------------------------

    elapsed_minutes = (time.time() - epoch_start) / 60

    print(f"Train Loss:      {average_train_loss:.4f}")
    print(f"Train Accuracy:  {train_accuracy:.4f}")
    print(f"Val Loss:        {average_val_loss:.4f}")
    print(f"Val Accuracy:    {val_accuracy:.4f}")
    print(f"Val Precision:   {val_precision:.4f}")
    print(f"Val Recall:      {val_recall:.4f}")
    print(f"Val F1-score:    {val_f1:.4f}")
    print(f"Val ROC-AUC:     {val_auc:.4f}")
    print(f"Epoch Time:      {elapsed_minutes:.2f} minutes")

    scheduler.step(val_auc)


    # ------------------------------------------------------------
    # Model checkpoint and early stopping
    # ------------------------------------------------------------

    if val_auc > best_val_auc:

        best_val_auc = val_auc
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            f"/kaggle/working/{CFG.model_name}"
        )

        print("Best model saved.")

    else:

        epochs_without_improvement += 1

        print(
            "No validation improvement for "
            f"{epochs_without_improvement} epoch(s)."
        )

    if epochs_without_improvement >= PATIENCE:

        print("Early stopping triggered.")
        break


print("\nTraining complete.")
print(f"Best validation ROC-AUC: {best_val_auc:.4f}")