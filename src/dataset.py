import os

import nibabel as nib
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from config import CFG
from preprocessing import normalize_slice


class BraTSClassificationDataset(Dataset):
    """
    PyTorch dataset for binary classification of BraTS MRI slices.

    Label 0: non-tumor slice
    Label 1: tumor-containing slice
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

        image_volume = nib.load(image_path).get_fdata()
        image = image_volume[:, :, slice_index]

        image = normalize_slice(image)

        # Add channel dimension: [H, W] -> [1, H, W]
        image = np.expand_dims(image, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


def create_dataloaders(train_df, val_df, test_df):
    """
    Create train, validation and test dataloaders.
    """

    train_dataset = BraTSClassificationDataset(train_df)
    val_dataset = BraTSClassificationDataset(val_df)
    test_dataset = BraTSClassificationDataset(test_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader