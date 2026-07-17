import torch
import torch.nn as nn


def create_weighted_loss(train_df, device):
    """
    Create a binary cross-entropy loss function with positive-class
    weighting based on the training dataset.
    """

    num_negative = int((train_df["label"] == 0).sum())
    num_positive = int((train_df["label"] == 1).sum())

    positive_weight = num_negative / num_positive

    pos_weight_tensor = torch.tensor(
        [positive_weight],
        dtype=torch.float32,
        device=device
    )

    print("Negative training slices:", num_negative)
    print("Positive training slices:", num_positive)
    print("Positive-class weight:", round(positive_weight, 4))

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_tensor
    )

    return criterion