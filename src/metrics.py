import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def calculate_binary_metrics(logits, labels, threshold=0.5):
    """
    Convert logits to probabilities and compute binary metrics.

    Returns
    -------
    dict
        Accuracy, precision, recall, F1-score and ROC-AUC.
    """

    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).long()

    labels_np = labels.detach().cpu().numpy().astype(int)
    predictions_np = predictions.detach().cpu().numpy().astype(int)
    probabilities_np = probabilities.detach().cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(labels_np, predictions_np),
        "precision": precision_score(
            labels_np,
            predictions_np,
            zero_division=0
        ),
        "recall": recall_score(
            labels_np,
            predictions_np,
            zero_division=0
        ),
        "f1": f1_score(
            labels_np,
            predictions_np,
            zero_division=0
        )
    }

    # ROC-AUC requires both classes to be present.
    if len(np.unique(labels_np)) == 2:
        metrics["roc_auc"] = roc_auc_score(
            labels_np,
            probabilities_np
        )
    else:
        metrics["roc_auc"] = np.nan

    return metrics