import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from tqdm import tqdm

from config import CFG
from model import BrainTumorCNN


# ------------------------------------------------
# Load the best validation model
# ------------------------------------------------

best_model_path = f"/kaggle/working/{CFG.model_name}"

model = BrainTumorCNN().to(CFG.device)

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=CFG.device
    )
)

model.eval()

print("Best model loaded successfully.")


# ------------------------------------------------
# Generate predictions on the independent test set
# ------------------------------------------------

test_labels_all = []
test_predictions_all = []
test_probabilities_all = []

with torch.no_grad():

    test_progress = tqdm(
        test_loader,
        desc="Testing",
        leave=False
    )

    for images_batch, labels_batch in test_progress:

        images_batch = images_batch.to(
            CFG.device,
            non_blocking=True
        )

        logits = model(images_batch)

        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()

        test_labels_all.extend(
            labels_batch.numpy().astype(int)
        )

        test_predictions_all.extend(
            predictions.cpu().numpy().astype(int)
        )

        test_probabilities_all.extend(
            probabilities.cpu().numpy()
        )


# Convert lists to NumPy arrays
test_labels_all = np.asarray(test_labels_all)
test_predictions_all = np.asarray(test_predictions_all)
test_probabilities_all = np.asarray(test_probabilities_all)


# ------------------------------------------------
# Calculate test metrics
# ------------------------------------------------

test_accuracy = accuracy_score(
    test_labels_all,
    test_predictions_all
)

test_precision = precision_score(
    test_labels_all,
    test_predictions_all,
    zero_division=0
)

test_recall = recall_score(
    test_labels_all,
    test_predictions_all,
    zero_division=0
)

test_f1 = f1_score(
    test_labels_all,
    test_predictions_all,
    zero_division=0
)

test_auc = roc_auc_score(
    test_labels_all,
    test_probabilities_all
)

test_confusion_matrix = confusion_matrix(
    test_labels_all,
    test_predictions_all
)


# ------------------------------------------------
# Print final results
# ------------------------------------------------

print("\nFinal Test Results")
print("-----------------------------")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-score:  {test_f1:.4f}")
print(f"ROC-AUC:   {test_auc:.4f}")

print("\nConfusion Matrix")
print("-----------------------------")
print(test_confusion_matrix)

print("\nClassification Report")
print("-----------------------------")

print(
    classification_report(
        test_labels_all,
        test_predictions_all,
        target_names=["Non-Tumor", "Tumor"],
        digits=4,
        zero_division=0
    )
)


# ================================================================
# CONFUSION MATRIX FIGURE
# ================================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    test_confusion_matrix,
    interpolation="nearest",
    cmap="Blues"
)

plt.title("Test-Set Confusion Matrix")
plt.colorbar()

class_names = ["Non-Tumor", "Tumor"]

tick_positions = np.arange(len(class_names))

plt.xticks(
    tick_positions,
    class_names
)

plt.yticks(
    tick_positions,
    class_names
)

threshold = test_confusion_matrix.max() / 2.0

for row_index in range(test_confusion_matrix.shape[0]):
    for column_index in range(test_confusion_matrix.shape[1]):

        value = test_confusion_matrix[
            row_index,
            column_index
        ]

        plt.text(
            column_index,
            row_index,
            str(value),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if value > threshold else "black",
            fontsize=12
        )

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

plt.savefig(
    "/kaggle/working/Figure5_Confusion_Matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ================================================================
# ROC CURVE FIGURE
# ================================================================

false_positive_rate, true_positive_rate, thresholds = roc_curve(
    test_labels_all,
    test_probabilities_all
)

plt.figure(figsize=(7, 5))

plt.plot(
    false_positive_rate,
    true_positive_rate,
    linewidth=2,
    label=f"CNN Classifier (AUC = {test_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "/kaggle/working/Figure6_ROC_Curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()