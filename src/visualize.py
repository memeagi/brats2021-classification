import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CFG
from model import BrainTumorCNN


# ================================================================
# 13. TRAINING AND VALIDATION CURVES
# Figure 4 for the classification paper
# ================================================================

# Epoch numbers start from 1 for publication-quality figures
epochs_completed = range(1, len(history["train_loss"]) + 1)

# ------------------------------------------------
# Plot training and validation loss
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_completed,
    history["train_loss"],
    marker="o",
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    epochs_completed,
    history["val_loss"],
    marker="o",
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.xticks(list(epochs_completed))
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "/kaggle/working/Figure4A_Training_Validation_Loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------
# Plot training and validation accuracy
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_completed,
    history["train_accuracy"],
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    epochs_completed,
    history["val_accuracy"],
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.xticks(list(epochs_completed))
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "/kaggle/working/Figure4B_Training_Validation_Accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------
# Plot validation classification metrics
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_completed,
    history["val_precision"],
    marker="o",
    linewidth=2,
    label="Precision"
)

plt.plot(
    epochs_completed,
    history["val_recall"],
    marker="o",
    linewidth=2,
    label="Recall"
)

plt.plot(
    epochs_completed,
    history["val_f1"],
    marker="o",
    linewidth=2,
    label="F1-score"
)

plt.plot(
    epochs_completed,
    history["val_auc"],
    marker="o",
    linewidth=2,
    label="ROC-AUC"
)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Classification Metrics")
plt.xticks(list(epochs_completed))
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "/kaggle/working/Figure4C_Validation_Metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("Figures saved successfully:")
print("/kaggle/working/Figure4A_Training_Validation_Loss.png")
print("/kaggle/working/Figure4B_Training_Validation_Accuracy.png")
print("/kaggle/working/Figure4C_Validation_Metrics.png")


# ================================================================
# 14. GRAD-CAM VISUALIZATION
# Figure 7 for the classification paper
# ================================================================

# ------------------------------------------------
# Grad-CAM helper
# ------------------------------------------------

class GradCAM:
    """
    Grad-CAM implementation for the BrainTumorCNN model.
    Uses the final convolutional layer of the feature extractor.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_hook = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input_data, output_data):
        self.activations = output_data.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """
        Generate a Grad-CAM heatmap for one MRI image.
        """

        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        # For binary classification, maximize the tumor logit
        score = logits[0]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        # Global-average-pool the gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


# ------------------------------------------------
# Load best model
# ------------------------------------------------

model = BrainTumorCNN().to(CFG.device)

model.load_state_dict(
    torch.load(
        f"/kaggle/working/{CFG.model_name}",
        map_location=CFG.device
    )
)

model.eval()

print("Best model loaded successfully.")


# ------------------------------------------------
# Select one correctly classified tumor example
# ------------------------------------------------

selected_image = None
selected_label = None
selected_probability = None

for image_tensor, label_tensor in test_dataset:

    if int(label_tensor.item()) != 1:
        continue

    input_tensor = image_tensor.unsqueeze(0).to(CFG.device)

    with torch.no_grad():
        logit = model(input_tensor)
        probability = torch.sigmoid(logit).item()

    prediction = int(probability >= 0.5)

    if prediction == 1:
        selected_image = image_tensor
        selected_label = int(label_tensor.item())
        selected_probability = probability
        break

if selected_image is None:
    raise RuntimeError(
        "No correctly classified tumor image was found in the test subset."
    )

print("Selected true label:", selected_label)
print("Tumor probability:", round(selected_probability, 4))


# ------------------------------------------------
# Generate Grad-CAM
# ------------------------------------------------
# The last convolutional layer is features[12]
# based on the BrainTumorCNN architecture.

target_layer = model.features[12]

grad_cam = GradCAM(
    model=model,
    target_layer=target_layer
)

input_tensor = selected_image.unsqueeze(0).to(CFG.device)
heatmap = grad_cam.generate(input_tensor)

grad_cam.remove_hooks()


# ------------------------------------------------
# Prepare MRI image and overlay
# ------------------------------------------------

mri_image = selected_image.squeeze().cpu().numpy()

# Normalize MRI image for display
mri_display = mri_image.copy()
mri_display -= mri_display.min()

if mri_display.max() > 0:
    mri_display /= mri_display.max()

# Resize Grad-CAM to MRI dimensions
heatmap_resized = cv2.resize(
    heatmap,
    (mri_display.shape[1], mri_display.shape[0])
)

# Convert heatmap to color
heatmap_color = cv2.applyColorMap(
    np.uint8(255 * heatmap_resized),
    cv2.COLORMAP_JET
)

heatmap_color = cv2.cvtColor(
    heatmap_color,
    cv2.COLOR_BGR2RGB
)

heatmap_color = heatmap_color.astype(np.float32) / 255.0

# Convert grayscale MRI to RGB
mri_rgb = np.stack(
    [mri_display, mri_display, mri_display],
    axis=-1
)

# Create overlay
overlay = 0.60 * mri_rgb + 0.40 * heatmap_color
overlay = np.clip(overlay, 0, 1)


# ------------------------------------------------
# Plot Grad-CAM figure
# ------------------------------------------------

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(mri_display, cmap="gray")
plt.title("Original FLAIR MRI")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_resized, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(
    f"Grad-CAM Overlay\nTumor Probability = {selected_probability:.3f}"
)
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "/kaggle/working/Figure7_GradCAM.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Grad-CAM figure saved to:",
    "/kaggle/working/Figure7_GradCAM.png"
)