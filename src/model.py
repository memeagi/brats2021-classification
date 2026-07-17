import torch
import torch.nn as nn

from config import CFG


class BrainTumorCNN(nn.Module):
    """
    Convolutional neural network for binary classification of
    BraTS 2021 FLAIR MRI slices.

    Input:
        [batch_size, 1, 240, 240]

    Output:
        One logit per image:
        0 = non-tumor
        1 = tumor
    """

    def __init__(self):
        super(BrainTumorCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Dropout2d(0.30)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)

        return x.squeeze(1)


# ------------------------------------------------
# Create model and test one forward pass
# ------------------------------------------------

if __name__ == "__main__":

    model = BrainTumorCNN().to(CFG.device)

    test_images = torch.randn(2, 1, 240, 240).to(CFG.device)

    with torch.no_grad():
        test_logits = model(test_images)

    print("Input shape:", test_images.shape)
    print("Output shape:", test_logits.shape)
    print("Model created successfully.")