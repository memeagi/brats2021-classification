import torch


class CFG:

    # -----------------------------
    # Random seed
    # -----------------------------
    seed = 42

    # -----------------------------
    # Dataset
    # -----------------------------
    image_size = 240

    # BraTS modalities
    modality = "flair"

    # -----------------------------
    # Training Parameters
    # -----------------------------
    batch_size = 16

    epochs = 15

    learning_rate = 1e-4

    num_workers = 2

    # -----------------------------
    # Device
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Dataset location
    # -----------------------------
    tar_path = (
        "/kaggle/input/datasets/dschettler8845/"
        "brats-2021-task1/BraTS2021_Training_Data.tar"
    )

    extract_path = "/kaggle/working/brats2021"

    # -----------------------------
    # Save model
    # -----------------------------
    model_name = "best_brain_tumor_classifier.pth"


print(CFG.device)