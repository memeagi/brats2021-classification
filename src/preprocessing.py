import numpy as np


def normalize_slice(image):
    """
    Apply z-score normalization to non-zero brain pixels.

    Parameters
    ----------
    image : np.ndarray
        Two-dimensional MRI slice.

    Returns
    -------
    np.ndarray
        Normalized MRI slice.
    """
    image = image.astype(np.float32)
    brain_mask = image > 0

    if brain_mask.any():
        mean = image[brain_mask].mean()
        std = image[brain_mask].std()
        image[brain_mask] = (
            image[brain_mask] - mean
        ) / (std + 1e-8)

    image[~brain_mask] = 0.0

    return image