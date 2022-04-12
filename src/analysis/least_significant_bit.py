"""Analysis of the least significant bit."""

import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk


@np.vectorize
def get_last_bit(number: int) -> int:
    """Get the last bit of a number."""
    return int(bin(number)[-1])


def least_significant_bit_image(image: np.ndarray) -> np.ndarray:
    """Least significant bit image."""
    image_ = image.copy()
    if image.dtype in [np.float32, np.float64]:
        image_ = (image_ * 255).astype(np.uint8)
    if image.ndim == 2:
        return _least_significant_bit_image_grayscale(image_)
    return np.stack(
        [_least_significant_bit_image_grayscale(image_[:, :, i]) for i in range(image.shape[2])],
        axis=2,
    )


def _least_significant_bit_image_grayscale(image: np.ndarray) -> np.ndarray:
    """Least significant bit image for grayscale images."""
    if image.ndim != 2:
        raise ValueError("Image must be grayscale.")
    return get_last_bit(image)


def generate_entropy_image(image: np.ndarray, disk_size: int = 10) -> np.ndarray:
    """Compute the entropy of an image."""
    if image.ndim == 2:
        return entropy(image, disk(disk_size))
    return np.stack(
        [entropy(image[:, :, i], disk(disk_size)) for i in range(image.shape[2])], axis=2
    )
