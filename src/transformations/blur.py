"""Blur image."""

import numpy as np
import torchvision.transforms as T

from .base import BaseTransformation


class BlurTransformation(BaseTransformation):
    """Blur transformation."""

    KERNEL_SIZE = (5, 9)
    SIGMA = (0.1, 1)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.array(
            T.GaussianBlur(kernel_size=self.KERNEL_SIZE, sigma=self.SIGMA)(T.ToPILImage()(image))
        )
