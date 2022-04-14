"""Flip image."""

import numpy as np
import torchvision.transforms as T

from .base import BaseTransformation


class FlipTransformation(BaseTransformation):
    """Flip transformation."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.array(T.RandomHorizontalFlip(p=1.0)(T.ToPILImage()(image)))
