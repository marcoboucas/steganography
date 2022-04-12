"""Identity transformation."""

import numpy as np

from .base import BaseTransformation


class IdentityTransformation(BaseTransformation):
    """Identity transformation."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image.copy()
