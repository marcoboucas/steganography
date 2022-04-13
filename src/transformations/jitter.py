"""Color jitter."""

import numpy as np
import torchvision.transforms as T

from .base import BaseTransformation


class JitterTransformation(BaseTransformation):
    """Jitter transformation."""

    BRIGHTNESS = 0.5
    HUE = 0.3

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.array(
            T.ColorJitter(brightness=self.BRIGHTNESS, hue=self.HUE)(T.ToPILImage()(image))
        )
