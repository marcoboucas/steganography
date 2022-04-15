"""Merge Model."""

import cv2
import numpy as np

from .base import BaseSteganographyModel


class MergeModel(BaseSteganographyModel):
    """Merge model."""

    def __init__(self, *args, width: int = 100, height: int = 100, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height

    def encode_img(self, image: np.ndarray, to_encode: np.ndarray) -> np.ndarray:
        """Encode one image."""
        new_image = image.copy()

        to_encode = cv2.resize(to_encode, (self.width, self.height))
        new_image[0 : to_encode.shape[0], 0 : to_encode.shape[1]] = to_encode

        return new_image

    def decode_img(self, image: np.ndarray) -> np.ndarray:
        """Decode message."""
        return image[0 : self.width, 0 : self.height]
