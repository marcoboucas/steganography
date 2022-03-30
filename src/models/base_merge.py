"""Merge Model."""

import cv2
import numpy as np

from .base import BaseSteganographyModel, MessageType


class MergeModel(BaseSteganographyModel):
    """Merge model."""

    def __init__(self, *args, width: int = 100, height: int = 100, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height

    def encode_img(self, image: np.ndarray, img_to_hide: np.ndarray) -> np.ndarray:
        """Encode one image."""
        new_image = image.copy()

        img_to_hide = cv2.resize(img_to_hide, (self.width, self.height))
        new_image[0 : img_to_hide.shape[0], 0 : img_to_hide.shape[1]] = img_to_hide

        return new_image

    def decode(self, img: np.ndarray) -> np.ndarray:
        """Decode message."""
        return img[0 : self.width, 0 : self.height]
