"""Discrete Cosine transform Model."""

import numpy as np

from src.utils.transforms import dct2, idct2

from .base import BaseSteganographyModel


class DCTModel(BaseSteganographyModel):
    """Discrete Cosine transform model."""

    def __init__(self, *args, factor: float = 0.1, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.factor = factor

    def encode_img(self, image: np.ndarray, to_encode: np.ndarray) -> np.ndarray:
        """Encode one image."""
        image, to_encode = self._resize_both_images(image, to_encode)
        image_c = dct2(image)
        message_c = dct2(to_encode)
        self._save_one(image_c, "cosine", "image_c")
        return idct2(image_c + self.factor * message_c).astype(np.uint8)

    def decode(self, image: np.ndarray) -> np.ndarray:
        """Decode message."""
        image_c = self._read_one("cosine", "image_c")
        return idct2((dct2(image) - image_c) / self.factor).astype(np.uint8)
