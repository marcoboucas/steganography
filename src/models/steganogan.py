"""Steganogan model."""
import os
import tempfile

import cv2
import numpy as np
from steganogan import SteganoGAN

from src.config import TMP_FOLDER

from .base import BaseSteganographyModel


class SteganoGanModel(BaseSteganographyModel):
    """SteganoGAN model."""

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.model = SteganoGAN.load(architecture="dense")

    def encode_str(self, image: np.ndarray, to_encode: str) -> np.ndarray:
        """Encode one string."""
        path = os.path.join(TMP_FOLDER, "output.png")
        with tempfile.NamedTemporaryFile(
            dir=TMP_FOLDER, mode="wb", suffix=".png", delete=False
        ) as image_:
            print(image_.name)
            cv2.imwrite(image_.name, image)
            self.model.encode(image_.name, path, to_encode)
        return cv2.imread(path)

    def decode(self, image: np.ndarray) -> str:
        """Decode message."""
        with tempfile.NamedTemporaryFile(
            dir=TMP_FOLDER, mode="wb", suffix=".png", delete=False
        ) as image_:
            cv2.imwrite(image_.name, image)
            print(image_.name)
            return self.model.decode(image_.name)
