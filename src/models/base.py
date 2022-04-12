"""Base Model for steganography."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np

from src.config import TMP_FOLDER
from src.types import MessageType


class BaseSteganographyModel(ABC):
    """Base model for steganography."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Initialize the model."""
        self.tmp_folder = kwargs.get("tmp_folder", TMP_FOLDER)
        self.logger = logging.getLogger(self.__class__.__name__)

    def encode(self, image: np.ndarray, message: MessageType):
        """Encode the image."""
        if isinstance(message, str):
            return self.encode_str(image, message)
        if isinstance(message, np.ndarray):
            return self.encode_img(image, message)
        raise ValueError("Not a valid message type")

    @abstractmethod
    def decode(self, image) -> MessageType:
        """Decode the image."""

    def save(self, *args, **kwargs):
        """Save the model."""

    def load(self, *args, **kwargs):
        """Load the model."""

    def encode_str(self, image: np.ndarray, to_encode: str) -> np.ndarray:
        """Encode one string."""
        raise NotImplementedError("We can't encode str in images")

    def encode_img(self, image: np.ndarray, to_encode: np.ndarray) -> np.ndarray:
        """Encode one string."""
        raise NotImplementedError("We can't encode image in images")

    def _save_one(self, matrix: np.ndarray, layer_name: str, name: str) -> None:
        """Save one matrix."""
        path = os.path.join(self.tmp_folder, f"{layer_name}_{name}.npy")
        np.save(path, matrix)

    def _read_one(self, layer_name: str, name: str) -> np.ndarray:
        """Read one matrix."""
        path = os.path.join(self.tmp_folder, f"{layer_name}_{name}.npy")
        element = np.load(path, allow_pickle=True)
        try:
            os.remove(path)
        except PermissionError:
            self.logger.warning("Can't delete the matrix file")
        return element

    @staticmethod
    def _resize_both_images(
        image1: np.ndarray, image2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize both images to same shape."""
        size = min(image1.shape[0], image1.shape[1], image2.shape[0], image2.shape[1])
        image1_resized = cv2.resize(image1, (size, size))
        image2_resized = cv2.resize(image2, (size, size))
        return image1_resized, image2_resized
