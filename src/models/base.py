"""Base Model for steganography."""


from abc import ABC, abstractmethod

import numpy as np

from src.types import MessageType


class BaseSteganographyModel(ABC):
    """Base model for steganography."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        pass

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
        pass

    def save(self, *args, **kwargs):
        """Save the model."""
        pass

    def load(self, *args, **kwargs):
        """Load the model."""
        pass

    def encode_str(self, image: np.ndarray, to_encode: str) -> np.ndarray:
        """Encode one string."""
        raise NotImplementedError("We can't encode str in images")

    def decode_img(self, image: np.ndarray, to_encode: np.ndarray) -> np.ndarray:
        """Encode one string."""
        raise NotImplementedError("We can't encode image in images")
