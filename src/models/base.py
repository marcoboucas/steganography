"""Base Model for stenography."""


from abc import ABC, abstractmethod

from src.types import MessageType


class BaseStenographyModel(ABC):
    """Base model for stenography."""

    HANDLE_TEXT = False
    HANDLE_IMAGE = False

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        pass

    @abstractmethod
    def encode(self, image, message: MessageType):
        """Encode the image."""
        pass

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
