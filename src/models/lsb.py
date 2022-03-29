"""LSB Model."""

import numpy as np

from .base import BaseStenographyModel, MessageType


class LSBModel(BaseStenographyModel):
    """LSB model."""

    HANDLE_TEXT = True

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)

    def encode(self, image: np.ndarray, message: MessageType):
        """Encode the message."""
        return super().encode(image, message)
