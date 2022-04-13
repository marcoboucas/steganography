"""LSB Model."""

import cv2
import numpy as np

from src.utils.text import bin_to_charac, convert_str_to_octet

from .base import BaseSteganographyModel


class LSBModel(BaseSteganographyModel):
    """LSB model."""

    def __init__(self, *args, end_token: str = "[END]", channel: int = 0, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.end_token = end_token
        self.channel = channel

    def encode_str(self, image: np.ndarray, to_encode: str) -> np.ndarray:
        """Encode one string."""
        text = to_encode + self.end_token
        new_image = image.copy()
        # Convert to bytes
        to_bytes = convert_str_to_octet(text)

        # Now add to the image the message
        channel_img = cv2.split(new_image)[self.channel]

        # Add the message to the red channel
        for i, byte in enumerate(to_bytes):
            x, y = i % channel_img.shape[0], i // channel_img.shape[0]
            channel_img[x, y] = self.set_last_bit(channel_img[x, y], byte)

        new_image[:, :, self.channel] = channel_img
        return new_image

    def decode(self, image: np.ndarray) -> str:
        """Decode message."""
        channel_img = cv2.split(image)[self.channel]

        # Get the message
        decoded_message = ""
        current_character = ""
        for i in range(channel_img.shape[0] * channel_img.shape[1]):
            x, y = i % channel_img.shape[0], i // channel_img.shape[0]
            current_character += self.read_last_bit(channel_img[x, y])

            if len(current_character) == 8:
                decoded_message += bin_to_charac(current_character)
                current_character = ""
                if decoded_message.endswith(self.end_token):
                    return decoded_message[: -len(self.end_token)]
        return f"Not found: '{decoded_message[:30]}'"

    @staticmethod
    def set_last_bit(number: int, bit: str) -> int:
        """Set the last bit."""
        return int(bin(number)[2:-1] + bit, 2)

    @staticmethod
    def read_last_bit(number: int) -> str:
        """Read the last bit."""
        return bin(number)[-1]
