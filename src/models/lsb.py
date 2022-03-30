"""LSB Model."""

import cv2
import numpy as np

from src.utils.text import bin_to_charac, convert_octet_to_str, convert_str_to_octet

from .base import BaseSteganographyModel, MessageType


class LSBModel(BaseSteganographyModel):
    """LSB model."""

    def __init__(self, *args, end_token: str = "[END]", channel: int = 0, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.end_token = end_token
        self.channel = channel

    def encode_str(self, image: np.ndarray, text: str) -> np.ndarray:
        """Encode one string."""
        text = text + self.end_token
        new_image = image.copy()
        # Convert to bytes
        to_bytes = convert_str_to_octet(text)

        # Now add to the image the message
        channel_img = cv2.split(new_image)[self.channel]

        # Add the message to the red channel
        for i in range(len(to_bytes)):
            x, y = i % channel_img.shape[0], i // channel_img.shape[0]
            channel_img[x, y] = self.set_last_bit(channel_img[x, y], to_bytes[i])

        new_image[:, :, self.channel] = channel_img
        return new_image

    def decode(self, img: np.ndarray, end_token: str = "[END]", channel: int = 0) -> str:
        """Decode message."""
        channel_img = cv2.split(img)[channel]

        # Get the message
        decoded_message = ""
        current_character = ""
        for i in range(channel_img.shape[0] * channel_img.shape[1]):
            x, y = i % channel_img.shape[0], i // channel_img.shape[0]
            c = channel_img[x, y]
            current_character += self.read_last_bit(channel_img[x, y])

            if len(current_character) == 8:
                decoded_message += bin_to_charac(current_character)
                current_character = ""
                if decoded_message.endswith(end_token):
                    return decoded_message[: -len(end_token)]
        return "Not found"

    @staticmethod
    def set_last_bit(number: int, bit: str) -> int:
        """Set the last bit."""
        return int(bin(number)[2:-1] + bit, 2)

    @staticmethod
    def read_last_bit(number: int) -> str:
        """Read the last bit."""
        return bin(number)[-1]
