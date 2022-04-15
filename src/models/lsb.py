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

    def encode_img(self, image: np.ndarray, to_encode: np.ndarray) -> np.ndarray:
        """Encode one image."""
        new_image = image.copy()

        height = min(to_encode.shape[0], image.shape[0])
        width = min(to_encode.shape[1], image.shape[1])

        for i in range(height):
            for j in range(width):
                binary_pixel_img_original = self.convert_pixel_to_binary(image[i, j])
                binary_pixel_img_to_hide = self.convert_pixel_to_binary(to_encode[i, j])

                binary_new_pixel = self.mix_pixel(
                    binary_pixel_img_original, binary_pixel_img_to_hide
                )
                new_pixel = self.convert_binary_to_pixel(binary_new_pixel)

                new_image[i, j] = new_pixel
        return new_image

    def decode_str(self, image: np.ndarray) -> str:
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

    def decode_img(self, image: np.ndarray) -> np.ndarray:
        """Decode image."""
        decode_img = image.copy()

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                binary_pixel_img = self.convert_pixel_to_binary(image[i, j])
                binary_pixel_img_hidden = self.unmix_pixel(binary_pixel_img)

                pixel_img_hidden = self.convert_binary_to_pixel(binary_pixel_img_hidden)
                decode_img[i, j] = pixel_img_hidden
        return decode_img

    @staticmethod
    def set_last_bit(number: int, bit: str) -> int:
        """Set the last bit."""
        return int(bin(number)[2:-1] + bit, 2)

    @staticmethod
    def read_last_bit(number: int) -> str:
        """Read the last bit."""
        return bin(number)[-1]

    @staticmethod
    def convert_pixel_to_binary(pixel):
        """Convert pixel to binary."""
        return [format(channel, "b").zfill(8) for channel in pixel]

    @staticmethod
    def convert_binary_to_pixel(binary_pixel):
        """Convert binary pixel to pixel."""
        return [int(channel, 2) for channel in binary_pixel]

    @staticmethod
    def mix_pixel(origin, hiden):
        """Return a new pixel by taking the 4 first bits of the original image pixel and the 4 first bits of the pixel of the image to hide"""
        return [origin[i][:4] + hiden[i][:4] for i in range(len(origin))]

    @staticmethod
    def unmix_pixel(pixel):
        """Return a pixel which looks like the one of the hidden image"""
        return [pixel[i][4:] + "0000" for i in range(len(pixel))]
