"""PVD Model."""

import math

import cv2
import numpy as np

from src.utils.text import convert_octet_to_str, convert_str_to_octet

from .base import BaseSteganographyModel


class GLMModel(BaseSteganographyModel):
    """GLM model."""

    def __init__(self, *args, end_token: str = "[END]", channel: int = 0, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.end_token = end_token
        self.channel = channel


    def encode_str(self, image: np.ndarray, text: str) -> np.ndarray:
        """Encode one string."""
        text = text + self.end_token
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert to bytes
        to_bytes = convert_str_to_octet(text)

        # Get the good channel
        channel_img = cv2.split(new_image)[self.channel]

        for index, bit in enumerate(to_bytes):
            row, col = GLMModel.__get_bit_position(index, channel_img.shape[0])

            if channel_img[col, row]%2 != 0:
                channel_img[col, row] += 1
            if bit == '1':
                channel_img[col, row] -= 1

        return np.dstack((channel_img, channel_img, channel_img))






    def decode(self, img: np.ndarray) -> str:
        """Decode message."""
        # Get the good channel
        channel_img = cv2.split(img)[self.channel]

        decoded_binary_message = ""
        decoded_message = ""
        current_character = ""

        index = 0

        while(decoded_message[-5:]!=self.end_token):
            row, col = GLMModel.__get_bit_position(index, channel_img.shape[0])

            if channel_img[col, row]%2 == 0:
                decoded_binary_message += '0'
            else:
                decoded_binary_message += '1'
            index+=1
            decoded_message = convert_octet_to_str(decoded_binary_message)

        return decoded_message[: -len(self.end_token)]

    @staticmethod
    def __get_bit_position(index, width):
        row=(2*index+5)%width
        col= (2*index+5)//width
        return row, col
