"""PVD Model."""

import math

import cv2
import numpy as np

from src.utils.text import bin_to_charac, convert_str_to_octet

from .base import BaseSteganographyModel


class PVDModel(BaseSteganographyModel):
    """PVD model."""

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

        # Get the good channel
        channel_img = cv2.split(new_image)[self.channel]

        for j in range (1, channel_img.shape[0]-1, 2):
            for i in range (1, channel_img.shape[0]-1, 2):

                if (len(to_bytes) <= 0):
                    return new_image
                else:
                    neighbors = self.get_neighbors(i,j, channel_img)
                    pvd = self.compute_pvd(neighbors)
                    nbits = self.classify(pvd)
                    new_pixel = self.modify_pixel(channel_img[j,i], nbits, to_bytes)

                    # Add the new pixel to the good channel
                    new_image[j, i, self.channel] = new_pixel
                    to_bytes = to_bytes[nbits:]

        return new_image



    def decode(self, img: np.ndarray) -> str:
        """Decode message."""
        # Get the good channel
        channel_img = cv2.split(img)[self.channel]

        decoded_message = ""
        current_character = ""

        for j in range (1, channel_img.shape[0]-1, 2):
            for i in range (1, channel_img.shape[0]-1, 2):

                neighbors = self.get_neighbors(i,j, channel_img)
                pvd = self.compute_pvd(neighbors)
                nbits = self.classify(pvd)

                decimal_value = self.get_decimal_value(channel_img[j,i], nbits)

                current_character += format(decimal_value, "b").zfill(nbits)

                if len(current_character) >= 8:
                    decoded_message += bin_to_charac(current_character[:8])
                    current_character = current_character[8:]
                if decoded_message.endswith(self.end_token):
                    return decoded_message[: -len(self.end_token)]

        return "Not found"


    @staticmethod
    def compute_pvd(neighbors):

        return max(neighbors)-min(neighbors)


    @staticmethod
    def classify(pvd):
        """The embedding capacity of a pixel depends on the pvd """
        nbits = 0

        if pvd <= 1:
            nbits = 1

        else:
            #log2(pvd) modulo 4
            #We don't want to hide more than 4 bits in a pixel
            nbits = int(math.log2(pvd)%4)+1

        return nbits

    @staticmethod
    def modify_pixel(pixel, nbits, text_to_hide):
        """We hide a sub-stream with n bits of our message inside our image """

        # We extract the last n bits of the message
        if (len(text_to_hide)<nbits):
            nbits = len(text_to_hide)

        text_to_hide_in_the_pixel = text_to_hide[:nbits]
        decimal_to_hide = int(text_to_hide_in_the_pixel, 2)
        newpixel = pixel - pixel%(2**nbits) + decimal_to_hide
        return newpixel

    @staticmethod
    def get_neighbors(i,j, channel_img):
        """Return 3 neighbors of a pixel (upper left, upper and left"""
        return [channel_img[j-1, i-1], channel_img[j-1,i], channel_img[j, i-1]]

    @staticmethod
    def get_decimal_value(pixel, nbits):
        return pixel%(2**nbits)
