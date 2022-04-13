"""Convert image to binary and back."""

from io import BytesIO

import numpy as np


def convert_img_to_bin(image: np.ndarray) -> str:
    """Convert image to binary."""
    with BytesIO() as out:
        np.save(out, image)
        return "".join([format(n, "08b") for n in out.getvalue()])


def convert_bin_to_img(binary: str) -> np.ndarray:
    """Convert binary to image."""
    decoded_b2 = bytes([int(binary[i : i + 8], 2) for i in range(0, len(binary), 8)])
    return np.load(BytesIO(decoded_b2), allow_pickle=True)


def convert_pixel_to_bin(pixel: np.ndarray) -> str:
    """Convert pixel to bin."""
    assert pixel.shape == (3,)
    return "".join([format(n, "08b") for n in pixel])


def convert_bin_to_pixel(binary: str) -> np.ndarray:
    """Convert bin to pixel."""
    assert len(binary) == 24
    return np.array([int(binary[i : i + 8], 2) for i in range(0, len(binary), 8)])
