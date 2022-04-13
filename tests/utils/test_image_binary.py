"""Test image_binary.py."""


import numpy as np

from src.utils.image_binary import (
    convert_bin_to_img,
    convert_bin_to_pixel,
    convert_img_to_bin,
    convert_pixel_to_bin,
)


def test_convert_bin_img():
    """Test convert_img_to_bin."""
    image = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.sum(np.abs(image - convert_bin_to_img(convert_img_to_bin(image)))) == 0.0


def test_convert_pixel():
    """Test convert pixel."""
    pixel = np.array([1, 2, 3])
    assert np.sum(np.abs(pixel - convert_bin_to_pixel(convert_pixel_to_bin(pixel)))) == 0.0
    x = convert_pixel_to_bin(pixel)
    assert len(x) == 3 * 8
