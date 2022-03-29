"""Test the LSB model."""

import pytest

from src.models.lsb import LSBModel
from src.utils.data_manipulation import load_image
from tests.config_test import IMAGE_PATHS


@pytest.mark.parametrize("image_path", IMAGE_PATHS)
def test_lsb_model(image_path: str):
    """Test."""
    model = LSBModel()
    print(image_path)
    image = load_image(image_path)
    message = "Hello world !"

    assert model.decode(model.encode(image, message)) == message
