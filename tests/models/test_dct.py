"""Test the DCT model."""


import pytest

from src.models.dct import DCTModel
from src.utils.data_manipulation import load_image
from tests.config_test import IMAGE_PATHS, MESSAGE_IMAGE


@pytest.mark.parametrize("image_path,message_path", [(x, MESSAGE_IMAGE) for x in IMAGE_PATHS])
def test_dct_model(image_path: str, message_path: str):
    """Test."""
    model = DCTModel()
    image = load_image(image_path)
    message = load_image(message_path)

    result = model.decode(model.encode(image, message))
    assert result.shape == message.shape

    # TODO: find a way to compare the images properly
