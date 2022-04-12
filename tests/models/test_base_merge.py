"""Test the base merge model."""


import cv2
import numpy as np
import pytest

from src.models.base_merge import MergeModel
from src.utils.data_manipulation import load_image
from tests.config_test import IMAGE_PATHS, MESSAGE_IMAGE


@pytest.mark.parametrize("image_path,message_path", [(x, MESSAGE_IMAGE) for x in IMAGE_PATHS])
def test_lsb_model(image_path: str, message_path: str):
    """Test."""
    model = MergeModel()
    image = load_image(image_path)
    message = load_image(message_path)

    result = model.decode(model.encode(image, message))
    expected = cv2.resize(message, (model.width, model.height))
    assert result.shape == expected.shape
    assert np.sum(np.abs(result - expected)) == 0
