"""Configuration file for tests."""


import os
from glob import glob

TEST_FOLDER = os.path.abspath(os.path.dirname(__file__))

IMAGE_PATHS = list(glob(os.path.join(TEST_FOLDER, "test_images", "*")))
MESSAGE_IMAGE = os.path.join(TEST_FOLDER, "test_messages", "cs.png")
