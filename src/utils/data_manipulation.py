"""Data manipulation functions."""

import cv2


def load_image(path: str):
    """Load one image."""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
