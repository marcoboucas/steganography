"""Base Transformation."""

import logging
from abc import ABC, abstractmethod

import numpy as np


class BaseTransformation(ABC):
    """Base Transformation."""

    def __init__(self):
        """Base Transformation."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Do the transformation."""
