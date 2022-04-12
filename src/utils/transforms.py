"""Transforms."""

import numpy as np
from scipy.fftpack import dct as _dct
from scipy.fftpack import idct as _idct


def dct2(matrix: np.ndarray) -> np.ndarray:
    """implement 2D DCT."""
    return _dct(_dct(matrix.T, norm="ortho").T, norm="ortho")


def idct2(matrix: np.ndarray) -> np.ndarray:
    """implement 2D DCT."""
    return _idct(_idct(matrix.T, norm="ortho").T, norm="ortho")
