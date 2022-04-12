"""Histogram analysis."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def generate_histogram(image: np.ndarray) -> np.ndarray:
    """Generate the histogram of an image."""
    return np.stack([cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(3)]).reshape(
        (3, 256)
    )


def plot_histogram(hist: np.ndarray) -> np.ndarray:
    """Plot the histogram."""
    assert hist.shape == (3, 256), hist.shape

    for color_hist, color in zip(hist, ["Red", "Green", "Blue"]):
        plt.plot(color_hist, color=color[0].lower(), label=color, marker="+")
    plt.legend()
    plt.xlim([0, 256])


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> np.ndarray:
    """Compare 2 histograms."""
    return np.abs(hist1 - hist2)


def compute_diff_imgs(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Compute the difference between the 2 images."""
    assert len(img1.shape) == 3 and len(img2.shape) == 3, "3 colors images"
    return (img1 == img2).astype(np.uint16).sum(axis=-1) != 3
