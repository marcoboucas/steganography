"""Wavelet Transform Model."""

import numpy as np
import pywt

from .base import BaseSteganographyModel


class WTModel(BaseSteganographyModel):
    """WT (Wavelet Transform) model."""

    def __init__(self, *args, factor_svd: float = 0.1, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.factor_svd = factor_svd

    def encode_img(self, image: np.ndarray, to_encode: np.ndarray) -> np.ndarray:
        """Encode one image."""
        new_image = image.copy()
        return np.stack(
            [
                self.__encode_grayscale(new_image[:, :, i], to_encode[:, :, i], f"layer_{i}")
                for i in range(3)
            ],
            axis=2,
        ).astype(np.uint8)

    def decode_img(self, image: np.ndarray) -> np.ndarray:
        """Decode message."""
        return np.stack(
            [self.__decode_grayscale(image[:, :, i], f"layer_{i}") for i in range(3)], axis=2
        ).astype(np.uint8)

    # pylint: disable=too-many-locals
    def __encode_grayscale(
        self, image: np.ndarray, img_to_hide: np.ndarray, layer_name: str
    ) -> np.ndarray:
        """Encode grayscale image."""
        if image.ndim != 2 or img_to_hide.ndim != 2:
            raise ValueError("Image or image to hide must be grayscale.")

        # Resize the images if needed to the smallest one
        image, img_to_hide = self._resize_both_images(image, img_to_hide)

        # Decompose the image into wavelet coefficients and apply SVD on the LL one
        c_ll, (c_lh, c_hl, c_hh) = pywt.dwt2(image, "haar")
        u_img1, s_img1, v_img1 = np.linalg.svd(c_ll)
        # Same for the image to hide
        w_ll, (w_lh, w_hl, w_hh) = pywt.dwt2(img_to_hide, "haar")
        u_img2, s_img2, v_img2 = np.linalg.svd(w_ll)

        # Incorporate the image to hide into the LL one and rebuild the LL
        s_wimg = s_img1 + (self.factor_svd * s_img2)
        new_c_ll = u_img1.dot(np.diag(s_wimg)).dot(v_img1)

        # Save the matrices
        self._save_one(w_lh, layer_name, "w_lh")
        self._save_one(w_hl, layer_name, "w_hl")
        self._save_one(w_hh, layer_name, "w_hh")
        self._save_one(u_img2, layer_name, "u_img2")
        self._save_one(v_img2, layer_name, "v_img2")
        self._save_one(s_img1, layer_name, "s_img1")

        return pywt.idwt2((new_c_ll, (c_lh, c_hl, c_hh)), "haar", mode="symmetric")

    def __decode_grayscale(self, img_with_message: np.ndarray, layer_name: str) -> np.ndarray:
        """Decode a grayscale image."""
        if img_with_message.ndim != 2:
            raise ValueError("Image must be grayscale.")

        # Load the needed matrices
        s_img1 = self._read_one(layer_name, "s_img1").reshape((-1,))
        u_img2 = self._read_one(layer_name, "u_img2")
        v_img2 = self._read_one(layer_name, "v_img2")
        w_lh = self._read_one(layer_name, "w_lh")
        w_hl = self._read_one(layer_name, "w_hl")
        w_hh = self._read_one(layer_name, "w_hh")

        # Decompose the image into wavelet coefficients and apply SVD on the LL one
        wm_ll, _ = pywt.dwt2(img_with_message, "haar")
        _, s_img3, _ = np.linalg.svd(wm_ll)

        # Decode the LL one
        s_ewat = (s_img3 - s_img1) / self.factor_svd
        ewat = u_img2.dot(np.diag(s_ewat)).dot(v_img2)
        return pywt.idwt2((ewat, (w_lh, w_hl, w_hh)), "haar", mode="symmetric")
