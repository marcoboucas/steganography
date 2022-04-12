"""Wavelet Transform Model."""

import os

import cv2
import numpy as np
import pywt

from src.config import TMP_FOLDER

from .base import BaseSteganographyModel


class WTModel(BaseSteganographyModel):
    """WT (Wavelet Transform) model."""

    def __init__(self, *args, tmp_path: str = TMP_FOLDER, factor_svd: float = 0.1, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.tmp_folder = tmp_path
        self.factor_svd = factor_svd

    def encode_img(self, image: np.ndarray, img_to_hide: np.ndarray) -> np.ndarray:
        """Encode one image."""
        new_image = image.copy()
        return np.stack(
            [
                self.__encode_grayscale(new_image[:, :, i], img_to_hide[:, :, i], f"layer_{i}")
                for i in range(3)
            ],
            axis=2,
        ).astype(np.uint8)

    def decode(self, img: np.ndarray) -> np.ndarray:
        """Decode message."""
        return np.stack(
            [self.__decode_grayscale(img[:, :, i], f"layer_{i}") for i in range(3)], axis=2
        ).astype(np.uint8)

    def __encode_grayscale(
        self, image: np.ndarray, img_to_hide: np.ndarray, layer_name: str
    ) -> np.ndarray:
        """Encode grayscale image."""
        if image.ndim != 2 or img_to_hide.ndim != 2:
            raise ValueError("Image or image to hide must be grayscale.")

        # Resize the images if needed to the smallest one
        size = min(image.shape[0], image.shape[1], img_to_hide.shape[0], img_to_hide.shape[1])
        size = 128
        image = cv2.resize(image, (size, size))
        img_to_hide = cv2.resize(img_to_hide, (size, size))

        # Decompose the image into wavelet coefficients and apply SVD on the LL one
        c_ll, (c_lh, c_hl, c_hh) = pywt.dwt2(image, "haar")
        u_img1, S_img1, v_img1 = np.linalg.svd(c_ll)
        # Same for the image to hide
        w_ll, (w_lh, w_hl, w_hh) = pywt.dwt2(img_to_hide, "haar")
        u_img2, S_img2, v_img2 = np.linalg.svd(w_ll)

        # Incorporate the image to hide into the LL one and rebuild the LL
        S_wimg = S_img1 + (self.factor_svd * S_img2)
        new_c_ll = u_img1.dot(np.diag(S_wimg)).dot(v_img1)

        # Save the matrices
        self.__save_one(w_lh, layer_name, "w_lh")
        self.__save_one(w_hl, layer_name, "w_hl")
        self.__save_one(w_hh, layer_name, "w_hh")
        self.__save_one(u_img2, layer_name, "u_img2")
        self.__save_one(v_img2, layer_name, "v_img2")
        self.__save_one(S_img1, layer_name, "S_img1")

        return pywt.idwt2((new_c_ll, (c_lh, c_hl, c_hh)), "haar", mode="symmetric")

    def __decode_grayscale(self, img_with_message: np.ndarray, layer_name: str) -> np.ndarray:
        """Decode a grayscale image."""
        if img_with_message.ndim != 2:
            raise ValueError("Image must be grayscale.")

        # Load the needed matrices
        s_img1 = self.__read_one(layer_name, "s_img1").reshape((-1,))
        u_img2 = self.__read_one(layer_name, "u_img2")
        v_img2 = self.__read_one(layer_name, "v_img2")
        w_lh = self.__read_one(layer_name, "w_lh")
        w_hl = self.__read_one(layer_name, "w_hl")
        w_hh = self.__read_one(layer_name, "w_hh")
        print(s_img1.shape, u_img2.shape, v_img2.shape, w_lh.shape, w_hl.shape, w_hh.shape)

        # Decompose the image into wavelet coefficients and apply SVD on the LL one
        wm_ll, _ = pywt.dwt2(img_with_message, "haar")
        _, S_img3, _ = np.linalg.svd(wm_ll)

        # Decode the LL one
        S_ewat = (S_img3 - s_img1) / self.factor_svd
        print(u_img2.shape, S_ewat.shape, v_img2.shape)
        ewat = u_img2.dot(np.diag(S_ewat)).dot(v_img2)
        return pywt.idwt2((ewat, (w_lh, w_hl, w_hh)), "haar", mode="symmetric")

    def __save_one(self, matrix: np.ndarray, layer_name: str, name: str) -> None:
        """Save one matrix."""
        path = os.path.join(self.tmp_folder, f"{layer_name}_{name}.npy")
        np.save(path, matrix)

    def __read_one(self, layer_name: str, name: str) -> np.ndarray:
        """Read one matrix."""
        path = os.path.join(self.tmp_folder, f"{layer_name}_{name}.npy")
        return np.load(path)
