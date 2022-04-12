"""Main file."""

import logging
from typing import Optional

import cv2
import numpy as np
from fire import Fire

from src.models import get_model
from src.types import MessageType
from src.utils.data_manipulation import load_image


class CLI:
    """Our CLI."""

    def __init__(self):
        """Initialize our CLI."""
        self.logger = logging.getLogger("CLI")
        self.logger.info("Starting your request !")

    def encode(self, model: str, image_path: str, message_path: str, output_path: str) -> None:
        """Encode a message in an image."""
        self.logger.info("Encoding message in image.")

        model_instance = get_model(model)

        image = self._load_image(image_path)
        message = self._load_message(message_path)

        encoded_image = model_instance.encode(image, message)

        self._save_image(encoded_image, output_path)

    def decode(self, model: str, image_path: str, output_path: Optional[str]) -> None:
        """Encode a message in an image."""
        self.logger.info("Encoding message in image.")

        model_instance = get_model(model)

        image = self._load_image(image_path)

        decoded_message = model_instance.decode(image)

        if output_path:
            self._save_message(decoded_message, output_path)
        else:
            print(f"Message: {decoded_message}")
        return decoded_message

    def _load_message(self, message_path: str) -> MessageType:
        """Load a message (can be a text or image)."""
        self.logger.info("Loading message to hide.")
        if message_path.endswith(".txt"):
            with open(message_path, "r", encoding="utf-8") as f:
                message = f.read()
                self.logger.info("Message loaded, type text with %i characters.", len(message))
                return message
        if any(message_path.endswith(x) for x in {".png", ".jpg"}):
            return load_image(message_path)
        self.logger.info("We consider this is a plain text")
        return message_path
        raise ValueError("Unknown file type.")

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load an image."""
        self.logger.info("Loading image.")
        return load_image(image_path)

    def _save_image(self, image: np.ndarray, image_path: str) -> None:
        """Save an image."""
        self.logger.info("Saving image with the hidden message inside.")
        cv2.imwrite(image_path, image)

    def _save_message(self, message: MessageType, message_path: str) -> None:
        """Save a message."""
        self.logger.info("Saving message.")
        if isinstance(message, str):
            with open(message_path, "w", encoding="utf-8") as f:
                f.write(message)
        elif isinstance(message, np.ndarray):
            cv2.imwrite(message_path, message)
        else:
            raise ValueError(f"Unknown message type: '{type(message)}'.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(CLI)
