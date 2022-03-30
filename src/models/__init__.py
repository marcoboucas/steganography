"""Models."""

from enum import Enum
from multiprocessing.sharedctypes import Value

from .base import BaseSteganographyModel


class SteganographyModels(Enum):
    """List of available models."""

    LSB = "LSB (Least Significant Bit)"


def get_model(model_name: str) -> BaseSteganographyModel:
    """Get one model."""
    try:
        corresponding_name = getattr(SteganographyModels, model_name.upper())
    except AttributeError:
        raise ValueError(
            (
                f"Model {model_name} does not exist. Available models:\n"
                "\n".join(list(map(lambda x: f"{x.name}", SteganographyModels)))
            )
        )

    if corresponding_name == SteganographyModels.LSB:
        from .lsb import LSBModel

        return LSBModel()
    raise NotImplementedError(f"Model {corresponding_name} is not implemented.")
