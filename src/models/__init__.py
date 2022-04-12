"""Models."""

from enum import Enum
from multiprocessing.sharedctypes import Value

from .base import BaseSteganographyModel


class SteganographyModels(Enum):
    """List of available models."""

    LSB = "LSB (Least Significant Bit)"
    BASE_MERGE = "Basic MERGE (A basic image merge)"


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
    if corresponding_name == SteganographyModels.BASE_MERGE:
        from .base_merge import MergeModel

        return MergeModel()
    raise NotImplementedError(f"Model {corresponding_name} is not implemented.")
