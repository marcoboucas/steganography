"""Transformations."""


from enum import Enum
from typing import List

from .base import BaseTransformation


class Transformations(Enum):
    """List of available transformations."""

    IDENTITY = "Identity (do nothing)"
    JITTER = "Jitter (color modification)"
    FLIP = "Flip (flip horizontal)"
    BLUR = "Blur (gaussian blur)"


def get_transformations(transformation_names: List[str]) -> List[BaseTransformation]:
    """Get a list of transformations."""
    corresponding_transformations = []
    for transformation_name in transformation_names:
        try:
            corresponding_transformations.append(
                getattr(Transformations, transformation_name.upper())
            )
        except AttributeError as err:
            raise ValueError(
                (
                    f"Model {transformation_name} does not exist. Available models:\n"
                    "\n".join(list(map(lambda x: f"{x.name}", Transformations)))
                )
            ) from err

    # Get the transformations
    return list(map(__get_one_transformation, corresponding_transformations))


# pylint: disable=import-outside-toplevel
def __get_one_transformation(transformation_name: Transformations) -> BaseTransformation:
    """Get one transformation."""
    if transformation_name == Transformations.IDENTITY:
        from .identity import IdentityTransformation

        return IdentityTransformation()
    if transformation_name == Transformations.JITTER:
        from .jitter import JitterTransformation

        return JitterTransformation()
    if transformation_name == Transformations.FLIP:
        from .flip import FlipTransformation

        return FlipTransformation()
    if transformation_name == Transformations.BLUR:
        from .blur import BlurTransformation

        return BlurTransformation()
    raise NotImplementedError(f"Transformation {transformation_name} is not implemented.")
