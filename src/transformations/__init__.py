"""Transformations."""


from enum import Enum
from typing import List

from .base import BaseTransformation


class Transformations(Enum):
    """List of available transformations."""

    IDENTITY = "Identity (do nothing)"


def get_transformations(transformation_names: List[str]) -> List[BaseTransformation]:
    """Get a list of transformations."""
    corresponding_transformations = []
    for transformation_name in transformation_names:
        try:
            corresponding_transformations.append(
                getattr(Transformations, transformation_name.upper())
            )
        except AttributeError:
            raise ValueError(
                (
                    f"Model {transformation_name} does not exist. Available models:\n"
                    "\n".join(list(map(lambda x: f"{x.name}", Transformations)))
                )
            )

    # Get the transformations
    return list(map(__get_one_transformation, corresponding_transformations))


def __get_one_transformation(transformation_name: Transformations) -> BaseTransformation:
    """Get one transformation."""
    if transformation_name == Transformations.IDENTITY:
        from .identity import IdentityTransformation

        return IdentityTransformation()
    raise NotImplementedError(f"Transformation {transformation_name} is not implemented.")
