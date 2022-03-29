"""Test about the utils functions about text."""

from string import ascii_letters

import pytest

from src.utils.text import (
    bin_to_charac,
    charac_to_bin,
    convert_octet_to_str,
    convert_str_to_octet,
)


@pytest.mark.parametrize("text", ascii_letters)
def test_charac_bin(text: str):
    """Test charac and bin conversion."""
    assert text == bin_to_charac(charac_to_bin(text))


@pytest.mark.parametrize("text", ["hello world", "good morning", "123"])
def test_str_to_octet(text: str):
    """Test charac and bin conversion."""
    assert text == convert_octet_to_str(convert_str_to_octet(text))
