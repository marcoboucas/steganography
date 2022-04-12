"""Configuration file."""

import os

# FOLDERS

SRC_FOLDER = os.path.abspath(os.path.dirname(__file__))
ROOT_FOLDER = os.path.dirname(SRC_FOLDER)
TMP_FOLDER = os.path.join(ROOT_FOLDER, "tmp")
os.makedirs(TMP_FOLDER, exist_ok=True)
