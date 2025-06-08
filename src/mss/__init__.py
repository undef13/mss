"""Lightweight utilities for music source separation."""

from pathlib import Path

# TODO: this should be in the io module and we should use user cache paths to store models
PATH_MODULE = Path(__file__).parent.parent
PATH_BASE = PATH_MODULE.parent
PATH_DATA = PATH_BASE / "data"
PATH_CONFIG = PATH_DATA / "config"
PATH_MODELS = PATH_DATA / "models"

# NOTE: not re-exporting because our structure is simple enough.
