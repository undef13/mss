"""
Core ML Model Intermediate Language.
"""

from pathlib import Path


def resolve_path(raw_path_str: str, model_path: Path) -> Path:
    assert raw_path_str.startswith("@model_path")
    rest = raw_path_str.removeprefix("@model_path/")
    relative = Path(rest)
    return model_path / relative
