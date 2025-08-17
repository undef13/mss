"""Data types and functions for the model registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import types as t


@dataclass(frozen=True)
class Author:
    name: str | None = None
    huggingface: str | None = None
    github: str | None = None


@dataclass(frozen=True)
class ModelSource:
    url: str


@dataclass(frozen=True)
class ModelInfo:
    identifier: str
    authors: tuple[Author, ...]
    model_sources: tuple[ModelSource, ...]
    model_sha256: str | None

    config_identifier: str
    model_type: t.ModelType
