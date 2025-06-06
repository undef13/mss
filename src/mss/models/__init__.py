from __future__ import annotations

from typing import Protocol, TypeAlias, runtime_checkable

ModelType: TypeAlias = str
"""The type of the model, e.g. `bs_roformer`, `demucs`"""

ModelOutputStemName: TypeAlias = str
"""The output stem name, e.g. `vocals`, `drums`, `bass`, etc."""


@runtime_checkable
class ModelConfigLike(Protocol):
    """A trait that must be implemented to be considered a model configuration."""

    chunk_size: int
    output_stem_names: tuple[ModelOutputStemName, ...]
