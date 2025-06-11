"""Source separation models."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from annotated_types import Gt
from torch import nn

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

ModelType: TypeAlias = str
"""The type of the model, e.g. `bs_roformer`, `demucs`"""


@runtime_checkable
class ModelConfigLike(Protocol):
    """A trait that must be implemented to be considered a model configuration."""

    chunk_size: ChunkSize
    output_stem_names: tuple[ModelOutputStemName, ...]


ChunkSize: TypeAlias = Annotated[int, Gt(0)]
"""The length of an audio segment, in samples, processed by the model at one time.

A full audio track is often too long to fit into GPU, instead we process it in fixed-size chunks.
A larger chunk size may allow the model to capture more temporal context at the cost of increased
memory usage.
"""

ModelOutputStemName: TypeAlias = str
"""The output stem name, e.g. `vocals`, `drums`, `bass`, etc."""

ModelConfigLikeT = TypeVar("ModelConfigLikeT", bound=ModelConfigLike)

ModelT = TypeVar("ModelT", bound=nn.Module)


@dataclass
class ModelMetadata(Generic[ModelT, ModelConfigLikeT]):
    """Metadata about a model, including its type, configuration class, and model class."""

    model_type: ModelType
    config: type[ModelConfigLikeT]
    model: type[ModelT]

    @classmethod
    def from_module(
        cls,
        module_name: str,
        model_cls_name: str,
        *,
        model_type: ModelType,
        package: str | None = None,
    ) -> ModelMetadata[nn.Module, ModelConfigLike]:
        """
        Dynamically import a model named `X` and its configuration dataclass `XConfig` under a
        given module name (e.g. `splifft.models.bs_roformer`).

        :param model_cls_name: The name of the model class to import, e.g. `BSRoformer`.
        :param module_name: The name of the module to import, e.g. `splifft.models.bs_roformer`.
        :param package: The package to use as the anchor point from which to resolve the relative import.
        to an absolute import. This is only required when performing a relative import.
        """
        _loc = f"{module_name=} under {package=}"
        try:
            module = importlib.import_module(module_name, package)
        except ImportError as e:
            raise ValueError(f"failed to find or import module for {_loc}") from e

        config_cls_name = f"{model_cls_name}Config"
        model_cls = getattr(module, model_cls_name, None)
        config_cls = getattr(module, config_cls_name, None)
        if model_cls is None or config_cls is None:
            raise AttributeError(
                f"expected to find a class named `{config_cls_name}` in {_loc}, but it was not found."
            )

        return ModelMetadata(
            model_type=model_type,
            model=model_cls,
            config=config_cls,  # type: ignore
        )
