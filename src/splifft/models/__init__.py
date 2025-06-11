"""Source separation models.

When definining a new model, you may be accustomed to the common practice of defining all model
configuration in the `__init__` method of a [torch.nn.Module][], then maybe add a `@beartype`
decorator to verify types at runtime. But the problem is we want to easily serialize and deserialize
it to/from a JSON file, and using `__init__` for 20+ arguments is not scalable.

Instead, follow this convention:

- each module lives in its own module, e.g. `src/splifft/models/{model_type}.py`.
    - it should have a [**standard library** `dataclass`][dataclasses.dataclass] named
      `{ModelName}Config` that implements the [splifft.models.ModelConfigLike][]
      [protocol](https://typing.python.org/en/latest/spec/protocol.html).
    - it should also have a top level [torch.nn.Module][] subclass named `ModelName` that
      accepts the aforementioned dataclass as the *sole* argument to its `__init__` method.

!!! question "Why do this???"

    If you follow this convention, [splifft.config][] can easily read the `{ModelName}Config`
    dataclass to 1) verify the configuration at runtime and 2) serialize/deserialize it to/from JSON
    using [pydantic.TypeAdapter][].
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar, runtime_checkable

from torch import nn

ModelType: TypeAlias = str
"""The type of the model, e.g. `bs_roformer`, `demucs`"""


@runtime_checkable
class ModelConfigLike(Protocol):
    """A trait that must be implemented to be considered a model configuration."""

    chunk_size: int
    output_stem_names: tuple[ModelOutputStemName, ...]


ModelConfigLikeT = TypeVar("ModelConfigLikeT", bound=ModelConfigLike)

ModelT = TypeVar("ModelT", bound=nn.Module)

ModelOutputStemName: TypeAlias = str
"""The output stem name, e.g. `vocals`, `drums`, `bass`, etc."""


# NOTE: a global state like
# ```
# MODEL_REGISTRY = {
#     "bs_roformer": (BSRoformer, BSRoformerConfig),
# }
# ```
# is a anti-pattern because it forces us to import all models at once. we dont want to force
# the users to install the dependencies of unused models. neither a factory any good:
# ```
# def load_model(model_type: ModelType, model_config: LazyModelConfig) -> nn.Module:
#     if model_type == "bs_roformer":
#         from splifft.models.bs_roformer import BSRoformer, BSRoformerConfig
#         return BSRoformer(model_config.to_concrete(BSRoformerConfig))
#     elif ...
# ```
# because every new model would require a change in this function, violating OCP.
#
@dataclass
class ModelMetadata(Generic[ModelT, ModelConfigLikeT]):
    """Metadata about a model, including its type, configuration class, and model class.

    To use it with a model that is part of the `splifft` library, first import the model and
    construct an instance of this class, e.g.:
    ```py
    from splifft.io import ModelMetadata

    def metadata_factory() -> ModelMetadata:
        from your_library.models.bs_roformer import BSRoformer, BSRoformerConfig  # external

        return ModelMetadata(
            model_type="bs_roformer",
            config=BSRoformerConfig,
            model=BSRoformer,
        )
    ```
    Alternatively, you can also use the [splifft.models.ModelMetadata.from_module][].
    """

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

        This technique is used by the CLI to import arbitrary models that are not part of the
        `splifft` package, e.g.
        ```py
        model_metadata = ModelMetadata.from_module(
            "splifft.models.bs_roformer",
            "BSRoformer",
            model_type="bs_roformer"
        )
        ```

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
