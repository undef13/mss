"""Source separation models.

When definining a new model, you may be accustomed to the common practice of defining all model
configuration in the `__init__` method of a [torch.nn.Module][], then maybe add a `@beartype`
decorator to verify types at runtime. But the problem is we want to easily serialize and deserialize
it to/from a JSON file, and using `__init__` for 20+ arguments is not scalable.

Instead, follow this convention:

- each module lives in its own module, e.g. `src/mss/models/{model_type}.py`.
    - it should have a [**standard library** `dataclass`][dataclasses.dataclass][] named
      `{ModelName}Config` that implements the [mss.models.ModelConfigLike][]
      [protocol](https://typing.python.org/en/latest/spec/protocol.html).
    - it should also have a top level [torch.nn.Module][] subclass named `ModelName` that
      accepts the aforementioned dataclass as the *sole* argument to its `__init__` method.

!!! question "Why do this???"

    If you follow this convention, [mss.config][] can easily read the `{ModelName}Config` dataclass
    to 1) verify the configuration at runtime and 2) serialize/deserialize it to/from JSON
    using [pydantic.TypeAdapter][].
"""

from __future__ import annotations

from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

ModelType: TypeAlias = str
"""The type of the model, e.g. `bs_roformer`, `demucs`"""

ModelOutputStemName: TypeAlias = str
"""The output stem name, e.g. `vocals`, `drums`, `bass`, etc."""


@runtime_checkable
class ModelConfigLike(Protocol):
    """A trait that must be implemented to be considered a model configuration."""

    chunk_size: int
    output_stem_names: tuple[ModelOutputStemName, ...]


ModelConfigLikeT = TypeVar("ModelConfigLikeT", bound=ModelConfigLike)
