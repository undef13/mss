"""Configuration"""

# NOTE: since this contains heavy validation logic with type annotations,
# we are not using `typing.TYPE_CHECKING` here.
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Hashable,
    Literal,
    Sequence,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
)

from annotated_types import Ge, Gt, Le, Len
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)
from pydantic_core import PydanticCustomError

from .models import ModelConfigLike, ModelType
from .models import ModelOutputStemName as _ModelOutputStemName

if TYPE_CHECKING:
    from typing_extensions import Self
    # TODO: continue moving down, but only if we are confident.

_Item = TypeVar("_Item", bound=Hashable)


def _validate_unique_sequence(sequence: Sequence[_Item]) -> Sequence[_Item]:
    if len(sequence) != len(set(sequence)):
        raise PydanticCustomError("unique_sequence", "Sequence must contain unique items")
    return sequence


_S = TypeVar("_S", bound=Sequence[Hashable])
NonEmptyUnique = Annotated[
    _S,
    Len(min_length=1),
    AfterValidator(_validate_unique_sequence),
    Field(json_schema_extra={"unique_items": True}),
]

ModelInputStemName: TypeAlias = Literal["mixture"]
ModelOutputStemName: TypeAlias = Annotated[_ModelOutputStemName, StringConstraints(min_length=1)]
_INPUT_STEM_NAMES = get_args(ModelInputStemName)


# NOTE: the ideal case would be to use an ADT whose variants are known at "compile" time:
# enum ModelConfig {
#     BsRoformer { param_x: ..., param_y: ... },
#     Demucs { param_x: ..., params_z: ... },
# }
# but downstream users may want to register their own models with different configurations,
# something that enums do not support.
# so, instead of validating the model configuration eagerly, defer it until it is actually needed.
class LazyModelConfig(BaseModel):
    """A lazily validated model configuration.

    Note that it is not guaranteed to be fully valid until `to_concrete` is called.
    """

    chunk_size: int
    output_stem_names: NonEmptyUnique[tuple[ModelOutputStemName, ...]]

    def to_concrete(
        self,
        model_config: type[ModelConfigLike],
        *,
        pydantic_config: ConfigDict = ConfigDict(extra="forbid"),
    ) -> ModelConfigLike:
        """Validate against a real model configuration and convert to it.

        By default, extra fields are not allowed."""

        return TypeAdapter(
            type(
                f"{model_config.__name__}Validator",
                (model_config,),
                {"__pydantic_config__": pydantic_config},
            )  #  needed for https://docs.pydantic.dev/latest/errors/usage_errors/#type-adapter-config-unused
        ).validate_python(self.model_dump())

    @property
    def stem_names(self) -> tuple[ModelInputStemName | ModelOutputStemName, ...]:
        """Returns the model's input and output stem names."""
        return (*_INPUT_STEM_NAMES, *self.output_stem_names)

    model_config = ConfigDict(extra="allow")  # extra fields are not validated until `to_concrete`


assert isinstance(
    LazyModelConfig(chunk_size=1024, output_stem_names=("dummy",)), ModelConfigLike
), "make sure LazyModelConfig has all fields to ModelConfigLike"  # TODO: move this to tests instead


class AudioIOConfig(BaseModel):
    target_sample_rate: Annotated[int, Gt(0)] = 44100
    force_channels: Literal[1, 2] | None = 2
    """Whether to force mono or stereo audio input. If None, keep original."""


class InferenceConfig(BaseModel):
    normalize_input_audio: bool = False
    batch_size: Annotated[int, Gt(0)] = 4
    apply_tta: bool = False


class ChunkingConfig(BaseModel):
    method: Literal["overlap_add_windowed"] = "overlap_add_windowed"
    chunk_duration: Annotated[float, Gt(0)] = 8
    overlap_ratio: Annotated[float, Ge(0), Le(1)] = 0.5
    window_shape: Literal["hann", "hamming", "linear_fade"] = "hann"


DerivedStemName: TypeAlias = Annotated[str, StringConstraints(min_length=1)]
"""The name of a derived stem, e.g. `vocals_minus_drums`."""
StemName: TypeAlias = Union[ModelOutputStemName, DerivedStemName]
"""A name of a stem, either a model output stem or a derived stem."""


class SubtractConfig(BaseModel):
    operation: Literal["subtract"]
    stem_name: StemName
    by_stem_name: StemName


class SumConfig(BaseModel):
    operation: Literal["sum"]
    stem_names: NonEmptyUnique[tuple[StemName, ...]]


DerivedStemRule: TypeAlias = Annotated[Union[SubtractConfig, SumConfig], Discriminator("operation")]
DerivedStemsConfig: TypeAlias = dict[DerivedStemName, DerivedStemRule]


class OutputConfig(BaseModel):
    stem_names: NonEmptyUnique[tuple[StemName, ...]]  # TODO: make sure these are valid stems
    file_format: Literal["wav", "flac"] = "flac"
    pcm_type: Literal["PCM_16", "PCM_24", "FLOAT"] = "PCM_24"  # TODO: validate this


# if we were to implement a model registry (which we shouldn't need)
# heavily consider https://peps.python.org/pep-0487/#subclass-registration


class Config(BaseModel):
    identifier: str
    """Unique identifier for this configuration"""
    model_type: ModelType
    model: LazyModelConfig
    audio_io: AudioIOConfig = Field(default_factory=AudioIOConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    chunking: ChunkingConfig | None = None
    derived_stems: DerivedStemsConfig | None = None
    output: OutputConfig | None = None
    experimental: dict[str, Any] | None = None
    """Any extra experimental configurations outside of the `mss` core."""

    @model_validator(mode="after")
    def check_derived_stems(self) -> Self:
        if self.derived_stems is None:
            return self
        # accumulate valid stem names
        existing_stem_names: list[StemName] = list(self.model.stem_names)
        for derived_stem_name, definition in self.derived_stems.items():
            if derived_stem_name in existing_stem_names:
                raise PydanticCustomError(
                    "derived_stem_name_conflict",
                    "Derived stem `{derived_stem_name}` must not conflict with existing stem names: `{existing_stem_names}`",
                    {
                        "derived_stem_name": derived_stem_name,
                        "existing_stem_names": existing_stem_names,
                    },
                )
            required_stems: tuple[StemName, ...] = tuple()
            if isinstance(definition, SubtractConfig):
                required_stems = (definition.stem_name, definition.by_stem_name)
            elif isinstance(definition, SumConfig):
                required_stems = definition.stem_names
            for stem_name in required_stems:
                if stem_name not in existing_stem_names:
                    raise PydanticCustomError(
                        "invalid_derived_stem",
                        "Derived stem `{derived_stem_name}` requires stem `{stem_name}` but is not found in `{existing_stem_names}`",
                        {
                            "derived_stem_name": derived_stem_name,
                            "stem_name": stem_name,
                            "existing_stem_names": existing_stem_names,
                        },
                    )
            existing_stem_names.append(derived_stem_name)
        return self

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # for .model
        strict=True,
        extra="forbid",
    )
