"""Configuration"""

from __future__ import annotations

import json
from pathlib import Path
from typing import (
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

from annotated_types import Len
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)
from pydantic_core import PydanticCustomError
from typing_extensions import Self

from .core import (
    BatchSize,
    BitRate,
    Channels,
    Dtype,
    FftSize,
    FileFormat,
    HopSize,
    OverlapRatio,
    PaddingMode,
    SampleRate,
    WindowShape,
)
from .models import ChunkSize, ModelParamsLikeT, ModelType
from .models import ModelOutputStemName as _ModelOutputStemName

# NOTE: we are not using typing.TYPE_CHECKING because pydantic relies on that

_PYDANTIC_STRICT_CONFIG = ConfigDict(strict=True, extra="forbid")

_Item = TypeVar("_Item", bound=Hashable)


def _to_tuple(sequence: Sequence[_Item]) -> tuple[_Item, ...]:
    # this is so json arrays are converted to tuples
    return tuple(sequence)


Tuple = Annotated[tuple[_Item, ...], BeforeValidator(_to_tuple)]


def _validate_unique_sequence(sequence: Sequence[_Item]) -> Sequence[_Item]:
    # e.g. to ensure there are no duplicate stem names
    if len(sequence) != len(set(sequence)):
        raise PydanticCustomError("unique_sequence", "Sequence must contain unique items")
    return sequence


_S = TypeVar("_S")
NonEmptyUnique = Annotated[
    _S,
    Len(min_length=1),
    AfterValidator(_validate_unique_sequence),
    Field(json_schema_extra={"unique_items": True}),
]

ModelInputStemName: TypeAlias = Literal["mixture"]
ModelOutputStemName: TypeAlias = Annotated[_ModelOutputStemName, StringConstraints(min_length=1)]
_INPUT_STEM_NAMES = get_args(ModelInputStemName)


# NOTE: the ideal case would be to use an ADT whose variants are known at "compile" time, e.g. in Rust:
# enum ModelConfig {
#     BsRoformer { param_x: ..., param_y: ... },
#     Demucs { param_x: ..., params_z: ... },
# }
# but downstream users may want to register their own models with different configurations,
# so a discriminated enum wouldn't work here.
# so, we effectively let Config.model_config be dyn ModelParamsLike (i.e. dict[str, Any])
# and defer the validation of the model configuration until it is actually needed instead of doing it eagerly.
class LazyModelConfig(BaseModel):
    """A lazily validated model configuration.

    Note that it is not guaranteed to be fully valid until `to_concrete` is called.
    """

    chunk_size: ChunkSize
    output_stem_names: NonEmptyUnique[Tuple[ModelOutputStemName]]

    def to_concrete(
        self,
        model_params: type[ModelParamsLikeT],
        *,
        pydantic_config: ConfigDict = ConfigDict(extra="forbid"),
    ) -> ModelParamsLikeT:
        """Validate against a real set of model parameters and convert to it.

        :raises pydantic.ValidationError: if extra fields are present in the model parameters
            that doesn't exist in the concrete model parameters.
        """
        config_dict = self.model_dump()
        # NOTE: a more scalable approach would be to use Annotated[] and get_origin but its overkill.
        # hardcoding for now.
        if "input_type" in config_dict:
            raise PydanticCustomError(
                "readonly_model_param", "`input_type` is a model parameter and cannot be modified"
            )
        if "output_type" in config_dict:
            raise PydanticCustomError(
                "readonly_model_param", "`output_type` is a model parameter and cannot be modified"
            )
        model_params_concrete: ModelParamsLikeT = TypeAdapter(
            type(
                f"{model_params.__name__}Validator",
                (model_params,),
                {"__pydantic_config__": pydantic_config},
            )  # needed for https://docs.pydantic.dev/latest/errors/usage_errors/#type-adapter-config-unused
        ).validate_python(self.model_dump())  # type: ignore
        return model_params_concrete

    @property
    def stem_names(self) -> tuple[ModelInputStemName | ModelOutputStemName, ...]:
        """Returns the model's input and output stem names."""
        return (*_INPUT_STEM_NAMES, *self.output_stem_names)

    model_config = ConfigDict(
        strict=True, extra="allow"
    )  # extra fields are not validated until `to_concrete`


class StftConfig(BaseModel):
    """configuration for the short-time fourier transform."""

    n_fft: FftSize
    hop_length: HopSize
    win_length: FftSize
    window_shape: WindowShape = "hann"
    normalized: bool = False

    model_config = _PYDANTIC_STRICT_CONFIG


class AudioIOConfig(BaseModel):
    target_sample_rate: SampleRate = 44100
    force_channels: Channels | None = 2
    """Whether to force mono or stereo audio input. If None, keep original."""

    model_config = _PYDANTIC_STRICT_CONFIG


class TorchCompileConfig(BaseModel):
    fullgraph: bool = True
    dynamic: bool = True
    mode: Literal["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"] = (
        "reduce-overhead"
    )


class InferenceConfig(BaseModel):
    normalize_input_audio: bool = False
    batch_size: BatchSize = 8
    force_weights_dtype: Dtype | None = None
    use_autocast_dtype: Dtype | None = None
    compile_model: TorchCompileConfig | None = None
    apply_tta: bool = False

    model_config = _PYDANTIC_STRICT_CONFIG


class ChunkingConfig(BaseModel):
    method: Literal["overlap_add_windowed"] = "overlap_add_windowed"
    overlap_ratio: OverlapRatio = 0.5
    window_shape: WindowShape = "hann"
    padding_mode: PaddingMode = "reflect"

    model_config = _PYDANTIC_STRICT_CONFIG


DerivedStemName: TypeAlias = Annotated[str, StringConstraints(min_length=1)]
"""The name of a derived stem, e.g. `vocals_minus_drums`."""
StemName: TypeAlias = Union[ModelOutputStemName, DerivedStemName]
"""A name of a stem, either a model output stem or a derived stem."""


class SubtractConfig(BaseModel):
    operation: Literal["subtract"]
    stem_name: StemName
    by_stem_name: StemName

    model_config = _PYDANTIC_STRICT_CONFIG


class SumConfig(BaseModel):
    operation: Literal["sum"]
    stem_names: NonEmptyUnique[Tuple[StemName]]

    model_config = _PYDANTIC_STRICT_CONFIG


DerivedStemRule: TypeAlias = Annotated[Union[SubtractConfig, SumConfig], Discriminator("operation")]
DerivedStemsConfig: TypeAlias = dict[DerivedStemName, DerivedStemRule]


class OutputConfig(BaseModel):
    stem_names: Literal["all"] | NonEmptyUnique[Tuple[StemName]] = "all"
    file_format: FileFormat = "wav"
    bit_rate: BitRate | None = None
    """Output bit rate for lossy formats. The default is chosen by FFmpeg."""

    model_config = _PYDANTIC_STRICT_CONFIG


# if we were to implement a model registry (which we shouldn't need)
# heavily consider https://peps.python.org/pep-0487/#subclass-registration


class Config(BaseModel):
    identifier: str
    """Unique identifier for this configuration"""
    model_type: ModelType
    model: LazyModelConfig
    stft: StftConfig | None = None
    audio_io: AudioIOConfig = Field(default_factory=AudioIOConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    derived_stems: DerivedStemsConfig | None = None
    output: OutputConfig = Field(default_factory=OutputConfig)
    experimental: dict[str, Any] | None = None
    """Any extra experimental configurations outside of the `splifft` core."""

    # NOTE: stft config can be none for models that operate on raw waveforms
    # they are checked in the processing step instead.
    @model_validator(mode="after")
    def check_derived_stems(self) -> Self:
        if self.derived_stems is None:
            return self
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

    @classmethod
    def from_file(cls, path: Path) -> Config:
        with open(path, "r") as f:
            config_data = json.load(f)
        return Config.model_validate(config_data)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # for .model
        strict=True,
        extra="forbid",
    )
