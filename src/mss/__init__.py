# ruff: noqa: I001
from pathlib import Path

from .config import (
    Config,
    # lazy model config
    LazyModelConfig,
    ModelInputStemName,
    ModelOutputStemName,
    # io
    AudioIOConfig,
    # inference
    InferenceConfig,
    # chunking
    ChunkingConfig,
    # derived stems
    DerivedStemsConfig,
    DerivedStemName,
    DerivedStemRule,
    SubtractConfig,
    SumConfig,
    OutputConfig,
)
from .models import ModelType, ModelConfigLike
from .io import read_config

# TODO: this should be in the io module and we should use user cache paths
PATH_MODULE = Path(__file__).parent.parent
PATH_BASE = PATH_MODULE.parent
PATH_DATA = PATH_BASE / "data"
PATH_CONFIG = PATH_DATA / "config"
PATH_MODELS = PATH_DATA / "models"

# re-exports
__all__ = [
    # config
    "Config",
    "ModelInputStemName",
    "ModelOutputStemName",
    "LazyModelConfig",
    "AudioIOConfig",
    "InferenceConfig",
    "ChunkingConfig",
    "DerivedStemName",
    "DerivedStemRule",
    "DerivedStemsConfig",
    "SubtractConfig",
    "SumConfig",
    "OutputConfig",
    # models
    "ModelType",
    "ModelConfigLike",
    # io
    "read_config",
    # data paths
    "PATH_DATA",
    "PATH_CONFIG",
    "PATH_MODELS",
]
