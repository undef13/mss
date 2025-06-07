"""Operations for reading and writing to disk.

!!! warning
    This module is incomplete.
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import Config
from .core import Audio


def read_config(path: Path) -> Config:
    with open(path, "r") as f:
        config_data = json.load(f)
    return Config.model_validate(config_data)


def read_audio(path: Path, target_sr: int, target_channels: int | None) -> Audio:
    """Loads, resamples, converts channels"""
    raise NotImplementedError


def write_audio(path: Path, audio: Audio, format: str, pcm_type: str) -> None:
    """Writes audio to disk"""
    raise NotImplementedError


def read_model_from_checkpoint(
    identifier: str, model_params_dict: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> nn.Module:
    # Uses a registry or if/else to get ModelClass and ModelParamsDataclass from identifier.
    # params_instance = ModelParamsDataclass(**model_params_dict).
    # model = ModelClass(params_instance).
    # Loads checkpoint, handling potential mismatches.
    raise NotImplementedError
