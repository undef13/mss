"""Operations for reading and writing to disk.

!!! warning
    This module is incomplete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn

from .core import Audio

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TypeAlias

    from .core import RawAudioTensor, SampleRate


def read_audio(
    path: Path,
    target_sr: SampleRate,
    target_channels: int | None,
    device: torch.device | None = None,
) -> Audio[RawAudioTensor]:
    """Loads, resamples, converts channels"""
    # we should probably just use torchaudio, not soundfile or librosa
    raise NotImplementedError


def write_audio(
    path: Path, audio: Audio[RawAudioTensor], file_format: FileFormat, audio_format: AudioFormat
) -> None:
    """Writes audio to disk"""
    raise NotImplementedError


def read_model_from_checkpoint(
    identifier: str, model_params_dict: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> nn.Module:
    # uses a registry or if/else to get ModelClass and ModelParamsDataclass from identifier.
    # params_instance = ModelParamsDataclass(**model_params_dict).
    # model = ModelClass(params_instance).

    raise NotImplementedError
    # for simplicity, do not handle mismatches (yet)


FileFormat: TypeAlias = Literal["flac", "wav"]  # TODO: support more, mp3 etc.
AudioFormat: TypeAlias = Literal[
    "PCM_16", "PCM_24", "FLOAT"
]  # TODO: consider https://trac.ffmpeg.org/wiki/audio%20types
