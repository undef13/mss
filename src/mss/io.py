"""Operations for reading and writing to disk. All side effects should go here."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import torch
import torchaudio

from .core import Audio, RawAudioTensor, SampleRate
from .models import ModelT

if TYPE_CHECKING:
    from typing import TypeAlias


FileLike: TypeAlias = Path | str | BinaryIO


def read_audio(
    file: FileLike,
    target_sr: SampleRate,
    target_channels: int | None,
    device: torch.device | None = None,
) -> Audio[RawAudioTensor]:
    """Loads, resamples and converts channels."""
    waveform, sr = torchaudio.load(file, channels_first=True)
    waveform = waveform.to(device)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr).to(device)
        waveform = resampler(waveform)

    current_channels = waveform.shape[0]
    if target_channels is not None and current_channels != target_channels:
        if target_channels == 1:  # stereo -> mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif target_channels == 2:  # mono -> stereo
            waveform = waveform.repeat(2, 1)
        else:
            raise ValueError(
                f"expected target_channels to be 1 or 2, got {target_channels=} with {current_channels=}."
            )

    return Audio(RawAudioTensor(waveform), target_sr)


# NOTE: torchaudio.save is simple enough and a wrapper is not needed.


#
# model loading
#


def load_weights(
    model: ModelT,
    checkpoint_file: FileLike,
    device: torch.device,
) -> ModelT:
    """Load the weights from a checkpoint into the given model."""

    state_dict = torch.load(checkpoint_file, map_location=device, weights_only=True)

    # TODO: DataParallel and `module.` prefix
    model.load_state_dict(state_dict)
    # NOTE: do not torch.compile here!

    return model.to(device)
