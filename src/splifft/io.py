"""Operations for reading and writing to disk. All side effects should go here."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import torch
from torchcodec.decoders import AudioDecoder

from .core import Audio

if TYPE_CHECKING:
    from typing import TypeAlias

    from . import types as t
    from .models import ModelT


FileLike: TypeAlias = Path | str | BinaryIO


def read_audio(
    file: FileLike,
    target_sr: t.SampleRate,
    target_channels: int | None,
    device: torch.device | None = None,
) -> Audio[t.RawAudioTensor]:
    """Loads, resamples and converts channels."""
    decoder = AudioDecoder(source=str(file), sample_rate=target_sr, num_channels=target_channels)
    samples = decoder.get_all_samples()
    waveform = samples.data.to(device)

    return Audio(t.RawAudioTensor(waveform), samples.sample_rate)


# NOTE: torchaudio.save is simple enough and a wrapper is not needed.


#
# model loading
#


def load_weights(
    model: ModelT, checkpoint_file: FileLike, device: torch.device | str, *, strict: bool = False
) -> ModelT:
    """Load the weights from a checkpoint into the given model."""

    state_dict = torch.load(checkpoint_file, map_location=device, weights_only=True)

    # TODO: DataParallel and `module.` prefix
    model.load_state_dict(state_dict, strict=strict)
    # NOTE: do not torch.compile here!

    return model.to(device)
