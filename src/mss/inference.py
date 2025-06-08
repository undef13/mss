"""High level orchestrator for model inference"""

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .core import (
    Audio,
    NormalizedAudio,
    RawAudioTensor,
    denormalize_audio,
    derive_stems,
    generate_chunks,
    normalize_audio,
    stitch_chunks,
)
from .io import read_audio, write_audio

if TYPE_CHECKING:
    from .config import ChunkingConfig, Config
    from .core import (
        Audio,
        BatchSize,
        NormalizationStats,
        NormalizedAudioTensor,
        NumModelStems,
    )
    from .models import ModelOutputStemName


def run_inference_on_file(
    mixture_path: Path,
    config: Config,
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Runs the full source separation pipeline on a single audio file."""
    mixture = read_audio(
        mixture_path,
        config.audio_io.target_sample_rate,
        config.audio_io.force_channels,
        device=device,
    )
    mixture_data: RawAudioTensor | NormalizedAudioTensor = mixture.data
    mixture_stats: NormalizationStats | None = None
    if config.inference.normalize_input_audio:
        norm_audio = normalize_audio(mixture)
        mixture_data = norm_audio.audio.data
        mixture_stats = norm_audio.stats

    separated_data = _core_separation_process(
        model=model,
        mixture_data=mixture_data,  # upload to gpu
        chunk_cfg=config.chunking,
        batch_size=config.inference.batch_size,
        num_model_stems=len(config.model.output_stem_names),
    )  # -> (num_model_stems, C, T)

    denormalized_stems: dict[ModelOutputStemName, RawAudioTensor] = {}
    for i, stem_name in enumerate(config.model.output_stem_names):
        stem_data = separated_data[i, ...]
        if mixture_stats is not None:
            stem_data = denormalize_audio(
                audio_data=NormalizedAudioTensor(stem_data),
                stats=mixture_stats,
            )
            denormalized_stems[stem_name] = stem_data
        else:
            denormalized_stems[stem_name] = RawAudioTensor(stem_data)

    if config.inference.apply_tta:
        raise NotImplementedError

    output_stems = denormalized_stems
    if config.derived_stems:
        output_stems = derive_stems(
            denormalized_stems,
            mixture.data,
            config.derived_stems,
        )

    if not config.output:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for stem_name, stem_data in output_stems.items():
        if stem_name not in config.output.stem_names:
            continue

        write_audio(
            output_dir / f"{stem_name}",
            Audio(RawAudioTensor(stem_data.cpu()), mixture.sample_rate),
            config.output.file_format,
            config.output.audio_format,
        )


def _core_separation_process(
    model: nn.Module,
    mixture_data: RawAudioTensor | NormalizedAudioTensor,
    chunk_cfg: ChunkingConfig | None,
    batch_size: BatchSize,
    num_model_stems: NumModelStems,
) -> Tensor:
    """Chunk, predict and stitch.

    Dimensions:
    - `N`: number of stems
    - `C`: [number of channels][mss.core.Channels]
    - `T`

    :param model: The separation model.
    :param audio_data: The audio tensor to be processed, of shape (C, T).
    :param num_model_stems: The number of stems the model outputs.
    :return: The separated audio tensor of shape (N, C, T).
    """
    device = mixture_data.device
    original_num_samples = mixture_data.shape[-1]

    if chunk_cfg is None:
        raise NotImplementedError
        # pad to model_cfg.chunk_size for the model

    sample_rate = 0  # FIXME
    chunk_size = int(chunk_cfg.chunk_duration * sample_rate)
    hop_size = int(chunk_size * (1 - chunk_cfg.overlap_ratio))
    window = torch.hann_window(chunk_size, device=device)

    chunk_generator = generate_chunks(
        audio_data=mixture_data,
        chunk_size=chunk_size,
        hop_size=hop_size,
        batch_size=batch_size,
    )

    processed_chunks = []
    with torch.inference_mode():
        for chunk_batch in chunk_generator:
            chunk_batch = chunk_batch.to(device)
            # (B, C, T) -> (B, N, C, T)
            separated_batch = model(chunk_batch)
            processed_chunks.append(separated_batch.cpu())

    return stitch_chunks(
        processed_chunks=processed_chunks,
        num_stems=num_model_stems,
        chunk_size=chunk_size,
        hop_size=hop_size,
        target_num_samples=original_num_samples,
        window=window,
    )
