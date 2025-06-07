from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .core import (
    denormalize_audio,
    derive_stems,
    generate_chunks,
    normalize_audio,
    stitch_chunks,
)
from .io import read_audio, write_audio

if TYPE_CHECKING:
    from .config import ChunkingConfig, Config
    from .core import Audio, BatchSize, NormalizedAudio, NumModelStems


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
    )
    mixture_data = mixture.data.to(device)

    norm_audio: NormalizedAudio | None = None
    if config.inference.normalize_input_audio:
        norm_audio = normalize_audio(mixture_data)
        mixture_data = norm_audio.data

    separated_data = _core_separation_process(
        model=model,
        audio_data=mixture_data,
        chunk_cfg=config.chunking,
        batch_size=config.inference.batch_size,
        num_model_stems=len(config.model.output_stem_names),
    )  # -> (num_model_stems, C, T)

    denormalized_stems = {}
    for i, stem_name in enumerate(config.model.output_stem_names):
        stem_tensor = separated_data[i, ...]
        if norm_audio:
            stem_tensor = denormalize_audio(
                NormalizedAudio(data=stem_tensor, stats=norm_audio.stats)
            )
        denormalized_stems[stem_name] = stem_tensor

    if config.inference.apply_tta:
        raise NotImplementedError

    output_stems = denormalized_stems
    if config.derived_stems:
        output_stems = derive_stems(
            denormalized_stems,
            mixture.data.to(device),
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
            Audio(stem_data.cpu(), mixture.sample_rate),
            config.output.file_format,
            config.output.pcm_type,
        )


def _core_separation_process(
    model: nn.Module,
    audio_data: Tensor,
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
    device = audio_data.device
    original_length = audio_data.shape[-1]

    if chunk_cfg is None:
        raise NotImplementedError
        # pad to model_cfg.chunk_size for the model

    sample_rate = 0  # FIXME
    chunk_size = int(chunk_cfg.chunk_duration * sample_rate)
    hop_size = int(chunk_size * (1 - chunk_cfg.overlap_ratio))
    window = torch.hann_window(chunk_size, device=device)

    chunk_generator = generate_chunks(
        audio=audio_data,
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
        target_length=original_length,
        chunk_size=chunk_size,
        hop_size=hop_size,
        window=window,
    )
