"""High level orchestrator for model inference"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import Tensor, nn

from .core import (
    NormalizedAudioTensor,
    RawAudioTensor,
    WindowTensor,
    denormalize_audio,
    derive_stems,
    generate_chunks,
    get_dtype,
    normalize_audio,
    stitch_chunks,
)

if TYPE_CHECKING:
    from .config import ChunkingConfig, Config, StemName
    from .core import Audio, BatchSize, ChunkSize, Dtype, NormalizationStats, NumModelStems
    from .models import ModelOutputStemName


def run_inference_on_file(
    mixture: Audio[RawAudioTensor], config: Config, model: nn.Module
) -> dict[StemName, RawAudioTensor]:
    """Runs the full source separation pipeline on a single audio file."""

    mixture_data: RawAudioTensor | NormalizedAudioTensor = mixture.data
    mixture_stats: NormalizationStats | None = None
    if config.inference.normalize_input_audio:
        norm_audio = normalize_audio(mixture)
        mixture_data = norm_audio.audio.data
        mixture_stats = norm_audio.stats

    separated_data = separate(
        mixture_data=mixture_data,
        chunk_cfg=config.chunking,
        model=model,
        batch_size=config.inference.batch_size,
        num_model_stems=len(config.model.output_stem_names),
        chunk_size=config.model.chunk_size,
        use_autocast_dtype=config.inference.use_autocast_dtype,
    )

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

    return output_stems


def separate(
    mixture_data: RawAudioTensor | NormalizedAudioTensor,
    chunk_cfg: ChunkingConfig,
    model: nn.Module,
    batch_size: BatchSize,
    num_model_stems: NumModelStems,
    chunk_size: ChunkSize,
    *,
    use_autocast_dtype: Dtype | None = None,
) -> Tensor:  # FIXME: update type hint.
    """Chunk, predict and stitch."""
    device = mixture_data.device
    original_num_samples = mixture_data.shape[-1]
    hop_size = int(chunk_size * (1 - chunk_cfg.overlap_ratio))

    padded_length = original_num_samples + 2 * (chunk_size - hop_size)
    num_chunks = max(0, (padded_length - chunk_size) // hop_size + 1)
    total_batches = math.ceil(num_chunks / batch_size)

    if chunk_cfg.window_shape == "hann":
        window = torch.hann_window(chunk_size, device=device)
    else:
        raise NotImplementedError(f"{chunk_cfg.window_shape=}")

    chunk_generator = generate_chunks(
        audio_data=mixture_data,
        chunk_size=chunk_size,
        hop_size=hop_size,
        batch_size=batch_size,
        padding_mode=chunk_cfg.padding_mode,
    )

    processed_chunks = []

    dtype_str = f" • {use_autocast_dtype}" if use_autocast_dtype else ""
    info_text = f"[cyan](bs=[bold]{batch_size}[/bold] • {device.type}{dtype_str})[/cyan]"

    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn(info_text),
    )

    with Progress(*progress_columns, transient=True) as progress:
        task = progress.add_task("processing chunks...", total=total_batches)

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                enabled=use_autocast_dtype is not None,
                dtype=(
                    get_dtype(use_autocast_dtype)
                    if use_autocast_dtype is not None
                    else torch.float32
                ),
            ),
        ):
            for chunk_batch in chunk_generator:
                separated_batch = model(chunk_batch)
                processed_chunks.append(separated_batch)
                progress.update(task, advance=1)

    return stitch_chunks(
        processed_chunks=processed_chunks,
        num_stems=num_model_stems,
        chunk_size=chunk_size,
        hop_size=hop_size,
        target_num_samples=original_num_samples,
        window=WindowTensor(window),
    )
