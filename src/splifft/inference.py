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
from torch import nn

from .core import (
    _get_window_fn,
    create_w2w_model,
    denormalize_audio,
    derive_stems,
    generate_chunks,
    normalize_audio,
    stitch_chunks,
)

if TYPE_CHECKING:
    from . import types as t
    from .config import ChunkingConfig, Config, MaskingConfig, StemName, StftConfig
    from .core import Audio, NormalizationStats
    from .models import (
        ModelParamsLike,
    )


def run_inference_on_file(
    mixture: Audio[t.RawAudioTensor],
    config: Config,
    model: nn.Module,
    model_params_concrete: ModelParamsLike,
) -> dict[StemName, t.RawAudioTensor]:
    """Runs the full source separation pipeline on a single audio file."""
    mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor = mixture.data
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
        model_input_type=model_params_concrete.input_type,
        model_output_type=model_params_concrete.output_type,
        stft_cfg=config.stft,
        masking_cfg=config.masking,
        use_autocast_dtype=config.inference.use_autocast_dtype,
    )

    denormalized_stems: dict[t.ModelOutputStemName, t.RawAudioTensor] = {}
    for i, stem_name in enumerate(config.model.output_stem_names):
        stem_data = separated_data[i, ...]
        if mixture_stats is not None:
            stem_data = denormalize_audio(
                audio_data=t.NormalizedAudioTensor(stem_data),
                stats=mixture_stats,
            )
        denormalized_stems[stem_name] = t.RawAudioTensor(stem_data)

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
    mixture_data: t.RawAudioTensor | t.NormalizedAudioTensor,
    chunk_cfg: ChunkingConfig,
    model: nn.Module,
    batch_size: t.BatchSize,
    num_model_stems: t.NumModelStems,
    chunk_size: t.ChunkSize,
    model_input_type: t.ModelInputType,
    model_output_type: t.ModelOutputType,
    stft_cfg: StftConfig | None,
    masking_cfg: MaskingConfig,
    *,
    use_autocast_dtype: torch.dtype | None = None,
) -> t.RawSeparatedTensor:
    """Chunk, predict and stitch."""
    device = mixture_data.device
    original_num_samples = mixture_data.shape[-1]
    hop_size = int(chunk_size * (1 - chunk_cfg.overlap_ratio))

    window = _get_window_fn(chunk_cfg.window_shape, chunk_size, device)

    padded_length = original_num_samples + 2 * (chunk_size - hop_size)
    num_chunks = max(0, (padded_length - chunk_size) // hop_size + 1)
    total_batches = math.ceil(num_chunks / batch_size)

    chunk_generator = generate_chunks(
        audio_data=mixture_data,
        chunk_size=chunk_size,
        hop_size=hop_size,
        batch_size=batch_size,
        padding_mode=chunk_cfg.padding_mode,
    )

    model_w2w = create_w2w_model(
        model=model,
        model_input_type=model_input_type,
        model_output_type=model_output_type,
        stft_cfg=stft_cfg,
        num_channels=mixture_data.shape[0],
        chunk_size=chunk_size,
        masking_cfg=masking_cfg,
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
                dtype=use_autocast_dtype,
            ),
        ):
            for chunk_batch in chunk_generator:
                separated_batch = model_w2w(chunk_batch)
                processed_chunks.append(separated_batch)
                progress.update(task, advance=1)

    return stitch_chunks(
        processed_chunks=processed_chunks,
        num_stems=num_model_stems,
        chunk_size=chunk_size,
        hop_size=hop_size,
        target_num_samples=original_num_samples,
        window=t.WindowTensor(window),
    )
