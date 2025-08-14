"""Reusable, pure algorithmic components for inference and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Generic, Iterator, Literal, NewType, TypeVar

import torch
import torch.nn.functional as F
from annotated_types import Ge, Gt, Lt
from torch import Tensor

if TYPE_CHECKING:
    from typing import Mapping, Sequence, TypeAlias

    from .config import DerivedStemsConfig, StemName
    from .models import ChunkSize, ModelOutputStemName


_AudioTensorLike = TypeVar("_AudioTensorLike")


@dataclass
class Audio(Generic[_AudioTensorLike]):
    data: _AudioTensorLike
    """This should either be an [raw][splifft.core.RawAudioTensor] or a
    [normalized][splifft.core.NormalizedAudioTensor] audio tensor."""
    sample_rate: SampleRate


#
# normalization
#


@dataclass
class NormalizationStats:
    """Statistics for [normalizing](https://en.wikipedia.org/wiki/Feature_scaling#Standardization_(Z-score_Normalization))
    and denormalizing audio.
    """

    mean: float
    r"""Mean $\mu$ of the mixture"""
    std: Annotated[float, Gt(0)]
    r"""Standard deviation $\sigma$ of the mixture"""


@dataclass
class NormalizedAudio:
    """Container for normalized audio and its original stats."""

    audio: Audio[NormalizedAudioTensor]  # NOTE: composition over inheritance.
    stats: NormalizationStats


def normalize_audio(audio: Audio[RawAudioTensor]) -> NormalizedAudio:
    """Preprocess the raw audio in the time domain to have a mean of 0 and a std of 1
    before passing it to the model.

    Operates on the mean of the [channels][splifft.core.Channels].
    """
    mono_audio = audio.data.mean(dim=0)
    mean = float(mono_audio.mean())
    std = float(mono_audio.std())

    if std <= 1e-8:  # silent audio
        return NormalizedAudio(
            audio=Audio(data=NormalizedAudioTensor(audio.data), sample_rate=audio.sample_rate),
            stats=NormalizationStats(mean, 1.0),
        )

    normalized_data = (audio.data - mean) / std
    return NormalizedAudio(
        audio=Audio(data=NormalizedAudioTensor(normalized_data), sample_rate=audio.sample_rate),
        stats=NormalizationStats(mean, std),
    )


def denormalize_audio(
    audio_data: NormalizedAudioTensor, stats: NormalizationStats
) -> RawAudioTensor:
    """Take the model output and restore them to their original loudness."""
    return RawAudioTensor((audio_data * stats.std) + stats.mean)


#
# chunking
#


def generate_chunks(
    audio_data: RawAudioTensor | NormalizedAudioTensor,
    chunk_size: ChunkSize,
    hop_size: HopSize,
    batch_size: BatchSize,
    *,
    padding_mode: PaddingMode = "reflect",
) -> Iterator[PaddedChunkedAudioTensor]:
    """Generates batches of overlapping chunks from an audio tensor.

    :return: An iterator that yields batches of chunks of shape (B, C, chunk_T).
    """
    padding = chunk_size - hop_size
    padded_audio = F.pad(audio_data, (padding, padding), mode=padding_mode)

    unfolded = padded_audio.unfold(
        dimension=-1, size=chunk_size, step=hop_size
    )  # (C, num_chunks, chunk_size)

    num_chunks = unfolded.shape[1]
    unfolded = unfolded.permute(1, 0, 2)  # (num_chunks, C, chunk_size)

    for i in range(0, num_chunks, batch_size):
        yield PaddedChunkedAudioTensor(unfolded[i : i + batch_size])


def stitch_chunks(
    processed_chunks: Sequence[SeparatedChunkedTensor],
    num_stems: NumModelStems,
    chunk_size: ChunkSize,
    hop_size: HopSize,
    target_num_samples: Samples,
    *,
    window: WindowTensor,
) -> RawSeparatedTensor:
    r"""Stitches processed audio chunks back together using the [overlap-add method](https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method).

    Reconstructs the full audio signal from a sequence of overlapping, processed chunks. Ensures
    that the sum of all overlapping windows is constant at every time step:
    $\sum_{m=-\infty}^{\infty} w[n - mH] = C$ where $H$ is the [hop size][splifft.core.HopSize].
    """
    all_chunks = torch.cat(tuple(processed_chunks), dim=0)
    total_chunks, _N, num_channels, _chunk_T = all_chunks.shape
    windowed_chunks = all_chunks * window.view(1, 1, 1, -1)

    # folding: (B, N * C * chunk_T) -> (1, N * C * chunk_T, total_chunks)
    reshaped_for_fold = windowed_chunks.permute(1, 2, 3, 0).reshape(
        1, num_stems * num_channels * chunk_size, total_chunks
    )

    total_length = (total_chunks - 1) * hop_size + chunk_size

    folded = F.fold(
        reshaped_for_fold,
        output_size=(1, total_length),
        kernel_size=(1, chunk_size),
        stride=(1, hop_size),
    )  # (1, N * C, 1, total_length)
    stitched = folded.view(num_stems, num_channels, total_length)

    # normalization for overlap-add
    windows_to_fold = window.expand(total_chunks, 1, chunk_size)
    reshaped_windows_for_fold = windows_to_fold.permute(1, 2, 0).reshape(
        1, chunk_size, total_chunks
    )
    norm_window = F.fold(
        reshaped_windows_for_fold,
        output_size=(1, total_length),
        kernel_size=(1, chunk_size),
        stride=(1, hop_size),
    ).squeeze(0)

    norm_window.clamp_min_(1e-8)  # for edges where the window sum might be zero
    stitched /= norm_window

    padding = chunk_size - hop_size
    return RawSeparatedTensor(stitched[..., padding : padding + target_num_samples])


#
# derive stems
#
def derive_stems(
    separated_stems: Mapping[ModelOutputStemName, RawAudioTensor],
    mixture_input: RawAudioTensor,
    stem_rules: DerivedStemsConfig,
) -> dict[StemName, RawAudioTensor]:
    """
    It is the caller's responsibility to ensure that all tensors are aligned and have the same shape.

    !!! note
        Mixture input and separated stems must first be [denormalized][splifft.core.denormalize_audio].
    """
    stems = {
        "mixture": RawAudioTensor(mixture_input),  # for subtraction
        **separated_stems,
    }

    for derived_name, rule in stem_rules.items():
        if rule.operation == "subtract":
            minuend = stems.get(rule.stem_name, mixture_input)
            subtrahend = stems.get(rule.by_stem_name, mixture_input)
            stems[derived_name] = RawAudioTensor(minuend - subtrahend)
        elif rule.operation == "sum":
            to_sum = tuple(stems[s] for s in rule.stem_names)
            stems[derived_name] = RawAudioTensor(torch.stack(to_sum).sum(dim=0))

    stems.pop("mixture", None)
    return stems


#
# misc
#


def get_dtype(dtype: Dtype) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"unsupported {dtype=}")


#
# The following used purely for type annotations and documentation.
# they provide semantic meaning *only* and we additionally use `NewType` for strong semantic distinction.
# to avoid mixing up different kinds of tensors.
#
# they are put right at the bottom for brevity and so no code implementations shall be placed beyond this point.
#

#
# key time domain concepts
#

Samples: TypeAlias = Annotated[int, Gt(0)]
"""Number of samples in the audio signal."""

SampleRate: TypeAlias = Annotated[int, Gt(0)]
"""The number of samples of audio recorded per second (hertz).

See [concepts](../concepts.md#introduction) for more details.
"""

Channels: TypeAlias = Annotated[int, Gt(0)]
"""Number of audio streams.

- 1: Mono audio
- 2: Stereo (left and right). Models are usually trained on stereo audio.
"""


FileFormat: TypeAlias = Literal["flac", "wav", "ogg"]
BitRate: TypeAlias = Literal[8, 16, 24, 32, 64]
"""Number of bits of information in each sample.

It determines the dynamic range of the audio signal: the difference between the quietest and loudest
possible sounds.

- 16-bit: Standard for CD audio: ~96 dB dynamic range.
- 24-bit: Common in professional audio, allowing for more headroom during mixing
- 32-bit float: Standard in digital audio workstations (DAWs) and deep learning models.
    The amplitude is represented by a floating-point number, which prevents clipping (distortion
    from exceeding the maximum value). This library primarily works with fp32 tensors.
"""

RawAudioTensor = NewType("RawAudioTensor", Tensor)
"""Time domain tensor of audio samples.
Shape ([channels][splifft.core.Channels], [samples][splifft.core.Samples])"""

NormalizedAudioTensor = NewType("NormalizedAudioTensor", Tensor)
"""A mixture tensor that has been normalized using [on-the-fly statistics][splifft.core.NormalizationStats].
Shape ([channels][splifft.core.Channels], [samples][splifft.core.Samples])"""

#
# key time-frequency domain concepts
#

ComplexSpectrogram = NewType("ComplexSpectrogram", Tensor)
r"""A complex-valued representation of audio's frequency content over time via the STFT.

Shape ([channels][splifft.core.Channels], [frequency bins][splifft.core.FftSize], [time frames][splifft.models.ChunkSize], 2)

See [concepts](../concepts.md#complex-spectrogram) for more details.
"""

HopSize: TypeAlias = Annotated[int, Gt(0)]
"""The step size, in samples, between the start of consecutive [chunks][splifft.models.ChunkSize].

To avoid artifacts at the edges of chunks, we process them with overlap. The hop size is the
distance we "slide" the chunking window forward. `ChunkSize < HopSize` implies overlap and the
overlap amount is `ChunkSize - HopSize`.
"""


# NOTE: sharing both for stft and overlap-add stitching for now
WindowShape: TypeAlias = Literal["hann", "hamming", "linear_fade"]
"""The shape of the window function applied to each chunk before computing the STFT."""


FftSize: TypeAlias = Annotated[int, Gt(0)]
"""The number of frequency bins in the STFT, controlling the [frequency resolution](../concepts.md#fft-size)."""

Bands: TypeAlias = Tensor
"""Groups of [adjacent frequency bins in the spectrogram](../concepts.md#bands)."""

#
# miscallaneous
#
BatchSize: TypeAlias = Annotated[int, Gt(0)]
"""The number of chunks processed simultaneously by the GPU.

Increasing the batch size can improve GPU utilisation and speed up training, but it requires more
memory.
"""

Dtype: TypeAlias = Literal["float32", "float16", "bfloat16"]

# preprocessing

PaddingMode: TypeAlias = Literal["reflect", "constant", "replicate"]
"""The method used to pad the audio before chunking, crucial for handling the edges of the audio signal.

- `reflect`: Pads the signal by reflecting the audio at the boundary. This creates a smooth
  continuation and often yields the best results for music.
- `constant`: Pads with zeros. Simpler, but can introduce silence at the edges.
- `replicate`: Repeats the last sample at the edge.
"""
# TODO: we should intelligently decide whether to choose reflect or constant.
# for songs that start with silence, we should use constant padding.


ChunkDuration: TypeAlias = Annotated[float, Gt(0)]
"""The length of an audio segment, in seconds, processed by the model at one time.

Equivalent to [chunk size][splifft.models.ChunkSize] divided by the [sample rate][splifft.core.SampleRate].
"""

OverlapRatio: TypeAlias = Annotated[float, Ge(0), Lt(1)]
r"""The fraction of a chunk that overlaps with the next one.

The relationship with [hop size][splifft.core.HopSize] is:
$$
\text{hop\_size} = \text{chunk\_size} \cdot (1 - \text{overlap\_ratio})
$$

- A ratio of `0.0` means no overlap (hop_size = chunk_size).
- A ratio of `0.5` means 50% overlap (hop_size = chunk_size / 2).
- A higher overlap ratio increases computational cost as more chunks are processed, but it can lead
  to smoother results by averaging more predictions for each time frame.
"""

Padding: TypeAlias = Annotated[int, Gt(0)]
"""Samples to add to the beginning and end of each chunk.

- To ensure that the very beginning and end of a track can be centerd within a chunk, we often may
  add "reflection padding" or "zero padding" before chunking.
- To ensure that the last chunk is full-size, we may pad the audio so its length is a multiple of
  the hop size. 
"""

PaddedChunkedAudioTensor = NewType("PaddedChunkedAudioTensor", Tensor)
"""A batch of audio chunks from a padded source.
Shape ([batch size][splifft.core.BatchSize], [channels][splifft.core.Channels], [chunk size][splifft.models.ChunkSize])"""

NumModelStems: TypeAlias = Annotated[int, Gt(0)]
"""The number of stems the model outputs. This should be the length of [splifft.models.ModelConfigLike.output_stem_names]."""

# post separation stitching

SeparatedChunkedTensor = NewType("SeparatedChunkedTensor", Tensor)
"""A batch of separated audio chunks from the model.
Shape ([batch size][splifft.core.BatchSize], [number of stems][splifft.core.NumModelStems], [channels][splifft.core.Channels], [chunk size][splifft.models.ChunkSize])"""

WindowTensor = NewType("WindowTensor", Tensor)
"""A 1D tensor representing a window function.
Shape ([chunk size][splifft.models.ChunkSize])"""

RawSeparatedTensor = NewType("RawSeparatedTensor", Tensor)
"""The final, stitched, raw-domain separated audio.
Shape ([number of stems][splifft.core.NumModelStems], [channels][splifft.core.Channels], [samples][splifft.core.Samples])"""

#
# evaluation metrics
# We use bold letters like $\mathbf{s}$ to denote the entire signal tensor.
# NOTE: once we implement these metrics, cut down on the docstrings.
#

Sdr: TypeAlias = float
r"""Signal-to-Distortion Ratio (decibels). Higher is better.

Measures the ratio of the power of clean reference signal to the power of all other error components
(interference, artifacts, and spatial distortion).

Definition:
$$
\text{SDR} = 10 \log_{10} \frac{|\mathbf{s}|^2}{|\mathbf{s} - \mathbf{\hat{s}}|^2},
$$
where:

- $\mathbf{s}$: ground truth source signal
- $\mathbf{\hat{s}}$: estimated source signal produced by the model
- $||\cdot||^2$: squared L2 norm (power) of the signal
"""

SiSdr: TypeAlias = float
r"""Scale-Invariant SDR (SI-SDR) is invariant to scaling errors (decibels). Higher is better.

It projects the estimate onto the reference to find the optimal scaling factor $\alpha$, creating a scaled reference that best matches the estimate's amplitude.

- Optimal scaling factor: $\alpha = \frac{\langle\mathbf{\hat{s}}, \mathbf{s}\rangle}{||\mathbf{s}||^2}$
- Scaled reference: $\mathbf{s}_\text{target} = \alpha \cdot \mathbf{s}$
- Error: $\mathbf{e} = \mathbf{\hat{s}} - \mathbf{s}_\text{target}$
- $\text{SI-SDR} = 10 \log_{10} \frac{||\mathbf{s}_\text{target}||^2}{||\mathbf{e}||^2}$
"""

L1Norm: TypeAlias = float
r"""L1 norm (mean absolute error) between two signals (dimensionless). Lower is better.

Measures the average absolute difference between the reference and estimated signals.

- Time domain: $\mathcal{L}_\text{L1} = \frac{1}{N}
\sum_{n=1}^{N} |\mathbf{s}[n] - \mathbf{\hat{s}}[n]|$,
- Frequency domain: $\mathcal{L}_\text{L1Freq} = \frac{1}{\text{MK}}\sum_{m=1}^{M}
\sum_{k=1}^{K} \left||S(m, k)| - |\hat{S}(m, k)|\right|$
"""  # NOTE: zfturbo scales by to 1-100

DbDifferenceMel: TypeAlias = float
r"""Difference in the dB-scaled mel spectrogram.
$$
\mathbf{D}(m, k) = \text{dB}(|\hat{S}_\text{mel}(m, k)|) - \text{dB}(|S_\text{mel}(m, k)|)
$$
"""

Bleedless: TypeAlias = float
r"""A metric to quantify the amount of "bleeding" from other sources. Higher is better.

Measures the average energy of the parts of the [mel spectrogram][splifft.core.DbDifferenceMel]
that are louder than the reference.
A high value indicates that the estimate contains unwanted energy (bleed) from other sources:
$$
\text{Bleed} = \text{mean}(\mathbf{D}(m, k)) \quad \forall \quad \mathbf{D}(m, k) > 0
$$
"""

Fullness: TypeAlias = float
r"""A metric to quantify how much of the original source is missing. Higher is better.

Complementary to [Bleedless][splifft.core.Bleedless].
Measures the average energy of the parts of the [mel spectrogram][splifft.core.DbDifferenceMel]
that are quieter than the reference.
A high value indicates that parts of the target loss were lost during the separation, indicating
that more of the original source's character is preserved.
$$
\text{Fullness} = \text{mean}(|\mathbf{D}(m, k)|) \quad \forall \quad \mathbf{D}(m, k) < 0
$$
"""
