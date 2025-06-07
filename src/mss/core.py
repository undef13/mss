"""Reusable, pure algorithmic components for inference and training.

!!! warning
    This module is incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence, TypeAlias

from torch import Tensor

from .config import DerivedStemsConfig


@dataclass
class Audio:
    data: Tensor
    sample_rate: SampleRate


#
# normalization
#


@dataclass
class NormalizationStats:
    """Statistics for normalizing and denormalizing audio.

    Neural networks are sensitive to the scale of input data and often perform better with inputs
    that have a consistent statistical distribution. Normalization helps to achieve this.
    """

    mean: float
    r"""Mean $\mu$ of the mixture"""
    std: float
    r"""Standard deviation $\sigma$ of the mixture"""


@dataclass
class NormalizedAudio:
    """Container for normalized audio and its original stats."""

    data: Tensor
    stats: NormalizationStats


def normalize_audio(audio_data: Tensor) -> NormalizedAudio:
    """Preprocess the raw audio in the time domain to have a mean of 0 and a std of 1
    before passing it to the model.

    Operates on the mean of the [channels][mss.core.Channels].
    """
    raise NotImplementedError


def denormalize_audio(normalized_audio: NormalizedAudio) -> Tensor:
    """Take the model output and restore them to their original loudness."""
    return (normalized_audio.data * normalized_audio.stats.std) + normalized_audio.stats.mean


#
# chunking
#


def generate_chunks(
    audio: Tensor,
    chunk_size: ChunkSize,
    hop_size: HopSize,
    batch_size: BatchSize,
    *,
    padding_mode: str = "reflect",
) -> Iterator[Tensor]:
    """Generates batches of overlapping chunks from an audio tensor.

    :return: An iterator that yields batches of chunks of shape (B, C, chunk_T).
    """
    # handles padding
    raise NotImplementedError


def stitch_chunks(
    processed_chunks: Sequence[Tensor],
    num_stems: NumModelStems,
    target_length: int,
    chunk_size: ChunkSize,
    hop_size: HopSize,
    window: Tensor,
) -> Tensor:
    """Stitches processed audio chunks back together using the overlap-add method.

    When chunks overlap, the overlapping sections are processed multiple times.
    To reconstruct the full audio signal without clicks or discontinuities at chunk boundaries, the
    overlapping parts are smoothly faded in and out.

    1. Multiply each processed chunk by a [window function][mss.core.WindowFunction].
    2. Add the windowed chunks together at their original positions.
    3. The overlapping sections were added, so their amplitude is higher.
       Divide by the sum of all applied windows to restore the correct amplitude.
       Such a constant is precalculated as the window is constant.

    Dimensions:

    - `B`: [batch size][mss.core.BatchSize]
    - `C`: [number of channels][mss.core.Channels]
    - `chunk_T`: [chunk length][mss.core.ChunkSize]
    - `target_T`: target length of the original audio signal
    """
    raise NotImplementedError


#
# derive stems
#
def derive_stems(
    separated_stems: dict[str, Tensor], mixture_input: Tensor, stem_rules: DerivedStemsConfig
) -> dict[str, Tensor]:
    raise NotImplementedError


#
# The following used purely for type annotations and documentation.
# they are put right at the bottom for brevity and no logic shall be placed below this point.
#

#
# key time domain concepts
#

SampleRate: TypeAlias = int
"""The number of samples of audio recorded per second (hertz).

According to the [Nyquist-Shannon sampling theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem),
the maximum frequency that can be accurately represented is half the sample rate. The full range of
human hearing is approximately 20 Hz to 20000 Hz.

- 44100 Hz: Standard for CD audio, most common sample rate for music.
- 48000 Hz: Standard in professional audio
- 16000 Hz: Common for voice recordings as it sufficiently captures the human voice
"""

BitDepth: TypeAlias = int
"""Number of bits of information in each sample.

It determines the dynamic range of the audio signal: the difference between the quietest and loudest
possible sounds.

- 16-bit: Standard for CD audio: ~96 dB dynamic range.
- 24-bit: Common in professional audio, allowing for more headroom during mixing
- 32-bit float: Standard in digital audio workstations (DAWs) and deep learning models.
    The amplitude is represented by a floating-point number, which prevents clipping (distortion
    from exceeding the maximum value). This library primarily works with fp32 tensors.
"""

Channels: TypeAlias = int
"""Number of audio streams.

- 1: Mono audio
- 2: Stereo (left and right). Models are usually trained on stereo audio.
"""

#
# key time-frequency domain concepts
#

Spectrogram: TypeAlias = Tensor
r"""A 2D representation of audio's frequency content over time.


While the time domain gives us the amplitude over time, it doesn't explicitly tell us about
frequency content. The [Short-Time Fourier Transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
(STFT) is the cornerstone of transforming a 1D discrete-time signal $x[n]$ into a 2D time-frequency
representation $X(m, k)$ of shape `(frequency_bins, time_frames, 2)`).

The STFT coefficient $X(m, k)$ is a complex number that can be decomposed into:

- the **magnitude** $|X(m, k)|$ (tells us "how much" of a frequency is present)
- the **phase** $\phi(m, k)$ (tells us "how it's aligned" in time)

of a specific time frame, where $m$ is the time frame index and $k$ is the frequency bin index.
Human hearing is highly sensitive to phase differences, crucial for sound localisation and timbre
perception.

```text
Freq (Hz)                                                  Magnitude
     ^                                                   -------------
8000 |   @  :  :  +  @   @@:#     @  :  +  =               @  high    
4000 |   @  :  +  @-#@#@## ##@#   @  :  :  +               #  medium  
2000 |   @+ : @#@-## @+    =  @#@@@+ :  +  =               -  low     
1000 |   @##+@-#  +  @##+= +     @@##+= :  :             -------------
 500 |   @@%%%###++= @@%%%###++=  @@%%%###++=    
   0 +----------------------------------------> time (s)

Freq (Hz)                                                    Phase
     ^ - ++++ -  -    -  +-   +-+- --+  - +-+ +          -------------
8000 |  +- +     -+-+-- +-  -  +    ++    +   +            +   2pi    
4000 | + +  -  + - +- - --+++-+- +-+ +-  +- --+                       
2000 |      +-+++ - -++ + +   -----   + -+  ++             -   -2pi   
1000 | -++--+ +  --  - ++  +-++--  - ---   +  +          -------------
 500 | +-++ + +  +  + --- ++     - -  +  + - -    
   0 +----------------------------------------> Time (s)
```

Phase is notoriously difficult to model: it behaves chaotically and wraps around from $-\pi$ to
$\pi$. Early models discarded phase information, focusing only on the magnitude spectrogram,
and used the [Griffin-Lim algorithm](https://ieeexplore.ieee.org/document/1164317) to estimate the
plausible phase. Modern models like [SCNet](https://arxiv.org/abs/2401.13276) use the complex valued
spectrogram as the loss function directly.

Definition:
$$
X(m, k) = \sum_{n=-\infty}^{\infty} x[n] \cdot w[n - mH] \cdot e^{-j \frac{2\pi kn}{N_\text{fft}}},
$$
where:

Practically, the process involves:

1. Dividing the audio signal into short, overlapping segments in time (chunks), parameterised by the
   [hop size][mss.core.HopSize] $H$
2. Applying a [window function][mss.core.WindowFunction] $w[n]$ (e.g.
   [Hann window][torch.hann_window]) to each chunk to reduce spectral leakage
3. Computing the Fast Fourier Transform (FFT) on each windowed segment to get its complex frequency
   spectrum. The [FFT size][mss.core.FftSize] $N_\text{fft}$ determines the number of frequency
   bins.
4. Stacking these spectra to form the 2D complex spectrogram.

Some models like [BS-Roformer][mss.models.bs_roformer.BSRoformer] use the linear frequency scale and
learn their own perceptually relevant [bandings][mss.core.Bands]. Other models like Mel-Roformer
is based on the [Mel scale](https://en.wikipedia.org/wiki/Mel_scale), which is a perceptual scale
of pitches that approximates human hearing.

Neural networks in source separation essentially learn to approximate an ideal ratio mask (or its 
complex equivalent): $\hat{S}_\text{source} = M_\text{complex} \odot S_\text{mixture}$.
"""

HopSize: TypeAlias = int
"""The step size, in samples, between the start of consecutive [chunks][mss.core.ChunkSize].

To avoid artifacts at the edges of chunks, we process them with overlap. The hop size is the
distance we "slide" the chunking window forward. `ChunkSize < HopSize` implies overlap and the
overlap amount is `ChunkSize - HopSize`.
"""


WindowFunction: TypeAlias = Callable[[Tensor], Tensor]
# TODO: add docs


FftSize: TypeAlias = int
r"""The number of frequency bins in the STFT.

The [time-frequency uncertainty principle](https://en.wikipedia.org/wiki/Uncertainty_principle#Signal_processing)
states that there is a fundamental tradeoff between the standard deviations in time and frequency
energy concentrations:
$$
\sigma_t \sigma_f \ge \frac{1}{4\pi}
$$

- A short window (small $N_\text{fft}$) gives good time resolution, excellent for capturing sharp
percussive sounds like drum hits (transients), but it blurs frequencies together, making it hard to
separate instruments with close pitches.
- A long window (large $N_\text{fft}$) gives good frequency resolution, excellent for separating the
fine harmonics of tonal instruments like a violin or piano, but it blurs the exact timing.

The `auraloss` library's `MultiResolutionSTFTLoss` calculates the loss on spectrograms with
multiple FFT sizes, forcing the model to optimise for both transient and tonal sources.
"""

Bands: TypeAlias = Tensor
"""Groups of adjacent frequency bins in the spectrogram.

Instead of processing every single frequency bin independently, we can group them into "bands".
This reduces the computational complexity and allows the model to learn relationships within
frequency regions, which often correspond to musical harmonics or instrument characteristics.
"""

#
# miscallaneous
#

ChunkSize: TypeAlias = int
"""The length of an audio segment, in samples, processed by the model at one time.

A full audio track is often too long to fit into GPU, instead we process it in fixed-size chunks.
A larger chunk size may allow the model to capture more temporal context at the cost of increased
memory usage.
"""

Padding: TypeAlias = int
"""Samples to add to the beginning and end of each chunk.

- To ensure that the very beginning and end of a track can be centerd within a chunk, we often may
  add "reflection padding" or "zero padding" before chunking.
- To ensure that the last chunk is full-size, we may pad the audio so its length is a multiple of
  the hop size. 
"""

BatchSize: TypeAlias = int
"""The number of chunks processed simultaneously by the GPU.

Increasing the batch size can improve GPU utilisation and speed up training, but it requires more
memory.
"""

NumModelStems: TypeAlias = int
"""The number of stems the model outputs. This should be the length of [mss.models.ModelConfigLike.output_stem_names]."""

#
# evaluation metrics
# We use bold letters like $\mathbf{s}$ to denote the entire signal tensor.
#

SDR: TypeAlias = float
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

SISDR: TypeAlias = float
r"""Scale-Invariant SDR (SI-SDR) is invariant to scaling errors (decibels). Higher is better.

It projects the estimate onto the reference to find the best possible scaling:

- Target projection: $\mathbf{s}_\text{target} = \frac{\langle\mathbf{s},
\mathbf{\hat{s}}\rangle}{||\mathbf{\hat{s}}||^2} \mathbf{\hat{s}}$, where $\langle\cdot, \cdot\rangle$ is the inner
product.
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

Measures the average energy of the parts of the [mel spectrogram][mss.core.DbDifferenceMel]
that are louder than the reference.
A high value indicates that the estimate contains unwanted energy (bleed) from other sources:
$$
\text{Bleed} = \text{mean}(\mathbf{D}(m, k)) \quad \forall \quad \mathbf{D}(m, k) > 0
$$
"""

Fullness: TypeAlias = float
r"""A metric to quantify how much of the original source is missing. Higher is better.

Complementary to [Bleedless][mss.core.Bleedless].
Measures the average energy of the parts of the [mel spectrogram][mss.core.DbDifferenceMel]
that are quieter than the reference.
A high value indicates that parts of the target loss were lost during the separation, indicating
that more of the original source's character is preserved.
$$
\text{Fullness} = \text{mean}(|\mathbf{D}(m, k)|) \quad \forall \quad \mathbf{D}(m, k) < 0
$$
"""
