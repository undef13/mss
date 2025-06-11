## Introduction

In the digital world, sound is captured as a discrete sequence of samples, a representation of the original continuous audio signal $x(t)$. We refer to this time-domain data as a [`RawAudioTensor`][splifft.core.RawAudioTensor]. This digital signal is defined by several key parameters: its [sample rate][splifft.core.SampleRate], number of [channels][splifft.core.Channels], [bit depth][splifft.core.BitDepth], [audio encoding][splifft.core.AudioEncoding], and [file format][splifft.core.FileFormat].

## Separation pipeline

### Normalization

Neural networks perform best when their input data has a consistent statistical distribution. To prepare the audio for the model, we first [normalize it][splifft.core.normalize_audio]. This process transforms the [`RawAudioTensor`][splifft.core.RawAudioTensor] into a [`NormalizedAudioTensor`][splifft.core.NormalizedAudioTensor] with a mean of 0 and a standard deviation of 1. The original statistics ($\mu$ and $\sigma$) are stored in [`NormalizationStats`][splifft.core.NormalizationStats] to be used later for denormalization.

### Time-Frequency Transformation

The model operates not on raw samples, but in the time-frequency domain, which reveals the signal's frequency content over time. This is achieved via the Short-Time Fourier Transform (STFT), which converts the 1D audio signal into a 2D [`ComplexSpectrogram`][splifft.core.ComplexSpectrogram].

The choice of [`FftSize`][splifft.core.FftSize] presents a fundamental [trade-off](https://en.wikipedia.org/wiki/Uncertainty_principle#Signal_processing): a larger FFT size provides better frequency resolution, ideal for separating tonal instruments, while a smaller size offers better time resolution, crucial for capturing sharp, percussive sounds (transients). A [`WindowShape`][splifft.core.WindowShape] is applied to each chunk before the FFT to minimize spectral leakage. The model's core task is to take the mixture's spectrogram and learn a complex-valued mask for each desired source, such that $\hat{S}_\text{source} = M_\text{complex} \odot S_\text{mixture}$.

### Chunking and Inference

Since a full audio track is too large for GPU memory, we process it in overlapping segments. The [`ChunkSize`][splifft.core.ChunkSize] defines the segment length, while the [`HopSize`][splifft.core.HopSize] dictates the step between them, controlled by the [`OverlapRatio`][splifft.core.OverlapRatio]. This process yields a stream of [`PaddedChunkedAudioTensor`][splifft.core.PaddedChunkedAudioTensor] batches, which are fed into the model. The model then outputs a corresponding stream of [`SeparatedChunkedTensor`][splifft.core.SeparatedChunkedTensor].

### Stitching and post-processing

After the model processes each chunk, we must reconstruct the full-length audio. The [`stitch_chunks`][splifft.core.stitch_chunks] function does this using the overlap-add method, applying a [`WindowTensor`][splifft.core.WindowTensor] to each chunk to ensure a seamless, artifact-free reconstruction. The final result is a [`RawSeparatedTensor`][splifft.core.RawSeparatedTensor] for each separated stem.

With the separated audio back in the time domain, the final steps are to reverse the normalization using the original [`NormalizationStats`][splifft.core.NormalizationStats] and, optionally, to create new stems (e.g., an "instrumental" track) using rules defined in [`DerivedStemsConfig`][splifft.config.DerivedStemsConfig].