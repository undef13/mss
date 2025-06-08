
## Time domain

In the real world, audio is effectively a *continuous* time series of sound pressure levels $x(t)$, where $t$ is the time in seconds.

In digital systems, it is represented as a *discrete* sequence of samples taken at regular intervals $x[n]$, where $n$ is the samples. It is parameterised by the [sample rate][mss.core.SampleRate], [number of channels][mss.core.Channels], [bit depth][mss.core.BitDepth], [audio encoding][mss.core.AudioEncoding] and the [file format][mss.core.FileFormat].

## Time-Frequency Domain


<!-- spectrogram, hop size, window function, fft size, bands, chunksize, padding, overlap. -->

## Core loop

Core loop includes:

- [padding][mss.core.Padding]
- [chunking][mss.core.ChunkSize] (`unfold`)
- model inferencing (`model(batch)`)
- windowing of processed chunks
- stitching (`fold`)
- [normalization/denormalization][mss.core.NormalizationStats]
