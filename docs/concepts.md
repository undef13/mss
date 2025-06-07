
## Time domain

In the real world, audio is effectively a *continuous* time series of sound pressure levels $x(t)$, where $t$ is the time in seconds.

In digital systems, it is represented as a *discrete* sequence of samples taken at regular intervals $x[n]$, where $n$ is the samples. It is parameterised by the [sample rate][mss.core.SampleRate], [bit depth][mss.core.BitDepth] and the [number of channels][mss.core.Channels].

## Time-Frequency Domain


<!-- spectrogram, hop size, window function, fft size, bands, chunksize, padding, overlap. -->

## Core loop

Core loop includes:

- [padding][mss.core.Padding]
- [chunking][mss.core.ChunkSize] (`unfold`)
- [model inferencing](#inference) (`model(batch)`)
- windowing of processed chunks
- stitching (`fold`)
- [normalization/denormalization][mss.core.NormalizationStats]
