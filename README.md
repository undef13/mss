# mss

Lightweight utilities for music source separation.

This library is a ground-up rewrite of the [zfturbo's MSST repo](https://github.com/ZFTurbo/Music-Source-Separation-Training), with a strong focus on robustness, simplicity and extensibility. While it is a fantastic collection of models and training scripts, this rewrite adopts a different architecture to address common pain points in research code.

Key principles:

- **Configuration as code**. We replace untyped dictionaries and `ConfigDict` with pydantic models. This provides static type safety, runtime data validation, IDE autocompletion, and a single, clear source of truth for all parameters.
- **Data-oriented and functional core**: We avoid complex class hierarchies and inheritance. The codebase is built on plain data structures (like `dataclasses`) and pure, stateless functions.
- **Semantic typing as documentation**: We leverage Python's type system to convey intent. Types like `RawAudioTensor` vs. `NormalizedAudioTensor` make function signatures self-documenting, reducing the need for verbose comments and ensuring correctness.
- **Extensibility without modification**: New models can be integrated from external packages without altering the core library. The dynamic model loading system allows easy plug-and-play adhering to the open/closed principle.

⚠️ This is pre-alpha software, expect significant breaking changes.

## Features and Roadmap

- [x] **Typed Configuration**: A robust configuration system powered by `pydantic`
- [x] **Core Inference Pipeline**:
    - [x] Normalization and denormalization
    - [x] Chunk generation: vectorized with `unfold`
    - [x] Chunk stitching: vectorized overlap-add with `fold`
    - [x] Flexible ruleset for stem deriving: add/subtract model outputs or any intermediate output (e.g., creating an `instrumental` track by subtracting `vocals` from the `mixture`).
- [x] **Web-based Documentation**: Generated with `mkdocs` with excellent crossrefs.
- [x] **Command-Line Interface**: A simple CLI for inferencing on a single audio file.
- [ ] Simple file-based cache
- [ ] **BS-Roformer Optimizations**:
  - [x] fp16 support
  - [ ] remove complex multiplication
  - [ ] support cormeltools and torch compilation
- [ ] **Evaluation Metrics**: Implement standardized evaluation metrics (SDR, bleedless, fullness, etc.).
- [ ] **Data Augmentation**: Introduce a data augmentation pipeline
- [ ] **Full Training Pipeline**: Implement a complete, configurable training loop.
- [ ] **Expanded Model Zoo**: Port additional models from the MSST (Demucs, SCNet, etc.).
- [ ] **Simple web-based GUI**: with FastAPI and Svelte.

PRs are very welcome!

## Installation & Usage

- [I just want to run it](#cli)
- [I want to add it as a library to my Python project](#library)
- [I want to hack around](#development)

Documentation on the config (amongst other details) can be found [here](https://undef13.github.io/mss/api/config/)

### CLI

There are three steps. You do not need to have Python installed.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already. It is an awesome Python package and library manager with `uv pip` comptability.
```sh
# Linux / MacOS
wget -qO- https://astral.sh/uv/install.sh | sh
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Open a new terminal and install the current project as a tool. It will install the Python interpreter and all necessary packages if you haven't already:
```sh
uv tool install "git+https://github.com/undef13/mss.git[config,inference,cli]"
```

3. Go into a new directory and place the [model checkpoint](https://github.com/undef13/mss/releases/download/v0.0.1/roformer-fp16.pt) and [configuration](https://raw.githubusercontent.com/undef13/mss/refs/heads/main/data/config/bs_roformer.json) inside it. Assuming your current directory has this structure (doesn't have to be exactly this):

<details>
   <summary>Grab an example audio from YouTube</summary>

```sh
uv tool install yt-dlp
yt-dlp -f bestaudio -o data/audio/input/3BFTio5296w.flac 3BFTio5296w
```
</details>

```
.
└── data
    ├── audio
    │   ├── input
    │   │   └── 3BFTio5296w.flac
    │   └── output
    ├── config
    │   └── bs_roformer.json
    └── models
        └── roformer-fp16.pt
```

Run:
```sh
mss separate data/audio/input/3BFTio5296w.flac --config data/config/bs_roformer.json --checkpoint data/models/roformer-fp16.pt
```
<details>
   <summary>Console output</summary>

```php
[00:00:41] INFO     using device=device(type='cuda')                                                 __main__.py:117
           INFO     loading configuration from                                                       __main__.py:119
                    config_path=PosixPath('data/config/bs_roformer.json')                                           
           INFO     loading model metadata `BSRoformer` from module `mss.models.bs_roformer`         __main__.py:122
[00:00:42] INFO     loading weights from checkpoint_path=PosixPath('data/models/roformer-fp16.pt')   __main__.py:131
           INFO     processing audio file:                                                           __main__.py:138
                    mixture_path=PosixPath('data/audio/input/3BFTio5296w.flac')                                     
[00:00:56] INFO     wrote stem `bass` to data/audio/output/3BFTio5296w/bass.flac                     __main__.py:168
           INFO     wrote stem `drums` to data/audio/output/3BFTio5296w/drums.flac                   __main__.py:168
           INFO     wrote stem `other` to data/audio/output/3BFTio5296w/other.flac                   __main__.py:168
[00:00:57] INFO     wrote stem `vocals` to data/audio/output/3BFTio5296w/vocals.flac                 __main__.py:168
           INFO     wrote stem `guitar` to data/audio/output/3BFTio5296w/guitar.flac                 __main__.py:168
           INFO     wrote stem `piano` to data/audio/output/3BFTio5296w/piano.flac                   __main__.py:168
[00:00:58] INFO     wrote stem `instrumental` to data/audio/output/3BFTio5296w/instrumental.flac     __main__.py:168
           INFO     wrote stem `drums_and_bass` to data/audio/output/3BFTio5296w/drums_and_bass.flac __main__.py:168
```
</details>

To update the tool:

```sh
uv tool upgrade mss --force-reinstall
```

### Library

Add the latest bleeding edge to your project:

```sh
uv add git+https://github.com/undef13/mss.git
```

This only installs absolutely minimal core dependencies for the `src/mss/models/` directory. It does not enable inference, training or CLI components. You must install the optional dependencies defined in `pyproject.toml`, for example:

```sh
# enable the built-in configuration, inference and CLI
uv add "git+https://github.com/undef13/mss.git[config,inference,cli]"
```

### Development

For a local dev build enabling all optional and developer dependencies:

```sh
git clone https://github.com/undef13/mss.git
cd mss
uv venv
uv sync --all-extras --all-groups
```

If you're using mss from another project, you may also want to use `--editable`.

```sh
# lint
uv run ruff check src tests
# format
uv run ruff format --check src tests
# build & host documentation
uv run mkdocs serve
# type check
uv run mypy src tests
```

This repo is no longer compatible with zfturbo's repo. The last version that does so is [`v0.0.1`](https://github.com/undef13/mss/tree/v0.0.1). To pin a specific version in `uv`, change your `pyproject.toml`:

```toml
[tool.uv.sources]
mss = { git = "https://github.com/undef13/mss.git", rev = "287235e520f3bb927b58f9f53749fe3ccc248fac" }
```
