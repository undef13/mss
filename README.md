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
- [ ] **True fp16 support for BSRoformer**: Partially done.
- [ ] **Evaluation Metrics**: Implement standardized evaluation metrics (SDR, bleedless, fullness, etc.).
- [ ] **Data Augmentation**: Introduce a data augmentation pipeline
- [ ] **Full Training Pipeline**: Implement a complete, configurable training loop.
- [ ] **Expanded Model Zoo**: Port additional models from the MSST (Demucs, SCNet, etc.).
- [ ] **Simple web-based GUI**: with FastAPI and Svelte.

PRs are very welcome!

## Installation

Add the latest bleeding edge to your project:

```sh
uv add git+https://github.com/undef13/mss.git
```

This only installs absolutely minimal core dependencies for the `src/mss/models/` directory. It does not enable inference, training or CLI components. You must install the optional dependencies defined in `pyproject.toml`, for example:

```sh
# enable the built-in configuration, inference and CLI
uv add "git+https://github.com/undef13/mss.git[config,inference,cli]"
```

For a local dev build enabling all optional dependencies and dev groups:

```sh
git clone https://github.com/undef13/mss.git
cd mss
uv venv
uv sync --all-extras --all-groups
```

You may want to install with `--editable`.

## Development

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

## Usage

First, [setup a local dev build](#installation) and cd into the root directory. v0.0.2 is compatible with v0.0.1 weights:

```sh
# download v0.0.1 model.
wget -P data/models/roformer.pt https://github.com/undef13/mss/releases/download/v0.0.1/roformer-fp16.pt
# download example audio.
uv tool install yt-dlp
yt-dlp -f bestaudio -o data/audio/input/3BFTio5296w.flac 3BFTio5296w
# process it.
uv run python3 -m src.mss data/audio/input/3BFTio5296w.flac --config data/config/bs_roformer.json --checkpoint data/models/roformer.pt
```

This repo is no longer compatible with zfturbo's repo. The last version that does so is [`v0.0.1`](https://github.com/undef13/mss/tree/v0.0.1). To pin the version in uv, change your `pyproject.toml`:

```toml
[tool.uv.sources]
mss = { git = "https://github.com/undef13/mss.git", rev = "287235e520f3bb927b58f9f53749fe3ccc248fac" }
```
