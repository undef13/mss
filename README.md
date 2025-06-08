# mss

Lightweight utilities for music source separation.

⚠️ This is pre-alpha software, expect significant breaking changes.

## Installation

Add the latest bleeding edge to your project:

```sh
uv add git+https://github.com/undef13/mss.git
```

This only installs absolutely minimal core dependencies. It does not enable inference or training components. You must install the optional dependencies as specified in `pyproject.toml`, for example:

```sh
# enable the built-in configuration module for easy inference
uv add "git+https://github.com/undef13/mss.git[config]"
```

## Development

```sh
git clone https://github.com/undef13/mss.git
cd mss
uv venv
uv sync --all-extras --all-groups
# serve documentation.
uv mkdocs serve
```

## Usage

This repo is no longer compatible with zfturbo's repo. The last version that does so is [`v0.0.1`](https://github.com/undef13/mss/tree/v0.0.1). To pin the version in uv, change your `pyproject.toml`:

```toml
[tool.uv.sources]
mss = { git = "https://github.com/undef13/mss.git", rev = "287235e520f3bb927b58f9f53749fe3ccc248fac" }
```
