# mss

Lightweight utilities for music source separation.

## Installation

```sh
uv add git+https://github.com/undef13/mss.git
```

## Development

```sh
uv venv
uv sync --all-extras --all-groups
# serve documentation.
uv mkdocs serve
```

## Usage

Note that this repo is no longer compatible with zfturbo's repo. The last version that does so is [`v0.0.1`](https://github.com/undef13/mss/tree/v0.0.1). To pin the version in uv, change your `pyproject.toml`:

```toml
[tool.uv.sources]
mss = { git = "https://github.com/undef13/mss.git", rev = "287235e520f3bb927b58f9f53749fe3ccc248fac" }
```
