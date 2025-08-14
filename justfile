check:
    uv run ruff check src tests
    uv run ruff format --check src tests
    # uv run --with pydantic mypy src tests

fmt:
    uv run ruff check src tests --fix
    uv run ruff format src tests