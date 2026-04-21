set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    @just --list

sync:
    uv sync --group dev

fmt:
    uvx ruff format .

fmt-check:
    uvx ruff format --check .

lint:
    uvx ruff check .

typecheck:
    uv run --group dev pyrefly check

test *ARGS:
    uv run --group dev pytest {{ARGS}}

coverage:
    uv run --group dev coverage run -m pytest
    uv run --group dev coverage report

precommit:
    uv run --group dev pre-commit run --all-files --show-diff-on-failure

docs:
    uv run --group dev sphinx-build -W --keep-going -b html docs docs/_build

docs-serve:
    uv run --group dev sphinx-autobuild --open-browser docs docs/_build
