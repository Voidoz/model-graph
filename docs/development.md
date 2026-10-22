# Development

## Requirements

- Python **3.12** (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/)
- Linux or macOS. Windows is via WSL2 for now (`Q-012`).

## Setup

```text
uv sync
uv run mg version
```

`uv sync` creates `.venv/` from `uv.lock`. Do not commit the virtual environment.

## Checks

```text
make format   # write formatting and safe lint fixes
make check    # format check + ruff + mypy + pytest
```

Run a single test:

```text
uv run pytest tests/test_smoke.py -q
uv run pytest tests/test_smoke.py::test_version_command -v
```

The test suite must run **offline**. Live Ollama tests are opt-in:

```text
make live-test
# equivalent to: MG_TEST_OLLAMA=1 uv run pytest tests/live -q
```

They skip by default and require a running Ollama server with the example models pulled.

## Adding dependencies

Add runtime dependencies to `[project].dependencies` in `pyproject.toml`, dev-only tooling to
the `dev` extra, then run `uv lock` and `uv sync`. A new dependency needs a reason in the
ticket; avoid adding a framework for something the standard library does adequately.

## Package layout

```text
src/modelgraph/
  __init__.py
  cli.py           # Typer commands
  contracts/       # schema loading, document models, duplicate-key YAML
  compiler/        # graph validation and canonical IR
  providers/       # mock, Ollama, OpenAI-compatible adapters
  runtime/         # envelope, renderer, executors, engine, trace, replay
tests/             # offline unit and integration tests
```

## Service checks

The CLI is the stable interface for now:

```text
uv run mg --help
uv run mg validate <graph.yaml>
uv run mg run <graph.yaml> --provider mock
uv run mg replay <run-id>
```

See the ticket board in `.scratch/tickets/BOARD.md` for what exists today.

## CI

`.github/workflows/ci.yml` runs `uv sync --locked` and `make check` on Python 3.12 for pushes to
`main` and pull requests. A change that only passes locally is not done.
