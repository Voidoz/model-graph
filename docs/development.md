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

## GUI development

The GUI is vanilla HTML/CSS/JS served by `modelgraph.web`; there is no build step and no Node
dependency at runtime. From the repository root:

```text
make ui-check              # tests/web plus a headless HTTP smoke check
uv run pytest tests/web -q # just the GUI suite
uv run python tests/web/smoke_ui.py
uv run mg ui --port 0 --no-browser
```

`make ui-check` binds `create_server` on an ephemeral loopback port, requests the dashboard and
core API endpoints, and shuts down. The same path runs in `tests/web/test_gui_integration.py`
alongside a mock run, manifest/trace/artifact/replay checks, error envelopes, traversal
rejection, static asset checks, and a real `mg ui` start/SIGINT/stop smoke.

When Node is installed, the suite also runs `node --check` on every page script; without Node
those checks skip and the Python tests still run. For a manual check, start `mg ui`, open the
dashboard, inspect a graph, run the example task with `mock`, open the trace/artifact, replay it,
then open Models & runs and follow a run link back to its recorded run. Browser devtools should
show no external requests and no JavaScript console errors.

Static assets live in `src/modelgraph/web/static/` and are served from `/static/`. Keep them
local: no CDNs, external fonts/images, or inline scripts/styles. The server sends a same-origin
Content-Security-Policy, so a new external reference will fail in the browser as well as in the
tests. API-derived text must be inserted with text nodes (`MG.el`/`textContent`), never
`innerHTML`. See [gui.md](gui.md) for the page and API contract.

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
  web/             # local stdlib server, JSON API, static frontend
tests/             # offline unit and integration tests
  web/             # GUI API, page, and integration tests
```

## Service checks

The CLI and local GUI are the stable interfaces for now:

```text
uv run mg --help
uv run mg validate <graph.yaml>
uv run mg run <graph.yaml> --provider mock
uv run mg replay <run-id>
uv run mg ui --port 0 --no-browser
```

See the ticket board in `.scratch/tickets/BOARD.md` for what exists today.

## CI

`.github/workflows/ci.yml` runs `uv sync --locked` and `make check` on Python 3.12 for pushes to
`main` and pull requests. A change that only passes locally is not done.
