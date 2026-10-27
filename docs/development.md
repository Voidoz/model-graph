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
make format      # write formatting and safe lint fixes
make check       # format check + ruff + mypy + pytest
make eval-gates  # offline mock eval suites + contribution gates
make audit       # network: pip-audit the locked runtime dependencies
```

`make check` is offline and deterministic; `make audit` is deliberately separate because it
downloads `pip-audit` and advisory data. It never runs as part of `check`. The failure policy
and waiver rule are in [security.md](security.md#3-failure-policy-for-dependency-audit).
Security tests and the run-directory scanner:

```text
uv run pytest tests/security -q
uv run python scripts/scan_secrets.py runs/
```

`make eval-gates` regenerates the node, routing, and e2e mock candidate reports under
`runs/evals/ci/`, compares each against the committed baselines in `evals/baselines/`, and
exits non-zero on a tolerance or hard-gate failure. It needs no model server. Baseline moves
are a separate reviewed commit:

```text
make baselines   # regenerate evals/baselines/{node,routing,e2e,mock}.json
```

The gate policy and waiver rules are in `evals/baselines/policy.json` and
[`evals/baselines/README.md`](../evals/baselines/README.md). Live suites are gated and run
only on a machine with a model server:

```text
make nightly
# equivalent to MG_EVAL_LIVE=1 mg eval node|routing|e2e ... --live --fail-on-gate
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
  evaluation/      # node/routing/e2e suites, metrics, reports, gates
  pruning.py       # run profiles, capability pruning, lazy model loading
  providers/       # mock, Ollama, OpenAI-compatible adapters
  runtime/         # envelope, renderer, executors, engine, trace, replay
  web/             # local stdlib server, JSON API, static frontend
evals/             # mock suites, e2e fixtures, committed gate baselines
examples/profiles/ # example run profiles (4GB and 8GB)
tests/             # offline unit and integration tests
  web/             # GUI API, page, and integration tests
```

## Service checks

The CLI and local GUI are the stable interfaces for now:

```text
uv run mg --help
uv run mg validate <graph.yaml>
uv run mg run <graph.yaml> --provider mock
uv run mg run <graph.yaml> --provider mock --profile local-4gb
uv run mg replay <run-id>
uv run mg ui --port 0 --no-browser
```

Evaluation and gate commands (all offline with `--provider mock`):

```text
uv run mg eval node --suite evals/mock/js-specialist --provider mock
uv run mg eval routing --suite evals/mock/triage --provider mock
uv run mg eval e2e --suite evals/e2e/composed-assistant \
  --graph examples/composed-assistant.graph.yaml --provider mock --profile local-8gb
uv run mg eval compare --baseline evals/baselines/e2e.json \
  --candidate runs/evals/ci/e2e.json
```

`mg eval compare` exits 0 on pass, 1 on a gate failure, and 3 on an input error (missing
baseline/policy, invalid report, or a suite-hash mismatch). Reports are written under
`runs/evals/<report_id>/` as `report.json`, `report.md`, `manifest.json`, per-case files, and
(for e2e) the runtime `runs/` manifests. See [evaluation.md](evaluation.md) and
[`evals/README.md`](../evals/README.md) for the suite and report contracts, and
[profiles.md](profiles.md) for run profiles and pruning.

See the repository issues and the README status for what exists today.

## CI

`.github/workflows/ci.yml` runs `uv sync --locked`, `make check`, and the offline
`make eval-gates` comparison on Python 3.12 for pushes to `main` and pull requests. The gate
job writes the comparison Markdown to the job summary and uploads `runs/evals/ci/` as an
artifact. The separate `dependency-audit` job runs `make audit` (network) and is a merge
blocker; see [security.md](security.md#3-failure-policy-for-dependency-audit). A real-model
job is scheduled/manual only and is skipped unless the repository variable `MG_LIVE_RUNNER`
names a self-hosted runner; it runs `make nightly` and uploads `runs/evals/nightly/`. Default
tests do not contact a model server. A change that only passes locally is not done.
