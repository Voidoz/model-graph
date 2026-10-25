# Local GUI

The MVP GUI is a **local-only web app** served by `mg ui`. It inspects graphs, runs the mock
provider, shows traces/artifacts/usage, browses models, and replays runs. It is deliberately a
zero-build, zero-framework application: Python's standard library serves JSON and static files;
vanilla HTML/CSS/JS renders the UI.

- Goals: immediate local use, no Node build step, no CDN, no telemetry, no auth surface.
- Non-goals for the first GUI: graph editing, remote hosting, multi-user accounts, streaming.
- Graph editing and swap UX are `T-0046`, built after the revision workflow.

## Architecture

```text
mg ui --host 127.0.0.1 --port 8765
        │
        ▼
ThreadingHTTPServer (Python stdlib)
  ├── /api/*          JSON endpoints
  │     ├── compiler / registry
  │     ├── runtime engine
  │     └── runs/ directory
  └── /static/*       index.html, graph.html, run.html, models.html, JS/CSS
```

The server reads the repository's default registry and the local `runs/` directory. It does not
accept arbitrary filesystem paths from the browser. Bind to loopback by default; the first GUI
has no authentication and must not be exposed to a network.

## Command

```text
mg ui [--host 127.0.0.1] [--port 8765] [--open | --no-browser] [--runs-dir PATH]
```

`--open` launches the default browser; `--no-browser` suppresses it. `--port 0` is allowed for
tests and prints the chosen port. `--runs-dir` selects the run history directory (the default is
`runs/` under the current working directory, or `MG_RUNS_DIR`). The server refuses non-loopback
hosts. If the UI assets are missing, a browser receives a local HTML error page while API and
asset requests keep the JSON error envelope.

## API contract

All responses are JSON, UTF-8. Errors use:

```json
{"error": {"code": "MG-XXXX", "message": "human-readable message"}}
```

HTTP status codes: `200` success, `201` run created, `400` invalid request, `404` not found,
`500` internal error, `502` provider failure.

### `GET /api/health`

```json
{"ok": true, "version": "0.5.0", "ui": "1"}
```

### `GET /api/graphs`

Compiles the graphs in the default registry and returns summaries.

```json
{
  "graphs": [
    {
      "id": "hello-triage",
      "version": "1.0.0",
      "name": "Hello Triage",
      "description": "A minimal router...",
      "source": "/path/to/examples/hello-triage.graph.yaml",
      "hash": "sha256:...",
      "node_count": 3,
      "edge_count": 2,
      "warnings": ["[MG-1016] ..."]
    }
  ]
}
```

### `GET /api/graphs/{id}`

`id` may be `hello-triage` or `hello-triage@1.0.0`. Resolve through the default registry.

```json
{
  "graph": { "...": "CompiledGraph.to_dict()" },
  "name": "Hello Triage",
  "hash": "sha256:...",
  "warnings": ["[MG-1016] ..."]
}
```

`name` and `hash` are additive convenience fields for the UI. The `graph` object remains exactly
`CompiledGraph.to_dict()`, so the compiled contract is unchanged.

### `GET /api/models`

```json
{
  "models": [
    {
      "id": "router-small",
      "version": "1.0.0",
      "provider": "ollama",
      "model": "qwen2.5:1.5b-instruct",
      "license": "apache-2.0",
      "description": "A small local model..."
    }
  ]
}
```

### `GET /api/runs?limit=20`

Reads `run.json` files under the local runs directory, newest first.

```json
{
  "runs": [
    {
      "run_id": "run_ab12cd34",
      "outcome": "succeeded",
      "graph_id": "hello-triage",
      "graph_version": "1.0.0",
      "started_at": "2026-10-22T10:00:00Z",
      "tokens": 883,
      "cost_usd": 0.0,
      "dir": "/path/to/runs/run_ab12cd34"
    }
  ]
}
```

### `GET /api/runs/{run_id}`

Returns the full `run.json` manifest.

### `GET /api/runs/{run_id}/trace`

```json
{"events": [{"ts": "...", "event": "node.entered", "node": "triage"}]}
```

### `GET /api/runs/{run_id}/artifacts/{name}`

Looks up the latest artifact with that name in the run manifest and returns its content.

```json
{
  "artifact": {
    "name": "answer",
    "type": "code",
    "producer": "coding",
    "hash": "sha256:...",
    "content": "function debounce(...) { ... }"
  }
}
```

### `POST /api/runs`

Request:

```json
{"graph_id": "hello-triage", "task": "Write a debounce function in TypeScript.", "provider": "mock"}
```

- `provider`: `mock` (default, fast/offline) or `auto` (use the manifest's local/provider).
- The server compiles with `strict_models=True`, runs synchronously, persists the run, and
  returns `201`:

```json
{
  "run_id": "run_...",
  "outcome": "succeeded",
  "path": [{"node": "triage", "kind": "router", "decision": "route", "to": "coding"}],
  "usage": {"steps": 2, "input_tokens": 500, "output_tokens": 300, "total_tokens": 800, "cost_usd": 0.0, "elapsed_seconds": 3.2},
  "artifacts": [{"name": "answer", "type": "code", "content": "..."}],
  "run_dir": "/path/to/runs/run_..."
}
```

The response is the same shape as `mg run --json`, excluding the manifest path convenience.
Provider failures return `502` with the error envelope; validation failures return `400`.

### `POST /api/runs/{run_id}/replay`

Replays the recorded run. Returns:

```json
{"ok": true, "run_id": "run_...", "replay_id": "replay_...", "mismatches": []}
```

If the recorded run no longer matches (graph hash, routing path, or exit artifact hashes), return
`409` with the mismatch in the error message unless an `allow_ir_mismatch` field is explicitly
true. The GUI does not expose that escape hatch by default.

## Static files and ownership

```text
src/modelgraph/web/static/
  index.html        dashboard / graph catalog      (T-0042)
  graph.html        graph canvas + node details    (T-0042)
  styles.css        shared layout/theme             (T-0042)
  common.js         API helper, formatting, nav     (T-0042)
  graph.js          canvas/detail logic             (T-0042)
  run.html          run console                     (T-0043)
  run.js            run form/trace/artifact logic   (T-0043)
  run.css           run-specific styles             (T-0043)
  models.html       model catalog + run history     (T-0044)
  models.js         model/run-list logic            (T-0044)
  models.css        model/history styles            (T-0044)
```

Shared UI contract (`common.js`):

- `MG.api(path, options)` → fetch JSON; throws an error with `.code` and `.message` on failure.
- `MG.el(tag, attrs, ...children)` → create elements safely; never use `innerHTML` with API data.
- `MG.text(value)` / `MG.number(value)` — formatting helpers.
- `MG.nav(active)` — render the top navigation and highlight the active page.
- All pages include `/static/styles.css` plus their own CSS if needed.
- No external fonts, scripts, styles, or images. No `eval`, `new Function`, or inline event
  handler attributes.

## Page requirements

### Dashboard + graph canvas (T-0042)

- Dashboard lists graphs from `/api/graphs`, with node/edge counts, warnings, and links.
- Graph page `graph.html?id=<id>` fetches `/api/graphs/{id}` and renders an SVG:
  - nodes colored by kind (`router`, `worker`, `joiner`, `subgraph`);
  - directed edges, labels for `when` (shortened), entry/exit markers;
  - a simple layered layout from `entry` (no external graph library);
  - click a node to show description, model, reads/writes, instruction, warnings;
  - a "Run this graph" link to `run.html?graph=<id>`.
- Handle loading, empty, and error states.

### Run console (T-0043)

- `run.html` supports `?graph=<id>` preselection, a graph select, task textarea, provider
  select (`mock` default, `auto` local), and a Run button.
- Show a clear running state; disable duplicate submits.
- Render the path as a timeline with router decisions/confidence/reason.
- Render exit artifacts with a copy button and a raw trace table.
- Link to the run directory and a Replay button (calls `/api/runs/{id}/replay`).
- `run.html?id=<run_id>` loads a recorded run from `/api/runs/{run_id}` so the run-history links
  show its manifest, artifacts, trace, and replay controls without executing the graph again.
- Never inject artifact content with `innerHTML`.

### Models + history (T-0044)

- `models.html` lists `/api/models` as cards with provider/model/license and description.
- `GET /api/runs` history table with outcome, graph, tokens, cost, started time, and links.
- Filters by graph/outcome are optional; empty/error states required.

## Security and testing

- Bind `127.0.0.1` by default. Refuse non-loopback hosts unless an explicit `--allow-remote`
  flag is added (not part of the first GUI; do not add it silently).
- Static serving uses a fixed file map / safe path join; reject traversal.
- UI escapes all API-derived text; artifact/trace content is untrusted.
- No external network calls from the browser; no CDN.
- Responses set a same-origin Content-Security-Policy plus `nosniff`, `DENY` framing, and
  `no-referrer` headers. The frontend has no inline scripts or styles.
- Tests:
  - API tests with `create_server(host, port=0)` and stdlib HTTP clients;
  - `tests/web/test_gui_integration.py` covers pages, assets, a full mock run, replay, error
    envelopes, loopback binding, traversal, and the CLI start/stop smoke;
  - static asset presence and no-external-URL checks;
  - JS syntax check via `node --check` when Node is available (skip otherwise);
  - `make ui-check` runs `tests/web` and a headless HTTP smoke check (`tests/web/smoke_ui.py`);
  - `mg ui` CLI smoke test (server starts and shuts down, or `--port 0 --no-browser`).

## MVP acceptance

1. `mg ui` starts on loopback and serves the dashboard.
2. Dashboard lists the example graphs; the graph canvas renders nodes/edges and details.
3. A user can run a task with the mock provider from the Run page, see the routing path,
   artifact, usage, and trace, then replay the run.
4. Models and run history are visible.
5. No external asset requests, no build step, no new runtime dependency.
6. `make check` passes, including the web tests.
