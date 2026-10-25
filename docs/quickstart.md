# Quickstart (MVP)

This is the first runnable slice of Model Graph. It proves the vertical path:

```text
graph YAML -> schema + semantic validation -> compiled IR -> routed execution
           -> scoped prompts -> provider -> artifacts -> trace + run manifest -> replay
```

**New here?** Follow [`tutorial.md`](tutorial.md) first. It is the tested, copy-paste path
through install, a mock run, a live local run, tracing, pruning, and a node evaluation. This
page is the fuller command reference. The starter library lives in [`graphs/`](../graphs/),
profiles in [`profiles/`](../profiles/), and model manifests in
[`registry/models/`](../registry/models/).

## 1. Install

```text
uv sync
uv run mg version
```

Python 3.12 is pinned in `.python-version`. The test suite runs offline.

## 2. Validate a graph

```text
uv run mg validate graphs/*.graph.yaml
uv run mg validate graphs/hello-triage.graph.yaml
uv run mg validate graphs/composed-assistant.graph.yaml
```

You should see `valid` plus an `MG-1016` warning (rule V-15) on the hello graph: its two
workers both write the `answer` exit artifact with different types (`code` and `text`). That
warning is intentional;
the graph is still valid.

A broken graph exits 1 and reports every error it can find. Machine-readable output:

```text
uv run mg validate graphs/hello-triage.graph.yaml --json
```

## 3. Run offline with the mock provider

```text
uv run mg run graphs/hello-triage.graph.yaml --provider mock
```

The mock provider is deterministic. The default mock route selects the first menu entry, which
is the coding branch for this example. The output includes:

- the routing path;
- the final artifact;
- token usage and cost;
- the run directory.

The run directory contains:

```text
runs/<run-id>/
  run.json        # graph hash, models, request, budget, usage, path, artifacts
  trace.jsonl     # append-only events: render, provider, decision, artifact, usage
  envelope.json   # request-scoped state, validated against the envelope schema
  artifacts/      # content-addressed artifact blobs
```

## 4. Inspect the run

```text
uv run mg runs list
uv run mg runs show runs/<run-id>
uv run mg trace show runs/<run-id>
uv run mg trace show runs/<run-id> --json
```

## 5. Replay it

```text
uv run mg replay runs/<run-id>
```

Replay serves the recorded provider responses in order. It does not call a model or the
network. It compares the routing path and exit artifact hashes with the original run and writes
`replay.json` into the run directory.

If the graph changed since the run, replay refuses with a graph-hash mismatch. Use
`--allow-ir-mismatch` only for deliberate experiments.

## 6. Run against a local model

The example manifests target Ollama. Install the model and run:

```text
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5-coder:7b
uv run mg run graphs/hello-triage.graph.yaml --registry registry/models
```

`--provider auto` (the default) uses the provider declared in each model manifest. You can
force a local adapter with `--provider ollama`. The endpoint comes from the manifest; it is not
embedded in the graph.

If Ollama is not running, the run fails with a typed `MG-3001` connection error and still writes
a trace/manifest for inspection.

The hello/composed/coding-specialist examples have been verified live on CPU and on an
RTX 3080 through Ollama Vulkan. See `.scratch/runs/2026-10-22-live-ollama.md` and
`.scratch/runs/2026-10-22-live-gpu.md`; timings are hardware-specific.

If `ollama ps` reports `CPU` after installing a GPU backend, restart the service and verify the
backend:

```text
systemctl restart ollama
journalctl -u ollama -n 30 --no-pager | grep -i 'inference compute'
ollama ps
```

A Vulkan service reports `library=Vulkan`. Installing `ollama-vulkan` without restarting the
service can leave the old CUDA runner (and its watchdog timeouts) in place.

## 7. Nested graph

```text
uv run mg run graphs/composed-assistant.graph.yaml --provider mock
uv run mg run graphs/coding-specialist.graph.yaml --provider mock
uv run mg run graphs/writing-studio.graph.yaml --provider mock
```

This routes `domain-triage -> coding subgraph -> coding-triage -> js-specialist`. The inner
graph's artifact is mapped to the outer `answer`; unmapped inner artifacts stay in the trace.

## 8. Local GUI

```text
uv run mg ui
```

Open `http://127.0.0.1:8765/`. The GUI is served by Python's standard library, binds loopback
only, and makes no external requests. Use `--port 0` to pick a free port (the chosen URL is
printed), `--no-browser` to suppress the browser, or `--runs-dir PATH` to point run history at a
different directory.

The pages are:

- **Dashboard** (`index.html`): compiled graphs with node/edge counts, warnings, and links.
- **Graph** (`graph.html?id=<graph-id>`): a local SVG canvas, node/edge details, warnings, and a
  link to run the graph.
- **Run console** (`run.html`): run a task with the deterministic `mock` provider or `auto` (each
  node's configured local provider), then inspect the routing path, artifacts, usage, trace, and
  replay. `run.html?id=<run-id>` opens a recorded run from history.
- **Models & runs** (`models.html`): the local model registry and recorded run history, newest
  first.

The mock provider is the quickest way to see the whole flow: it is offline, deterministic, and
does not need Ollama. `auto` is the same provider choice `mg run` uses and needs the configured
model server. See [gui.md](gui.md) for the JSON API and security contract.

## 9. Evaluate and prune

```text
uv run mg eval node --suite evals/mock/js-specialist \
  --graph graphs/coding-specialist.graph.yaml --provider mock
uv run mg eval routing --suite evals/mock/triage --provider mock
uv run mg eval e2e --suite evals/e2e/composed-assistant \
  --graph graphs/composed-assistant.graph.yaml --provider mock --profile local-8gb
make eval-gates
```

`mg eval` writes JSON, Markdown, per-case files, and an eval manifest under `runs/evals/`.
`make eval-gates` regenerates the offline mock candidate reports, compares them with
`evals/baselines/`, and exits non-zero on a regression or hard-gate failure.

Profiles prune branches the deployment cannot serve:

```text
uv run mg run graphs/composed-assistant.graph.yaml --provider mock \
  --profile profiles/local-4gb.yaml
uv run mg run graphs/hello-triage.graph.yaml --provider mock \
  --profile profiles/no-network.yaml   # audit: fails MG-1005, no in-process provider in v0
```

The 4GB profile removes the 6GB story and generic-coder branches, records the `PruneReport`
in the run manifest, and leaves the authored YAML unchanged. The 8GB profile removes nothing
from the starter library. `no-network` prunes every starter node because local Ollama is
loopback network access; see [`tutorial.md`](tutorial.md#7-prune-with-a-profile). The mock run
works offline; without the profile, `auto`/`ollama` uses the configured local models.

## What the MVP includes

- YAML/JSON loading with duplicate-key detection.
- Canonical JSON Schema validation and semantic rules V-01..V-20.
- Model manifests resolved through a local registry, with manifest hashes in the run.
- Router, worker, joiner, and subgraph execution.
- Structured route decisions with confidence, fallback, and one repair attempt.
- Single- and multi-artifact output protocols.
- Hard budgets: steps, depth, tokens, cost, wall time.
- Scoped views (a node sees only its declared reads).
- Deterministic mock provider and OpenAI-compatible/Ollama HTTP adapters.
- Content-addressed artifacts, trace, envelope, run manifest, and replay.
- CLI commands: `validate`, `compile`, `run`, `replay`, `trace show`, `runs`, `models`, `ui`,
  `eval`, `data export`, `status`, `graph diff`, `swap`, `change`, `promote`, and `rollback`.
- Node/routing/e2e evaluation, safety gates, baselines, and offline contribution checks.
- Change sets, impact analysis, immutable revisions, atomic promotion, and rollback.
- Consented, redacted trace-to-dataset export with frozen grouped splits.
- A local GUI for graph inspection, mock runs, traces, artifacts, models, history, and replay.
- A tested starter library and tutorial.

## What is not in the MVP yet

- Route flattening, AOT prefix compilation, cross-request batching.
- Canary/shadow, in-flight pinning, and hot swap (`T-0039`).
- Resident/fused/compiled targets and graph distillation.
- Adapter submission and specialist/router training recipes.
- Bounded fan-out/ensemble joins.
- Provider streaming and tools/code execution.
- Graph editing and model-swap UX (`T-0046`).

The next milestones are the human live tutorial pass and the training flywheel. The performance
track is Phase 5 and deliberately does not block the usable local workflow.

## Test it

```text
make check
uv run pytest tests/runtime -q
uv run pytest tests/cli -q
```
