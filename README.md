# Model Graph

**Compose small, independently trainable language models into a routed, observable graph.**

Model Graph breaks an AI system into discrete nodes: routers that decide where a request should
go, workers that answer, joiners that combine results, and subgraphs that compose teams. The edges
carry natural language, not tensors. A graph can use local Ollama models, an OpenAI-compatible
endpoint, or the deterministic mock provider.

This is a local-first Python project for people who want to inspect, evaluate, and change a model
workflow instead of treating it as one opaque endpoint.

## Status

**v1.0.0 — usable local release.** The portable runtime, evaluation, revisions, dataset export,
registry, training recipes, security controls, observability, static inspector, and local GUI are
implemented and tested. Live local execution has been validated with Ollama on CPU and on an
RTX 3080 via Vulkan, including a composed graph with 4GB pruning.

Not in this release: resident/fused/compiled GPU backends, distilled deployment students, canary
or hot-swap serving, and a full visual graph builder. The repository documents those boundaries
rather than faking them. Real LoRA training and real-hardware 8GB/24GB claims remain unverified
in this environment; the specialist kit explicitly marks synthetic artifacts as non-promotable.

## Why

Scaling one model is not the only way to improve a system. A router can send a narrow request to a
narrow model, a specialist can be trained by the person who owns its data, and a deployment can
drop branches it cannot run. The trade-offs are real:

- routing and handoff can lose quality;
- a graph can cost more than a single model;
- a specialist can regress off-distribution;
- local hardware, licenses, and privacy still matter.

The project treats those as measurable properties, not marketing claims. Every run produces a
trace and a manifest; graph changes can produce evaluation evidence; deployments are immutable
revisions.

## Features

- **Graph contract:** YAML/JSON, JSON Schema validation, canonical IR hashing, topology,
  prompt, binding, policy, and deployment-target diffs.
- **Runtime:** router, worker, joiner, and subgraph execution; bounded fan-out/ensemble joins;
  hard step/token/cost/wall budgets; deterministic traces; replay.
- **Providers:** deterministic mock, Ollama, OpenAI-compatible HTTP, and a local trained-router
  provider.
- **Profiles:** 4GB, 8GB, and no-network starter profiles; capability pruning with a recorded
  `PruneReport`.
- **Evaluation:** node, routing, e2e, and safety suites; oracle/confusion/calibration metrics;
  bootstrap confidence intervals; committed baselines; contribution gates.
- **Revisions:** change sets, impact analysis, immutable deployment revisions, atomic promotion,
  rollback, and retention policy.
- **Training:** trace-to-dataset export with consent/redaction/dedupe; trained-router recipe with
  calibration; a synthetic specialist/LoRA/distillation kit; a versioned local model registry.
- **Inspection:** self-contained offline HTML inspector with paths, pruned branches, artifacts,
  impact/stale/revision panels, and side-by-side run comparison.
- **Operations:** exact-match response cache, structured JSONL/console logs, trace-derived metrics,
  `mg stats`, optional OpenTelemetry spans, and a dependency audit.
- **GUI:** loopback-only zero-build web app for graph inspection, mock/live runs, traces,
  artifacts, model pull, YAML graph editing, candidate validation, and run history.

## Requirements

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/)
- Optional live models: [Ollama](https://ollama.com/) and any starter tags you want
- Optional GUI: a modern browser; no Node build step

## Install

```bash
git clone https://github.com/Voidoz/model-graph.git
cd model-graph
uv sync
uv run mg version
make check
```

`make check` runs formatting, lint, mypy, and the offline test suite. It does not need a model,
network, API key, or GPU.

## Quickstart

Validate and run the smallest graph with the deterministic mock provider:

```bash
uv run mg validate graphs/hello-triage.graph.yaml
uv run mg run graphs/hello-triage.graph.yaml --provider mock \
  --task "Write a debounce function in TypeScript."
```

Inspect the artifacts, trace, and manifest, then replay the run:

```bash
uv run mg runs list
uv run mg replay runs/<run-id>
uv run mg trace show runs/<run-id>
```

Try the starter library:

```bash
uv run mg run graphs/coding-specialist.graph.yaml --provider mock
uv run mg run graphs/composed-assistant.graph.yaml --provider mock
uv run mg run graphs/writing-studio.graph.yaml --provider mock
```

The guided path, including a live Ollama run, pruning, and a node evaluation, is
[`docs/tutorial.md`](docs/tutorial.md). The command reference is
[`docs/quickstart.md`](docs/quickstart.md).

## Local models

The starter manifests map graph model ids to Ollama tags:

| Manifest | Ollama tag | Declared need |
| --- | --- | --- |
| `router-small` | `qwen2.5:1.5b-instruct` | 2GB VRAM |
| `js-specialist-3b` | `qwen2.5-coder:3b` | 4GB VRAM |
| `coder-7b` | `qwen2.5-coder:7b` | 6GB VRAM |
| `story-7b` | `mistral:7b-instruct` | 6GB VRAM |
| `writer-7b` | `qwen2.5:7b-instruct` | 6GB VRAM |

Pull a tag with Ollama:

```bash
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5-coder:7b
uv run mg run graphs/hello-triage.graph.yaml --registry registry/models
```

You can also pull models from the local GUI: open **Models & runs**, enter a tag, and press
**Pull model**. The server invokes `ollama pull` directly (no shell), shows the job log, and
refreshes the installed-tag list when it completes. Pulls are explicit local actions; nothing is
uploaded by the GUI.

Pruning is how a smaller machine stays inside its budget:

```bash
uv run mg run graphs/composed-assistant.graph.yaml \
  --profile profiles/local-4gb.yaml --registry registry/models
```

The 4GB profile removes the 6GB story/generic-coder branches and records the removals. A pruned
fallback is reported; router abstention then fails safely.

## Local GUI

```bash
uv run mg ui
# Model Graph UI: http://127.0.0.1:8765/
```

The GUI is served by Python's standard library. It binds to loopback only, has no CDN or telemetry,
and has no authentication; do not expose it to a network.

Pages:

- **Dashboard** — compiled graphs, node/edge counts, warnings.
- **Graph** — SVG topology, node details, run link, edit link.
- **Run console** — mock/live run, routing timeline, artifacts, usage, trace, replay.
- **Models & runs** — registry catalog, installed Ollama tags, model pull, run history.
- **Graph editor** — edit YAML, validate through the real compiler, preview compiled nodes/edges,
  save a candidate under `runs/gui-candidates/`, and smoke-test it with the mock provider.

Saving a candidate does **not** change the authored graph, the live deployment, or a revision.
Candidates must go through the normal evaluation and revision workflow in the CLI. Promotion, rollback, and canary are intentionally CLI-only; the editor never mutates the live deployment.

Use `--port 0` for a free port, `--no-browser` to suppress the browser, and `--runs-dir PATH` to
point history at another directory. See [`docs/gui.md`](docs/gui.md) for the API, page contract,
and security model.

## Evaluation

```bash
uv run mg eval node --suite evals/mock/js-specialist \
  --graph graphs/coding-specialist.graph.yaml --provider mock
uv run mg eval routing --suite evals/mock/triage --provider mock
uv run mg eval e2e --suite evals/e2e/composed-assistant \
  --graph graphs/composed-assistant.graph.yaml --provider mock --profile local-8gb
uv run mg eval compare --baseline evals/baselines/e2e.json \
  --candidate runs/evals/ci/e2e.json
make eval-gates
```

`mg eval compare` reports quality, validity, cost, latency, slices, confidence intervals, and an
optional candidate/baseline/oracle reference when the reports carry e2e baselines. A failed gate
is a release blocker; a waiver is visible and requires a reason. `make eval-gates` runs the four
offline gates used by CI.

## Revisions and iteration

A model swap is a candidate revision, not a live mutation:

```bash
uv run mg status --revisions
uv run mg swap node js-specialist --to js-specialist-3b --dry-run
uv run mg change explain <change-id>
uv run mg promote --revision <revision> --dry-run
uv run mg rollback --to <previous-revision>
```

Iteration without loading a model:

```bash
uv run mg watch graphs/hello-triage.graph.yaml --once
uv run mg graph diff graphs/hello-triage.graph.yaml /tmp/candidate.graph.yaml
uv run mg inspect --html runs/<run-id> --graph graphs/hello-triage.graph.yaml \
  --out /tmp/run.html
```

`mg inspect --change` adds the impact/stale/revision timeline; `--compare RUN_B` adds a
side-by-side run view. The inspector is read-only and offline. See
[`docs/iteration.md`](docs/iteration.md) and [`docs/inspector.md`](docs/inspector.md).

## Training and registry

Export accepted traces into a consented, redacted dataset:

```bash
uv run mg data export --run runs/<run-id> --out /tmp/dataset \
  --consent "operator approved local training"
```

Train a calibrated router on a dataset:

```bash
uv run mg train dataset --dataset /tmp/dataset --out runs/training/router-dataset
uv run mg train router --config training/configs/router-triage.yaml
```

Build the synthetic specialist/LoRA/distillation rehearsal and its manifest, card, and eval
report:

```bash
uv run mg train specialist --config training/configs/specialist-synthetic.yaml --eval
```

The 8GB/24GB recipe configs are under `training/configs/`. The real LoRA path requires PyTorch,
PEFT, and a supported GPU; this environment does not have that stack, so real-hardware runs are
explicitly unverified and synthetic artifacts are never promotable.

The local registry stores versioned manifests, model cards, and evidence:

```bash
uv run mg registry add /path/to/contribution
uv run mg registry list
uv run mg registry inspect js-specialist-3b@0.3.0
uv run mg registry verify
```

Registry operations work offline. Weights are referenced by URI/hash, never downloaded. A license
field is a declaration, not legal advice.

## Operations

```bash
uv run mg stats --last 5
MG_LOG_FORMAT=json MG_LOG_FILE=runs/run.jsonl uv run mg run graphs/hello-triage.graph.yaml
make audit
```

The exact-match response cache is on by default for `mg run`; `--no-cache` bypasses it and
`--cache-mode readonly` serves existing entries without writing. Cache hits consume zero tokens
and report avoided usage. Structured logs and optional OpenTelemetry spans carry identifiers and
timings, not prompts or artifacts.

`make audit` is a separate network-dependent dependency audit. It is not part of the offline
`make check` path.

## Architecture

```text
graph YAML -> schema + semantic validation -> compiled IR
           -> scoped envelopes and rendered prompts
           -> router / worker / joiner / subgraph execution
           -> providers (mock, Ollama, OpenAI-compatible)
           -> artifacts + trace + run manifest + replay
```

The contracts are in `schemas/`; the runtime and compiler are under `src/modelgraph/`; the starter
graphs, profiles, model manifests, examples, evaluations, and training recipes are top-level
directories. See [`docs/architecture.md`](docs/architecture.md).

## Security and limits

- The GUI is local-only and unauthenticated; keep it on loopback.
- Untrusted content is delimited, protocol markers are neutralized, and persisted traces,
  envelopes, manifests, and errors are redacted. Delimiters are not injection immunity.
- Artifact blobs and evaluation reports are not rewritten by runtime redaction.
- There is no sandbox for tools/code execution in this release and no signing for manifests or
  weights.
- Redaction can break replay. Cache entries are local, unencrypted sensitive data.
- Resident/fused/compiled/canary deployment is not implemented. Do not read the performance
  design docs as measured release claims.

See [`docs/security.md`](docs/security.md) and [`SECURITY.md`](SECURITY.md).

## Development

```bash
make format
make lint
make type
make test
make check
make eval-gates
make ui-check
```

`make check` is offline and deterministic. A live model test is opt-in. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/development.md`](docs/development.md).

## Documentation

- [`docs/tutorial.md`](docs/tutorial.md) — guided install → run → prune → evaluate
- [`docs/quickstart.md`](docs/quickstart.md) — command reference
- [`docs/architecture.md`](docs/architecture.md) — components and request lifecycle
- [`docs/graph-model.md`](docs/graph-model.md) — YAML, node kinds, validation
- [`docs/protocol.md`](docs/protocol.md) — envelope, renderer, routing contract
- [`docs/execution.md`](docs/execution.md) — scheduler, budgets, caching, replay
- [`docs/evaluation.md`](docs/evaluation.md) — suites, metrics, gates
- [`docs/iteration.md`](docs/iteration.md) — change sets, revisions, rollback
- [`docs/training.md`](docs/training.md) — datasets, routers, specialists
- [`docs/gui.md`](docs/gui.md) — local GUI/API
- [`docs/security.md`](docs/security.md) — threat model and release checklist
- [`docs/observability.md`](docs/observability.md) — logs, metrics, spans
- [`docs/adr/`](docs/adr/) — accepted decisions and trade-offs
- [`docs/glossary.md`](docs/glossary.md) — vocabulary

## Contributing

Contributions are welcome under the project license. Read [`CONTRIBUTING.md`](CONTRIBUTING.md),
the [Code of Conduct](CODE_OF_CONDUCT.md), and the evaluation-evidence requirements. Security
reports should follow [`SECURITY.md`](SECURITY.md), not a public issue.

## License

Model Graph is licensed under the **GNU General Public License, version 3 or later**
(`GPL-3.0-or-later`). See [`LICENSE`](LICENSE).

Third-party models, adapters, datasets, and model cards retain their own licenses. The project
does not redistribute model weights; check upstream terms before use or redistribution.
