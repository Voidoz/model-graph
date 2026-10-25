# Model Graph

**Compose small, independently trainable language models into a routed graph.**

Model Graph is a system for breaking an AI system into discrete nodes — deciders that
route a request and specialists that handle it — connected only by natural language.
Instead of scaling one monolithic model, you compose many small ones: a router decides
which expert should see the request, the expert answers, and the graph returns the
result. Experts can be trained by different people on ordinary hardware, versioned
independently, and pruned when a deployment does not need them.

> **Status: v0.5.0 — hardened, observable graph runtime.** The repository has a
> runnable vertical slice — schema and semantic validation, router/worker/joiner/subgraph
> execution, deterministic mock and local HTTP providers, hard budgets, traces, run manifests,
> and offline replay — plus a loopback-only `mg ui` GUI for graph inspection, mock runs,
> traces, artifacts, models, history, and replay. The measurement layer includes node, routing,
> end-to-end, and safety gates; model changes go through impact analysis and immutable revisions
> with atomic promotion and rollback. Trace-to-dataset export, a tested starter library, an
> exact-match response cache, structured logs/metrics, and a security threat model are in place.
> Resident/compiled targets and canaries are the next tracks. Live local-model validation passed
> on CPU and on an RTX 3080 via Ollama Vulkan, including a composed run with 4GB pruning. Start
> with [docs/tutorial.md](docs/tutorial.md) for the guided path,
> [docs/quickstart.md](docs/quickstart.md) for the command reference, and
> [`.scratch/tickets/BOARD.md`](.scratch/tickets/BOARD.md) for what comes next.

## The idea in one picture

```text
                 ┌──────────────┐
   request ─────▶│  triage      │  decider
                 │  (router)    │
                 └──┬────────┬──┘
                    │        │
          "coding"  │        │  "story"
                    ▼        ▼
             ┌──────────┐  ┌──────────┐
             │ coding   │  │ story    │
             │ decider  │  │ writer   │
             └──┬────┬──┘  └────┬─────┘
       "JS"      │    │ "general" │
                 ▼    ▼           ▼
            ┌──────┐ ┌────────┐ ┌────────┐
            │  JS  │ │ coder  │ │ story  │
            │ 3B   │ │ 7B     │ │ 7B     │
            └──────┘ └────────┘ └────────┘
                    \     |     /
                     \    |    /
                    answer artifacts
```

The edges carry language, not tensors. Each model sees a rendered prompt it can
understand and returns a result or a routing decision. The graph is the composition;
the language is the bus.

## Quickstart (MVP)

The first runnable slice is in place. It validates graphs, runs them with a deterministic
mock provider or a local Ollama endpoint, records a trace and run manifest, and replays a
recorded run offline. The guided version of this section, including pruning and a node
evaluation, is [docs/tutorial.md](docs/tutorial.md).

```text
uv sync
uv run mg validate graphs/hello-triage.graph.yaml
uv run mg run graphs/hello-triage.graph.yaml --provider mock
uv run mg runs list
uv run mg replay runs/<run-id>
uv run mg trace show runs/<run-id>
```

To use a local model instead of the mock provider, pull the model in Ollama and run with the
default provider:

```text
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5-coder:7b
uv run mg run graphs/hello-triage.graph.yaml --registry registry/models
```

The starter library also includes a nested coding team and a writing/research graph:

```text
uv run mg run graphs/coding-specialist.graph.yaml --provider mock
uv run mg run graphs/composed-assistant.graph.yaml --provider mock
uv run mg run graphs/writing-studio.graph.yaml --provider mock
```

See [docs/quickstart.md](docs/quickstart.md) for the full command reference and
[docs/tutorial.md](docs/tutorial.md) for the tested first-run path.

## Local GUI

`mg ui` serves a zero-build, loopback-only web app from Python's standard library. It inspects
compiled graphs, runs the deterministic mock provider, shows traces/artifacts/usage, browses
models and run history, and replays recorded runs. No CDN, telemetry, auth, or Node build step.

```text
uv run mg ui
# Model Graph UI: http://127.0.0.1:8765/
```

Use `--port 0` for a free port, `--no-browser` to suppress the browser, and `--runs-dir PATH` to
point history at another directory. The run console's `mock` provider works offline; `auto`
uses each node's configured local provider. See [docs/gui.md](docs/gui.md) for the API,
page requirements, and security model.

## Evaluation and profiles

The measurement layer is runnable offline:

```text
uv run mg eval node --suite evals/mock/js-specialist \
  --graph graphs/coding-specialist.graph.yaml --provider mock
uv run mg eval routing --suite evals/mock/triage --provider mock
uv run mg eval e2e --suite evals/e2e/composed-assistant \
  --graph graphs/composed-assistant.graph.yaml --provider mock --profile local-8gb
make eval-gates
```

`mg eval compare` enforces the contribution gates against committed baselines. Run profiles
prune branches the machine cannot serve; `mg run --profile profiles/local-4gb.yaml` removes
the 6GB story/coder branches and reports the removals.

## Revisions and training data

A model swap is a candidate revision, not a live mutation. Inspect its impact before loading a
model, require the change-class evidence, then promote atomically or roll back:

```text
uv run mg status --revisions
uv run mg swap node js-specialist --to js-specialist-3b --dry-run
uv run mg change explain <change-id>
uv run mg promote --revision <revision> --dry-run
```

Accepted traces can become a consented, redacted, deduplicated dataset with frozen grouped
splits:

```text
uv run mg data export --run runs/<run-id> --out /tmp/dataset \
  --consent "operator approved local training"
```

See [docs/iteration.md](docs/iteration.md) and [docs/training.md](docs/training.md).

## Operations and safety

Repeated provider calls hit a local exact-match cache by default; `--no-cache` bypasses it and
`--cache-mode readonly` serves existing entries without writing. Cache hits consume zero tokens
and report avoided usage separately.

```text
uv run mg stats --last 5
MG_LOG_FORMAT=json MG_LOG_FILE=runs/run.jsonl uv run mg run graphs/hello-triage.graph.yaml
make audit
```

Structured logs and optional OpenTelemetry spans carry identifiers and timings, not prompts or
artifacts. The renderer labels untrusted content, neutralizes protocol markers, and caps oversized
inputs; persisted traces, envelopes, and errors are redacted. `make audit` is network-dependent
and separate from the offline gate. See [docs/security.md](docs/security.md) and
[docs/observability.md](docs/observability.md) for the threat model, limitations, and export
controls. Redaction can break replay and does not rewrite artifact blobs; do not treat delimiters
as injection immunity.

## Design principles

1. **Language is the wire.** Any model that can chat can join the graph. No shared
   embeddings, no weight surgery, no vendor lock-in (see [ADR-0001](docs/adr/0001-language-as-the-wire.md)).
2. **Nodes are independently ownable.** A node has a model manifest, an eval suite,
   and a version. You can improve one node without retraining the system
   (see [ADR-0010](docs/adr/0010-evaluation-gated-contributions.md)).
3. **The graph is declared, the route is runtime.** YAML declares the topology and is
   statically validated; the router chooses a branch per request
   (see [ADR-0005](docs/adr/0005-compile-time-graph-runtime-routing.md)).
4. **Local-first, provider-neutral.** Weights stay where their owners put them.
   A manifest points at an endpoint; adapters do the rest
   (see [ADR-0009](docs/adr/0009-local-first-provider-neutral.md)).
5. **Budgets are part of the type system.** Every run has hard limits on steps, depth,
   tokens, money, and wall time. Graphs must be safe to run by default
   (see [ADR-0008](docs/adr/0008-hard-budgets.md)).
6. **No proof, no merge.** A node, model, or prompt change ships with a reproducible
   evaluation report. Quality, cost, and latency are reported together.
7. **Portable by default, compiled for speed.** The same graph can be interpreted, GPU-resident,
   fused, or distilled without semantic drift. Performance claims require a quality-matched
   benchmark (see [ADR-0011](docs/adr/0011-gpu-resident-backends.md) and
   [ADR-0012](docs/adr/0012-compiled-deployment-targets.md)).

## What this repository contains

| Path | Purpose |
| --- | --- |
| [`docs/tutorial.md`](docs/tutorial.md) | Guided install → validate → mock/live run → trace → prune → node eval |
| [`docs/quickstart.md`](docs/quickstart.md) | Install, validate, run, inspect, and replay the MVP |
| [`docs/vision.md`](docs/vision.md) | Problem, thesis, hypotheses, kill criteria, what this is not |
| [`docs/architecture.md`](docs/architecture.md) | System architecture, components, request lifecycle, deployment |
| [`docs/graph-model.md`](docs/graph-model.md) | Node kinds, edges, artifact types, YAML anatomy, validation rules |
| [`docs/protocol.md`](docs/protocol.md) | Envelope, renderers, router decisions, error model |
| [`docs/execution.md`](docs/execution.md) | Scheduler, budgets, pruning, caching, replay, observability |
| [`docs/security.md`](docs/security.md) | Threat model, untrusted input, redaction, trust policy, release checklist |
| [`docs/observability.md`](docs/observability.md) | Structured logs, trace-derived metrics, `mg stats`, optional spans |
| [`docs/gpu-execution.md`](docs/gpu-execution.md) | GPU-resident and fused same-base backends, overhead accounting, limits |
| [`docs/performance.md`](docs/performance.md) | LLM-parity analysis, options compared, overhead budget, compilation targets |
| [`docs/iteration.md`](docs/iteration.md) | Iteration loops, swap impact, atomic revisions, canary/rollback |
| [`docs/gui.md`](docs/gui.md) | Local GUI architecture, JSON API, page contract, security |
| [`docs/profiles.md`](docs/profiles.md) | Run profiles, capability pruning, and lazy loading |
| [`docs/training.md`](docs/training.md) | Independent specialist and router training, data flywheel |
| [`docs/evaluation.md`](docs/evaluation.md) | Metrics, suites, baselines, contribution gates |
| [`docs/roadmap.md`](docs/roadmap.md) | Phases, milestones, sequencing, risk register summary |
| [`docs/glossary.md`](docs/glossary.md) | Shared vocabulary |
| [`docs/development.md`](docs/development.md) | Toolchain, checks, and repository layout |
| [`docs/adr/`](docs/adr/) | Accepted architectural decisions and their trade-offs |
| [`schemas/`](schemas/) | Machine-checkable contracts: graph, model manifest, envelope, ticket, eval suite, run manifest, dataset, change set |
| [`graphs/`](graphs/) | Tested starter library: hello-triage, coding-specialist, composed-assistant, writing-studio |
| [`profiles/`](profiles/) | Starter run profiles: 4GB, 8GB, and an air-gapped audit profile |
| [`registry/models/`](registry/models/) | Model manifests for the starter/library tutorial models |
| [`examples/`](examples/) | Design fixtures and the envelope example used by eval suites |
| [`.scratch/`](.scratch/) | Working ideas, open questions, ticket backlog, journal, handoff |

## Read in this order

0. [`docs/tutorial.md`](docs/tutorial.md) — run the starter library before reading the design.
1. This README and [`docs/vision.md`](docs/vision.md) — why the project exists.
2. [`docs/architecture.md`](docs/architecture.md) — how the pieces fit.
3. [`docs/graph-model.md`](docs/graph-model.md) and [`graphs/`](graphs/) — what you author.
4. [`docs/protocol.md`](docs/protocol.md) and [`docs/execution.md`](docs/execution.md) — what runs.
5. [`docs/gpu-execution.md`](docs/gpu-execution.md) — how a graph can run GPU-resident, and what that cannot fix.
6. [`docs/performance.md`](docs/performance.md) — how the graph compares to an LLM, what to compile, and what parity requires.
7. [`docs/iteration.md`](docs/iteration.md) — how a user swaps models safely while building a workflow.
8. [`docs/security.md`](docs/security.md) and [`docs/observability.md`](docs/observability.md) — what is untrusted, what is persisted, and what is exported.
9. [`docs/training.md`](docs/training.md) and [`docs/evaluation.md`](docs/evaluation.md) — how nodes improve.
10. [`docs/roadmap.md`](docs/roadmap.md) and [`.scratch/tickets/BOARD.md`](.scratch/tickets/BOARD.md) — what to build next.

## Working in the repository

- **Humans:** start with the ticket board, pick a `ready` ticket, and follow the
  definition of done in the ticket.
- **AI agents and external workflows:** read [`AGENTS.md`](AGENTS.md) and
  [`.scratch/AGENT-PROTOCOL.md`](.scratch/AGENT-PROTOCOL.md). The `.scratch` folder is
  the shared interface: stable ticket format, explicit status transitions, append-only
  journal, and a current handoff note.
- **Contracts:** `schemas/*.schema.json` are the canonical shapes. Change a contract by
  changing the schema and the docs together, with an ADR if the decision is significant.

## Assumptions and open decisions

These are explicit so they can be challenged, not buried:

- The working name is **Model Graph**; the CLI prefix is `mg`. Naming is open (`Q-002`).
- The implementation language is **Python 3.12+**; the contracts are language-neutral
  ([ADR-0002](docs/adr/0002-python-core-neutral-contracts.md)).
- The license is **undecided** (`Q-001`). Do not assume reuse rights yet.
- v0 is **text-in, text-out** (language, code, JSON) and **local-first**.
- Composition can *lose* quality as well as gain it. Routing and handoff fidelity are
  the central risks; the design treats them as measurable (`docs/evaluation.md`).

## License

TBD — see [`.scratch/ideas/open-questions.md`](.scratch/ideas/open-questions.md#q-001--license-and-contribution-terms).
