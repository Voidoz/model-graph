# Model Graph

**Compose small, independently trainable language models into a routed graph.**

Model Graph is a design for breaking an AI system into discrete nodes — deciders that
route a request and specialists that handle it — connected only by natural language.
Instead of scaling one monolithic model, you compose many small ones: a router decides
which expert should see the request, the expert answers, and the graph returns the
result. Experts can be trained by different people on ordinary hardware, versioned
independently, and pruned when a deployment does not need them.

> **Status: design v0.** This repository currently contains the design, contracts,
> examples, architectural decisions, and a ticket backlog. There is no runtime code yet.
> The first implementation ticket is `T-0001` — see [`.scratch/tickets/BOARD.md`](.scratch/tickets/BOARD.md).

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
| [`docs/vision.md`](docs/vision.md) | Problem, thesis, hypotheses, kill criteria, what this is not |
| [`docs/architecture.md`](docs/architecture.md) | System architecture, components, request lifecycle, deployment |
| [`docs/graph-model.md`](docs/graph-model.md) | Node kinds, edges, artifact types, YAML anatomy, validation rules |
| [`docs/protocol.md`](docs/protocol.md) | Envelope, renderers, router decisions, error model |
| [`docs/execution.md`](docs/execution.md) | Scheduler, budgets, pruning, caching, replay, observability |
| [`docs/gpu-execution.md`](docs/gpu-execution.md) | GPU-resident and fused same-base backends, overhead accounting, limits |
| [`docs/performance.md`](docs/performance.md) | LLM-parity analysis, options compared, overhead budget, compilation targets |
| [`docs/iteration.md`](docs/iteration.md) | Iteration loops, swap impact, atomic revisions, canary/rollback |
| [`docs/training.md`](docs/training.md) | Independent specialist and router training, data flywheel |
| [`docs/evaluation.md`](docs/evaluation.md) | Metrics, suites, baselines, contribution gates |
| [`docs/roadmap.md`](docs/roadmap.md) | Phases, milestones, sequencing, risk register summary |
| [`docs/glossary.md`](docs/glossary.md) | Shared vocabulary |
| [`docs/adr/`](docs/adr/) | Accepted architectural decisions and their trade-offs |
| [`schemas/`](schemas/) | Machine-checkable contracts (graph, model manifest, envelope, ticket) |
| [`examples/`](examples/) | Worked graph, manifest, and envelope examples |
| [`.scratch/`](.scratch/) | Working ideas, open questions, ticket backlog, journal, handoff |

## Read in this order

1. This README and [`docs/vision.md`](docs/vision.md) — why the project exists.
2. [`docs/architecture.md`](docs/architecture.md) — how the pieces fit.
3. [`docs/graph-model.md`](docs/graph-model.md) and [`examples/`](examples/) — what you author.
4. [`docs/protocol.md`](docs/protocol.md) and [`docs/execution.md`](docs/execution.md) — what runs.
5. [`docs/gpu-execution.md`](docs/gpu-execution.md) — how a graph can run GPU-resident, and what that cannot fix.
6. [`docs/performance.md`](docs/performance.md) — how the graph compares to an LLM, what to compile, and what parity requires.
7. [`docs/iteration.md`](docs/iteration.md) — how a user swaps models safely while building a workflow.
8. [`docs/training.md`](docs/training.md) and [`docs/evaluation.md`](docs/evaluation.md) — how nodes improve.
9. [`docs/roadmap.md`](docs/roadmap.md) and [`.scratch/tickets/BOARD.md`](.scratch/tickets/BOARD.md) — what to build next.

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
