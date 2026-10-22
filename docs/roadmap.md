# Roadmap

This roadmap is organized by **exit criteria**, not calendar dates. Work is expected to stop
and start; the ticket board is the live view of what is in progress, and every ticket is
written to be independently executable.

- Live board: [`../.scratch/tickets/BOARD.md`](../.scratch/tickets/BOARD.md)
- Risks: [`../.scratch/ideas/risks.md`](../.scratch/ideas/risks.md)
- Open questions: [`../.scratch/ideas/open-questions.md`](../.scratch/ideas/open-questions.md)

## Sequencing principles

1. **Contracts before code.** A schema and a validator are cheaper to change than a runtime.
2. **Mock before model.** Deterministic fixtures expose scheduler bugs that model
   non-determinism hides.
3. **Trace before training.** No flywheel without a trace format and outcome labels.
4. **Evaluation before training.** You cannot improve what you cannot measure.
5. **Single branch before fan-out.** Concurrency multiplies scheduling bugs.
6. **Local before hosted.** The core value is ownership and portability; a server can come
   later.
7. **One domain deeply before many domains shallowly.** Prove the thesis on a domain where
   quality is checkable (code) before expanding to judged tasks.

## Phase 0 — Foundations

**Goal:** the repository can validate a graph and establish a deterministic provider boundary.

| Ticket | Deliverable |
| --- | --- |
| [T-0001](../.scratch/tickets/T-0001-dev-environment.md) | Python project, uv, lint/format, pytest, CI, pre-commit |
| [T-0002](../.scratch/tickets/T-0002-contract-tests.md) | Schema validation tests and a fixture corpus |
| [T-0003](../.scratch/tickets/T-0003-graph-compiler.md) | Compiler and static validator with V-01..V-20 |
| [T-0004](../.scratch/tickets/T-0004-cli.md) | `mg validate`, `mg run`, `mg eval` skeletons and config resolution |
| [T-0005](../.scratch/tickets/T-0005-mock-provider.md) | Deterministic mock provider, canned responses, failure scenarios |

**Exit criteria**

- A fresh checkout runs `mg validate examples/*.graph.yaml` with no network.
- A deliberately broken graph reports every validation error in one pass.
- The compiler produces a stable IR hash for identical inputs.
- The mock provider passes its contract tests and can script route, worker, joiner, and
  provider-failure scenarios.

The end-to-end mock walk is M0, delivered by the run engine in Phase 1 (T-0010), because it
needs a scheduler. Phase 0 stops at "the pieces can be validated and the provider boundary is
deterministic", not at "a graph runs".

## Phase 1 — Runtime core

**Goal:** a real local graph runs, is pruned for the machine, is fully traceable, and can be
replayed.

| Ticket | Deliverable |
| --- | --- |
| [T-0006](../.scratch/tickets/T-0006-envelope-artifacts.md) | Envelope, artifact store, scoped views |
| [T-0007](../.scratch/tickets/T-0007-renderer.md) | Deterministic renderer and prompt templates |
| [T-0008](../.scratch/tickets/T-0008-router-executor.md) | Router executor: structured decisions, repair, abstain, fallback |
| [T-0009](../.scratch/tickets/T-0009-worker-joiner.md) | Worker and joiner executors and artifact parsing |
| [T-0010](../.scratch/tickets/T-0010-run-engine.md) | Run engine: walk, budgets, error model |
| [T-0011](../.scratch/tickets/T-0011-trace-replay.md) | Trace capture and deterministic trace replay |
| [T-0012](../.scratch/tickets/T-0012-remote-adapters.md) | OpenAI-compatible adapter |
| [T-0013](../.scratch/tickets/T-0013-local-adapters.md) | Ollama and llama.cpp adapters |
| [T-0014](../.scratch/tickets/T-0014-accounting.md) | Token, cost, latency accounting |
| [T-0015](../.scratch/tickets/T-0015-pruning-lazy-loading.md) | Capability profiles, pruning, lazy loading |
| [T-0016](../.scratch/tickets/T-0016-cache.md) | Exact-match response cache |
| [T-0017](../.scratch/tickets/T-0017-subgraphs.md) | Subgraph nodes and boundary mapping |
| [T-0018](../.scratch/tickets/T-0018-fanout-ensemble.md) | Fan-out and ensemble joins (post-v0) |
| [T-0019](../.scratch/tickets/T-0019-run-manifests.md) | Run manifests and version pinning |

**Exit criteria**

- The hello graph runs end to end against the mock provider with a deterministic node
  sequence and artifact hashes (**M0 — "Mock walk"**).
- The composed-assistant example runs on a local Ollama endpoint with real models (**M1 —
  "Local walk"**).
- Every run produces a run manifest, a trace, and content-addressed artifacts.
- Replaying a trace reproduces the node sequence and artifacts without network access.
- A profile that excludes one specialist runs a pruned graph and reports exactly what was
  removed, or fails loudly if an exit becomes unreachable.
- A budget-exceeding run stops before the call that would cross the limit.

**Milestone M1 — "Local walk"**: `mg run examples/composed-assistant.graph.yaml
--profile examples/profiles/local-8gb.yaml` returns an answer with a per-node cost and latency
breakdown.

## Phase 2 — Measurement

**Goal:** claims about routing and composition become evidence.

| Ticket | Deliverable |
| --- | --- |
| [T-0020](../.scratch/tickets/T-0020-node-eval.md) | Node eval harness, routing metrics, calibration report |
| [T-0021](../.scratch/tickets/T-0021-e2e-eval.md) | E2E suite, baselines, Pareto report |
| [T-0022](../.scratch/tickets/T-0022-ci-gates.md) | CI regression gates and contribution checks |

**Exit criteria**

- The coding example has a node suite, a routing suite with oracle labels, and an e2e suite.
- A report compares the graph against the best single model, oracle routing, and random
  routing at matched cost.
- A pull request that regresses e2e quality beyond tolerance fails CI.
- Routing reports include a confusion matrix and calibration, not just accuracy.

**Milestone M2 — "Measured"**: the repository can answer "is this graph better than the
monolith, and where does it fail?" with a reproducible report.

## Phase 3 — Training flywheel

**Goal:** a contributor can improve a node on consumer hardware and prove it.

| Ticket | Deliverable |
| --- | --- |
| [T-0023](../.scratch/tickets/T-0023-dataset-export.md) | Trace-to-dataset export with redaction and manifests |
| [T-0024](../.scratch/tickets/T-0024-router-training.md) | Router labeling, training, calibration recipe |
| [T-0025](../.scratch/tickets/T-0025-specialist-training.md) | Specialist LoRA/distillation kit and model cards |
| [T-0026](../.scratch/tickets/T-0026-model-registry.md) | Registry, contribution cards, verification |

**Exit criteria**

- At least one router is trained from exported traces and improves oracle agreement.
- At least one specialist is trained with LoRA on a 24GB GPU and improves its node suite.
- Both improvements pass graph regression and are promoted via the registry with cards.
- Dataset and model provenance is complete enough for another person to reproduce the run.

**Milestone M3 — "Trainable"**: a documented end-to-end flywheel from traces to a promoted
node, executed once on real hardware.

## Phase 4 — Experience and hardening

**Goal:** the system is usable, inspectable, and not embarrassing to run.

| Ticket | Deliverable |
| --- | --- |
| [T-0027](../.scratch/tickets/T-0027-starter-library.md) | Starter graph library and tutorial |
| [T-0028](../.scratch/tickets/T-0028-visual-inspector.md) | Read-only graph/trace inspector spike |
| [T-0029](../.scratch/tickets/T-0029-security-hardening.md) | Injection defenses, redaction, untrusted-node policy |
| [T-0030](../.scratch/tickets/T-0030-observability.md) | Structured logs, metrics, OpenTelemetry export |
| [T-0038](../.scratch/tickets/T-0038-change-sets-swaps.md) | Change sets, impact analysis, atomic revisions, rollback (v0.1) |
| [T-0040](../.scratch/tickets/T-0040-iteration-ux.md) | Watch mode, diff/compare, impact panel, stale badges (v0.1) |

**Exit criteria**

- A new user can go from clone to a running local graph by following one tutorial.
- The inspector renders a graph and a trace, including decisions and artifacts.
- The security checklist passes for a local release: no secrets in traces, delimited
  content, budget enforcement, dependency audit.
- `mg run` emits structured events suitable for external monitoring.

**Milestone M4 — "Usable v0"**. T-0038/T-0040 are the v0.1 iteration track; they improve model swapping but are not required for M4.

## Definition of usable v0

The first release is real when all of the following are true:

1. A user can write a graph in YAML with routers, workers, a subgraph, and a fallback; run
   it against at least two providers (one local) without code changes.
2. Every run is budgeted, traced, replayable, and reports quality/cost/latency for the
   cases that were evaluated.
3. Capability pruning works and is explained, including failing safely.
4. A node can be evaluated and contributed with a manifest, card, and report.
5. The documentation describes what the system does *not* do, with the same clarity as what
   it does.
6. The security baseline is met and the open questions that block a public release (license,
   contribution terms, privacy) are resolved.

## Phase 5 — LLM-class execution (post-v0)

**Goal:** make a configured graph a credible performance competitor to a dense LLM, not merely
a modular runtime with lower transport overhead. The graph is treated as source; deployments
may be resident, fused, compiled, or distilled.

This phase is deliberately after usable v0. It must not delay M1–M4, and the logical graph,
artifact, budget, and trace contracts do not change. The analysis is in
[performance.md](performance.md); the decisions are
[ADR-0011](adr/0011-gpu-resident-backends.md) and
[ADR-0012](adr/0012-compiled-deployment-targets.md).

| Ticket | Deliverable |
| --- | --- |
| [T-0031](../.scratch/tickets/T-0031-resident-backend.md) | `ExecutionBackend` contract + resident heterogeneous spike (B1) |
| [T-0032](../.scratch/tickets/T-0032-fused-multilora.md) | Fused same-base multi-LoRA backend (B2) |
| [T-0033](../.scratch/tickets/T-0033-transition-benchmark.md) | Transition and LLM-parity benchmark, baselines, gates |
| [T-0034](../.scratch/tickets/T-0034-aot-plan-compiler.md) | AOT execution plan: tokenized templates, prefix KV, memory plan |
| [T-0035](../.scratch/tickets/T-0035-fast-routing.md) | Classifier/head routing, cascade flattening, route cache |
| [T-0036](../.scratch/tickets/T-0036-batched-scheduler.md) | Multi-request scheduler and continuous batching |
| [T-0037](../.scratch/tickets/T-0037-graph-distillation.md) | Single-model distilled deployment target (B4) |
| [T-0039](../.scratch/tickets/T-0039-canary-hot-swap.md) | Canary/shadow, in-flight pinning, hot swap/drain, rollback |

**Exit criteria**

- A 3-node heterogeneous graph runs with all models resident, zero weight reloads, and no
  HTTP between nodes.
- A shared-base graph runs with at least two adapters, token-native handoff, and artifacts
  logically identical to the portable backend.
- A compiled plan eliminates static re-tokenization, reuses scoped prefix KV, and invalidates
  safely when the graph/model changes.
- Fast routing meets its latency target without losing oracle agreement beyond tolerance.
- Cross-request batching improves throughput without breaking per-request budgets or isolation.
- The benchmark reports TTFT, ITL, end-to-end p95, throughput, cost, VRAM, cold load, and
  transition phases for B0/B1/B2 and dense/MoE/cascade baselines at matched quality.
- A distilled student is evaluated against the graph; it is promoted only if quality holds,
  and its staleness/re-distillation policy is documented.
- A model/adapter swap can be staged, promoted, and rolled back without changing in-flight requests;
  adapter hot swap works where the engine supports it.
- Backend admission refuses an infeasible plan with an actionable report; no silent paging.
- The docs state clearly what still costs what: per-node prefill, routing TTFT, batching loss,
  and the runtime-modularity cost of distillation.

**Milestone M5 — "LLM-class or evidence"**: a quality-matched report either shows the compiled
graph within the parity budget on at least three decomposable domains, or documents precisely
where it loses. A negative result is a valid, valuable outcome; the phase can be cancelled
rather than rationalized.

## Critical path

```text
T-0001 → T-0002 → T-0003 → T-0006 → T-0007 → T-0008/T-0009 → T-0010
      → T-0011 → T-0013 → T-0015 → T-0017 → T-0020 → T-0021 → T-0022
      → T-0023 → T-0024/T-0025 → T-0026 → T-0027 → T-0029
```

T-0004 and T-0005 run in parallel with the compiler. T-0012 can proceed once T-0010 lands.
T-0018 (fan-out) is explicitly off the critical path and should not delay v0.

Optional post-v0 performance track:

```text
T-0021 → T-0031 → T-0032 → T-0034 → T-0035
                         ↘ T-0036
T-0033 (parity benchmark, rerun as targets land)
T-0033 + T-0023/T-0025 → T-0037 (distillation)
```

T-0031 can start after T-0013/T-0015/T-0019 even if Phase 2 is still running. It must not
consume the capacity needed for M2/M3. T-0033 can be run incrementally; T-0037 is the last
step because it needs both the benchmark methodology and the training data.

## Deliberately not scheduled

| Idea | Why not now |
| --- | --- |
| Hosted marketplace and payments | License, trust, and serving economics are unresolved (`Q-001`, `Q-003`) |
| Arbitrary tool/code-execution nodes | Security and sandboxing dominate; needs a separate design (`Q-004`) |
| Multi-modal nodes | The protocol is text-only; images/audio require schema and cost work (`Q-005`) |
| Recursive graph composition | Termination and budget semantics need proof (`Q-007`) |
| Auto-discovering graph topology | Evaluation is too weak to trust search yet |
| Training a base model | Explicitly not the project's thesis |
| Jointly trained MoE / expert merging | Strong single-model competitors, but they require shared training or lossy weight merging; compared in benchmarks, not adopted as the core |
| Latent handoff between models | Requires co-designed representations and breaks independent ownership (ADR-0001); research only |

## Decision points

| When | Decision | Owner |
| --- | --- | --- |
| Before public release | License and contribution terms | Maintainers (`Q-001`) |
| After Phase 2 | Is the oracle gap large enough to justify training routers? | Maintainers |
| After Phase 2 | Does composition beat the best single model at matched cost on 3 domains? | Maintainers (kill criteria) |
| Before Phase 3 | Dataset consent/redaction policy | Maintainers (`Q-010`) |
| Before a hosted mode | Trust model for untrusted graphs and models | Maintainers (`Q-003`) |
| Before fan-out | Cost/quality justification from eval data | Maintainers |
| Before Phase 5 | Does the measured overhead justify a resident backend, and which tier (B1/B2)? | Maintainers (T-0033) |
| Before B2 | Is a shared base model acceptable for the target graph, given the quality trade-off? | Graph author + maintainers |
| Before B4 | Does distillation preserve quality on rare routes, and is a hybrid graph fallback required? | Maintainers (`Q-016`) |

## Risk summary

The full register is in [`../.scratch/ideas/risks.md`](../.scratch/ideas/risks.md). The
roadmap is shaped by these risks:

| Risk | Mitigation in this roadmap |
| --- | --- |
| Routing errors erase specialist gains | Oracle labels and routing suite in Phase 2; router training in Phase 3; fallbacks from day one |
| Handoff loss makes composition worse than one model | Typed artifacts, scoped views, E2E baselines, kill criteria |
| Cost/latency stack up across hops | Budgets, pruning, exact cache, Pareto reporting |
| Evaluation is noisy or gameable | Frozen suites, confidence intervals, hard gates, private slices |
| Scope sprawls into an agent framework | Explicit non-goals, deferred tools/recursion, node-kind restraint |
| GPU residency forces silent paging or OOM | Pre-load admission, explicit fallback, no reload guarantee, T-0033 matrix |
| Fused backend locks the graph to one base model | Opt-in backend, compatibility checks, quality eval, heterogeneous B1 remains |
| Distilled target loses rare routes or goes stale | Slice eval, provenance, re-distill policy, graph fallback (`Q-016`) |
| Throughput loses to batching/queueing | Cross-request scheduler, per-model queues, isolation tests (T-0036) |
| Parity claim rests on a weak baseline | Pinned quality-matched dense/MoE baselines and a published report (`Q-017`) |
| Stale plans/students or a bad swap reach users | Immutable revisions, impact analysis, hash invalidation, canary, rollback (T-0038/T-0039) |
| Iteration is too slow to modularize | Fast authoring loop, watch/diff/compare, incremental eval (T-0040) |

## How to use this roadmap when work stops

1. Read `.scratch/HANDOFF.md` for the last known state.
2. Take the highest-priority `ready` ticket on the board whose dependencies are done.
3. Read the ticket's linked ADRs and docs before writing code.
4. Follow the ticket's acceptance criteria and test plan, then update the board and handoff.

A phase is complete only when its exit criteria are demonstrated, not when its tickets are
marked done.
