# ADR-0010: Evaluation-gated contributions and per-node ownership

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [evaluation.md](../evaluation.md), [training.md](../training.md), ADR-0006

## Context

The project expects contributions from many people and models: new specialists, routers,
prompts, graph topologies, adapters, and eval suites. Without a shared standard of evidence:

- a contributor can claim an improvement that is noise, overfitting, or a benchmark leak;
- a model that is better in isolation can regress the end-to-end system;
- reviewers cannot reproduce claims;
- the component ecosystem becomes a pile of unverified checkpoints;
- the project's central promise — that small, independently trained components can be
  trusted and composed — collapses.

Evaluation is also expensive. If every change requires a full benchmark, contribution becomes
inaccessible. The gate must be strong enough to trust and cheap enough to run locally.

## Decision

**Contributions are gated by reproducible, cost-aware evaluation, and each node has a clear
owner and version.**

1. **Every node has an owner**, a manifest, a version, and an eval suite. Ownership can be a
   person or a team; it implies responsibility for regressions, not exclusive control.
2. **Every model/prompt/graph change ships with an eval report** referencing the run
   manifest, suite hash, command, commit, and hardware.
3. **Evaluation is leveled**: node suite → routing suite → graph regression. A change must
   pass its own level and not regress the graph beyond tolerance.
4. **Quality, cost, and latency are reported together.** A quality gain at unbounded cost is
   not automatically accepted.
5. **Baselines are mandatory**: best single model, incumbent, and (for routing) oracle and
   random.
6. **Suites are versioned and frozen.** Changing a suite invalidates comparability and
   requires review. Private held-out slices exist for release decisions.
7. **Hard gates cannot be traded**: validity, schema, safety, no secret leakage, budget
   enforcement.
8. **Initial tolerances** (documented in [evaluation.md](../evaluation.md)) are defaults, not
   eternal truths. Changing a gate requires a documented decision, not an exception in a PR.
9. **Reproducibility is part of done.** If another person cannot rerun the report, the claim
   is provisional.

## Consequences

**Positive**

- Claims are auditable and comparable across contributors.
- Regressions are caught near the node that caused them, not after a release.
- The registry can rank nodes by evidence rather than popularity.
- Contributors get a clear, local path to prove value (node suite + graph regression).
- Evaluation data doubles as a learning signal for routers and specialists.

**Negative**

- Up-front cost: suites, fixtures, and reports must be built before contributions scale.
- Evaluation can be gamed or overfit; private slices and contamination checks are necessary
  but imperfect.
- Small contributions may not justify the eval overhead, encouraging batching.
- Judge-based metrics are noisy and require calibration.
- Node ownership can create bottlenecks if owners disappear; the project needs a stewardship
  policy.
- Strict gates can slow research; the escape hatch is a clearly marked research branch, not
  a silent exception.

**Neutral**

- Gates apply to releases and the registry, not to every local experiment. Researchers can
  run ungated graphs locally and only submit evidence when claiming a result.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Merge on review alone | Does not scale; reviewers cannot detect subtle regressions or contamination. |
| Full benchmark for every change | Too expensive; kills contribution. Leveled suites are the compromise. |
| Trust model cards from contributors | Cards are self-reported and often incomplete; reports must be reproducible. |
| Centralized evaluation only | Creates a bottleneck and hides methodology from contributors; local suites are required. |
| No cost/latency reporting | Optimizes quality in a vacuum; the project's thesis is quality per dollar. |

## Follow-ups

- [T-0020](../../.scratch/tickets/T-0020-node-eval.md), [T-0021](../../.scratch/tickets/T-0021-e2e-eval.md),
  and [T-0022](../../.scratch/tickets/T-0022-ci-gates.md) implement the levels and gates.
- [T-0026](../../.scratch/tickets/T-0026-model-registry.md) enforces cards and reports at
  registry entry.
- Publish a contribution guide with a worked example once Phase 2 is complete.
- Revisit tolerances with measured variance, not intuition.
