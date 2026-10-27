# ADR-0014: Multi-route selection, bounded fan-out, and fail-closed joins

- **Status:** Accepted
- **Date:** 2026-10-25
- **Deciders:** Project maintainers
- **Related:** [graph-model.md](../graph-model.md), [execution.md](../execution.md), ADR-0003, ADR-0006, ADR-0008

## Context

The v0 graph model already allowed a worker or subgraph to point at a joiner, but the
scheduler activated exactly one router route. That made the topology inspectable but not
executable: an ensemble could be authored, never run. T-0018 makes it executable without
changing the default single-route behavior.

Two decisions are not free:

1. A router decision currently names one target. Fan-out needs a rule for which routes run
   when the router can only rank, not assign independent probabilities.
2. Concurrent branches share one hard budget. A check-then-spend race can start branches
   that cannot all finish, and a joiner can wait forever for a branch that never arrives.

## Decision

**Add additive `routing.selection` modes `single` (default), `top_k` (`k` required), and
`all`; execute independent branches on a bounded thread pool; join only when every
declared input is present; and reserve a fan-out's first-call budget before starting any
branch.**

Details:

- `single` is byte-for-byte the old behavior and remains the default. Graphs without a
  multi-selection router use the original sequential walk.
- `top_k` accepts an ordered `routes` ranking in the router decision. Selection orders by
  confidence descending, edge `priority` descending, then target ID lexical ascending.
  Unranked routes fill missing slots in the same deterministic menu order. Router output
  order alone never breaks a tie.
- `all` activates every menu route in deterministic menu order. The router still returns a
  valid route/abstain decision so confidence-threshold and fallback policy keep working.
- Concurrency is a runtime limit (`max_concurrency`, default 4), not a graph budget field.
  The hard budgets remain the only authority on spend.
- Before a ready batch starts, the scheduler renders each first prompt, estimates tokens
  and cost with the same rules the executor uses, and reserves the sum atomically. If the
  sum does not fit, the run fails with `MG-4002`/`MG-4003` and no branch call starts.
  Reservations are settled with reported usage; the budget is lock-protected.
- A joiner runs only when all incoming edges are resolved and every declared `read` exists.
  The v0 contract has no `allow_partial`; a missing input is `MG-2201`. A join that waits
  past `join_timeout_seconds` (or the wall budget when unset) is `MG-2105`.
- Trace node groups are persisted in deterministic start order. Actual completion order is
  recorded separately. Artifacts and manifests use deterministic producer order.
- Branch usage is reported per branch in `RunOutcome.branches` and in `branch.completed`,
  `joiner.joined`, and `run.finished` trace events.

## Consequences

**Positive**

- The fan-out topology the data model described is executable and auditable.
- Single-route runs keep their old path, traces, and determinism guarantees.
- A fan-out that cannot fit the budget is rejected before it spends anything.
- A missing join input or a branch failure fails the run with a typed error instead of
  hanging or silently returning partial output.
- Per-branch cost makes the ensemble's quality/cost trade-off measurable.

**Negative**

- Fan-out multiplies token and cost usage by roughly the number of activated branches;
  `top_k`/`all` are opt-in and need evaluation against the best single branch.
- Concurrency adds real thread scheduling to the runtime. Budget, cache stats, envelope
  writes, and trace output need locks and deterministic flushing; completion order is
  inherently variable.
- `join_timeout_seconds` and wall-clock budgets are cooperative: a provider call that
  cannot be cancelled is allowed to return before the run fails.
- The pre-flight reserves each branch's *first* call, not its entire downstream cost.
  A subgraph branch reserves through its inner scheduler; the outer batch cannot see inside
  it. Later calls are still vetoed individually by the hard budget.
- A joiner cannot request partial results in v0 even when a branch is optional; adding
  that later needs a contract field and an explicit partial-join policy.

**Neutral**

- The router's ranking is a preference signal, not a calibrated distribution. The
  evaluation suite must still measure whether the ensemble beats one branch.
- The limit is per scheduler; a nested subgraph has its own pool.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Always activate all routes whenever a router has several edges | Changes existing single-route behavior and spends money without an author opt-in. |
| Select top-k by router output order only | Model output order is unstable and makes ties non-deterministic across providers. |
| Put `concurrency` in the graph contract | Concurrency is a deployment/runtime property; the graph budget is the spend authority. |
| Check the budget per branch just before each call | Two branches can pass the check and then overspend; a partial fan-out start is exactly what the veto must prevent. |
| Treat a missing joiner input as a partial result | It silently changes the answer contract. v0 has no `allow_partial` field, so fail closed. |
| Kill in-flight provider threads on join timeout | Python threads cannot be safely interrupted mid-call; a cooperative timeout is honest about the limit. |
| Use async tasks instead of threads | Provider adapters are synchronous in v0; threads add real concurrency without rewriting every adapter. |

## Follow-ups

- [T-0018](../../.scratch/tickets/T-0018-fanout-ensemble.md) implements the contract,
  scheduler, tests, and docs.
- Evaluate ensembles against the best single branch before promoting them.
- A future partial-join option needs a contract field, a recorded event, and an eval story;
  it is not a hidden runtime flag.
