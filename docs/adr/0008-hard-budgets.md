# ADR-0008: Hard budgets and loop guards by default

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [execution.md](../execution.md), [graph-model.md](../graph-model.md), ADR-0003, ADR-0005

## Context

A graph of models can spend real money and time. Multi-hop routing multiplies calls; a
misconfigured graph can route in a cycle, overflow context, retry a failing provider
forever, or fan out to an ensemble that costs more than the task is worth. Local models
also consume GPU memory and wall time, so "free" inference is not free.

Most agent frameworks treat limits as optional configuration. That makes the default unsafe:
the first time a user runs a graph, a bug can produce a surprisingly large bill or a run that
never ends.

## Decision

**Every run has hard budgets and loop guards, enabled by default and enforced before each
call.**

Defaults (a graph may change them, but cannot disable enforcement):

| Budget | Default |
| --- | --- |
| `max_steps` | 8 node entries |
| `max_depth` | 3 subgraph levels |
| `max_total_tokens` | 32,000 input + output |
| `max_cost_usd` | 0.25 estimated |
| `max_wall_seconds` | 300 |
| `on_budget_exhausted` | `fail` (optionally `best_effort`) |

Rules:

1. **Pre-flight veto.** Before a provider call, the runtime estimates the minimum tokens and
   cost. If the call cannot fit, it is not started.
2. **Hard ceiling.** The runtime does not exceed a limit to finish a request, even if one
   more call would produce an answer.
3. **Graphs are acyclic.** Cycles are a validation error. Subgraph nesting is depth-limited.
4. **Retries consume budget** and are themselves capped.
5. **Retry/repair limits are explicit** per node and never infinite.
6. **Subgraphs share the parent budget.** Nesting cannot reset counters.
7. **Every run records its effective budget** and the reason for exhaustion.
8. **`best_effort` is opt-in** and marks the run as incomplete; it never silently returns a
   partial answer as if it were complete.
9. **Unknown cost** does not count as zero unless the profile explicitly allows it; the run
   warns or fails.

## Consequences

**Positive**

- Safe by default: a new user cannot accidentally spend unbounded money on a first run.
- Latency and cost are predictable and reportable.
- Termination is a static property, not a hope.
- Evaluation can include cost as a first-class metric because cost is measured consistently.
- The same mechanism supports local resource limits (VRAM, concurrent model loads) later.

**Negative**

- A legitimate long task may need budget tuning before it works.
- Pre-flight estimates can be wrong (output length, tokenizer differences), causing a veto
  that is slightly conservative.
- `best_effort` results are harder to interpret than clean success.
- Cost estimation depends on manifest accuracy; local models with unknown pricing are
  approximated.
- Extra bookkeeping in every execution path.

**Neutral**

- Budgets are policy, not physics; profiles can adjust them deliberately and record the
  change.
- A future dynamic budget allocator could redistribute across branches without weakening
  the ceiling.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| No limits; trust the graph author | One bad router or retry loop produces real bills and hangs. |
| Only wall-clock timeout | Prevents hangs but not cost blowups or token overruns. |
| Only cost limit | Local models have zero marginal cost by default but still consume GPU time and memory. |
| Soft warnings | Users ignore warnings; the runtime must enforce. |
| Per-node limits only | Global compounding across nodes is exactly the failure mode to prevent. |

## Follow-ups

- T-0010 implements the budget accountant and
  veto.
- T-0014 implements cost/token/latency
  accounting.
- T-0015 adds resource-aware
  profiles.
- Revisit defaults after real traces; publish the rationale for any change.
