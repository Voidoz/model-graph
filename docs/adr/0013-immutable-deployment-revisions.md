# ADR-0013: Swaps are immutable deployment revisions with explicit invalidation

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [iteration.md](../iteration.md), [performance.md](../performance.md),
  [execution.md](../execution.md), ADR-0007, ADR-0010, ADR-0012

## Context

Users will constantly change a graph: swap a specialist, train a new adapter, edit a prompt,
add a router, change a base model, or re-distill a hot path. The system has to make those changes
fast, comparable, and reversible. It also has compiled plans, prefix caches, route caches, and
distilled students that depend on exact graph/model hashes.

Two naive approaches fail:

1. **Mutate the live deployment.** Edit the model binding or prompt in place. There is no
  revision identity, in-flight requests can change behavior mid-run, and rollback is impossible.
2. **Rebuild everything on every edit.** Correct but slow; users stop iterating and stop
  modularizing, which defeats the product.

A third failure is subtle: **stale artifacts**. A compiled plan or distilled student continues
to serve after the source graph changes because nothing forces invalidation. The user believes
they are testing the new workflow, but the deployment is running the old one.

## Decision

**Every behavior-changing swap produces a new immutable deployment revision. Iteration is
explicit, impact is analyzed before building, and promotion is atomic and reversible.**

1. **Revision identity.** A deployment revision is a hash over the graph hash, all model/adapter
   hashes, the target, the plan/student hash, and the compiler version. It is immutable.
2. **No in-place mutation.** Editing a graph or swapping a model creates a candidate change set.
   The active revision changes only through promotion.
3. **Impact before build.** The system computes compatibility (base/tokenizer/adapter), affected
   nodes and routes, invalidated plans/caches/students, VRAM/admission delta, and required eval
   coverage. A change that cannot be proven compatible is not promotable.
4. **Hash-based invalidation.** Compiled prompts, prefix caches, route caches, and distilled
   students are keyed by the hashes they depend on. A stale artifact is never silently reused.
5. **In-flight pinning.** Requests and sessions pin to the revision they started on. New requests
   use the new revision after promotion. Mid-request behavior does not change.
6. **Promotion is atomic and evidence-gated.** The new revision must pass node eval, routing eval
   when decisions/menus change, e2e regression, and a parity benchmark when performance-relevant.
   Local experiments may skip gates and are marked experimental; promotion may not.
7. **Rollback is a pointer switch.** The previous revision and its artifacts remain addressable
   for a documented grace period. Rollback does not require a recompile.
8. **Hot swap is conditional, not assumed.** Adapter swaps and same-engine model swaps may be hot
   if both fit in VRAM and the engine can switch atomically. Base/tokenizer changes are
   migrations. The system does not promise zero downtime for a migration.
9. **Distilled target staleness is visible.** Any source change marks B4 artifacts stale. The
   system may keep serving them, but status and traces show the drift; re-distillation is an
   explicit step.
10. **Diagnostics use revisions.** Run manifests, traces, eval reports, and comparisons record
    the deployment revision. "It got worse after the swap" is answerable.

## Consequences

**Positive**

- Rollback is reliable and fast.
- Candidates are comparable because the suite, graph, and model hashes are pinned.
- Stale plans/students are detected rather than silently served.
- Hot adapter swaps become a genuine advantage of the fused backend.
- In-flight requests are protected from behavior changes.
- The user can iterate aggressively without compromising the deployed revision.

**Negative**

- More artifacts and revisions to store and garbage-collect.
- Every swap carries impact-analysis and promotion overhead.
- Hot swap requires VRAM headroom for two revisions; otherwise a drain is needed.
- Distilled targets can lag the source graph by design, creating a visible but real drift.
- A strict gate can slow down exploratory users; the escape hatch is an explicit local
  experimental mode, not a silent bypass.
- The revision model adds concepts the user must learn (change set, revision, promotion).

**Neutral**

- Git remains the source of graph revisions; the deployment revision is the runtime identity.
- A future hosted deployment service can use the same revision semantics.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Mutate the live graph in place | No rollback, no in-flight safety, no reliable comparison. |
| Rebuild/redeploy everything on every edit | Correct but too slow for the authoring loop; encourages monolithic prompts. |
| Treat model swaps as simple manifest edits | Ignores tokenizer/base compatibility, cache invalidation, routing menu changes, and distilled staleness. |
| Automatic hot-reload of any change | Silent behavior changes and VRAM surprises; unsafe for base/tokenizer migrations. |
| Trust a version string for cache invalidation | Tags can move; only content hashes are reliable. |
| Never allow running stale distilled students | Safe but operationally rigid; a visible stale mode with a re-distill policy is more practical. |

## Follow-ups

- [T-0038](../../.scratch/tickets/T-0038-change-sets-swaps.md): change sets, impact analysis,
  atomic promotion, rollback (Phase 4).
- [T-0039](../../.scratch/tickets/T-0039-canary-hot-swap.md): canary/shadow, in-flight pinning,
  hot swap and drain for resident/fused targets (Phase 5).
- [T-0040](../../.scratch/tickets/T-0040-iteration-ux.md): watch mode, candidate comparison,
  impact panel, stale badges (Phase 4/5).
- [T-0034](../../.scratch/tickets/T-0034-aot-plan-compiler.md) supplies plan hashes and
  invalidation. [T-0037](../../.scratch/tickets/T-0037-graph-distillation.md) supplies the
  distilled-student staleness policy.
- Resolve `Q-016` (hybrid distilled fallback) and `Q-018` (revision retention/GC) before a
  production deployment workflow.
