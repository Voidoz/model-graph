# ADR-0003: Four node kinds, no arbitrary code nodes

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [graph-model.md](../graph-model.md), ADR-0005, ADR-0008

## Context

A graph engine can offer a fixed vocabulary of node kinds or a plugin system where arbitrary
code participates. A plugin system is powerful, but every plugin becomes a security,
versioning, and evaluation surface. It also tends to turn the project into a general agent
framework, which is explicitly not the goal.

The system needs just enough primitives to express the intended compositions:

- choose a branch;
- do work;
- combine branches;
- package a composition.

## Decision

**v0 has exactly four node kinds:**

| Kind | Role |
| --- | --- |
| `router` | Chooses one outgoing edge or abstains. Writes no artifacts. |
| `worker` | One model call, reads declared artifacts, writes named artifacts. Leaf. |
| `joiner` | Combines multiple upstream artifacts using an explicit strategy. |
| `subgraph` | Invokes another graph as a node with explicit input/output mapping. |

Rules that follow:

1. **No arbitrary code execution inside a graph.** There is no `python`, `shell`, or
   `plugin` node in v0. Tool use and code execution are deferred and would require a
   separate security design.
2. **Only routers route.** Workers and subgraphs are leaves; they may connect to a joiner
   for fan-out. This keeps the walk auditable and termination obvious.
3. **Recursion is explicit.** Nesting happens through `subgraph`, is visible at compile
   time, and is depth-budgeted. A worker cannot delegate.
4. **Node kinds are semantic, not cosmetic.** Adding one requires an ADR because the
   scheduler, validator, protocol, trace, and evaluation all branch on kind.

## Consequences

**Positive**

- Small surface for contributors to learn and for the runtime to secure.
- Static acyclicity and reachability are meaningful; the graph is analyzable before it runs.
- Evaluation can report per-kind metrics and budgets cleanly.
- No hidden execution path for malicious graph authors in the core.

**Negative**

- Some workflows need tools, retrieval, or code execution and cannot be expressed in v0.
- A subgraph boundary is heavier than an in-process function call; simple compositions pay
  serialization and mapping costs.
- Joiners are mostly pass-through until fan-out exists.
- Users coming from agent frameworks may expect arbitrary nodes and be disappointed.

**Neutral**

- The vocabulary can grow. A future `io` or `tool` kind is a deliberate extension, not a
  breaking change to existing graphs.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| One universal node with a `type` string | Blurs contracts and makes validation and scheduling a bag of conditionals. |
| Python/JS plugin nodes | Security, reproducibility, and evaluation complexity; turns the project into an agent runtime. |
| Separate `classifier` node kind | A router can be implemented by a classifier adapter; a separate kind adds no semantics. |
| Implicit recursion from any node | Unbounded, hard to trace, and impossible to validate statically. |
| Treat subgraphs as textual includes | Loses input/output contracts, version pinning, and boundary scoping. |

## Follow-ups

- [T-0003](../../.scratch/tickets/T-0003-graph-compiler.md) implements the validator rules
  for node kinds.
- [T-0009](../../.scratch/tickets/T-0009-worker-joiner.md) implements worker/joiner executors.
- [T-0017](../../.scratch/tickets/T-0017-subgraphs.md) implements subgraph boundaries.
- Revisit tool/IO nodes only after v0 with a threat model and a sandboxing plan (`Q-004`).
