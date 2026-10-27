# ADR-0005: Compile-time graph, runtime routing

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [graph-model.md](../graph-model.md), [execution.md](../execution.md), ADR-0006, ADR-0008

## Context

Two broad designs are possible:

1. **Static pipeline.** The graph topology is fixed before execution; every branch's
   existence is known and validated. The only dynamic choice is which branch to take.
2. **Dynamic graph.** Nodes can create, delete, or rewire nodes at runtime, like an agent
   that writes its own workflow.

Dynamic graphs are flexible but make termination, cost, security, and evaluation very hard.
Static graphs are analyzable but may feel rigid for open-ended tasks.

## Decision

**The graph is declared and compiled before execution. Routing decisions happen at
runtime.**

- Authors declare nodes, edges, inputs, exits, defaults, and a fallback in YAML.
- The compiler validates the whole topology (acyclicity, reachability, artifact
  availability, budgets) and produces an immutable Graph IR.
- At runtime, a router chooses among *existing* outgoing edges. It cannot create a node,
  name a target outside its menu, or change the topology.
- The runtime may prune unreachable/unsupported branches under a profile, but the authored
  graph is unchanged and the pruning is reported.
- The set of possible paths is finite and enumerable at compile time; the actual path is
  recorded in the trace.

## Consequences

**Positive**

- Termination and cost bounds are analyzable. A loop is a validation error, not a runtime
  surprise.
- The whole graph can be evaluated offline, including oracle routes over candidate paths.
- Security review is tractable: the set of nodes and capabilities is known before a run.
- Traces are comparable across runs because the topology is stable.
- Subgraphs compose without giving up these properties.

**Negative**

- The system cannot invent a new specialist at runtime or rewrite its own workflow. Some
  agentic tasks are out of scope.
- Adding a branch requires a graph edit, validation, and an eval run.
- A poorly designed topology (e.g., an under-specified router) cannot be rescued by runtime
  improvisation.
- Pruning changes the runnable topology, so the run manifest must record the effective
  graph, not just the authored one.

**Neutral**

- Dynamic *selection* is still rich: routers, confidence, fallback, and later fan-out all
  happen within the declared topology.
- A future "graph synthesis" tool could propose topology edits offline; it would not change
  the runtime guarantee.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Fully dynamic graph (agents create nodes) | Unbounded cost/latency, hard to validate, security nightmare, evaluation nearly impossible. |
| Hard-coded pipeline per task | Loses reuse and the ability to swap experts; no graph contract to standardize. |
| Runtime graph mutation with a cap | Still requires a policy for new capabilities, permissions, and evaluation; not worth v0 complexity. |
| Declarative graph with no runtime choice (DAG execution only) | Loses the central idea: a decider choosing an expert. |

## Follow-ups

- T-0003 implements compile-time checks.
- T-0010 implements the runtime walk.
- T-0015 implements profile pruning
  and records the effective graph.
- If dynamic topology is ever pursued, it needs an ADR covering termination, permissions,
  cost, and evaluation.
