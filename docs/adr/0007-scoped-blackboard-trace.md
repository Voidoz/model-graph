# ADR-0007: Scoped blackboard and content-addressed trace

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [protocol.md](../protocol.md), [execution.md](../execution.md), ADR-0001

## Context

Nodes need shared state: the request, artifacts produced by earlier nodes, and routing
context. There are two naive extremes:

1. **Pass the full transcript to every node.** Easy to implement and preserves context, but
   token cost grows with each hop, unrelated branches leak into each other, privacy suffers,
   and small models are easily distracted.
2. **Pass only the immediate predecessor's output.** Cheap, but a joiner or a nested
   specialist may need an artifact produced several steps earlier, and re-deriving it is
   wasteful and lossy.

Debugging and training add another requirement: the system must be able to reconstruct what
each node saw and why each route was chosen, without storing a giant mutable blob that can
be corrupted by later steps.

## Decision

**Use a request-scoped blackboard (the envelope) with explicit per-node scoped views, plus
an append-only, content-addressed trace.**

- The envelope holds inputs, artifacts, notes, path, budget, and provenance for one run.
- A node sees only what its `reads` declares, the request, its own instruction, and (for a
  router) its menu. It never sees the full trace or unrelated artifacts.
- Artifacts are immutable once written; a later write with the same name shadows but does
  not mutate the earlier one. Shadowing is recorded.
- The trace is append-only JSONL. Events reference artifact and prompt content by SHA-256
  hash; blobs are stored separately and deduplicated.
- Rendered prompts are hashed; full prompts are optional and redacted.
- Replay reads the trace and requires a matching IR hash unless explicitly overridden.
- Subgraph boundaries map a subset of artifacts in and out; everything else stays inside the
  child trace.

## Consequences

**Positive**

- Token cost is bounded by the node contract, not by transcript length.
- Branch isolation: a coding node cannot accidentally see a story artifact unless declared.
- Privacy: less data crosses nodes and trace storage can be minimized by keeping hashes.
- Exact reconstruction of a run for debugging and evaluation.
- The trace becomes a clean dataset: one record per node call with its exact input.
- Deduplication saves disk when the same artifact is read by many nodes.

**Negative**

- Authors must design `reads` carefully; an omitted read causes a hard failure at runtime.
- Debugging sometimes requires following artifact names across events rather than reading a
  single transcript.
- Hashing and blob storage add implementation complexity and filesystem overhead.
- Redaction can make replay of sensitive runs impossible by design; that is a feature but
  must be documented.
- Shadowing names can surprise authors; the validator only catches some cases.

**Neutral**

- The envelope is an internal contract; models never see it raw. It can evolve with the
  schema version.
- The trace format is deliberately boring JSONL so external tools and AI workflows can
  consume it without a custom library.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Full transcript to every node | Cost scales with path length; privacy and distraction problems; defeats scoping. |
| Predecessor-only handoff | Breaks joins and multi-step dependencies; forces re-derivation. |
| Mutable global state without versioning | Cannot reconstruct what a node saw; replay and evaluation become unreliable. |
| Database-first trace | Adds operations burden for a local-first tool; JSONL is portable and inspectable. |
| Store all content inline in the trace | Simple, but duplicates large artifacts and complicates redaction and dedup. |

## Follow-ups

- T-0006 implements the envelope and
  artifact store.
- T-0011 implements trace capture and replay.
- T-0029 implements redaction and
  secret scanning.
- Measure shadowing and missing-read rates in real graphs; if they are common, improve the
  validator rather than loosening the runtime.
