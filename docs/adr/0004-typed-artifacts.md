# ADR-0004: Semantic artifact types on a text substrate

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [graph-model.md](../graph-model.md), [protocol.md](../protocol.md), ADR-0001

## Context

If every edge carries untyped text, the compiler cannot check whether a worker's output is
the kind of thing the next node expects, evaluation cannot report validity by type, and
routers lack a useful signal about what a branch produces. If every edge carries a rich
schema, small models fail at the boundary, prompts become brittle, and the flexibility of
language is lost.

The project needs a middle ground: enough typing to validate and evaluate, not so much that
models must emit perfect structured data.

## Decision

**Artifacts have a small semantic type, but all content is text.**

Types in v0: `text`, `code`, `patch`, `plan`, `json`, `report`.

- A node declares what it writes: `{name, type, media_type?}`.
- An edge may declare the expected artifact type.
- Compatibility is permissive: a consumer expecting `text` accepts anything; a consumer
  expecting a specific type accepts that type (and the compiler warns on a mismatch rather
  than failing, because models sometimes return a plan when asked for code).
- Types are **semantic hints for validation and evaluation**, not a security boundary.
  `json` is parsed opportunistically; it is not trusted.
- Rich schemas are allowed inside a `json` artifact as an evaluation concern, not a
  transport requirement in v0.
- Adding a type requires an ADR and a schema change. The list stays small on purpose.

## Consequences

**Positive**

- The validator can catch "this edge promises code but the producer writes a plan" at
  compile time (as a warning) and at runtime (as a recorded mismatch).
- Evaluation can report validity and quality per type, and datasets can be sliced by type.
- Routers can use the produced type as a routing signal.
- The wire stays text, so any model can participate.

**Negative**

- The types are coarse; they cannot express "a TypeScript module with an exported function
  and no imports" without a schema.
- Warnings about mismatches may be ignored, reducing the value of typing.
- Model behavior can drift across types (asking for `patch` may produce `code`), and the
  runtime records rather than fixes it.
- A `json` artifact may be syntactically valid but semantically wrong; typing does not help.

**Neutral**

- A future optional `schema` field on writes could validate JSON artifacts more strictly
  without changing the wire.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Untyped text everywhere | No validation, weak routing cards, poor evaluation slices. |
| Full JSON Schema on every edge | Small models fail; prompt and parsing complexity; kills open-ended language. |
| MIME types as the primary type | Too many, too ambiguous for semantic routing; retained as a hint. |
| Separate channels per type (code channel, prose channel) | Complexity without a demonstrated need; the envelope can carry multiple artifacts instead. |

## Follow-ups

- [T-0002](../../.scratch/tickets/T-0002-contract-tests.md) pins the type enum and tests
  compatibility.
- [T-0009](../../.scratch/tickets/T-0009-worker-joiner.md) implements runtime type checks and
  mismatch events.
- Collect mismatch rates in real traces; if a type is consistently ignored, remove or
  redefine it rather than adding more.
