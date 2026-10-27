# ADR-0001: Language is the inter-node wire format

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [protocol.md](../protocol.md), [vision.md](../vision.md), ADR-0004, ADR-0007

## Context

The system must connect models that were trained independently, possibly by different
people, on different hardware, with different architectures and tokenizers. The connection
could be:

- shared hidden states, embeddings, or KV caches;
- logits or probability distributions;
- tool-call/JSON protocol;
- natural language (plus code and structured text).

Anything beyond language requires the models to share architecture, tokenizer, and often
training. That would force every contributor to adopt one base model and one training
stack, recreating the centralization the project exists to avoid. It would also make
nodes non-portable across providers.

## Decision

**Natural language is the payload crossing an edge. A typed envelope is the frame. A
rendered prompt is what each model sees.**

Concretely:

1. Nodes exchange named **artifacts** whose content is UTF-8 text with a semantic type
   (`text`, `code`, `patch`, `plan`, `json`, `report`).
2. No hidden state, logits, embeddings, or KV cache crosses a node boundary in v0.
3. The runtime may pass structured metadata (routing decisions, usage, provenance) alongside
   language, but the model-visible payload is rendered text.
4. Each handoff is compressed deliberately: nodes receive a **scoped view**, not the full
   transcript. Compact artifacts are preferred over raw chain-of-thought.
5. The design accepts the bandwidth limit as a first-class constraint and measures handoff
   loss rather than pretending it away.

## Consequences

**Positive**

- Any model with a chat interface can participate. No shared tokenizer or architecture.
- A node can be hosted anywhere: local GPU, another machine, a hosted API.
- Handoffs are human-readable, auditable, and usable as training data without decoding.
- The graph is inspectable with ordinary text tools and debuggable by a person.

**Negative**

- **Low bandwidth.** Language carries far less information than a shared latent state. Some
  tasks cannot be decomposed this way without loss.
- **Token cost.** Re-sending context at each hop is expensive; the envelope must be scoped
  aggressively.
- **Semantic drift.** A worker may misunderstand the router's intent; a joiner may receive
  inconsistent artifacts.
- **Lossy compression.** Summaries and typed artifacts discard information that a
  monolithic model would have kept internally.
- **Prompt injection surface.** Any text crossing an edge can carry adversarial content.

**Neutral**

- The design is compatible with future latent handoffs between nodes that share a base
  model, but that is not v0 and would require a new ADR.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Shared embeddings / hidden states | Requires a shared encoder or alignment training; breaks provider neutrality and independent ownership. |
| Shared KV cache | Requires identical tokenizer and model; brittle and security-sensitive; no portability. |
| Logit-level routing | Could be powerful for a single model family, but cannot route across independent models and is opaque. |
| Tool-call/JSON-only protocol | Too rigid for open-ended language work and still needs a text field; rejected as the *only* medium, retained as a structured artifact type. |
| Raw transcript passing | Simple, but token cost and context dilution make it scale poorly and leak unrelated information between branches. |

## Follow-ups

- Measure handoff loss: compare worker quality with a full vs. scoped view
  (T-0020).
- Revisit latent communication only for same-base-model nodes, behind a versioned protocol
  and an ADR, after v0 evidence.
- Keep artifacts small and named; add a semantic-compression guideline when real traces
  show where tokens are wasted.
