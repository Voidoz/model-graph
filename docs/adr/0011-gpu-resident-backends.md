# ADR-0011: GPU-resident execution is an optional backend, not new semantics

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [gpu-execution.md](../gpu-execution.md), [performance.md](../performance.md), ADR-0001, ADR-0002, ADR-0009, ADR-0010, ADR-0012

## Context

The portable design moves language between nodes through provider adapters, often over a
local HTTP API. That is portable and provider-neutral, but it has avoidable runtime overhead:

- HTTP/JSON round-trips and process scheduling between nodes;
- detokenizing one model's output and re-tokenizing it for the next;
- reloading or swapping model weights if not resident;
- framework dispatch and synchronization gaps.

A user reasonably asks: can all models and the handoff live on the GPU, so a configured graph
runs as if it were a single model?

The expensive part of a handoff is not the transport. It is **per-node prefill and KV
recomputation**. Independent models have different weights, tokenizers, and KV caches; they
cannot share a forward pass. A shared-base adapter graph can share weights and tokenizer, but
each node still performs its own forward pass. Anything beyond that is latent communication,
which [ADR-0001](0001-language-as-the-wire.md) deliberately excludes.

## Decision

**GPU residency is an optional execution backend with two modes. The logical graph, language
artifacts, routing decisions, budgets, and traces do not change.**

1. **Introduce an `ExecutionBackend` contract** between the scheduler/executors and model
   runners. The portable provider stack is backend **B0** and remains the default.
2. **B1 — resident heterogeneous backend.** Load all models in a graph in one process and pin
   them in VRAM; invoke them in-process; bridge between different tokenizers with text on the
   host; cache prefixes per model. Weight reloads after `load()` are forbidden.
3. **B2 — fused same-base multi-LoRA backend.** Use one resident base model, one tokenizer,
   and per-node LoRA adapters. Pass token IDs on-device on the fast path; switch adapters per
   node; materialize artifact text at boundaries for trace/evaluation. This is the realistic
   "single model feel" path.
4. **No latent handoff.** Hidden states, soft prompts, or cross-model KV sharing require a
   new ADR and a co-design of the models. B2 shares tokenizer and base weights, not latent
   state.
5. **Backend selection is explicit.** A run profile may request a backend; the compiler
   checks graph/model compatibility and VRAM admission. If the plan does not fit or is
   incompatible, the runtime either fails with an actionable report or falls back to B0
   explicitly. Silent paging, offload, or model substitution is forbidden.
6. **Logical parity is mandatory.** Every backend must pass the same conformance suite:
   identical routing decisions and artifact text for scripted responses, identical budget
   accounting, and replayable traces.
7. **Measure before claiming.** Backends report cold load, TTFT, inter-token latency,
   transition latency by phase, overhead ratio, VRAM peak, and reload count. No "runs like a
   single model" claim without a paired report (T-0033).
8. **Artifacts remain language.** A token-native backend may avoid host text internally, but
   it must produce the declared text artifacts (or a deterministic detokenization reference)
   for the trace, evaluation, and caller.

This ADR does not supersede ADR-0001. Token IDs are an encoding of language, not a hidden
latent state. KV caches stay inside the engine serving one model/adapter set; they do not
cross a graph boundary as a new artifact type.

## Consequences

**Positive**

- The runtime can eliminate most non-model overhead: HTTP, serialization, weight reloads,
  and re-tokenization.
- Heterogeneous experts remain possible in one process (B1).
- A shared-base adapter graph can behave like one model in wall-clock terms (B2), with one
  engine and one tokenizer.
- Adapters are independently trainable, which strengthens the community ownership story.
- The same graph can be deployed portably or resident without rewriting it.
- Performance claims become measurable and comparable.

**Negative**

- VRAM becomes a hard admission constraint; a graph may not fit even when its logic is valid.
- B2 constrains all nodes to one base model and tokenizer, trading model diversity for speed.
- Adapter capacity and quality may be lower than a heterogeneous specialist.
- In-process runners tie the project to inference frameworks and their APIs.
- Token-native paths add a second execution path that must be kept logically identical.
- Multi-model residency increases fragmentation and OOM risk.
- The word "single model" is easy to misuse; the docs must keep the three-level distinction.

**Neutral**

- B3 (latent handoff) remains a research direction, not a roadmap commitment.
- A true mixture-of-experts model is a separate comparison, not an implementation of this
  contract.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Keep HTTP-only execution and optimize prompts | Portable, but cannot remove weight reloads, process hops, or re-tokenization; leaves the performance question unanswered. |
| One monolithic model with a single prompt | Fast and simple, but abandons independent experts, routing, pruning, and node ownership. |
| True MoE trained jointly | Strong single-model execution, but requires joint training and a shared router; incompatible with independently contributed experts in v0. |
| Latent handoff between models | Best theoretical bandwidth, but requires shared dimensions/tokenizers or trained projectors; breaks the independence premise. |
| Merge experts into one weight set | Cheap at runtime but lossy, hard to evaluate, and makes per-node rollback impossible. |
| Silent CPU offload / weight paging | Preserves "it runs" but destroys the latency property; hides a performance failure. |

## Follow-ups

- T-0031: backend contract and B1 spike.
- T-0032: B2 and adapter manifest support.
- T-0033: transition measurement and
  gates.
- B4 (distilled single-model target) is defined in [ADR-0012](0012-compiled-deployment-targets.md); it is not a multi-model GPU backend.
- Revisit B3 only after B1/B2 data and with a new ADR.
- Resolve `Q-014` (which in-process runners) and `Q-015` (GPU platform support) before
  promising cross-platform residency.
