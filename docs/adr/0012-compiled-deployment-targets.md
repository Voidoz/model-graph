# ADR-0012: Graphs compile to deployment targets, including a distilled single model

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [performance.md](../performance.md), [gpu-execution.md](../gpu-execution.md),
  ADR-0001, ADR-0005, ADR-0006, ADR-0010, ADR-0011

## Context

The graph is a powerful authoring and training abstraction, but at runtime it pays costs a
monolithic LLM does not:

- a routing pass before work starts;
- a prefill for every node, even when prompts are short;
- decoded intermediate artifacts that a dense model would have produced internally;
- host marshaling and adapter switching;
- lower batching efficiency when requests scatter across experts.

ADR-0011 removes transport and residency overhead, but it explicitly does not remove per-node
prefill. If the project wants to be a serious contender against dense LLMs on latency and
throughput — not only cost and ownership — it needs a stronger answer than "make the same
graph faster."

The user's requirement is explicit: allow preprocessing and compilation. That changes the
design space. The graph does not have to be the runtime artifact.

## Decision

**The graph is source. The runtime executes a compiled deployment target chosen for the
workload. One valid target is a distilled single model.**

1. **Deployment targets.**
   - **Interpreted (B0):** the portable node-by-node runtime; reference semantics.
   - **Resident (B1):** all models resident and in-process.
   - **Fused (B2):** one base model + per-node LoRA adapters.
   - **Compiled fused (B2+):** B2 plus ahead-of-time prompt/tokenization, prefix-KV
     compilation, route flattening/classification, route caching, memory planning, and
     engine configuration.
   - **Distilled (B4):** a single student model trained to reproduce the graph's behavior on a
     target distribution. Used for stable high-volume hot paths.
2. **Compilation is explicit and versioned.** A compiled artifact records compiler version,
   graph hash, model/adapter hashes, target, feature flags, and the eval report. It is
   content-addressed and invalidated when its inputs change; a stale plan must not be reused
   silently.
3. **AOT optimizations are allowed and expected.** Pre-tokenize static prompts, precompute
   prefix KV where the model/adapter permits it, flatten separable routing hierarchies, use a
   classifier/head as the default fast router, cache route decisions for repeated/session
   requests, and plan memory/CUDA graphs ahead of the first request.
4. **Distillation is a legitimate deployment target, not a betrayal of modularity.** The graph
   remains the authoring, training, evaluation, and customization layer. B4 trades runtime
   modularity for single-model speed. It must pass the same quality gate as any other target,
   and the source graph remains runnable for debugging, rare routes, and re-distillation.
5. **Runtime observability is target-aware.** B4 traces record the target, student model hash,
   input/output, and eval evidence; they do not pretend to have per-node decisions. Debugging a
   B4 regression falls back to the graph target. B2+ traces record the compiled route plan and
   prefix-cache use.
6. **No latent cross-model handoff.** B4 is one model; B1/B2 remain language-based. This ADR
   does not reopen ADR-0001.
7. **Performance claims are gated.** The parity metrics and budgets in
   [performance.md](../performance.md) are required. No "faster than an LLM" claim without a
   quality-matched, hardware-pinned benchmark.
8. **Batching is part of the target runtime.** A multi-model graph that processes one request
   at a time cannot credibly compete on throughput. The compiled/runtime target must support
   per-model queues and cross-request continuous batching.

## Consequences

**Positive**

- The system can target different points on the latency/throughput/cost/modularity frontier
  without changing the graph source.
- AOT compilation removes most avoidable per-request overhead.
- Fast routing removes the most obvious TTFT penalty.
- B4 can match a dense LLM's runtime shape exactly: one model, one forward pass.
- Modularity survives where it matters: authoring, contribution, training, evaluation, and
  regeneration.
- The project can make honest, workload-specific performance claims.

**Negative**

- The compiler becomes a substantial component with its own correctness and versioning burden.
- B4 loses runtime modularity: you cannot swap a branch inside the student without retraining.
- Distillation can lose rare-route quality, inherit teacher bias, and go stale as the graph
  changes.
- Compiled artifacts can become stale or invalid if invalidation is imperfect.
- More targets mean more test matrices and more ways to misconfigure a deployment.
- Multi-request batching adds scheduler, fairness, and isolation complexity.
- The project must resist claiming universal parity; the truth is workload-specific.

**Neutral**

- The graph remains the canonical source; B0 is always available as the reference.
- Model merging is a possible cheap target for static experts but is not adopted here.
- Speculative/pipelined handoff remains a research option, not a target.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Only optimize the interpreted graph | Removes transport but not prefill, routing, or batch loss; cannot reach LLM-class throughput. |
| Only fused multi-LoRA | Strong middle ground, but still pays per-node prefill and constrains all experts to one base. |
| Treat MoE as the answer | Excellent single-model execution, but requires joint training and centralizes expert ownership; it is a competitor, not a deployment target for independently trained nodes. |
| Merge weights into one model | Cheap inference but lossy, unattributable, and no per-node rollback. |
| Latent handoff | Requires co-designed/shared representation; breaks portability. |
| Declare graph execution "as fast" without compilation | False. The overhead is real and measurable. |
| Distill everything | Loses modularity and rare-route quality; not every workload is stable enough. |

## Follow-ups

- T-0034: AOT plan, prompt/tokenization,
  prefix KV, memory plan.
- T-0035: classifier/head routing, flattening,
  route cache.
- T-0036: multi-request scheduler and
  continuous batching.
- T-0037: B4 student training and
  promotion.
- T-0033: LLM-parity benchmark and
  gates (expand to dense/MoE/cascade baselines).
- Revisit `Q-016` (distillation acceptance/hybrid fallback) before B4 promotion.
