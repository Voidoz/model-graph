# GPU-resident execution

This document answers a specific performance question:

> Can all models, and the handoff between them, live on the GPU so a configured graph runs
> as if it were a single model?

**Short answer:** yes for *runtime overhead*; not for *literal fusion*. A graph can eliminate
HTTP, process hops, repeated weight loading, and re-tokenization. It cannot eliminate the
fact that every node is a separate model invocation with its own prefill and KV cache, unless
the nodes share a base model and tokenizer — and even then, each node still performs a
forward pass. The design therefore treats GPU residency as an **execution backend**, not as a
new graph semantics.

See [ADR-0011](adr/0011-gpu-resident-backends.md).

## 1. Three meanings of "single model"

| Meaning | Achievable? | Notes |
| --- | --- | --- |
| **Logical equivalence** — the graph behaves the same as the portable backend | Yes, required | Same decisions, artifacts, budgets, traces. A backend swap must not change semantics. |
| **Runtime equivalence** — no weight reload, no HTTP, no process hop, no re-tokenization | Mostly | Resident heterogeneous models remove reloads and HTTP; fused multi-LoRA also removes re-tokenization. |
| **Physical equivalence** — one forward pass, one weight set, one KV stream | No (except true MoE/merging) | Each node is still an invocation. Independent models cannot share KV. A fused adapter graph shares weights but not one continuous forward pass. |

The project should only claim the first two, and only after [T-0033](../.scratch/tickets/T-0033-transition-benchmark.md)
measures them.

## 2. Where the time actually goes

A transition from node A to node B contains several costs with very different magnitudes:

| Stage | Typical order of magnitude | Can a GPU-resident backend remove it? |
| --- | --- | --- |
| HTTP request/response and JSON marshal (local server) | ~1–5 ms | Yes, by running in-process |
| Python/framework dispatch and synchronization | sub-ms to a few ms | Reduced, not always zero |
| Detokenize A's output, build B's prompt, tokenize | ~sub-ms to a few ms for typical prompts | Fused same-tokenizer path: yes, keep token IDs |
| Host-to-device transfer of tokens | negligible (kilobytes) | Yes, but it was never the problem |
| **Prefill of B's prompt** | **tens to hundreds of ms** for ~1k tokens on consumer hardware | **No**; this is model compute, and it repeats per node |
| Decode of B's output | proportional to output length | No; it is the task |
| Weight load/swap if B is not resident | **seconds** | Yes, by pinning all models in VRAM |
| Adapter switch in a multi-LoRA engine | small, but not zero | Reduced, not eliminated |
| KV cache allocation / re-prefill | can be large | Partially, via prefix caching; not across different models |

The counterintuitive result: **moving text between models is cheap; making each model read
the conversation again is expensive.** Removing HTTP is worthwhile, but it will not make a
graph feel like one model if the graph passes long prompts through many hops.

Two consequences drive the design:

1. Keep prompts short and artifacts compact.
2. Reuse prefixes where possible, and do not pretend that different models can share KV.

## 3. Backend tiers

| Tier | Name | What is resident | Handoff | Best for |
| --- | --- | --- | --- | --- |
| **B0** | Portable providers | One model at a time, usually via HTTP | UTF-8 text, tokenized per call | Heterogeneous models, remote APIs, portability baseline |
| **B1** | Resident heterogeneous | All models in the graph, in one process, pinned in VRAM | Text/token bridge on host; in-process calls | A deployment set that fits on one GPU and uses different model families |
| **B2** | Fused same-base multi-LoRA | One base model + all node adapters + one tokenizer | Token IDs stay on device; adapter switch | Graphs whose specialists are LoRA adapters on one base; the closest to a single-model feel |
| **B2+** | Compiled fused | B2 + pre-tokenized templates + prefix KV + fast/flattened routes | Token IDs; compiled decisions | Latency-critical shared-base graphs with stable prompts |
| **B3** | Latent handoff (research) | Shared representation or trained projectors | Hidden states / soft prompts | Not v0; requires a new ADR and co-designed models |
| **B4** | Distilled single model | One student model | None at runtime | Stable high-volume hot paths; the only target that exactly matches a dense LLM's runtime shape |

B0 is the default and the portability guarantee. B1, B2, and B2+ are opt-in deployments selected
by a profile. B4 is a **compiled deployment target** ([ADR-0012](adr/0012-compiled-deployment-targets.md)):
the graph is used for authoring/training/evaluation, and a student model imitates it at runtime.
It gives up runtime modularity for single-model speed. B3 is deliberately out of scope: it
would break independent model ownership unless the models were trained to share a latent space.

## 4. B1 — Resident heterogeneous backend

Goal: keep every model used by a graph loaded and run calls in-process, without changing the
logical graph.

Requirements:

- **One process, many runners.** The scheduler calls a model runner directly. No loopback
  HTTP, no JSON marshaling of prompts.
- **Load once, keep resident.** Weight loading happens at backend `load()`, not per call.
  Cold-start time is measured and reported separately.
- **Memory admission.** Before loading, estimate:
  ```text
  vram_needed = Σ model_weights
              + KV_cache_budget
              + workspace/activations
              + CUDA graph captures
              + allocator fragmentation margin
  ```
  If the plan does not fit, the backend must either select a smaller/quantized compatible
  variant, prune the graph (T-0015), or fail with an actionable report. **Silent weight
  swapping or CPU offload is forbidden** because it destroys the latency property the
  backend exists to provide.
- **Text bridge.** Different model families usually have different tokenizers, so the
  handoff is detokenize → build prompt → tokenize. This is acceptable; it is not the dominant
  cost. Record its time separately.
- **Per-model prefix cache.** Cache system prompts and stable prefixes per model where the
  runner supports it. Do not attempt cross-model KV reuse.
- **Concurrency.** A single active branch has no GPU concurrency to exploit. Fan-out may
  overlap independent calls, subject to memory and T-0018.
- **Fallback.** If residency is infeasible, fall back to B0 explicitly, record the reason,
  and report the latency class. No silent degradation.

## 5. B2 — Fused same-base multi-LoRA backend

This is the path that can genuinely feel like one model, because most of the system shares one
set of base weights and one tokenizer.

Shape:

```text
                 one base model (resident)
                 one tokenizer (shared)
                 one engine / KV cache pool
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
   router adapter    JS adapter         general-coder adapter
   (or classifier)   (LoRA)             (LoRA)
        │                 │                  │
        └───────── tokens stay on device ────┘
```

What improves:

- **No weight reload.** The base is loaded once; adapters are resident or paged with a known,
  small switch cost.
- **No host text round-trip** when both nodes use the same tokenizer: the next prompt can be
  assembled from token IDs and template fragments already on the device.
- **No re-tokenization** on the fast path; tokenization happens once for the user request.
- **Shared prefix caching** per adapter for system prompts and stable instructions.
- **One engine** can batch, schedule, and manage KV more efficiently than a pool of servers.

What does **not** improve:

- **Each node still prefills its own prompt.** The router and the worker each perform a
  forward pass. The adapter changes the computation; KV computed under one adapter is not
  valid under another.
- **The base model constrains the experts.** Nodes must be adapters compatible with the same
  base, tokenizer, and context window. A best-in-class external model cannot join this
  backend.
- **Adapter capacity is finite.** LoRA rank and training data limit specialization. This is a
  quality trade-off, not only a performance one.
- **Artifacts still need materialization.** The contract is language; the trace and evaluator
  need text (or a deterministic detokenization) at artifact boundaries. Keeping tokens on
  device is an internal optimization, not a new artifact type.

### 5.1 Compatibility contract

A graph may run on B2 only if every node's model binding declares:

- the same base model manifest hash;
- the same tokenizer identity;
- compatible context limits and special-token handling;
- an adapter artifact with an integrity hash.

The backend must fail closed when these do not match. It must not silently substitute a
different base or drop an adapter.

### 5.2 Router in a fused graph

The router can be:

- a LoRA adapter that emits a constrained route token or JSON;
- a small classification head over the base model's hidden state;
- a separate tiny embedding/classifier model, if the extra latency is acceptable.

A full generative router model defeats much of the purpose. The choice is measured in T-0033,
not assumed.

## 6. Runtime interface

The backend sits *under* the scheduler and executors, above individual runners. It does not
make routing decisions and does not parse artifacts.

```python
class ExecutionBackend(Protocol):
    def plan(self, graph_ir, profile) -> BackendPlan: ...
    def load(self, plan: BackendPlan) -> "LoadedBackend": ...
    def invoke(self, node_id: str, view: ScopedView, params: GenerationParams) -> BackendResponse: ...
    def metrics(self) -> BackendMetrics: ...
    def close(self) -> None: ...
```

`BackendPlan` contains:

- mode (`portable`, `resident`, `fused`);
- resolved models/adapters and manifest hashes;
- estimated VRAM and whether admission passed;
- compatibility warnings/errors;
- fallback decision and reason;
- a content hash for the run manifest.

`BackendResponse` normalizes text, token usage, latency, and optional token IDs. A
token-native backend may expose an additional fast path, but the executor must still produce
the declared artifacts and the trace must store their text (or a deterministic
detokenization reference).

Backend selection is recorded in the run manifest and cannot change silently between replay
and original execution.

## 7. Memory and KV management

- **Admission before load.** Compute the plan against a profile's VRAM ceiling. Reserved
  memory, KV block size, and adapter overhead must be included.
- **KV budget.** Per-token KV size depends on layers, KV heads, head dimension, dtype, and
  context. Use the engine's reported capacity where possible; otherwise estimate
  conservatively.
- **Prefix caching.** Key by `(base_hash, adapter_hash, token_prefix)`; never reuse a block
  across a different adapter or base.
- **No silent paging.** If memory is exhausted during a run, fail with the budget and plan
  attached. A slow fallback can be offered by a profile, but it must be explicit and recorded.
- **Unload policy.** LRU unloading is allowed only for models not in the active profile's
  resident set; unloading a resident model turns a latency guarantee into a surprise.
- **Fragmentation.** Leave headroom; multi-model residency is more sensitive to allocator
  fragmentation than one model.

## 8. Measurement

The only way to justify a backend is measurement. Required metrics:

| Metric | Definition |
| --- | --- |
| **Cold load time** | Time to load weights/adapters before the first call |
| **TTFT** | Request start to first output token |
| **Inter-token latency (ITL)** | Steady-state decode time per token |
| **Transition latency** | Last token of node A to first prefill token of node B, split into CPU, transfer, adapter switch, and prefill |
| **Runtime overhead ratio** | Non-model wall time / total wall time |
| **GPU idle ratio** | Fraction of wall time with no kernel executing (best effort) |
| **VRAM peak** | Actual peak during a run, plus the admission estimate |
| **Weight reload count** | Must be zero after `load()` for a resident plan |
| **Logical parity** | Decisions/artifacts equal to B0/B1 under scripted responses |

Initial targets (to be confirmed, not claimed):

- **B1:** transition overhead excluding prefill p95 < 15 ms; zero weight reloads; zero
  unexpected OOMs in the supported matrix; explicit failure when admission fails.
- **B2:** token-native path avoids host detokenize/retokenize; transition excluding prefill
  p95 < 5 ms; TTFT within 1.25× of a single-model baseline on the same base at equal output;
  artifact/decision parity with B1 within the documented tolerance.
- **Both:** runtime overhead measured and reported for every benchmark; no "single model"
  claim without a paired comparison.

These numbers are hypotheses. They will be accepted or revised by T-0033.

## 9. What still costs what

Even with a perfect backend:

- A router pass adds TTFT. A classifier/head is cheaper than a generative router, but not
  free.
- A worker pass re-reads its prompt. Prefix caching helps only where the prefix is shared.
- Long artifacts passed through many hops multiply prefill cost. The architecture's
  semantic-compression discipline matters more than the transport.
- Different model families force a host-side text bridge. This is small but measurable.
- Quantization reduces weight memory but changes quality; it must be evaluated, not assumed.
- A graph with many sequential hops will feel like many sequential calls, because it is.

## 10. Recommendation

1. Keep the logical graph, artifact, budget, and trace contracts unchanged.
2. Add the backend abstraction now; B0 is the reference implementation.
3. Build **B1** first: it proves that model residency and in-process execution work, and it
   supports heterogeneous experts.
4. Build **B2** for one reference base model with LoRA adapters: it is the realistic "single
   model feel" path and aligns with independent adapter training.
5. Add **B2+ compilation** (AOT prefixes, fast/flattened routing) and **cross-request
   batching**. These are the two biggest remaining levers after residency.
6. Offer **B4 distillation** for stable high-volume hot paths. It is the only target that
   exactly matches a dense LLM's runtime shape.
7. Gate every performance claim on T-0033/T-0037 reports. Do not put "runs like a single
   model" in the README before the paired report exists.

The full option comparison, overhead budget, and parity gates are in
[performance.md](performance.md).

## 11. Roadmap

| Ticket | Deliverable |
| --- | --- |
| [T-0031](../.scratch/tickets/T-0031-resident-backend.md) | Backend abstraction + resident heterogeneous spike |
| [T-0032](../.scratch/tickets/T-0032-fused-multilora.md) | Fused same-base multi-LoRA backend |
| [T-0033](../.scratch/tickets/T-0033-transition-benchmark.md) | Transition and LLM-parity benchmark, gates |
| [T-0034](../.scratch/tickets/T-0034-aot-plan-compiler.md) | AOT plan, prompt/tokenization, prefix KV, memory plan |
| [T-0035](../.scratch/tickets/T-0035-fast-routing.md) | Classifier/head routing, flattening, route cache |
| [T-0036](../.scratch/tickets/T-0036-batched-scheduler.md) | Multi-request scheduler and continuous batching |
| [T-0037](../.scratch/tickets/T-0037-graph-distillation.md) | Distilled single-model deployment target |

Phase 5 is post-v0: it must not delay the usable local release. It is the performance track
that makes the "single model" experience measurable. Not every graph should run on every target;
the compiler and the benchmark decide.
