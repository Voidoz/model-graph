# Performance and LLM parity

This document asks the hard question:

> Can a graph of models match or beat a monolithic LLM on latency, throughput, cost, and
> efficiency — and if not, what can we change to get there?

It is a design analysis, not a benchmark report. It defines the target, decomposes the cost,
compares the viable architectures, and specifies the redesigns and gates that would make an
LLM-parity claim honest.

**The central strategic move:** treat the graph as **source code**, not necessarily as the
runtime. Compile it to the best deployment target for the workload: portable, resident, fused,
or a distilled single model. Modularity is preserved where it matters — authoring, training,
evaluation, and customization — while the hot path can collapse to one model.

See [ADR-0012](adr/0012-compiled-deployment-targets.md).

## 1. What "match an LLM" means

"LLM speed" is not one number. A claim of parity must name the metric and the workload:

| Metric | Why it matters | Parity condition (initial) |
| --- | --- | --- |
| **TTFT** (time to first token) | Perceived responsiveness; routing is visible here | ≤1.25× a quality-matched dense baseline for a compiled graph; ≤1.10× for a distilled model |
| **Inter-token latency (ITL)** | Streaming smoothness once generation starts | Within 10% of the active specialist's baseline |
| **End-to-end p50/p95** | Total task latency, including intermediate artifacts | ≤1.25× dense at matched quality |
| **Throughput** | Requests/tokens per GPU-second under load | ≥0.9× dense for compiled fused; ≥1.0× for distilled |
| **Cost per task** | The project's north star | ≤1.0× dense at matched quality |
| **VRAM** | Whether the thing can run at all | ≤1.5× dense for compiled fused; lower for distilled |
| **Runtime overhead ratio** | Non-model time / total time | <10% on the reference workloads |
| **Quality** | The claim is meaningless without it | Within the suite's confidence interval / agreed tolerance |

Two claims must stay separate:

1. **Same-model speed parity** — the graph's worker is as fast as a dense model on the same
   task. This is easy if the worker is smaller; it is not the interesting claim.
2. **Quality-matched parity** — at equal output quality on the same task distribution, the
   system is as fast or faster. This is the claim that matters, and the one that requires
   benchmarks.

A graph is not universally faster. It wins in some regimes and loses in others. The project
should aim for **Pareto dominance on decomposable, evaluable tasks**, not a universal speed
record.

## 2. The two timelines

Let:

- `n_i` = input tokens at node `i`
- `m_i` = output tokens at node `i`
- `P_i` = prefill time per input token for node `i`
- `D_i` = decode time per output token for node `i`
- `R_i` = routing/decision latency before node `i`
- `H_i` = handoff overhead (host marshaling, tokenization, transfer, adapter switch, queueing)
- `N` = number of model calls on the path

A dense baseline:

```text
T_dense = P_dense(n_task) + D_dense(m_answer)
```

A graph:

```text
T_graph = Σ_i [P_i(n_i) + D_i(m_i)] + Σ_i R_i + Σ_i H_i
```

The key difference is that the graph **pays for intermediate work**. Routers generate or
classify, workers generate artifacts, and joiners may generate again. Every intermediate token
is a real cost, not just overhead.

A useful first-order comparison is active parameter-token work:

```text
W_dense ≈ params_dense × (n_task + m_answer)
W_graph ≈ Σ_i active_params_i × (n_i + m_i)
```

If each specialist is much smaller than the dense baseline and the path is short, the graph can
do less work. If there are many hops or long intermediate artifacts, the graph can do more work
than the monolith, regardless of how good the GPU backend is.

### 2.1 Prefill is the tax

At batch 1, decode is often memory-bandwidth bound, while prefill is compute bound. That means:

- A smaller active model can decode faster in tokens/second.
- A long prompt can still make a small model slow, because prefill work scales with prompt
  tokens.
- Re-sending the request and context at each hop is the most expensive "transition" in
  practice.

This is why the design's artifact-compression rule is a performance rule, not just a taste
rule.

### 2.2 Batching changes the conclusion

For a single request, the graph is a sequential chain. For many concurrent requests, a serving
engine can batch calls to the **same model** across requests. The graph runtime should therefore
be a **multi-request scheduler with per-model queues**, not one request at a time:

- a router call from request A can batch with a router call from request B;
- worker calls to the same expert can batch, even if their requests arrived at different times;
- a request waiting on a slow expert must not block others;
- fairness and per-request budgets must remain enforceable.

Without cross-request batching, a multi-model graph will lose to a dense server on throughput
even when it wins on single-request latency. This is a required redesign, not an optimization
detail.

## 3. Overhead taxonomy

| Cost | What it is | Size | Removable? |
| --- | --- | --- | --- |
| **Routing compute** | A model call to choose a branch | 1–200+ ms depending on router type | Reduce with a classifier/head; eliminate at runtime with a compiled route table or distillation |
| **Duplicate prefill** | The same task/context is read by the router and the worker | Tens–hundreds of ms | Cannot be removed across different models; can be reduced by a cheap classifier and shared prefixes |
| **Intermediate artifact decode** | Tokens a node generates before the next node can start | Proportional to output | Reduce with compact artifacts and direct answers; not free |
| **Host marshaling/tokenization** | Detokenize, concatenate, tokenize, transfer | sub-ms to a few ms | Eliminate with token-native same-tokenizer execution; reduce with AOT templates |
| **Adapter switch** | Multi-LoRA weight activation | small, but non-zero | Keep adapters resident; batch; avoid per-token switching |
| **Queueing/scheduling** | Waiting for a model slot or event loop | Can be large under load | Multi-request scheduler, continuous batching, priority |
| **Batch loss** | Small batches caused by routing to different experts | Up to 2× or more | Cross-request batching; model/adaptor grouping |
| **Memory pressure** | Weights + KV competing for VRAM; paging/offload | Can dominate | Admission control, quantization, fewer resident models, multi-LoRA |
| **Distillation loss** | Quality lost by collapsing the graph to a student | Task-dependent | Eval gates; hybrid fallback; periodic re-distillation |

The table makes the priority clear:

1. **Reduce hops and intermediate tokens.**
2. **Make routing cheap.**
3. **Keep models resident and eliminate host text work.**
4. **Batch across requests.**
5. **If the workload is stable, distill the graph to one model.**

## 4. When a graph wins and loses

### 4.1 Win conditions

- The task decomposes into steps with short, structured artifacts.
- Each specialist is substantially smaller than the quality-matched dense baseline.
- The path is short (2–4 model calls) or the router can be compiled away.
- Contexts are short, or prefixes are shared/cached.
- The deployment reuses the same branch for many requests (route caching/session pinning).
- The serving engine batches across requests.
- The workload is customized to a domain where a specialist achieves equal quality with fewer
  active parameters.

### 4.2 Lose conditions

- Many sequential hops, each generating a long artifact.
- A generative router that adds a full model call to TTFT.
- Long, duplicated prompts at every hop.
- Heterogeneous models with different tokenizers and no resident weight pool.
- Single-request, single-batch serving with an idle GPU between calls.
- Large intermediate artifacts that a dense model would have produced internally without
  materializing.
- VRAM pressure forcing paging or a smaller, worse model.
- A quality-matched dense baseline that is already excellent and needs only one pass.

### 4.3 The honest boundary

A graph is a **modularity and specialization** play. It can beat a dense LLM on quality per
dollar, customizability, and per-node ownership. It beats it on latency only when the
decomposition reduces total work and the runtime overhead is small. The design should publish
both sides.

## 5. Options compared

Qualitative comparison; the symbols are a design vocabulary, not measured results. Every row
must be validated by the benchmark in §9.

Legend: `++` strong, `+` better than dense baseline, `0` comparable, `−` worse, `−−` much worse.

| Option | Model calls / pass | TTFT | Throughput | Cost/task | VRAM | Runtime modularity | Quality ceiling | Main risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Dense LLM** | 1 | baseline | `++` | `−−` | `−−` | `−−` | `++` | Cost, centralization |
| **MoE LLM** | 1 (sparse) | `+` | `++` | `+` | `−` | `−−` | `++` | Joint training, expert balance |
| **Cascade (cheap→expensive)** | 1–2 | `+` | `+` | `+` | `−` | `−` | `++` | Hard cases pay twice; router quality |
| **B0 interpreted graph** | N + routers | `−−` | `−` | `−`/`0` | `−` | `++` | `+` | HTTP, reloads, re-tokenization |
| **B1 resident heterogeneous** | N + routers | `−` | `0`/`+` | `0`/`+` | `−` | `++` | `++` | Prefill, tokenizer bridge, VRAM |
| **B2 fused multi-LoRA** | N + routers | `0`/`−` | `+` | `+` | `0`/`+` | `+` | `+` | Per-node prefill, base lock-in |
| **B2 + AOT + fast routing** | N, cheap routing | `0`/`+` | `+` | `+` | `0`/`+` | `+` | `+` | Compiler complexity, classifier errors |
| **B4 distilled single model** | 1 | `++` | `++` | `+` | `+` | `−−` at runtime | `+`/`0` | Quality loss, staleness, re-distill cost |
| **Ensemble / fan-out** | K + join | `−−` | `−−` | `−−` | `−` | `++` | `++` | Cost and latency; only for quality |
| **Model merging** | 1 | `++` | `++` | `+` | `+` | `−−` | `−`/`0` | Lossy, hard to attribute |
| **Latent handoff** | N | `0`/`+` | `+` | `+` | `0` | `+` | `+` | Requires shared latent space; not portable |

### 5.1 Dense LLM

One model, one pass, best batching and simplest serving. It is the baseline to beat, not a
straw man. A dense model of equal quality has no routing, no handoff, and one KV cache.

### 5.2 MoE LLM

The closest single-model competitor. It gets sparse activation and one forward pass, with a
jointly trained router. It is excellent if you can train or adopt one. It centralizes expert
ownership, makes post-hoc expert addition hard, and requires a shared architecture. This is why
the graph must state where it is different: independent contribution, local training, provider
neutrality, and runtime pruning.

### 5.3 Cascade

Cheap model first, escalate if confidence is low. It can cut average cost, but hard cases pay
for both calls, and the router still adds latency. A graph can include a cascade as a policy;
it is not a substitute for the graph.

### 5.4 B0 interpreted graph

Maximum portability, worst latency. Useful for development, CI, hosted APIs, and heterogeneous
remote models. It is the reference semantics, not the performance target.

### 5.5 B1 resident heterogeneous

Removes HTTP, process hops, and reloads. Supports different model families. Still pays
per-model prefill, tokenizer bridging, and a large VRAM footprint. It is the right choice when
the best experts are heterogeneous and all fit on the machine.

### 5.6 B2 fused multi-LoRA

One base model, one tokenizer, per-node adapters. Eliminates weight reloads and re-tokenization;
adapter switches are cheap. Still pays per-node prefill; the base model constrains quality.
This is the best "modular but fast" target when the experts share a base.

### 5.7 B2 + AOT + fast routing

The compiled fused target: pre-tokenized templates, precomputed prefix KV, flattened routing,
a classifier/head instead of a generative router, route caching, and a memory plan. This is the
most likely graph-native path to LLM-class latency without giving up adapters.

### 5.8 B4 distilled single model

The graph is used to generate training data and behavior; a single student model is trained to
imitate the graph on a target distribution. At runtime there is no routing, no handoff, and one
forward pass. This can match LLM speed by definition, but it gives up runtime modularity and
may lose quality on rare routes. It is the strongest throughput/VRAM option and a legitimate
deployment target if it passes eval.

### 5.9 Ensemble / fan-out

Useful for quality and uncertainty, terrible for latency and cost. Keep it opt-in and
evaluation-driven.

### 5.10 Model merging

Cheap way to collapse compatible experts into one weight set. Lossy, hard to attribute, and no
per-node rollback. A valid experiment, not the core plan.

### 5.11 Latent handoff

The theoretically best bandwidth, but requires co-trained models or projectors. It would make
the graph less portable and less independently ownable. It remains research (B3), not v0.

## 6. Compilation targets

The graph is source. The compiler chooses a target:

| Target | What the runtime executes | Modularity at runtime | Modularity at authoring/training |
| --- | --- | --- | --- |
| **Interpreted (B0)** | Node-by-node provider calls | Full | Full |
| **Resident (B1)** | Node-by-node in-process calls | Full | Full |
| **Fused (B2)** | One engine, adapters per node | Full (adapter choices) | Full |
| **Compiled fused (B2+)** | Fused plan with AOT prompts, prefix KV, fast routes | Mostly (compiled decisions) | Full |
| **Distilled (B4)** | One student model | None | Full (source graph retained) |

"Compiled" can include:

1. **Prompt/template tokenization** — static fragments tokenized once.
2. **Prefix KV compilation** — stable system prompts/contexts precomputed per model/adapter.
3. **Route compilation** — flatten hierarchies, train a classifier/head, cache routes.
4. **Plan compilation** — memory layout, adapter assignment, CUDA graph capture, engine config.
5. **Artifact contract compilation** — enforce compact artifacts and token budgets.
6. **Optional distillation** — train a student on graph traces and deploy it as one model.

A compiled artifact records the compiler version, graph hash, model/adapters hashes, target,
and eval report. If the graph or model changes, the artifact is stale, not silently reused.

## 7. Redesigns that remove the cons

These are the concrete changes to the current design, in priority order.

### 7.1 Make routing cheap by default

A full generative router is the most expensive possible way to choose a branch. Use:

- an embedding + classifier;
- a small linear head on the base model's hidden state;
- a fine-tuned adapter that emits a constrained route token;
- a precompiled route table for deterministic workflows.

Keep the structured decision contract (target, confidence, reason) so traces and evaluation
still work. A classifier can produce it. Measure the accuracy/latency trade-off; do not assume
a 1.5B router is necessary.

### 7.2 Flatten routing for deployment

Authoring may use a domain router → sub-domain router cascade. At compile time, if the leaf
domains are separable, train one classifier to route directly to the leaf. This removes a
model call and reduces TTFT. Keep the hierarchy in source for clarity and retraining.

### 7.3 Compile prefixes and tokenize once

Per-node system prompts and stable instructions should be tokenized and prefetched at plan
time. For same-base/adapter execution, store prefix KV caches and reuse them across requests.
This directly attacks the largest non-model cost. It does not let different models share KV.

### 7.4 Keep token IDs on device when compatible

B2's token-native path removes detokenize/re-tokenize. Materialize text only for trace,
evaluation, and the caller. This is a small absolute win but it also removes a synchronization
point that can stall the GPU.

### 7.5 Batch across requests, not just within one request

Implement per-model queues and continuous batching. A graph server that processes one request
at a time wastes the GPU whenever a small model is running below capacity. Batching is how the
system competes on throughput, not just single-request latency.

### 7.6 Reduce hops and artifact size

Set a hop budget and a per-hop token budget. Prefer a direct answer from a specialist over an
intermediate plan that another node reads. Make artifacts compact and structured. The cheapest
handoff is the one that does not exist.

### 7.7 Speculate or pipeline where the payoff is large

- **Speculative routing/prefill**: start prefill for the most likely branch while the router
  decides. If wrong, discard. This trades compute for latency and is only sensible for
  high-confidence, high-cost branches.
- **Pipelined handoff**: let the next node start on a partial artifact while the previous node
  finishes. This requires a streaming artifact contract and is a research spike, not a v0
  feature.

Both must be evaluated for cost and quality; they are not free.

### 7.8 Distill the hot path

If a workload is stable and high-volume, the strongest speed answer is a single student model
that imitates the graph. Keep the graph for authoring, training, evaluation, rare routes, and
re-distillation. This is "modularity at build time, performance at runtime."

### 7.9 Choose models for the deployment target

A graph is not a reason to use large models everywhere. Prefer the smallest expert that passes
the node suite, quantize when quality allows, and prune branches that the deployment cannot
use. Efficiency is a model-selection problem as much as a runtime problem.

## 8. Overhead budget

A design target, not a measured result. The benchmark may revise it, but the project needs a
number to optimize against.

| Budget | Target | Applies to |
| --- | --- | --- |
| Routing latency | ≤20 ms for a classifier/head; ≤150 ms for a small SLM router | Per decision |
| Non-model transition | ≤10 ms p95, excluding prefill | B1/B2 compiled |
| Runtime overhead ratio | <10% of end-to-end wall time | Reference workloads |
| Hop count | ≤4 for latency-sensitive graphs; report every hop above 2 | Compiled graphs |
| Per-hop prompt tokens | ≤512 for workers unless evaluation justifies more | Generated artifacts |
| Weight reloads | 0 after load for resident targets | B1/B2 |
| TTFT ratio | ≤1.25× dense (compiled graph); ≤1.10× (distilled) | Quality-matched |
| Throughput ratio | ≥0.9× dense (compiled); ≥1.0× (distilled) | High-QPS workload |
| Cost per task | ≤1.0× dense at matched quality | All |
| VRAM | ≤1.5× dense (compiled fused); lower for distilled | Reference hardware |

These targets are deliberately strict. If the system cannot meet them, it can still be useful
for customization and ownership, but it should not claim LLM-class efficiency.

## 9. Benchmark plan

The benchmark must compare systems at **matched quality**, on fixed hardware, with enough
samples to see a difference.

### 9.1 Workloads

| Workload | What it stresses |
| --- | --- |
| Single-domain code generation | Specialist advantage, short path |
| Mixed-domain requests | Routing accuracy and TTFT |
| Multi-step task with a joiner | Intermediate artifacts, aggregation |
| Long-context task | Prefill sensitivity, prefix caching |
| Repeated/session tasks | Route caching, prefix reuse |
| High-QPS serving | Batching, throughput, queueing |
| Rare/ambiguous requests | Abstain, fallback, distillation failure |

### 9.2 Systems

- Dense generalist at matched quality (the baseline).
- MoE model, if a comparable open model is available.
- Cascade baseline (cheap → expensive).
- B0 interpreted graph.
- B1 resident heterogeneous.
- B2 fused multi-LoRA.
- B2 + AOT + fast routing.
- B4 distilled student.
- Ensemble (quality ceiling reference).

### 9.3 Metrics and controls

- Quality on the same suite; report confidence intervals.
- TTFT, ITL, end-to-end p50/p95, throughput, cost per task, VRAM, cold load, reload count.
- Hardware, quantization, driver, engine version, context length, batch size, concurrency.
- Warm and cold starts separately.
- At least five runs for timing; more for rare routes.
- Paired/blocked design where possible; same prompts, same hardware.

### 9.4 Gates

- **Quality gate:** no regression beyond the suite's tolerance; otherwise the target is not a
  replacement (it may be a hot-path accelerator).
- **Latency gate:** TTFT and p95 within the budget.
- **Throughput gate:** at least parity for the server claim.
- **Cost gate:** at most parity at matched quality.
- **Parity gate:** compiled/fused targets produce the same decisions/artifacts as B0 under
  scripted responses.
- **Honesty gate:** the report states what is still slower and on which slice.

## 10. Strategic recommendation

1. **Default to B2 (fused multi-LoRA) for graphs whose experts share a base.** It is the best
   balance of modularity and speed.
2. **Add AOT compilation** (prefix KV, tokenized templates, flattened/classifier routing,
   memory plan) before claiming LLM-class latency.
3. **Build the multi-request scheduler**; without batching, throughput parity is not credible.
4. **Keep B1 for heterogeneous experts** and be honest that it pays a prefill/tokenizer tax.
5. **Offer B4 distillation for high-volume, stable hot paths.** It is the only target that can
   fully match a dense LLM's runtime shape, at the cost of runtime modularity.
6. **Publish a Pareto report, not a slogan.** The project's claim is quality per dollar and
   customization, with measured speed in the regimes where it holds.

If after this the compiled graph still loses to a dense model on latency and throughput at
matched quality for the target workloads, the correct claim is "a modular alternative with
better ownership and cost," not "faster than an LLM." The benchmark decides.

## 11. Risks and falsification

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Per-node prefill dominates | No latency parity despite perfect transport | Fewer hops, shorter prompts, prefix caching, distillation |
| Router is too expensive | TTFT worse than dense | Classifier/head, flattening, route cache |
| VRAM forces paging or smaller models | Quality or latency collapse | Admission, quantization, multi-LoRA, B4 |
| Throughput loses due to small batches | Server economics fail | Cross-request batching, model grouping |
| Distillation loses quality or goes stale | B4 is not a replacement | Eval gates, periodic re-distill, hybrid graph fallback |
| Compiled plans invalidate silently | Wrong behavior | Content hashes, stale-plan detection, rebuild |
| Benchmark overfits to one workload | False confidence | Matrix, quality matching, held-out tasks |
| The fused base constrains quality | Modularity without a quality win | Compare to B1/dense; keep heterogeneous path |

**Falsification conditions:**

- A compiled B2 graph fails to get within 1.25× TTFT or 0.9× throughput of a quality-matched
  dense baseline across three decomposable domains.
- A B4 student loses more than the agreed quality tolerance, or must be re-distilled so often
  that it is not a stable deployment.
- Per-node prefill remains dominant after prefix caching and hop reduction, and no distillation
  target is acceptable.
- The graph only wins on cost, never on latency or throughput; then the claim should be narrowed
  to cost/ownership, not speed.

These are recorded so the project cannot quietly move the goalposts.
