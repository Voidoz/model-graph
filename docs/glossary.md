# Glossary

Terms are defined once here and used consistently across the design. When a term conflicts
with an ADR, the ADR wins; fix the glossary.

| Term | Definition |
| --- | --- |
| **Abstain** | A router decision to decline routing because no route is sufficiently likely to be correct. Policy decides whether this becomes an error, a fallback, or a retry. |
| **Adapter** | The implementation that talks to one provider (OpenAI-compatible HTTP, Ollama, llama.cpp, mock). Adapters are graph-unaware. |
| **Artifact** | Named language with a semantic type (`text`, `code`, `patch`, `plan`, `json`, `report`), a producer, and a content hash. The unit of handoff and evaluation. |
| **Backlog** | Unscoped ideas in `.scratch/ideas/backlog.md` that have not become tickets. |
| **Budget** | A hard limit on steps, depth, tokens, cost, or wall time for a run. |
| **Capability filter** | A run-profile rule (no network, max VRAM, license allowlist) used to decide which nodes can run on a machine. |
| **Capability card** | The compact description of a child shown to a router: target description, `when` clause, examples, artifact type, and cost/latency hints. |
| **Compile** | Parse, validate, resolve manifests, prune, and canonicalize a graph into a Graph IR with a stable hash. |
| **Decider** | See **Router**. |
| **Distillation** | Training a smaller specialist on filtered outputs of a larger teacher model. |
| **E2E (end-to-end)** | Evaluation of a complete graph on real tasks, including routing and handoffs. |
| **Envelope** | The request-scoped runtime state: run metadata, request, inputs, artifacts, notes, and budget. Never shown raw to a model. |
| **Execution backend** | The component that owns model residency and invocation: portable provider calls (B0), resident heterogeneous models (B1), fused same-base adapters (B2), or a compiled B2+ plan. It does not make routing decisions. B4 is a compiled target that bypasses graph execution. |
| **Admission control** | The pre-load check that a graph's weights, KV cache, workspace, and adapter overhead fit the profile's VRAM. Failure is explicit, never silent paging. |
| **AOT (ahead-of-time) plan** | A compiled execution plan containing tokenized static prompts, prefix KV caches, routing/compiler decisions, memory layout, and hashes. It is a cache, not a new source of truth. |
| **Compiled target** | A deployment artifact produced from a graph: resident, fused, compiled-fused, or distilled. The graph remains the authoritative source. |
| **Continuous batching** | Scheduling new requests into an active model batch as slots free up, instead of processing one request to completion. Required for throughput parity. |
| **Distilled target (B4)** | A single student model trained to imitate a graph's behavior on a target distribution. It removes runtime routing/prefill at the cost of runtime modularity and rare-route quality. |
| **Fast router** | A classifier, embedding model, or linear head that produces the structured routing decision at millisecond scale instead of a generative router call. |
| **LLM parity** | A quality-matched comparison of TTFT, end-to-end latency, throughput, cost, and VRAM against a dense LLM baseline. It is workload-specific and must be measured. |
| **Prefix KV compilation** | Precomputing and storing KV state for stable prompt prefixes so requests do not re-prefill them. Valid only under the same base, adapter, and tokenizer. |
| **Route flattening** | Replacing a hierarchical router cascade with a direct leaf classifier at compile time, when evaluation shows the boundary is separable. |
| **Cold load time** | Time to load weights/adapters before a run's first model call, reported separately from model time. |
| **Fused backend (B2)** | A GPU-resident backend using one base model, one tokenizer, and per-node LoRA adapters. Token IDs may stay on-device; artifacts are still materialized as language. |
| **Logical parity** | The requirement that a backend produce the same routing decisions and artifact text as the reference backend under scripted responses. |
| **Multi-LoRA** | Serving multiple low-rank adapters over one resident base model, switching adapters per request or node. |
| **Prefix cache** | Cached KV/state for a shared token prefix, keyed by base, adapter, and token sequence. It cannot be shared across different models or adapters. |
| **Resident backend (B1)** | A GPU-resident backend that pins all models used by a graph in VRAM and invokes them in-process, bridging different tokenizers with text. |
| **Runtime overhead ratio** | Non-model wall time (transport, tokenization, scheduling) divided by total wall time. |
| **Token-native** | An internal fast path that passes token IDs between compatible nodes without detokenizing and re-tokenizing; the logical artifact is still text. |
| **Transition latency** | Time from the last token of one node to the first prefill token of the next, split into host, transfer, adapter-switch, and prefill components. |
| **Exit artifact** | An artifact named in `spec.exit` that is returned to the caller. |
| **Fallback** | A node invoked when a router abstains or falls below its confidence threshold. Declared once per graph in v0. |
| **Fan-out** | Routing one request to multiple children and joining their results. Designed, not executed in v0. |
| **Graph** | A declarative composition of nodes and edges with inputs, exits, and budgets. Versioned by `id@version`. |
| **Graph hash** | A hash of the canonical Graph IR, used to identify exactly what ran. |
| **Graph IR** | The compiled, self-contained, canonical representation consumed by the runtime. |
| **Handoff** | The act of passing artifacts from one node to another through a rendered prompt. |
| **Joiner** | A node that combines artifacts from multiple upstream branches into one. |
| **Manifest (model)** | The pinned description of a model: provider, resolved identifier, limits, cost, resources, license, integrity. |
| **Manifest (run)** | The immutable record of a specific run: graph hash, model hashes, profile, seed, budget, outcome, usage. |
| **Menu** | The set of outgoing routes presented to a router, derived from edges and filtered by pruning. |
| **Node** | A router, worker, joiner, or subgraph in a graph. |
| **Oracle route** | The route that would maximize the evaluation objective for a request, computed offline by trying candidates. |
| **Oracle gap** | The quality difference between oracle routing and actual routing. The key routing diagnostic. |
| **Profile (run)** | The deployment constraints and budget overrides applied to a run. Recorded in the run manifest. |
| **Pruning** | Removing nodes and edges that cannot run under a profile, then ensuring the entry and exits survive. |
| **Provider** | A service or process that serves a model (hosted API, Ollama, llama.cpp server, mock). |
| **Render** | The deterministic process of turning an envelope and node definition into provider messages. Versioned. |
| **Router** | A node that chooses the next route (or abstains) based on the request and its menu. Also called a decider. |
| **Routing card** | Optional hand-tuned capability card override on a node. |
| **Run** | One execution of a graph for one request, identified by a run ID and recorded by a trace. |
| **Scoped view** | The subset of envelope state rendered for a particular node. |
| **Semantic compression** | The deliberate reduction of state at a handoff into a compact named artifact rather than a raw transcript. |
| **SLM** | Small language model. The term is relative; the design budgets on local hardware rather than a parameter count. |
| **Specialist** | A worker model trained or prompted for a narrow domain. |
| **Subgraph** | A graph invoked as a node, with explicit input/output mappings. |
| **Suite** | A versioned set of evaluation cases, a metric, and thresholds. |
| **Time to first token (TTFT)** | Request start to the first output token; includes routing latency and is the metric most affected by an extra decider hop. |
| **Trace** | The append-only event log of a run: node entries, renders, provider calls, decisions, artifacts, budgets, and outcome. |
| **Ticket** | A unit of work in `.scratch/tickets/` with a stable ID, scope, acceptance criteria, and test plan. |
| **Validity** | The fraction of outputs that satisfy the node's artifact contract, independent of quality. |
| **Worker** | A leaf node that performs the work with one model call and writes artifacts. |
| **Working memory** | Informal name for the envelope/blackboard: the state nodes read from and write to during a run. |
