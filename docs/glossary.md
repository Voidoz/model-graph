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
| **Trace** | The append-only event log of a run: node entries, renders, provider calls, decisions, artifacts, budgets, and outcome. |
| **Ticket** | A unit of work in `.scratch/tickets/` with a stable ID, scope, acceptance criteria, and test plan. |
| **Validity** | The fraction of outputs that satisfy the node's artifact contract, independent of quality. |
| **Worker** | A leaf node that performs the work with one model call and writes artifacts. |
| **Working memory** | Informal name for the envelope/blackboard: the state nodes read from and write to during a run. |
