# Execution

This document describes how a compiled graph becomes a run: scheduling, budgets, pruning,
caching, replay, and observability. The contracts it consumes are in
[graph-model.md](graph-model.md) and [protocol.md](protocol.md).

## Contents

1. [From YAML to a run](#1-from-yaml-to-a-run)
2. [The walk](#2-the-walk)
3. [Runtime state](#3-runtime-state)
4. [Budgets](#4-budgets)
5. [Determinism and replay](#5-determinism-and-replay)
6. [Pruning and lazy loading](#6-pruning-and-lazy-loading)
7. [Caching](#7-caching)
8. [Concurrency and streaming](#8-concurrency-and-streaming)
9. [Observability](#9-observability)
10. [Failure semantics](#10-failure-semantics)
11. [Deferred execution features](#11-deferred-execution-features)

## 1. From YAML to a run

```text
graph.yaml ──parse──▶ document ──schema──▶ valid document
                                              │
                    registry ──resolve───────┤
                                              ▼
                                     static validation (V-01..V-22)
                                              │
                       profile ──prune────────┤
                                              ▼
                                          Graph IR
                                    (canonical, hashed)
                                              │
                                     run manifest (pins)
                                              ▼
                                           scheduler
```

The compiler produces a **Graph IR** with these properties:

- **Self-contained.** Resolved nodes, edges, model references (with manifest hashes),
  renderer version, default policy, and the artifact contract. The runtime never re-reads
  YAML.
- **Canonical.** Nodes are stored in a deterministic order (topological, tie-broken by
  ID); maps are sorted; defaults are materialized. The IR hash is stable across machines.
- **Pinned.** Every model reference resolves to an exact manifest version and content
  hash. Floating tags like `latest` are resolved at compile time and recorded.

A run has an explicit lifecycle:

```text
created → compiling → ready → running → (succeeded | failed | aborted | budget_exhausted)
```

`aborted` means a caller cancelled it. `failed` means an unhandled error. `budget_exhausted`
means a hard limit stopped it; partial artifacts may exist. Every terminal state writes a
run manifest.

## 2. The walk

A graph with `selection: single` everywhere keeps the sequential walk below: one active
node at a time. A graph that declares `top_k` or `all` uses the fan-out scheduler
([§2.2](#22-fan-out-and-joins)): a node-level dependency DAG executes on a bounded thread
pool, and the same walk semantics apply inside each branch. The single-selection path is
deliberately unchanged so existing graphs and traces do not move.

```python
def run(graph, request, profile):
    ctx = Envelope.create(graph, request, profile)
    backend = ExecutionBackend.plan_and_load(graph, profile)  # B0/B1/B2
    frontier = [Frame(node=graph.entry, path=[], depth=0)]

    while frontier:
        frame = frontier.pop(0)
        budget.check_before(frame)                  # may raise MG-4xxx

        node = graph.nodes[frame.node]
        trace.enter(node, frame)

        if node.kind == "router":
            view = render_router(ctx, node, graph, frame)
            decision = call_router(view, node, backend)   # repair/retry policy
            target = resolve_route(decision, node, graph)
            trace.decision(decision, target)
            frontier.append(frame.advance(target))

        elif node.kind == "worker":
            view = render_worker(ctx, node, frame)
            text, usage = backend.invoke(node.id, view, node.params)
            budget.charge(usage)
            artifacts = parse_artifacts(text, node.writes)
            ctx.write(artifacts, producer=node.id)
            for target in graph.successors(node):   # only joiners in v0
                frontier.append(frame.advance(target))

        elif node.kind == "joiner":
            view = render_joiner(ctx, node, frame)
            text, usage = backend.invoke(node.id, view, node.params)
            budget.charge(usage)
            ctx.write(parse_artifacts(text, node.writes), producer=node.id)
            frontier.extend(frame.advance(t) for t in graph.successors(node))

        elif node.kind == "subgraph":
            result = run_subgraph(ctx, node, frame, backend)  # shares budget, depth + 1
            ctx.write(result.mapped_artifacts, producer=node.id)
            frontier.extend(frame.advance(t) for t in graph.successors(node))

        trace.exit(node, frame)

    return ctx.result_artifacts(graph.exit)
```

Important properties of this loop:

- **No hidden recursion.** Only a `subgraph` node recurses, and depth is checked before
  entry. A router cannot call a router outside its declared edges.
- **The envelope is shared, the prompt is not.** Artifacts accumulate; each render selects.
- **Budget is checked before every call**, and charged after. A provider call cannot start
  if the remaining budget is below the node's declared minimum estimate.
- **The backend owns model invocation, not semantics.** The portable backend (B0) wraps
  provider adapters. A resident or fused backend (B1/B2) implements the same call contract
  in-process and must pass logical-parity tests.
- **One branch by default.** In the sequential walk, a router activates exactly one
  successor. A router with `selection: top_k` or `all` switches the graph to the fan-out
  scheduler, which may activate several successors from one decision.

### 2.2 Fan-out and joins

Fan-out is scheduled as a DAG, not by re-entering the sequential walk. Each node is
runnable when every incoming edge has been resolved: delivered by a branch that ran, or
marked inactive because an upstream router did not select it. This is what makes a joiner
wait for all declared inputs without a second, ad-hoc barrier.

```python
def fan_out_run(graph, request):
    ctx = Envelope.create(graph, request)
    frontier = [Frame(node=graph.entry)]
    pending = {}                       # node -> first-call reservation
    while frontier or pending:
        ready = [frame for frame in frontier if frame.is_runnable()]
        # Atomic fan-out pre-flight: all first calls or none; veto before start.
        budget.reserve_many([estimate_first_call(frame) for frame in ready])
        for frame in ready[:budget.concurrency_limit]:
            pool.submit(run_node, frame)   # real threads, shared budget/trace
        for result in pool.wait_for_one():
            trace.record(result)           # deterministic start order, actual completion order
            frontier.extend(graph.activate(result.selected_edges))
    return ctx.result_artifacts(graph.exit)
```

Rules:

1. **Selection is deterministic.** `all` uses menu order (edge `priority` descending, then
   target ID). `top_k` uses confidence, then priority, then target ID, and fills missing
   slots from menu order. See [graph-model.md §5.1](graph-model.md#51-multi-selection-decisions).
2. **Concurrency is bounded.** The default limit is 4 in-flight branch calls
   (`run_graph(..., max_concurrency=N)` or `mg run --max-concurrency N`). The limit bounds
   provider calls, not the number of selected branches; branches beyond the limit wait.
3. **The budget is shared and reserved.** Before a ready batch starts, the scheduler
   renders each first prompt, estimates tokens and cost, and reserves the sum atomically.
   If the sum does not fit, the run fails with `MG-4002`/`MG-4003` and **no branch call
   starts**. Reservations are replaced by reported usage when the call returns.
4. **A joiner waits for its declared inputs.** It runs only when every incoming edge is
   resolved and every `read` artifact exists. A missing read is `MG-2201`; the contract
   has no `allow_partial`, so partial joins are not accepted. A join that waits past
   `join_timeout_seconds` (or the wall budget when unset) fails with `MG-2105`.
5. **A branch failure fails the run, not the process.** The scheduler stops starting new
   nodes, lets in-flight calls settle, records the failing branch, and raises the first
   branch failure by start order. The joiner does not run; the run does not hang.
6. **Accounting is per branch.** Each branch has its own usage ledger (tokens, cost,
   cache hits). `RunOutcome.branches` and the `branch.completed` / `run.finished` trace
   events report direct and total usage. The root branch covers work before the first
   fan-out; each fan-out branch reports its own work; the join branch covers the joiner.
   A branch whose target is a subgraph reports the inner branches as descendants and
   exposes `subtree_total_tokens`/`subtree_cost_usd`; inner single-selection subgraphs get
   their own branch record so their cost does not disappear. `contributed` marks the
   branches whose artifacts reached a joiner, and `joiner.joined` repeats their usage.
7. **Trace order is reconstructable.** Node events are flushed in deterministic start
   order; each `node.entered` carries `start_order` and each `node.exited` carries
   `completion_order`. Completion order is real (it can vary between runs) and is recorded
   separately rather than being used as trace order. Artifacts, envelopes, and manifests
   use deterministic producer order for the same reason.

### 2.1 Subgraph execution

A subgraph invocation:

1. Checks `depth + 1 <= max_depth` (else `MG-4006`).
2. Creates a child envelope whose `run.parent_run_id` is the current run, whose inputs are
   the mapped outer artifacts, and which shares the parent's budget object.
3. Runs the inner graph with the same scheduler and a nested path.
4. Maps the inner graph's exit artifacts to the node's `writes` via `output_map`.
5. Discards all other inner artifacts. They are available in the trace, not to the parent.

A subgraph failure propagates as a node failure in the parent, with the child trace linked.
The parent may have a fallback only at router level; there is no per-node catch in v0.

### 2.3 Backend execution

Before the walk, the runtime asks the backend to plan and load:

1. **Plan.** Resolve models/adapters, check graph compatibility (especially for a fused
   backend), estimate VRAM, and decide admission. If the plan fails, the run either falls
   back to B0 explicitly or stops with the plan attached.
2. **Load.** Pin weights/adapters and initialize runners. Cold load time is recorded
   separately. A resident plan may not reload a weight after this point.
3. **Invoke.** Executors call `backend.invoke(node_id, view, params)`. The returned text,
   tokens, and usage are normalized identically across backends.
4. **Measure.** The backend records transition phases (host parse, tokenize, transfer,
   adapter switch, prefill) and exposes them as trace metrics. Prefill is reported separately
   from transport overhead because it is model compute, not marshaling.
5. **Close.** Release VRAM at the end of the run. Replay uses the recorded backend and plan;
   a backend change is a run-manifest difference, not a silent substitution.

Backend details and admission rules are in [gpu-execution.md](gpu-execution.md); compiled plans
and distilled targets are analyzed in [performance.md](performance.md). B2+ is selected by the
compiler; B4 bypasses the scheduler at runtime and is recorded as a deployment target, not as
graph execution. The scheduler does
not branch on backend mode; only the executors' call target changes.

## 3. Runtime state

| Object | Lifetime | Persisted | Notes |
| --- | --- | --- | --- |
| Graph IR | Run | Yes (hash in run manifest) | Immutable |
| Envelope | Run | Serialized at node boundaries (redacted) | Mutable; lock-protected writes under fan-out |
| Artifacts | Run | Content-addressed blobs | Append-only |
| Budget | Run | Snapshot per event | Shared with subgraphs |
| Path/frames | Run | Trace events | Ordered |
| Trace | Run | `trace.jsonl` | Append-only; redacted on write |
| Cache | Cross-run | `cache/responses/` | Exact-match; keyed by content hashes; no TTL |
| Model manifests | Cross-run | Registry | Immutable per version |

There is no global mutable state across runs. Two runs can execute concurrently in one
process if they share only the read-only registry and cache (cache writes are atomic).

## 4. Budgets

Budgets make cost a first-class constraint instead of a surprise.

| Budget | Counts | Charged |
| --- | --- | --- |
| `max_steps` | Node entries (routers included); a retry is another provider call but not a new step | Before node entry |
| `max_depth` | Subgraph nesting | Before subgraph entry |
| `max_total_tokens` | Input + output tokens reported by providers | After each call |
| `max_cost_usd` | Estimated cost from manifest prices and usage | After each call |
| `max_wall_seconds` | Monotonic elapsed time | Before each call and checked between |

Rules:

1. **Hard ceiling.** The runtime never exceeds a budget to finish a request. It fails,
   unless `on_budget_exhausted: best_effort`, in which case it returns artifacts produced
   so far and marks the run `budget_exhausted`.
2. **Pre-flight estimate and reservation.** Before a call, the runtime estimates the
   minimum cost and tokens (input tokens known from the render, output at least 1) and
   **reserves** them under the budget lock. If that already exceeds the remainder, the
   call is vetoed with `MG-4xxx` rather than started and abandoned. A fan-out batch
   reserves all of its first calls atomically before any branch starts; an unaffordable
   ensemble is refused with no partial start. Reservations are settled with the provider's
   reported usage and released on failure.
   Fan-out can multiply cost by roughly the number of activated branches; this is why the
   veto is mandatory and why `top_k`/`all` are opt-in.
3. **No negative budgets.** A provider that reports more usage than estimated is charged
   as reported; the next pre-flight check fails. Under-reporting is a provider bug and is
   recorded in `provider_meta`.
4. **Node limits tighten, never loosen.** A node may set a lower `max_output_tokens` or
   timeout. Only graph defaults (or the run profile) may raise the graph-level ceiling.
5. **Subgraphs share the budget.** A nested graph cannot reset the counter. This is what
   prevents recursive composition from becoming a denial-of-service.
6. **Wall-clock checks are cooperative.** A long provider call cannot be interrupted if
   the adapter does not support cancellation; the budget is checked immediately after and
   the run fails. Adapters should honor timeouts.
7. **Retries and repairs are bounded.** A retryable transport failure gets at most two
   retries (three attempts). Invalid router decisions and malformed artifacts get one repair
   attempt. A router's configured `max_retries` bounds low-confidence re-prompts, and an
   executor has a defense-in-depth cap of eight provider calls per node. A retry is charged
   to the shared budget; it does not reset it.
8. **Profile overrides cannot break the ceiling.** Overrides tighten by default. Even with
   `--allow-budget-override`, a profile cannot raise a run above the schema-aligned hard
   maxima for steps (64), depth (8), total tokens (10,000,000), or wall seconds (86,400).
   See [security.md](security.md#24-budget-and-denial-of-service-abuse).

### 4.1 Budget math

```text
cost_call = (input_tokens  / 1e6) * manifest.cost.input_per_million_usd
          + (output_tokens / 1e6) * manifest.cost.output_per_million_usd
```

- Local models default to zero marginal cost in the manifest, but their latency and
  resource use are still recorded. Cost is not the only scarce resource.
- If a provider uses per-request pricing, the manifest declares it and the accountant
  charges per call.
- Unknown pricing is recorded as `cost_unknown` and cannot satisfy a `max_cost_usd` check
  by assumption; the runtime treats unknown cost as zero only if the profile explicitly
  allows it. Otherwise it warns.

### 4.2 Defaults and profiles

Graph defaults are opinions. A run profile can tighten them (`--profile local-8gb`) or
change policy filters, but the profile used is recorded in the run manifest. Profiles do
not silently raise the graph's declared ceiling unless `--allow-budget-override` is passed.

## 5. Determinism and replay

Determinism is a spectrum. The runtime is explicit about what it can guarantee.

| Layer | Guarantee |
| --- | --- |
| Compilation | Bit-identical IR for identical inputs and registry versions (canonical form). |
| Scheduling | Identical node activation order for identical graph and decisions; start order is recorded. |
| Fan-out completion | Wall-clock completion order can vary; it is recorded separately (`completion_order`) and never used as trace order. |
| Prompts | Identical rendered messages for identical envelopes, node, and renderer version. |
| Routing | Identical decisions if the provider is deterministic (temperature 0, seed support). |
| Worker output | Not guaranteed by non-deterministic providers. Replay re-sends identical requests. |
| Artifacts/manifests | Deterministic producer order for scripted responses, independent of branch completion order. |
| Cost/latency | Varies with provider and hardware. |

**Replay** has two modes:

- **Trace replay** (default): re-execute using recorded model responses. No network. Used
  for debugging, evaluation of scheduler/renderer changes, and auditing. If a node or
  template changed, the runtime refuses to replay unless `--allow-ir-mismatch` is passed,
  because the recorded responses may no longer correspond to the new prompts.
- **Re-send replay**: execute the same requests against the providers again. Used to
  measure provider drift. The report compares outputs and usage rather than claiming
  equality.

A cache hit emits a normal `provider.response` event, so trace replay of a cached run works
without the cache or a network. See [§7](#7-caching).

A **run manifest** records everything needed to distinguish these, including the
immutable deployment revision that served the request:

```json
{
  "run_id": "run_01J9Z3K4",
  "deployment_revision": "rev_6f2a1c9b4e8d2077",
  "target": "portable",
  "status": "final",
  "outcome": "succeeded",
  "graph": {"id": "composed-assistant", "version": "1.0.0", "hash": "sha256:...", "source": "examples/composed-assistant.graph.yaml"},
  "ir_version": "modelgraph/ir/v0",
  "renderer_version": "3",
  "parser_version": "2",
  "models": [
    {"node": "domain-triage", "ref": "router-small", "version": "1.0.0", "hash": "sha256:...", "provider": "ollama", "model": "qwen2.5:1.5b-instruct", "resolved_revision": null, "revision_reason": "model manifest does not pin spec.integrity.revision"}
  ],
  "profile": {"name": "local-8gb", "providers": ["ollama", "llamacpp"], "licenses": ["apache-2.0", "mit"]},
  "profile_hash": "sha256:...",
  "prune": {"profile": "local-8gb", "removed_nodes": [], "warnings": []},
  "prune_hash": "sha256:...",
  "budget": {"max_steps": 8, "max_depth": 3, "max_total_tokens": 32000, "max_cost_usd": 0.05},
  "seed": 42,
  "environment": {"python_version": "3.12.6", "platform": "Linux-...", "providers": [{"name": "ollama", "version": "1.0.0"}], "hostname": "workstation", "pid": 4242},
  "started_at": "2026-10-22T10:00:00Z",
  "finished_at": "2026-10-22T10:00:07Z",
  "usage": {"steps": 3, "input_tokens": 2011, "output_tokens": 842, "total_tokens": 2853, "cost_usd": 0.0},
  "artifacts": [{"name": "answer", "type": "code", "hash": "sha256:...", "producer": "js-specialist", "bytes": 812, "exit": true}],
  "exit_artifacts": [{"name": "answer", "hash": "sha256:..."}],
  "trace": "trace.jsonl",
  "parent_run_id": null,
  "subgraphs": [],
  "error": null
}
```

### 5.1 Manifest lifecycle, linkage, and recovery

A persisted run writes `runs/<run-id>/run.json` twice:

1. **Provisional**, before the first node runs: `status: provisional`, `outcome: running`,
   `finished_at: null`. A crash leaves this file behind instead of no evidence.
2. **Final**, at the end: the outcome, usage, error, exit artifact hashes, and timestamps are
   filled in. The write uses a temporary file, `fsync`, and `os.replace`, so a reader never sees
   a half-written manifest. Failed and budget-exhausted runs finalize too.

The manifest pins:

- graph id, version, hash, and source;
- every resolved model ref with its manifest version/hash, provider, resolved model, and
  resolved revision. When the revision is unknown the field is `null` and `revision_reason`
  says why;
- renderer, parser, and IR versions, plus a `deployment_revision` hash over the graph hash,
  model hashes, target, and compiler/IR identity (ADR-0013);
- the effective profile and prune report, with canonical hashes when present;
- budget, seed, usage, and environment (Python, platform, CPU/memory, provider adapter
  versions, hostname, and PID);
- outcome, error, artifacts, exit artifact hashes, and the trace path. `request.task`,
  router `path[].reason`, and the `error` payload are redacted before the manifest is
  written; the redaction report contains pattern names, counts, and hashes only;
- `parent_run_id` and a `subgraphs` index. A subgraph invocation also writes its own final
  manifest at `runs/<run-id>/subgraphs/<child-run-id>/run.json` with `parent_run_id` set and
  the parent's trace at `../../trace.jsonl`.

The graph hash is the source of truth for replay. `mg replay` recompiles the graph source and
refuses when the manifest's `graph.hash` differs, unless `--allow-ir-mismatch` is passed; the
override is recorded in `replay.json` rather than silently applied. The trace replay compares
the recorded path and exit artifact hashes, while re-send replay compares provider output and
usage instead of claiming equality.

Recovery is a library operation in `modelgraph.runtime.manifest`:

- `recover_interrupted_run(run_dir)` turns a provisional manifest into `status: interrupted`,
  `outcome: interrupted`, a finish timestamp, and `error.code: MG-2001` when the recorded PID
  is dead on the same host. A live process is left alone. If process identity is unavailable,
  recovery requires an explicit `stale_after_seconds`.
- `recover_interrupted_runs(runs_dir)` scans top-level and nested subgraph run directories.
  `run_graph` calls it before writing a new provisional manifest, so a later run repairs
  abandoned evidence without a separate daemon. `mg runs show RUN` recovers one run and
  `mg runs list` recovers the directory before listing, so inspection itself turns a crashed
  provisional file into `interrupted` rather than leaving it reported as `running` forever.

Interrupted means "the runtime never reached finalization"; the partial trace, path, and
artifacts that exist are still valid evidence. It does not mean the run was cancelled by a
user or that a provider failed. `MG-2001` is the runtime code for recovery of a run that never
finalized.

The canonical contract is `schemas/run-manifest.schema.json`; the contract suite validates the
schema plus final, provisional, and interrupted fixtures.


## 6. Pruning and lazy loading

The authored graph is an inventory of capability, not a deployment. A **profile** describes
what the current machine or policy can run:

```yaml
name: local-8gb
network: loopback            # none | loopback | internet
gpu: true
max_vram_gb: 8
providers: [ollama, llamacpp]
licenses: [apache-2.0, mit]
max_cost_usd: 0.0
budget_overrides: {max_steps: 8}
backend: auto                # portable | resident | fused | compiled | distilled | auto
```

The pruner runs after validation and before compilation:

1. Resolve every node's model manifest.
2. Mark a node unsupported if its provider is unavailable, its resource requirements exceed
   the profile, its license is not allowed, or its `network_access` exceeds the profile's
   `network` ceiling (`none` < `loopback` < `internet`).
3. Remove unsupported nodes and all edges incident to them.
4. Remove nodes and edges that can no longer reach an exit artifact from the entry.
5. If the entry is removed, fail with `MG-1005` and a list of causes.
6. For each surviving router, filter its menu to surviving targets (with at least one route
   per router; otherwise remove the router's branch).
7. Return a `PruneReport`:

```json
{
  "profile": "local-8gb",
  "removed_nodes": [{"id": "python-specialist", "reason": "model requires 14GB VRAM; profile allows 8GB"}],
  "removed_edges": [{"from": "coding-triage", "to": "python-specialist"}],
  "removed_exit_artifacts": [],
  "warnings": ["router coding-triage now has one route; consider bypassing it"]
}
```

Pruning never edits the YAML and never silently changes an exit contract. If an exit
artifact becomes unreachable, the run fails with the report rather than returning partial
output that looks successful.

**Lazy loading** is the runtime complement:

- Weights are loaded on first use of a node, not at startup.
- The runtime may unload an idle model under memory pressure (LRU), recording the event.
- A graph with a warm cache of loaded models starts faster; cold start is reported
  separately from model time so benchmarks are not misleading.
- Pruned nodes are never loaded, which is where a meaningful share of the savings comes
  from on a machine that would otherwise swap.

### 6.1 Backend admission

Lazy loading is the portable (B0) behavior. Resident backends invert it: they decide whether
the whole plan fits before loading anything, then keep the active set pinned.

- **Plan-time VRAM estimate.** Weights + KV budget + workspace + CUDA graphs + fragmentation
  headroom + adapter overhead. The estimate and its inputs go into the run manifest.
- **Admission.** If the plan does not fit, the backend may select a smaller/quantized
  compatible manifest, prune branches (T-0015), fall back to B0, or fail. It may not silently
  page weights or offload a model mid-run.
- **No reloads.** Once a resident plan is loaded, a model invocation must not reload weights.
  A reload is a performance failure and is recorded as such.
- **Explicit fallback.** Falling back to B0 is recorded with a reason and changes the latency
  class of the run; it is not hidden.
- **Unload policy.** Only models outside the active resident set may be unloaded. The fused
  backend keeps one base model plus adapters resident; evicting the base is equivalent to
  abandoning the backend.

See [gpu-execution.md](gpu-execution.md) for the backend contract, KV rules, and measurement
plan.

## 7. Caching

v0 supports **exact-match response caching** only. Semantic caching is deferred because it
can silently change behavior and is hard to evaluate (backlog B-01). There is no TTL, no
shared/remote cache, and invalidation is by key only.

### 7.1 Cache key

```text
sha256(
  canonical_json({
    node_id, graph_hash, renderer_version,
    model_manifest_hash, resolved_model_revision, model_manifest_version,
    effective_provider, model_id,
    canonical_messages, generation_params, seed
  })
)
```

- `generation_params` is `temperature`, `max_output_tokens`, `json_mode`, `stop`, and
  `seed`. `timeout_seconds` is not part of the key because it cannot change the generated
  content.
- `effective_provider` is the adapter that actually served the call (`mock`, `ollama`, …),
  not only the provider named in the manifest. Running the same graph with `--provider mock`
  and `--provider ollama` must not share responses.
- `graph_hash` scopes every node, including subgraph nodes; there is no cross-graph hit.
- Canonical JSON and SHA-256 are used. Python's process-randomized `hash()` is never used.
- There is no profile field: the model manifest hash, resolved revision, and effective
  provider already distinguish a local response from a hosted one.

### 7.2 Modes and CLI

| Mode | Reads | Writes | Typical use |
| --- | --- | --- | --- |
| `on` | yes | yes | Default for `mg run` |
| `readonly` | yes | no | Reproduce from a populated local cache without growing it |
| `off` | no | no | Hermetic tests, evaluation, or deliberate cold runs |

```text
uv run mg run examples/hello-triage.graph.yaml --provider mock
uv run mg run examples/hello-triage.graph.yaml --provider mock --no-cache
uv run mg run examples/hello-triage.graph.yaml --provider mock --cache-mode readonly
uv run mg run examples/hello-triage.graph.yaml --provider mock --cache-dir .cache/responses
```

- `mg run` defaults to `on`. `--no-cache` is shorthand for `--cache-mode off`; it bypasses
  both reads and writes. `--cache-dir PATH` sets the location; the default is
  `cache/responses/` under the working directory and can be overridden with `MG_CACHE_DIR`.
- The **library** default is off. `run_graph(..., cache=ResponseCache(...))` must be
  explicit, so tests, evaluation suites, and other callers stay hermetic and reproducible
  by default.
- `readonly` reads are safe on a cache populated by another run. A miss is not written; it
  is recorded as a miss and the provider is called normally.
- The cache is local and per-user. It is not a shared service; a future shared cache needs a
  privacy design. `cache/` is git-ignored and can be deleted at any time.

### 7.3 Storage and atomicity

One JSON file per key under `<cache-dir>/<sha256>.json`. An entry contains:

- the response text, normalized usage (`input_tokens`, `output_tokens`,
  `approximated`), `finish_reason`, and provider metadata;
- `created_at` and the cache-key metadata (node/graph/renderer/model hashes, resolved
  revision and version, effective provider, model id, generation params, and a
  `messages_hash`).

Entries **never** contain request messages, prompts, envelopes, artifacts, or credentials.
The prompt is represented only by a hash.

- Writes use a uniquely named temporary file in the cache directory, `fsync`, and
  `os.replace`. A concurrent reader sees the previous complete entry or the new complete
  entry, never a half-written file.
- A corrupt, truncated, or partial entry is ignored on read. When writes are allowed it is
  removed and the next successful response atomically rewrites the key; in `readonly` it is
  ignored and left alone. A cache problem never fails a run.
- A miss writes only after the provider returns a response that the budget accepted.
  Provider failures and budget-exhausted calls leave no entry.
- The cache is inspectable plain JSON. Treat it as sensitive local data: response text can
  contain user content, and there is no encryption at rest.

### 7.4 Hits, budget, and trace

A hit is resolved in `Executor.call` **before** the provider is called:

- It consumes zero tokens and zero cost and is not charged to the budget. The run report
  shows avoided usage separately: `cache.hits`, `cache.misses`, `cache.writes`,
  `cache.skipped`, `cache.avoided_tokens`, and `cache.avoided_cost_usd` (in `mg run --json`
  and the human run summary).
- Step and wall-clock budgets still apply to the node. Token/cost pre-flight checks do not
  veto a hit because it consumes neither.
- The trace emits `provider.response` with the cached `text` and `cached: true` so trace
  replay works, plus a distinct `provider.response.cached` event carrying the original usage
  and avoided tokens/cost. A `cache.miss` event marks an enabled miss. A paired hit is
  counted once by `mg stats`; both events share a `cache_hit_id`.
- No `budget.charged` event is emitted for a hit. A cache hit does not reset or increase
  budget usage.

### 7.5 Privacy, redaction, and replay

- A response that matches any default secret/PII pattern is **not cached**. The runtime
  records `cache.skipped` with `reason: default-secret-pattern` and pattern names only
  (never the matched text), and the next identical request misses. Redacting only the cache
  metadata was rejected: the response body is the sensitive part. API keys are never written
  to the cache.
- Persistence redaction still applies to the trace, envelope, and manifest. A cached hit
  still goes through the normal trace writer, so a run that contains a secret-shaped prompt
  or response is redacted exactly like an uncached run.
- Trace replay works for cached runs because the hit emits a normal `provider.response`
  with the response text. A redacted trace can still break replay (the T-0029 trade-off);
  skipping secret-shaped responses from the cache means the cache itself does not introduce
  a second replay failure.
- The cache is not rewritten by redaction, and it is not scanned by
  `scripts/scan_secrets.py` by default. Delete it, or point `--cache-dir` at a scratch
  directory, when response text must not persist.
- Cache files are trusted local data. Anyone who can write the cache directory can change a
  cached response, so protect the directory with filesystem permissions; entries are created
  mode `0600` by the atomic writer.

### 7.6 Scope

- No semantic/embedding caching, no cross-graph reuse, no shared or remote backend, and no
  TTL in v0.
- Caching routing decisions separately from provider responses is out of scope; a router
  response is cached like any other provider response.
- Evaluation and the baseline gates leave the cache off so baselines remain reproducible.

## 8. Concurrency and streaming

Graphs with only `selection: single` still execute sequentially. Fan-out graphs execute
independent branches concurrently on a bounded thread pool:

- **Limit.** `max_concurrency` defaults to 4 in-flight branch calls and is configurable per
  run (`run_graph(max_concurrency=...)`, `mg run --max-concurrency`). The limit is per
  scheduler; a nested subgraph has its own limit.
- **Real threads, shared state.** Branches use real worker threads because provider
  adapters are synchronous. Budget accounting, cache statistics, envelope writes, and the
  provider registry are lock-protected. The trace is buffered per node and flushed in
  deterministic start order.
- **No speculative execution.** A branch starts only after the router decision selects it
  and its first-call reservation is granted. Branches are not started early and discarded.
- **Join timeout.** A joiner may wait up to `join_timeout_seconds`; when unset, the
  wall-clock budget is the outer bound. In-flight provider calls are not killed (threads
  cannot be interrupted safely), so a timeout fails the run after the call returns; the
  same cooperative limitation as the wall budget.
- **Streaming** (deferred): the CLI may stream the final artifact from the last provider
  call for UX, but intermediate nodes produce structured artifacts. A `stream: true` option
  must not change graph semantics. See `Q-008`.

## 9. Observability

A trace is an append-only JSONL event stream. It is the primary debugging interface and
the source of training data.

Event types:

| Event | Key fields |
| --- | --- |
| `run.started` | run id, graph hash, profile, budget, seed |
| `node.entered` | node, kind, depth, step, path length |
| `render.completed` | prompt hash, message hashes, token estimate, truncated artifacts |
| `render.truncated` | one event per capped untrusted block: kind, name, limit, bytes omitted |
| `backend.plan` | mode, resolved models/adapters, VRAM estimate, admission result, fallback |
| `backend.loaded` | cold load ms, resident models, VRAM reserved |
| `backend.transition` | from/to node, host parse ms, tokenize ms, transfer ms, adapter switch ms, prefill ms |
| `provider.request` | provider, model, params hash, attempt |
| `provider.response` | usage, latency, finish reason, cached, provider meta; text for replay |
| `provider.response.cached` | original usage, avoided tokens/cost, cache hit id |
| `cache.miss` | node, cache key, mode, whether a write was allowed |
| `cache.skipped` | node, reason (`default-secret-pattern` or `write-failed`), pattern names |
| `decision.made` | decision, target(s), selection, confidence, reason, threshold |
| `fanout.decision` | router, selection, targets, branch ids |
| `fanout.preflight` | nodes, branches, estimated tokens/cost, accepted, code |
| `branch.started` / `branch.completed` | branch id/kind, node, start order, outcome, per-branch tokens/cost |
| `joiner.inputs` / `joiner.incomplete` / `joiner.joined` | declared reads, received/missing inputs, contributing branches and their cost |
| `join.timeout` | joiner, timeout, waited seconds, pending branches |
| `artifact.written` | name, type, hash, producer, bytes |
| `budget.charged` | tokens, cost, elapsed, remaining |
| `prune.applied` | profile, removed nodes/edges, warnings |
| `node.exited` | node, duration, outcome |
| `run.finished` / `run.failed` | outcome, usage, exit artifacts, error |

Rules:

- Events are timestamped with a monotonic clock and a wall-clock timestamp.
- Node event groups are persisted in deterministic `start_order`, not completion order.
  `node.exited` and `branch.completed` carry a real `completion_order`; the two orders are
  separate so a concurrent trace remains reconstructable.
- Content is referenced by hash; full prompts may be stored separately under `prompts/`
  when `--save-prompts` is enabled, subject to redaction.
- The trace never contains API keys or unredacted secrets. Persistence applies the default
  secret/PII pattern set to each event before writing the JSONL; in-memory events are left
  intact for the producing run. A `redaction.json` sidecar records pattern names, counts,
  field paths, and truncated hashes (never matched text) when persistence redacted
  something. Artifact blobs under `artifacts/` are content-addressed run data and are not
  rewritten; treat a run directory as sensitive.
- Backend transition metrics are recorded per hop; prefill is a separate field from host
  marshaling so the report does not blame transport for model compute.
- Logging is structured; human-readable logs are a rendering of events, not a second
  source of truth.
- OpenTelemetry export is optional and off by default; the core runs without the dependency
  (T-0030).
- Operational details — JSONL logs, trace-to-metric projection, `mg stats`, and the optional
  span exporter — are in [observability.md](observability.md).

## 10. Failure semantics

| Scenario | Result |
| --- | --- |
| Validation failure | No run; error list; nothing persisted except the compile report. |
| Pruning removes the entry or an exit | No run; `PruneReport`; explicit failure. |
| Router low confidence with `error` | Run failed; partial artifacts and decision preserved. |
| Worker provider failure after retries | Run failed at that node; partial artifacts preserved. |
| Fan-out branch failure | Run failed with the first branch error by start order; the joiner does not run; in-flight branches settle. |
| Joiner missing a declared input | Run failed `MG-2201`; no partial join; `joiner.incomplete` names the missing reads. |
| Joiner wait timeout | Run failed `MG-2105`; in-flight calls are allowed to return; `join.timeout` records the pending branches. |
| Budget exhausted with `fail` | Run `budget_exhausted`; no result returned; artifacts preserved. |
| Budget exhausted with `best_effort` | Run `budget_exhausted`; artifacts written so far returned with a warning. |
| Caller cancellation | Run `aborted`; in-flight call allowed to finish or cancelled if supported. |
| Internal error | Run `failed`; trace flushed; error includes stack and run id. |

A failed run is still valuable: it is replayable, inspectable, and can be exported as
training data. The runtime must never delete a trace to make a failure look clean.

## 11. Deferred execution features

| Feature | Why deferred | Ticket / question |
| --- | --- | --- |
| Semantic response cache | Behavior can change silently; needs evaluation | Backlog B-01 |
| Tool/code execution nodes | Security and sandboxing dominate the design | `Q-004` |
| Streaming intermediate events | UX, not semantics | `Q-008` |
| Multi-turn conversations | Envelope history and routing state need design | `Q-009` |
| Distributed model placement | Keep v0 single-machine; adapters are already network-capable | Backlog B-09 |
| GPU-resident execution backends | Post-v0 performance track; v0 uses the portable backend | T-0031 |
| Recursive subgraphs | Loop guards and termination proofs are unsolved | `Q-007` |
| Speculative execution of branches | Cost model and cache semantics need work | Backlog B-02 |
