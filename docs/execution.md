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
                                     static validation (V-01..V-20)
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

v0 executes one active branch at a time. The scheduler is still written around a
**frontier** (a set of active paths) so fan-out can be added without changing the core
loop.

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
- **One branch in v0.** If a node has multiple successors, only a fan-out-capable node
  (router with `selection: top_k`, deferred) may activate more than one. The loop's
  frontier supports it; validation prevents it until the scheduler does.

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

### 2.2 Backend execution

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
| Envelope | Run | Serialized at node boundaries (redacted) | Mutable, single-writer in v0 |
| Artifacts | Run | Content-addressed blobs | Append-only |
| Budget | Run | Snapshot per event | Shared with subgraphs |
| Path/frames | Run | Trace events | Ordered |
| Trace | Run | `trace.jsonl` | Append-only |
| Cache | Cross-run | `cache/` | Keyed by content hashes |
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
2. **Pre-flight estimate.** Before a call, the runtime estimates the minimum cost and
   tokens (input tokens known from the render, output at least 1). If that already exceeds
   the remainder, the call is vetoed with `MG-4xxx` rather than started and abandoned.
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
| Scheduling | Identical node sequence for identical graph and decisions. |
| Prompts | Identical rendered messages for identical envelopes, node, and renderer version. |
| Routing | Identical decisions if the provider is deterministic (temperature 0, seed support). |
| Worker output | Not guaranteed by non-deterministic providers. Replay re-sends identical requests. |
| Cost/latency | Varies with provider and hardware. |

**Replay** has two modes:

- **Trace replay** (default): re-execute using recorded model responses. No network. Used
  for debugging, evaluation of scheduler/renderer changes, and auditing. If a node or
  template changed, the runtime refuses to replay unless `--allow-ir-mismatch` is passed,
  because the recorded responses may no longer correspond to the new prompts.
- **Re-send replay**: execute the same requests against the providers again. Used to
  measure provider drift. The report compares outputs and usage rather than claiming
  equality.

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
  "renderer_version": "2",
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
  "environment": {"python_version": "3.12.6", "platform": "Linux-...", "providers": [{"name": "ollama", "version": "0.4.0"}], "hostname": "workstation", "pid": 4242},
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
- outcome, error, artifacts, exit artifact hashes, and the trace path;
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
can silently change behavior and is hard to evaluate.

Cache key:

```text
sha256(
  node_id + graph_hash + renderer_version
  + model_manifest_hash + resolved_model_revision
  + canonical_messages + generation_params + seed
)
```

Rules:

- Cache entries store the response text and normalized usage, not the envelope.
- A hit is recorded as `provider.response.cached` and charges zero cost, but the original
  token usage is displayed in the report as avoided cost.
- Cache is scoped by profile? No: the key already includes the model hash and messages.
  A local profile cannot accidentally hit a hosted response because the manifest hash
  differs.
- Secrets and user-identifying content must not be written to a shared cache. The default
  cache is per-user and local. A run may set `cache: off`.
- Cache invalidation is by key, not time. There is no TTL in v0.

## 8. Concurrency and streaming

v0 executes a single route sequentially. That is a deliberate simplification, not an
architectural limit.

- **Fan-out/ensemble** (deferred): the scheduler can run independent branches concurrently;
  budget accounting and trace ordering must become concurrency-safe first
  ([T-0018](../.scratch/tickets/T-0018-fanout-ensemble.md)).
- **Streaming** (deferred): the CLI may stream the final artifact from the last provider
  call for UX, but intermediate nodes produce structured artifacts. A `stream: true` option
  must not change graph semantics. See `Q-008`.
- **Parallel subgraphs** are not allowed in v0; one front branch only.
- **Thread/async model**: the runtime is async internally so provider calls do not block
  the event loop, even though v0 awaits one call at a time.

## 9. Observability

A trace is an append-only JSONL event stream. It is the primary debugging interface and
the source of training data.

Event types:

| Event | Key fields |
| --- | --- |
| `run.started` | run id, graph hash, profile, budget, seed |
| `node.entered` | node, kind, depth, step, path length |
| `render.completed` | prompt hash, message hashes, token estimate, truncated artifacts |
| `backend.plan` | mode, resolved models/adapters, VRAM estimate, admission result, fallback |
| `backend.loaded` | cold load ms, resident models, VRAM reserved |
| `backend.transition` | from/to node, host parse ms, tokenize ms, transfer ms, adapter switch ms, prefill ms |
| `provider.request` | provider, model, params hash, attempt |
| `provider.response` | usage, latency, finish reason, cached, provider meta |
| `decision.made` | decision, target, confidence, reason, threshold, tie-break |
| `artifact.written` | name, type, hash, producer, bytes |
| `budget.charged` | tokens, cost, elapsed, remaining |
| `prune.applied` | profile, removed nodes/edges, warnings |
| `node.exited` | node, duration, outcome |
| `run.finished` / `run.failed` | outcome, usage, exit artifacts, error |

Rules:

- Events are timestamped with a monotonic clock and a wall-clock timestamp.
- Content is referenced by hash; full prompts may be stored separately under `prompts/`
  when `--save-prompts` is enabled, subject to redaction.
- The trace never contains API keys or unredacted secrets.
- Backend transition metrics are recorded per hop; prefill is a separate field from host
  marshaling so the report does not blame transport for model compute.
- Logging is structured; human-readable logs are a rendering of events, not a second
  source of truth.
- OpenTelemetry export is a later integration ([T-0030](../.scratch/tickets/T-0030-observability.md)).

## 10. Failure semantics

| Scenario | Result |
| --- | --- |
| Validation failure | No run; error list; nothing persisted except the compile report. |
| Pruning removes the entry or an exit | No run; `PruneReport`; explicit failure. |
| Router low confidence with `error` | Run failed; partial artifacts and decision preserved. |
| Worker provider failure after retries | Run failed at that node; partial artifacts preserved. |
| Budget exhausted with `fail` | Run `budget_exhausted`; no result returned; artifacts preserved. |
| Budget exhausted with `best_effort` | Run `budget_exhausted`; artifacts written so far returned with a warning. |
| Caller cancellation | Run `aborted`; in-flight call allowed to finish or cancelled if supported. |
| Internal error | Run `failed`; trace flushed; error includes stack and run id. |

A failed run is still valuable: it is replayable, inspectable, and can be exported as
training data. The runtime must never delete a trace to make a failure look clean.

## 11. Deferred execution features

| Feature | Why deferred | Ticket / question |
| --- | --- | --- |
| Fan-out and ensemble joins | Needs concurrency-safe budgets, trace ordering, and a clear quality story | [T-0018](../.scratch/tickets/T-0018-fanout-ensemble.md) |
| Semantic response cache | Behavior can change silently; needs evaluation | Backlog B-01 |
| Tool/code execution nodes | Security and sandboxing dominate the design | `Q-004` |
| Streaming intermediate events | UX, not semantics | `Q-008` |
| Multi-turn conversations | Envelope history and routing state need design | `Q-009` |
| Distributed model placement | Keep v0 single-machine; adapters are already network-capable | Backlog B-09 |
| GPU-resident execution backends | Post-v0 performance track; v0 uses the portable backend | [T-0031](../.scratch/tickets/T-0031-resident-backend.md) |
| Recursive subgraphs | Loop guards and termination proofs are unsolved | `Q-007` |
| Speculative execution of branches | Cost model and cache semantics need work | Backlog B-02 |
