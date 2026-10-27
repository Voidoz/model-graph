# Graph model

A graph is a declarative description of how work moves between language models. This
document defines the node kinds, the edge and artifact semantics, the YAML anatomy, and
the rules a graph must satisfy before it can run.

- Schema: [`schemas/graph.schema.json`](../schemas/graph.schema.json)
- Examples: [`examples/hello-triage.graph.yaml`](../examples/hello-triage.graph.yaml),
  [`examples/coding-specialist.graph.yaml`](../examples/coding-specialist.graph.yaml),
  [`examples/composed-assistant.graph.yaml`](../examples/composed-assistant.graph.yaml),
  [`examples/ensemble-triage.graph.yaml`](../examples/ensemble-triage.graph.yaml)
- Decisions: [ADR-0003](adr/0003-four-node-kinds.md), [ADR-0004](adr/0004-typed-artifacts.md),
  [ADR-0005](adr/0005-compile-time-graph-runtime-routing.md),
  [ADR-0014](adr/0014-fanout-selection-and-joins.md)

## 1. Nodes

There are exactly four node kinds in v0. Each kind exists because it has a different
contract with the runtime; they are not cosmetic labels.

| Kind | Purpose | Must declare | May have outgoing edges? | Produces artifacts? |
| --- | --- | --- | --- | --- |
| `router` | Choose the next node for a request. | `model`, `description`; `routing` optional (defaults apply) | Yes, one or more | No |
| `worker` | Do the work with one model call. | `model`, `instruction`, `writes` | Only to a `joiner` (fan-out mode) | Yes |
| `joiner` | Combine artifacts from multiple upstream branches. | `model`, `writes`; `strategy` optional | Only to a `joiner` | Yes |
| `subgraph` | Run another graph as a node. | `graph`, `input_map`, `output_map`, `writes` | Only to a `joiner` | Yes |

### 1.1 Router

A router is a decider. It receives a scoped view of the request and the menu of its
outgoing edges, and returns a structured decision: route to one or more children, or
abstain. It does not write artifacts and normally does not answer the user.

```yaml
- id: coding-triage
  kind: router
  description: Distinguishes JavaScript/TypeScript work from general programming work.
  model: router-small
  reads: [task]
  routing:
    selection: single            # single | top_k | all
    confidence_threshold: 0.7
    on_low_confidence: fallback  # error | fallback | retry
    max_retries: 1
```

`selection` says how many menu routes one decision activates:

| `selection` | Activates | Requires | Use when |
| --- | --- | --- | --- |
| `single` (default) | Exactly the chosen route. | Nothing else. | The ordinary cascade. Behavior is unchanged from v0. |
| `top_k` | Exactly `k` routes, ranked by the router. | `k` (1..number of outgoing edges). | A small ensemble where the router can rank candidates. |
| `all` | Every outgoing route in deterministic menu order. | Nothing else. | A fixed ensemble; the router only decides whether to run it. |

`top_k` and `all` are fan-out modes: independent branches may run concurrently and a
downstream joiner waits for their declared inputs. They are opt-in because fan-out
multiplies token and cost usage; the runtime's budget pre-flight vetoes a fan-out that
cannot fit the remaining budget before any branch starts (see
[execution.md](execution.md#4-budgets)).

Router quality is the system's quality ceiling. A router is not a prompt hack; it is a
first-class, evaluated, trainable component (see [training.md](training.md#4-router-training)).

### 1.2 Worker

A worker is a leaf specialist. It receives its declared `reads`, its `instruction`, and
the user request, and writes one or more named artifacts.

```yaml
- id: js-specialist
  kind: worker
  description: Writes modern JavaScript and TypeScript, including tests.
  model: js-specialist-3b
  instruction: |
    You are a JavaScript and TypeScript specialist. Prefer standard library solutions.
    Write the implementation first, then a short list of assumptions.
    Put the final deliverable in the artifact named "answer".
  reads: [task]
  writes:
    - name: answer
      type: code
      media_type: text/typescript
```

Workers are leaves in v0 because arbitrary recursive delegation is how agent frameworks
become unbounded and unauditable. Nesting is expressed explicitly with a `subgraph`
node, which is statically visible and depth-budgeted.

### 1.3 Joiner

A joiner takes artifacts from several upstream branches and produces one. Under
single-selection routing a joiner usually has one incoming path at runtime; under
`top_k`/`all` it is the barrier where the selected branches meet. The scheduler runs the
joiner only after every declared `read` is available, records which branches contributed
and their token/cost usage, and fails closed with `MG-2201` if a declared input is missing.
The v0 graph contract has no `allow_partial` option; a partial join is not silently
accepted. See [execution.md](execution.md#22-fan-out-and-joins).

```yaml
- id: merge-answers
  kind: joiner
  description: Merges independent candidate solutions into one reviewed answer.
  model: coder-7b
  strategy: synthesize          # merge | pick_best | synthesize
  reads: [js_answer, generic_answer]
  writes:
    - name: answer
      type: code
```

A joiner's prompt includes each input artifact *with its name and producer*, and the
strategy. It must never receive an unbounded transcript; it receives the declared inputs.

### 1.4 Subgraph

A subgraph packages a graph as a node. This is the feature that makes "replace the coding
model with a coding decider plus specialists" a local edit rather than a rewrite.

```yaml
- id: coding
  kind: subgraph
  description: A nested graph for programming tasks.
  graph: coding-specialist@1.0.0
  input_map:
    task: task               # outer name: inner name
  output_map:
    answer: answer           # outer name: inner name
  reads: [task]
  writes:
    - name: answer
      type: code
```

Boundary rules:

- `input_map` keys are artifact names in the outer graph; values are input names of the
  inner graph. Missing required inner inputs make the graph invalid.
- `output_map` keys are artifact names exposed to the outer graph; values are inner graph
  exit artifacts. Unmapped inner artifacts are discarded.
- The subgraph gets its own envelope and path segment but shares the run's budget and
  depth counter. A nested graph cannot exceed the outer budget.
- Cycles through subgraph references are rejected. Recursion is not supported in v0.
- The subgraph's inner graph is pinned by `id@version`; the compiler resolves it to a hash
  and records the resolution in the run manifest.

## 2. Edges

An edge moves control (and, implicitly, language) from one node to another.

```yaml
edges:
  - from: triage
    to: coding
    when: The user asks for code, debugging, architecture, or technical implementation.
    examples:
      - Write a TypeScript function that debounces a callback.
      - Why does my Python test fail with a KeyError?
    artifact: code
    priority: 10
```

| Field | Meaning | Rules |
| --- | --- | --- |
| `from`, `to` | Node IDs. | Must exist; no self-loops; no duplicate pairs. |
| `when` | Natural-language routing criterion shown to the router. | Required on outgoing edges of a router; forbidden or ignored elsewhere. |
| `examples` | 1–5 short examples that disambiguate the edge. | Optional but strongly recommended. |
| `artifact` | Semantic type expected to flow on this edge. | Must be compatible with the target's writes; `text` is permissive. |
| `priority` | Tie-break order when the router is equally confident. | Higher wins; default 0; must be deterministic. |

Edges are the *only* place routing criteria live. A router's model does not get to
invent targets that are not in its menu.

## 3. Artifacts and types

Information crossing an edge is an **artifact**: named language with a semantic type. A
prompt is not an artifact; it is a rendering. A model's hidden state is never an artifact.

| Type | Meaning | Typical consumer |
| --- | --- | --- |
| `text` | Unstructured natural language. | Any node |
| `code` | Source code, optionally with prose notes. | Code reviewer, joiner, app |
| `patch` | A diff or a precise set of edits. | Apply/review nodes, app |
| `plan` | Ordered steps, dependencies, acceptance checks. | Workers, joiners |
| `json` | Machine-readable structured result. | App, joiner, validator |
| `report` | Findings, rationale, evaluation, or diagnosis. | Joiner, human, evaluator |

Compatibility is a deliberately small lattice:

```text
text  ⊇  code
text  ⊇  patch
text  ⊇  plan
text  ⊇  json
text  ⊇  report
```

- A consumer that accepts `text` accepts any artifact (everything is text on the wire).
- A consumer that requires `code` accepts only `code`.
- Edge `artifact` declares the expected type. If a producer writes a different type, the
  compiler warns and runtime records a type mismatch. It is not a hard failure by
  default, because models sometimes return a plan when asked for code; the mismatch is
  evidence for evaluation.
- Types are **semantic, not a security boundary**. Runtime validation is limited to what
  can be checked cheaply (JSON parseability, non-empty content, optional regex). A
  `json` artifact is not trusted input.

Why not free text only? Because the graph needs to validate what it can and because
evaluation and routing both improve when the deliverable has a name and a shape. Why not
strict schemas everywhere? Because models cannot reliably satisfy rich schemas at the
boundary, and over-constraining handoffs destroys the flexibility that makes language a
universal bus. See [ADR-0004](adr/0004-typed-artifacts.md).

## 4. Graph anatomy

A graph has identity, input/output contracts, defaults, nodes, and edges.

```yaml
apiVersion: modelgraph/v0
kind: Graph
metadata:
  id: hello-triage
  name: Hello Triage
  version: 1.0.0
  description: Routes between a coding specialist and a storyteller.
  labels:
    domain: general
spec:
  entry: triage
  inputs:
    - name: task
      type: text
      required: true
  exit: [answer]
  fallback: coding
  defaults:
    max_steps: 6
    max_depth: 2
    max_cost_usd: 0.05
    max_wall_seconds: 120
    max_total_tokens: 24000
  nodes: [...]
  edges: [...]
```

| Field | Meaning |
| --- | --- |
| `metadata.id` | Stable, lowercase, URL-safe identifier; unique in a registry. |
| `metadata.version` | Semantic version of the graph. Changing topology or contracts is at least a minor bump; changing behavior of a prompt is at least a patch. |
| `spec.entry` | First node executed. Must have no incoming edges. |
| `spec.inputs` | Named artifacts the graph requires. Defaults to one `text` artifact named `task`. |
| `spec.exit` | Artifact names returned to the caller. Each must be written by a sink node (or mapped by a subgraph). |
| `spec.fallback` | Node invoked when a router abstains or is below confidence and `on_low_confidence: fallback`. Must be reachable from the entry. |
| `spec.defaults` | Budgets and policy defaults. Node settings may tighten but not silently exceed them. |
| `spec.nodes` | 1..N nodes, unique IDs. |
| `spec.edges` | 0..N edges. A single-node graph with no edges is valid. |

### 4.1 Budget defaults

| Field | Default | Meaning |
| --- | --- | --- |
| `max_steps` | 8 | Node invocations per run (routers and workers both count). |
| `max_depth` | 3 | Maximum subgraph nesting. |
| `max_total_tokens` | 32,000 | Sum of input + output tokens across all calls. |
| `max_cost_usd` | 0.25 | Estimated cost ceiling using manifest prices. |
| `max_wall_seconds` | 300 | End-to-end wall clock ceiling. |
| `on_budget_exhausted` | `fail` | `fail` or `best_effort` (return artifacts produced so far). |

Defaults are opinions, not physics. A graph may raise them, but the runtime always
enforces whatever is in effect and records it in the run manifest.

## 5. Routing semantics

At runtime a router receives:

- the request text and any declared `reads` it asked for;
- a compact **route menu** derived from its outgoing edges and the target nodes'
  descriptions, examples, artifact types, and cost/latency metadata;
- instructions describing the decision JSON format.

It returns one of:

```json
{"decision": "route", "to": "coding-triage", "confidence": 0.87, "reason": "The request asks for a TypeScript implementation."}
```

```json
{"decision": "abstain", "confidence": 0.42, "reason": "The request is ambiguous between a tutorial and fictional dialogue."}
```

Policy:

1. If `decision == route` and the target is in the menu and confidence ≥ threshold, route.
2. If confidence is below threshold: `on_low_confidence` decides between `error`,
   `fallback` (graph-level fallback node), and `retry` (re-ask with a clarification or a
   stricter prompt, up to `max_retries`).
3. If the model returns an invalid decision, retry once with a repair prompt; then fail
   with a typed `MG-2101` router protocol error.
4. Ties are broken deterministically: confidence, then edge `priority`, then the menu
   order defined by node ID lexical order. The choice is recorded.
5. A router may be configured with `allow_direct_answer: true` only if the graph author
   accepts that a router can answer without a specialist; v0 defaults to false and the
   runtime treats a direct answer as a protocol error.

### 5.1 Multi-selection decisions

`single` keeps the decision shape above. For `top_k`, a router may return a ranking:

```json
{
  "decision": "route",
  "routes": [
    {"to": "js-specialist", "confidence": 0.82},
    {"to": "generic-coder", "confidence": 0.74}
  ],
  "confidence": 0.82,
  "reason": "A TypeScript task that also needs general review."
}
```

The runtime validates every target against the menu and selects exactly `k`. Ordering is
**deterministic and documented**: confidence descending, then edge `priority` descending,
then target ID lexical ascending. A router that returns fewer than `k` ranked routes is
filled from the same deterministic menu order. Router output order alone never decides a
tie. For `all`, the runtime ignores the chosen subset and activates every outgoing route in
that menu order; the decision's confidence still drives the confidence threshold and
`on_low_confidence` policy. Abstain and fallback keep their single-route meaning.

The uncertainty here is honest: a generative router's ranking is a preference signal, not
a calibrated distribution. Evaluate an ensemble against the best single branch before
using it ([evaluation.md](evaluation.md)).

The *routing card* is what the router sees about a child. By default it is derived from
the target node's `description`, edge `when` and `examples`, and model metadata. An
optional `routing_card` on a node can supply a shorter, hand-tuned card. Cards are part
of the compiled IR and are hashed for replay.

## 6. Composition patterns

### 6.1 Domain cascade

```text
domain router → sub-domain router → specialist
```

Use when domains separate cleanly and the cost of an extra small router call is worth
avoiding a large generalist. This is the canonical example in
[`examples/composed-assistant.graph.yaml`](../examples/composed-assistant.graph.yaml).

### 6.2 Specialist split

```text
router → {js specialist, python specialist, general coder}
```

Use when sibling specialists are genuinely different in training data or style. If the
`when` clauses overlap heavily, the router will be unreliable; merge the siblings.

### 6.3 Nested capability

```text
outer graph → subgraph node → inner graph
```

Use when a branch is itself a team. The boundary is an API: stable inputs, stable
outputs. Inner changes do not ripple outward if the mapping holds.

### 6.4 Fallback

```yaml
fallback: general-coder
```

Use when abstention should degrade to a known generalist rather than fail. The fallback
node is a normal node, reachable in the graph and evaluated with the rest.

### 6.5 Ensemble / fan-out

```text
router (selection: top_k) → several workers → joiner
router (selection: all)   → every worker     → joiner
```

Use when independent specialists produce useful disagreement or when a joiner can
synthesize a better answer than one branch. The scheduler activates the selected branches,
runs them with bounded concurrency, and waits at the joiner for all declared inputs. The
full worked topology is
[`examples/ensemble-triage.graph.yaml`](../examples/ensemble-triage.graph.yaml).

Fan-out is not free: `top_k` can multiply cost by roughly `k`, and `all` by the number of
outgoing routes, before the joiner's own call. The budget veto is mandatory, not advisory;
a fan-out that cannot fit the remaining budget fails before its branches start. Use fan-out
only when an eval report shows a quality or cost win over the best single branch.

## 7. Validation rules

The JSON Schema checks shape. The static validator checks meaning. Errors block
execution; warnings are reported and may be promoted to errors by CI.

| ID | Rule | Severity |
| --- | --- | --- |
| V-01 | Document validates against `graph.schema.json`. | Error |
| V-02 | Node IDs are unique and match the ID pattern. | Error |
| V-03 | `entry` exists and has no incoming edges. | Error |
| V-04 | Every edge endpoint exists. | Error |
| V-05 | No duplicate edge pairs and no self-loops. | Error |
| V-06 | The graph is acyclic (including subgraph references). | Error |
| V-07 | Every node is reachable from the entry. | Error |
| V-08 | Every sink node writes at least one artifact. | Error |
| V-09 | Every name in `spec.exit` is written by at least one sink, or mapped out of a subgraph. | Error |
| V-10 | A router has at least one outgoing edge and writes no artifacts. | Error |
| V-11 | Only routers have ordinary outgoing edges; workers, joiners, and subgraphs may point only to a joiner (fan-out mode) or terminate. | Error |
| V-12 | `when` is present on every router outgoing edge and absent elsewhere. | Error |
| V-13 | `when` texts and examples are non-empty; sibling examples are not near-duplicates. | Warning |
| V-14 | Each `reads` name is a graph input or is written by an ancestor node. | Error |
| V-15 | If multiple nodes write the same exit artifact with different types, the graph is still valid but the conflict is reported. | Warning |
| V-16 | `fallback`, if present, exists, is reachable, and writes every exit artifact it may need to return. | Error |
| V-17 | Model references resolve in the registry (when a registry is supplied). | Error (compile with registry) / Warning (offline validate) |
| V-18 | Subgraph references resolve and their input/output maps are complete. | Error (compile) / Warning (offline validate) |
| V-19 | Budgets are positive; thresholds are in `[0,1]`; node limits do not exceed graph defaults. | Error |
| V-20 | A joiner has at least one incoming edge (warning if fewer than two outside fan-out mode). | Warning |
| V-21 | `routing.k` is present exactly when `selection` is `top_k`, and `1 <= k <=` the number of outgoing edges. | Error |
| V-22 | Every route of a `top_k`/`all` router reaches a joiner; otherwise branch output may be ambiguous. | Warning |

The validator always reports **all** errors it can find, with a JSON path and a human
explanation, rather than stopping at the first. A graph that cannot be validated cannot
be compiled; a graph that is only warned about can run with `--allow-warnings`.

### 7.1 Pruning

Validation asks "is this graph coherent?". Pruning asks "is this graph runnable *here*?".
A capability filter (for example `no_network`, `max_vram_gb: 8`, `licenses:
[apache-2.0, mit]`) removes nodes whose models cannot run under the profile. The pruner
then:

1. Marks unsupported nodes.
2. Removes edges to them and, recursively, branches that can no longer reach an exit.
3. Verifies the entry and every exit artifact survive.
4. Emits a `PruneReport` listing removed nodes, removed edges, and the reason.
5. Recomputes router menus so the runtime never offers a branch that cannot run.

Pruning is a runtime policy, not a graph edit. The authored graph stays intact; the run
records the filter and the report. Details in [execution.md](execution.md#6-pruning-and-lazy-loading).

## 8. Graph versioning and identity

- A graph's identity is `metadata.id@metadata.version`.
- The compiler computes a **graph hash** over the canonical IR (nodes, edges, resolved
  manifests, prompts, defaults). Two runs with the same hash and the same model manifest
  hashes are comparable.
- `metadata.version` follows semver: breaking input/output or topology changes are major;
  adding a node or edge is minor; prompt-only changes are patch.
- A run records the graph hash, the version, and the resolved manifest hashes. Moving a
  tag without changing a version is prohibited; the hash is the source of truth.

## 9. Anti-patterns

| Anti-pattern | Why it hurts | Better move |
| --- | --- | --- |
| A router in front of every model call. | Adds latency and error probability for no quality gain. | Route only where domains are separable; measure oracle gain first. |
| Overlapping sibling `when` clauses. | Router cannot learn a boundary that does not exist. | Merge siblings or make the boundary a sub-decider. |
| Passing the entire transcript to every node. | Token cost, privacy leakage, context dilution. | Declare `reads`; render scoped views. |
| One worker per file or per obvious subtask. | Decomposition cost exceeds the benefit. | Keep sequential work in one prompt until evaluation says otherwise. |
| Fine-tuning a router before measuring the oracle gap. | Optimizes a problem that may not be the bottleneck. | Run the eval suite with oracle routing first. |
| A graph without an eval suite. | Cannot tell improvement from noise or regression. | Add node and e2e suites before promoting. |
| Using the graph to hide a bad model. | Routing cannot create capability that no node has. | Train or swap the specialist. |

## 10. Relationship to the other documents

- [protocol.md](protocol.md) — what exactly crosses an edge and how it is rendered.
- [execution.md](execution.md) — how the scheduler walks this graph under budgets.
- [training.md](training.md) — how routers and workers are produced.
- [evaluation.md](evaluation.md) — how a graph and its nodes are judged.
