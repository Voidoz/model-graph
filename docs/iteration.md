# Iteration and swapping models

The central promise of Model Graph is not that a graph runs once. It is that a user can keep
changing it — swap an expert, add a branch, retrain an adapter, recompile for speed — without
losing trust in what is running. Iteration is the product.

This document defines the lifecycle: what a user edits, what a swap invalidates, how candidates
are evaluated, how a revision is promoted, and how rollback works. It applies to every deployment
target: interpreted (B0), resident (B1), fused (B2), compiled (B2+), and distilled (B4).

See [ADR-0013](adr/0013-immutable-deployment-revisions.md) and
[performance.md](performance.md).

## 1. Three loops

A user moves between three loops, from cheap and fast to expensive and production-shaped.

| Loop | Question | Feedback | Typical gate |
| --- | --- | --- | --- |
| **Authoring** | Is the graph well-formed and does it behave plausibly? | Validation, mock run, trace, static analysis | None; local |
| **Improvement** | Is this candidate model/prompt/adapter better? | Node eval, routing eval, e2e comparison | Candidate vs incumbent, with uncertainty |
| **Deployment** | Is it safe and fast enough to be the new revision? | Compile, parity benchmark, canary, rollback plan | Promotion gates |

Distillation adds an **offline** loop: collect data, train a student, evaluate, and promote it as
a new deployment revision. It is not part of the inner authoring loop.

The loops must not be conflated:

- Local experiments should be fast and ungated.
- Promotion should be slow, evidence-based, and reversible.
- A compiled or distilled artifact should never be silently reused after its source changes.

## 2. Identity: graph revision, model revision, deployment revision

| Object | Identity | Changes when |
| --- | --- | --- |
| **Graph revision** | `graph.id@version` + canonical hash | YAML topology, prompts, edges, defaults change |
| **Model revision** | `model.id@version` + manifest hash + weights/adapter hash | Weights, adapter, tokenizer, or manifest change |
| **Deployment revision** | hash of `(graph hash, all model/adapter hashes, target, plan/student hash, compiler version)` | Any of the above, or a target/compiler change |
| **Change set** | A proposed diff + candidate models + eval evidence | The user is iterating, before promotion |

A **deployment revision is immutable**. Editing the graph or swapping a model produces a new
revision; it does not mutate the running one. Requests in flight pin to the revision they
started on. New requests use the new revision after an atomic promotion.

This is what makes rollback trivial: point at the previous revision. It is also what makes stale
artifacts detectable: a revision references the exact hashes it was built from.

## 3. What a swap changes

Not all swaps are equal. The blast radius depends on the node's role, the model family, the
deployment target, and whether a distilled student exists.

| Change | Affects | Must re-evaluate | Invalidates | Usual swap mode |
| --- | --- | --- | --- | --- |
| **Prompt/instruction only** | One node | Node suite; replay diff | Compiled prompt fragments; node prefix cache; distilled student (stale) | Hot plan rebuild; seconds–minutes |
| **Adapter swap, same base** | One node | Node suite | Adapter prefix cache; plan; distilled student | Hot if the engine supports concurrent adapters; minutes |
| **Model swap, same tokenizer** | Node + routing cards | Node suite; routing if menu/cost/latency changed; e2e | Plan; model/prefix caches; routing cache; distilled student | Hot if VRAM allows both; otherwise drain |
| **Model swap, different tokenizer** | Node + handoff | Node, routing, e2e | Plan; all B2 compatibility; prefix caches; distilled student | Not fused; use B1/B0 or rebuild the family; cold |
| **Router swap** | Routing decisions | Routing suite; e2e | Compiled route table/classifier; route cache; plan; distilled student | Hot; minutes |
| **Classifier/head swap** | Routing decisions | Routing suite; calibration | Route cache; plan | Hot; seconds |
| **Edge/topology change** | Routes and reachability | Validation, routing, e2e | All plans; route caches; distilled student | New revision; minutes–hours |
| **Base model swap in B2** | Every adapter | All node suites; routing; e2e; quality | Every adapter, plan, prefix cache, student | Migration, not a swap; retrain adapters; cold |
| **Target change (e.g., B2→B4)** | Deployment shape | Full parity benchmark; student eval | Student/plan; deployment trace shape | New target; offline training; canary |
| **Student refresh (B4)** | One deployment | Student eval vs graph and incumbent | New student artifact | Normal model deploy; canary |

The table encodes the main rule: **changing a node's weights does not just change that node.**
It changes the router's menu, the plan that pre-tokenized the prompt, the prefix cache, the
route cache, and any distilled student that learned the old behavior.

## 4. The safe swap workflow

```text
1. Propose       edit graph/model binding; create a change set
2. Explain       validate; show compatibility, invalidation, VRAM, eval coverage
3. Evaluate      node suite; routing suite if decisions changed; e2e regression
4. Compile       build the target plan/artifact; conformance with the reference target
5. Benchmark     TTFT, p95, throughput, cost, VRAM vs dense and incumbent
6. Stage         shadow or canary the new revision
7. Promote       atomic pointer switch; record revision and evidence
8. Observe       traces, outcomes, cost/latency by revision
9. Rollback      point back to the previous revision; keep artifacts for a grace period
```

Steps 1–3 are the improvement loop and are usually local. Steps 4–6 are the deployment loop
and are mandatory before a promotion that changes running behavior.

### 4.1 Step 1 — Propose

The user edits the graph or points a node at a new model/adapter. The edit is a **candidate**,
not an active revision.

```text
mg swap node js-specialist --to js-specialist-3b@0.4.0 --dry-run
mg swap node js-specialist --to js-specialist-3b@0.4.0 --open-change
```

For a no-op experiment, the user can run the candidate locally without creating a deployment
revision. A candidate becomes deployable only after the evaluation and compilation steps.

### 4.2 Step 2 — Explain before doing anything

`mg change explain` is the most important iteration command. It answers "what breaks if I do
this?" before the user spends time on evals.

```text
Change: js-specialist  js-specialist-3b@0.3.0 -> @0.4.0

Compatibility
  base model:        same (qwen2.5-coder-3b)
  tokenizer:         same
  adapter:           same rank, new weights
  license:           apache-2.0, allowed by profile
  VRAM delta:        +0.08 GB (adapter resident alongside incumbent)

Invalidates
  compiled plan:     fused-v3 (prompt tokens + adapter hash)
  prefix cache:      js-specialist adapter prefixes only
  route cache:       entries for js-specialist
  distilled student: hot-path-v2 is now stale (source graph changed)

Evaluation required before promotion
  node:    js-specialist suite (v1.2.0)
  routing: not required (menu, routing_card, cost, latency unchanged)
  e2e:     coding-graph suite (v3.1.0)
  perf:    required if the deployment target changes

Risk
  adapter swap is hot-swappable; old adapter remains resident for rollback
  no base/tokenizer migration

Rollback
  revision rev_01J... (current); artifacts retained for 7 days
```

The impact report is not advisory decoration. It drives which suites run and which artifacts
are rebuilt. If it cannot prove compatibility, the change is not promotable.

### 4.3 Step 3 — Evaluate

```text
mg eval node --node js-specialist --candidate js-specialist-3b@0.4.0 --baseline current
mg eval routing --suite evals/routing/triage          # only if decisions/menu changed
mg eval e2e --suite evals/e2e/coding-graph --baseline rev_01J...
```

The report must include quality, validity, cost, latency, and the slices where the candidate
loses. A candidate that improves the average but regresses a critical slice is not a win.

For prompt changes, also run a replay comparison: same inputs, old vs new prompts, diff the
artifacts. For routing changes, report oracle agreement, confusion, abstention, and calibration.

### 4.4 Step 4 — Compile and verify logical parity

```text
mg compile --target fused --change <change-id>
mg plan verify --against portable           # scripted conformance
```

Compilation is where a swap becomes deployable for a target. It also produces the invalidation
report: what changed, what was rebuilt, what remained cached, and what is stale.

A compiled target must pass the same logical-parity suite as the portable target under scripted
responses. A speedup that changes decisions or artifact text is a behavior change and needs
another evaluation loop.

### 4.5 Step 5 — Benchmark

Only if the change is performance-relevant (model size, target, routing, batching, prefixing):

```text
mg bench parity --revision <candidate-rev> --baseline dense --targets dense,fused
```

The result is quality-matched. If the candidate is faster but worse, it is a different operating
point, not a speed win.

### 4.6 Step 6 — Stage

For a local single-user workflow, "stage" can mean running both revisions on the eval set and,
optionally, a shadow replay of recent traces. For a served deployment, use shadow traffic and
then a canary.

```text
mg deploy --revision <candidate-rev> --shadow 100% --duration 1h
mg deploy --revision <candidate-rev> --canary 5%
```

A canary compares quality proxies, error rate, latency, and cost per revision. It does not
replace the eval suite; it catches distribution surprises.

### 4.7 Step 7 — Promote

```text
mg promote --revision <candidate-rev>
```

Promotion is an atomic pointer switch. In-flight requests finish on their original revision.
The previous revision remains addressable for rollback. The run manifest records the deployment
revision and the evidence used to promote it.

### 4.8 Step 8 — Observe

The revision stays visible in traces, run manifests, and evaluation reports. This is what makes
"it got worse after the swap" debuggable instead of anecdotal.

### 4.9 Step 9 — Rollback

```text
mg rollback --to <previous-rev>
```

Rollback is a pointer switch, not a recompile. Keep the previous artifacts, prefix caches, and
model revisions for the documented grace period. Delete them only after the new revision is
stable.

## 5. Hot, warm, and cold swaps

| Mode | When | Cost | Risk |
| --- | --- | --- | --- |
| **Hot** | Both revisions fit in VRAM and the engine can switch atomically (especially adapters) | Lowest disruption | Two revisions occupy memory; cache isolation must be exact |
| **Warm** | Candidate is loaded while the old revision drains; overlapping briefly | Brief double memory | Admission must reserve headroom |
| **Cold** | Base/tokenizer/target change, or VRAM cannot hold both | Downtime or drain | In-flight requests must be pinned or rejected |
| **Shadow** | Candidate runs on copies of requests without serving responses | Extra compute | Privacy and cost; not a user-visible rollout |
| **Canary** | A fraction of traffic uses the candidate | Extra compute/memory for both | Small sample may miss rare regressions |

Rules:

- **Adapters** are the easiest hot swap: load the new adapter beside the old, switch per request,
  keep the base resident. This is the strongest reason to standardize on B2.
- **Heterogeneous models** may require a warm or cold swap because both full models must fit.
- **Base/tokenizer changes** in a fused graph are migrations, not swaps. Do not promise
  zero-downtime; require retraining/re-binding and a new deployment revision.
- **In-flight requests** always pin to the revision they started on. A model swap must never
  change the semantics of a request halfway through.
- **Sessions** pin to a revision until the next turn by default; a re-route policy decides
  whether a new turn may move to the new revision. Without that, a conversation can switch
  branches without warning.
- **Caches** are keyed by the revision's model/plan hashes. Old cache entries are not reused by
  the new revision; they remain for rollback.

## 6. What a distilled target changes about iteration

B4 collapses the runtime to one student model. This makes iteration easier in one sense and
harder in another:

- **Easier:** swapping the student is a normal model deployment; there is no plan graph, route
  cache, or prefix tree to reason about at runtime.
- **Harder:** the student is **stale as soon as the source graph changes**. There is no automatic
  correspondence between "I swapped the JS expert" and "the student now routes like the new
  graph."
- **Non-negotiable:** the system must mark the student stale, retain the source graph revision,
  and require an explicit re-distill or an explicit acknowledgment that the old student remains
  the deployment.

Recommended policy:

1. Any source change marks B4 artifacts `stale`.
2. Stale students can continue serving, but `mg status` and the inspector show the drift.
3. Re-distillation is scheduled by change significance, drift, or a fixed cadence.
4. A new student is evaluated against the graph and the incumbent before promotion.
5. Rollback points to the previous student revision and its source graph.

An automatic hybrid ("send uncertain requests to the graph") is deferred (`Q-016`); it adds a
router and a second runtime path, which must be measured rather than assumed.

## 7. The UX surfaces

### 7.1 CLI

| Command | Purpose |
| --- | --- |
| `mg watch GRAPH [--once] [--interval S]` | Validate on save; rerun with the mock provider; fast authoring loop |
| `mg graph diff A B [--before-target T] [--after-target T]` | Topology, prompt, model-binding, policy, and deployment-target diff |
| `mg swap node NODE --to MODEL --dry-run` | Propose a swap and show impact |
| `mg change explain` | Compatibility, invalidations, eval coverage, VRAM, risk |
| `mg eval compare` | Candidate vs incumbent with CIs, slices, and an oracle reference when available |
| `mg compile --target ...` | Build the deployment target and report invalidations |
| `mg bench parity` | Quality-matched latency/throughput/cost/VRAM report |
| `mg deploy --shadow/--canary` | Stage a revision |
| `mg promote`, `mg rollback` | Atomic revision changes |
| `mg status --stale` | Plans, caches, students, and revisions that no longer match source |
| `mg inspect --html ... [--change ...] [--compare ...]` | Static impact panel, stale badges, timeline, and side-by-side runs |

### 7.2 Inspector

The visual inspector should make the impact visible before the user commits:

- node card: current model, candidate versions, eval delta, cost/latency delta, compatibility
  badges (base, tokenizer, license, VRAM);
- route view: what the router menu looks like after the swap;
- invalidation panel: plan, prefix cache, route cache, distilled student;
- "compare runs" side-by-side with artifacts and decisions;
- stale badges and a one-click "recompile" or "re-distill";
- revision timeline with promote/rollback markers.

### 7.3 Registry

Model versions are immutable. A version can be deprecated or yanked, not overwritten. A candidate
is a manifest plus evidence; selecting it in the UI is still a change set, not a live mutation.

## 8. A worked story

**Start.** A user has a single worker: `coder-7b`. They iterate its instruction, validate with
the mock, and run the node suite. This is the authoring loop; nothing is compiled.

**Add a specialist.** They add `coding-triage` and a `js-specialist-3b`. The compiler validates
reachability and routing edges. The mock run shows the decision path. The router suite needs
oracle labels; they run it and see JavaScript/TypeScript separation is reliable.

**Swap the JS model.** They propose `js-specialist-3b@0.4.0`. `mg change explain` says: same base,
same tokenizer, adapter-only, hot-swappable, plan invalid, JS prefix cache invalid, student not
yet in use, e2e regression suite required. They run node + e2e; the candidate wins 6 points on
JS and costs 20% more on that slice. They promote it locally.

**Compile for speed.** The graph runs on B2 with the base and adapters. `mg compile --target
fused` pre-tokenizes the prompts, builds prefix caches, and records a plan. `mg bench parity`
shows TTFT is within the target but the router still costs too much, so they flatten the
two-level cascade (`T-0035`) and compile again. This is a compiled revision, not a new model.

**Swap the base.** Later they want a stronger base. The impact report says all adapters are
incompatible, the compiled plan and prefix caches are invalid, and re-distillation is required.
This is a migration: retrain the adapters, run all node suites, rebuild the fused target, and
cold-swap or run a parallel deployment. The system refuses to treat it as an adapter swap.

**Distill the hot path.** For a stable high-volume workflow, they collect traces and train a
student. The student becomes a new deployment revision. A prompt tweak to the source graph marks
it stale; the old student keeps serving until a new student passes the gate. `mg status` shows
the drift. Rollback is a pointer to the previous student and its source revision.

## 9. Anti-patterns

| Anti-pattern | Why it fails | Better |
| --- | --- | --- |
| Editing a live deployment's model binding in place | No revision identity, no rollback, requests see different behavior mid-flight | New change set + deployment revision |
| Reusing a compiled plan after a model/prompt change | Silent stale prompts or wrong adapter | Hash-based invalidation; recompile |
| Promoting on node eval alone | Routing/e2e can regress | Node + routing-if-changed + e2e |
| Comparing candidates on different suite versions | Numbers are not comparable | Pin the suite hash; report both if changed |
| Swapping router and worker at once | Confounded results | One change at a time; then a combined confirmation run |
| Hot-swapping a base/tokenizer under load | KV/tokenizer incompatibility; unpredictable behavior | Migration with a new target, cold/warm swap |
| Re-distilling after every prompt edit | Wasteful and noisy | Change-significance policy; batch refresh |
| Keeping a stale student with no badge | Source/deployment drift | Stale status; re-distill policy |
| Deleting old artifacts immediately | Rollback is impossible | Grace period + retention policy |

## 10. Iteration metrics

The system should measure its own iteration behavior:

| Metric | What it tells you |
| --- | --- |
| Time from edit to validation | Authoring-loop friction |
| Time from candidate to node eval | Improvement-loop friction |
| Time to compile and load a target | Deployment friction |
| Time to rollback | Safety |
| Fraction of promotions with complete eval evidence | Trust |
| Number of stale plans/students/caches | Drift debt |
| Regressions caught before promotion | Value of the gates |
| Evaluation cost per promotion | Whether iteration is affordable |
| Candidate acceptance rate | Whether the registry is useful |

These are product metrics, not just engineering metrics. If iteration is slow, users will stop
modularizing and go back to a single prompt.

## 11. Relationship to the rest of the design

- [performance.md](performance.md) — why compilation and distillation exist, and what parity
  costs.
- [gpu-execution.md](gpu-execution.md) — backend residency, prefix caches, and admission.
- [evaluation.md](evaluation.md) — candidate comparison, gates, and uncertainty.
- [training.md](training.md) — adapters, specialist retraining, and graph distillation.
- [roadmap.md](roadmap.md) — Phase 4 change management and Phase 5 hot swap/canary.

The rule that ties them together: **the source graph is what you iterate; the deployment
revision is what you run. A swap is a new revision with an impact report, not a mutation.**

## 12. Implementation status (v0, T-0038/T-0040)

This section records what the repository actually implements today. It does not change the
accepted semantics above; where v0 can only approximate a concept, the approximation is named.

### 12.1 What is implemented

- `src/modelgraph/revision.py` — change sets, graph diff (including routing-menu highlighting),
  impact analysis, an immutable local revision store, an atomic active pointer,
  promote/rollback, evidence coverage, retention, and a read-only revision timeline.
- `src/modelgraph/watch.py` — the offline authoring loop used by `mg watch`: validate/compile,
  deterministic mock execution, content-based save detection, and previous-success comparison.
- `src/modelgraph/inspector.py` — the T-0028 inspector plus the T-0040 impact panel, stale
  artifact badges, revision timeline, and side-by-side run comparison.
- `schemas/change-set.schema.json` — the JSON contract for a persisted change set.
- CLI: `mg watch`, `mg graph diff`, `mg swap node`, `mg change create|list|show|explain`,
  `mg promote`, `mg rollback`, `mg status --revisions`, and
  `mg inspect --html ... [--change ...] [--compare ...]`.
- Tests: `tests/iteration/test_revisions.py`, `tests/iteration/test_changesets.py`,
  `tests/iteration/test_cli_changesets.py`, `tests/experience/test_inspector.py`, and
  `tests/experience/test_iteration_ux.py`. They use the mock provider and never touch the
  network.

The store defaults to `.mg/revisions/` (git-ignored) and is overridable with `--store` or
`MG_REVISION_STORE`:

```text
<store>/
  active.json                    # atomic pointer: revision_id + previous_revision
  events.jsonl                   # append-only promote/rollback/delete audit events
  revisions/<rev_id>/
    revision.json                # immutable revision manifest
    graph.yaml                   # the exact top-level graph source
    graphs/<subgraph>.graph.yaml # copied subgraph sources
    models/<model-id>.yaml       # resolved model manifests
    ir.json                      # canonical compiled IR
  changes/<chg_id>/
    change.json                  # status, bindings, evidence, promotion record
    impact.json                  # the impact report the user reviewed
```

A revision id is the existing run-manifest identity:
`deployment_revision(graph, target)`, a hash over the graph hash, resolved model hashes,
target, renderer/parser, and compiler version. Consequently `runs/<run-id>/run.json` already
records the same `deployment_revision` a revision would use. `mg run` additionally appends one
`revision.pinned` event to `trace.jsonl`, so traces name the revision without changing the
runtime trace schema or rewriting recorded provider events.

### 12.2 Commands

```text
# Bootstrap the first deployment revision (no prior active revision).
mg promote --graph graphs/coding-specialist.graph.yaml --store .mg/revisions

# Propose a swap. This persists a candidate revision and change set; active is unchanged.
mg swap node js-specialist --to js-specialist-3b@0.4.0 --store .mg/revisions \
  --registry tests/iteration/fixtures/models

mg change explain <change-id> --store .mg/revisions
mg graph diff <base-rev> <candidate-rev> --store .mg/revisions
mg change list --store .mg/revisions
mg change show <change-id> --store .mg/revisions

# Promotion requires evidence for the change class.
mg promote --revision <candidate-rev> --store .mg/revisions \
  --evidence runs/.../node/report.json \
  --evidence runs/.../routing/report.json \
  --evidence runs/.../e2e/report.json

mg rollback --to <previous-rev> --store .mg/revisions
mg status --revisions --store .mg/revisions
```

`mg swap ... --dry-run` compiles the candidate and prints the impact without writing a
revision or change set. `mg promote --dry-run` validates evidence and compatibility without
moving the pointer. `mg rollback --dry-run` shows the switch without moving it.

The v0 registry allows one version per model id. To version-swap with `id@version`, pass the
requested manifest path to `--to`, or pass an extra `--registry` directory containing that
version. The revision module loads target manifests through the normal `ModelManifest`
contract; it does not invent a version or mutate the active manifest.

### 12.3 Change classes and required evidence

The single mapping is `CHANGE_REQUIRED_EVIDENCE` / `required_evidence()` in
`modelgraph.revision`; `mg change explain` and `mg promote` call the same function.

| Change class | Required eval levels |
| --- | --- |
| `prompt` | node |
| `adapter_swap` | node |
| `model_swap` | node, e2e; routing too when a router menu/card/cost/latency changed |
| `base_swap` | node, routing, e2e, safety |
| `router_swap` | routing, e2e |
| `topology` | e2e |
| `target` | e2e, safety |
| `student` | e2e, safety |
| `initial` | none (explicit bootstrap only) |

Evidence is a normal evaluation `report.json`, loaded through the existing eval-report
contract and validated by `validate_report`. It counts only when all of the following hold:

1. the report schema and required fields validate;
2. its `graph.hash` (and graph id, when present) equals the candidate revision's graph;
3. its gate passed; and
4. it has at least one case result.

The suite id, report id, content hash, gate result, and any validation errors are recorded in
the change set. A report with a different graph hash, a failed gate, or no cases is listed as
unsatisfied evidence, not silently ignored. This is evidence-gated promotion, not a fresh
execution: the suites are run by `mg eval ...`; promotion does not re-run them.

### 12.4 Compatibility derivation and its limit

A model manifest in v0 has no dedicated `base_model` or `tokenizer` field. Impact analysis
uses `metadata.labels.base_model` (or `base`) and `metadata.labels.tokenizer` when they are
present. Otherwise it derives:

- base identity from `spec.model` (for example `qwen2.5-coder:3b`);
- tokenizer identity from the model family (`qwen2.5-coder`).

The derived match is conservative but not proof. Two manifests that happen to share a family
string are not guaranteed to share a tokenizer. Add the explicit labels before a shared
deployment. For `fused`/`compiled`/`distilled` targets, a changed base or tokenizer blocks the
change with an actionable migration message: use `--target portable` for a cold swap, or
retrain/rebind adapters and rebuild the model family. A portable target records the change as
a cold migration rather than pretending it is a hot swap.

### 12.5 Invalidations in the portable target

The portable target has no compiled plan, prefix cache, route cache, or distilled student.
The impact report therefore marks those artifacts `not_applicable` (with the changed scope and
a reason) instead of claiming an existing artifact was invalidated. For a `fused`, `compiled`,
or `distilled` target, the same graph/model/prompt change reports the plan and prefix cache as
`invalidated`, with `present: false`, because no plan is built in this ticket. Any source
change marks a distilled student `stale` (ADR-0013); no student artifact is tracked yet. This
is deliberately honest: a future compiled target must rebuild these artifacts, and no revision
manifest fakes a plan or student hash.

### 12.6 Atomicity, promotion, and rollback

- A revision directory is built in a temporary sibling and renamed into place. It is immutable
  after creation; the same inputs return the existing revision.
- A swap or graph change writes only a candidate revision and change set. The active pointer is
  untouched.
- Promotion first validates compatibility and evidence, then writes `active.json` with
  temp-file + `fsync` + `os.replace`, recording the previous revision. In-flight pinning and
  hot swap remain out of scope (T-0039); the pointer is the only switch v0 implements.
- A `fused`, `compiled`, or `distilled` revision records `plan_status: not_built`; it can be
  described by impact analysis, but `mg promote` and `mg promote --graph` refuse to make it
  active (`MG-6001`) because no such plan exists in v0. Those targets become promotable only
  after a compiled-target ticket builds and records the plan.
- Rollback is the same pointer write to a retained revision, with no recompile.
- `events.jsonl` is append-only and records the pointer history.

### 12.7 Retention (conservative Q-018 default)

The default grace period is **7 days**. Deleting a revision is refused when it is active, when
it is inside the grace period, or when an open (`draft`/`compiled`/`evaluated`/`promotable`)
change set references it. `RevisionStore.gc(dry_run=True)` reports eligible and blocked
revisions; the CLI does not delete automatically. This is the conservative default recorded
for `Q-018`; reference-counted GC across eval reports and the rest of the artifact store remains
a follow-up.

## 13. One concrete v0 loop: author, improve, evaluate, promote

This is one complete iteration using the commands implemented today. It is offline and uses
the deterministic mock provider for authoring and evaluation. Replace placeholder ids
(`<base-rev>`, `<change-id>`, `<candidate-rev>`, `<run-id>`) with the values printed by the
previous command.

### 13.1 Author with `mg watch`

```text
uv run mg watch examples/hello-triage.graph.yaml --once
uv run mg watch examples/hello-triage.graph.yaml --interval 1.0
```

The first command validates, compiles, and mock-runs once. The second keeps watching: on start
and after each content-changing save it validates, compiles, and mock-runs again. The report
shows the graph hash, the chosen path and decisions, artifact names/hashes, tokens, cost,
provider latency, and whether the last successful cycle differs. The run is recorded under
`--runs-dir` (default `runs/`).

Two honest limits:

- `mg watch` always uses `MockProvider`. It never selects a manifest's `ollama` or remote
  provider, never loads weights, and never calls the network. A save that needs a real model
  must go through `mg run` or an eval command.
- The loop reports recorded provider latency, not wall-clock time, so the JSON is deterministic.
  Wall-clock time is not reproducible and would make two identical cycles compare unequal.

`--json` prints one machine-readable cycle document (compact JSONL in loop mode):

```text
uv run mg watch examples/hello-triage.graph.yaml --once --json
```

A validation failure exits non-zero in `--once` mode and prints the compiler errors. In loop
mode the error is printed and the watcher keeps polling; `Ctrl-C` stops it cleanly with exit 0.

### 13.2 Improve and inspect the impact

```text
cp examples/hello-triage.graph.yaml /tmp/hello-triage.candidate.yaml
# edit the candidate YAML: prompt, topology, routing edge, or model binding
uv run mg watch /tmp/hello-triage.candidate.yaml --once
uv run mg graph diff examples/hello-triage.graph.yaml /tmp/hello-triage.candidate.yaml
```

`mg graph diff` reports topology, prompts, model bindings, policy, and deployment-target
differences, including the routing-menu changes that a topology-only reading can miss. Revision
ids carry their own target; for graph files, state it explicitly:

```text
uv run mg graph diff base.graph.yaml candidate.graph.yaml \
  --before-target portable --after-target resident
```

Bootstrap the active revision once, then propose the candidate as a change set:

```text
uv run mg promote --graph examples/hello-triage.graph.yaml --store .mg/revisions

# Preview the candidate without persisting a revision or change set.
uv run mg change create --graph /tmp/hello-triage.candidate.yaml \
  --base-revision <base-rev> --store .mg/revisions --dry-run

# Persist it, then explain it.
uv run mg change create --graph /tmp/hello-triage.candidate.yaml \
  --base-revision <base-rev> --store .mg/revisions \
  --summary "tighten the router prompt"
uv run mg change explain <change-id> --store .mg/revisions
```

`mg change explain` is the impact report: change class, base/tokenizer compatibility, required
evaluation levels and current coverage, invalidations, VRAM admission, rollback target, and
risks.

Render the same stored impact report in the inspector, next to a recorded run:

```text
uv run mg inspect runs/<run-id> --html \
  --graph examples/hello-triage.graph.yaml \
  --out /tmp/hello-triage.html \
  --change <change-id> --store .mg/revisions

uv run mg inspect runs/<run-id> --html \
  --graph examples/hello-triage.graph.yaml \
  --out /tmp/hello-triage-compare.html \
  --compare runs/<other-run-id>
```

The page's embedded `modelgraph/inspector/v0` document and the read-only/no-network boundary are
documented in [`inspector.md`](inspector.md).

The inspector shows the impact before promotion, stale badges for plan/cache/student artifacts,
a revision timeline with active/previous/candidate/stale roles, and safe follow-up commands as
text. It never promotes, rolls back, recompiles, or re-distills. In v0 there is no compiled plan
or distilled student artifact, so those badges read `not_applicable` or `none` rather than
inventing a hash.

### 13.3 Evaluate with uncertainty

```text
uv run mg eval node --suite evals/mock/js-specialist --provider mock \
  --reports-dir runs/evals/ci --json > runs/evals/ci/node.json

uv run mg eval compare \
  --baseline evals/baselines/node.json \
  --candidate runs/evals/ci/node.json \
  --reports-dir runs/evals/ci/comparisons
```

`mg eval compare` prints the primary delta and its 95% bootstrap interval, cost and latency
deltas with intervals, the gate checks, and per-slice paired deltas with per-slice intervals.
When both reports carry e2e data and the candidate carries oracle or best-single baselines, it
also prints a candidate/baseline/oracle/best-single reference table; this is a comparison of
measured reports, not a claim that the oracle is deployable. The persisted comparison report
keeps the same data under `metrics`, `comparisons`, `gate`, `costs`, `latency`, and `slices`.
The CLI `--json` output additionally includes `reference` when available; the persisted
`report.json` does not. A slice is a tag or a `domain:`/`difficulty:` metadata dimension from
the suite cases.

### 13.4 Promote with evidence, or dry-run first

```text
uv run mg promote --revision <candidate-rev> --store .mg/revisions \
  --evidence runs/evals/ci/node.json \
  --dry-run

uv run mg promote --revision <candidate-rev> --store .mg/revisions \
  --evidence runs/evals/ci/node.json
```

The dry-run validates compatibility and evidence coverage without moving the active pointer.
The real promotion writes `active.json` atomically and records the previous revision for
rollback. A change whose required evidence is missing is refused, not silently promoted:

```text
uv run mg promote --revision <candidate-rev> --store .mg/revisions --dry-run
# blocked: missing passing evidence for e2e (and routing, when the menu changed)

uv run mg rollback --to <base-rev> --store .mg/revisions --dry-run
uv run mg rollback --to <base-rev> --store .mg/revisions
```

### 13.5 What the loop intentionally does not do in v0

- It does not build compiled or fused plans, prefix/route caches, or distilled students. Impact
  analysis and the inspector mark those artifacts `not_applicable`/`none`; they never fake a
  plan or student hash.
- It does not promote a `fused`, `compiled`, or `distilled` target, because no plan exists to
  make active. Use `--target portable` until a compiled-target ticket builds one.
- It does not auto-promote, auto-recompile, or auto-re-distill. The inspector's follow-up
  commands are text; promotion remains an explicit, evidence-gated command.
- It does not add a filesystem-watch dependency. `mg watch` polls the graph file's bytes with
  the standard library and reruns only when the content changes.
