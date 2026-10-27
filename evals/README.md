# Evaluation suites

This directory contains the node, routing, and end-to-end evaluation fixtures for the
harness in `src/modelgraph/evaluation/`, plus the committed contribution-gate baselines. The
harness is offline-first: suites default to the deterministic mock provider, and every live
provider path is gated.

```bash
export UV_CACHE_DIR=$PWD/.uv-cache UV_PYTHON_INSTALL_DIR=$PWD/.uv-python
uv run mg eval node --suite evals/mock/js-specialist --provider mock
uv run mg eval routing --suite evals/mock/triage --provider mock
uv run mg eval e2e --suite evals/e2e/composed-assistant \
  --graph examples/composed-assistant.graph.yaml --provider mock --profile local-8gb
uv run mg eval e2e --suite evals/e2e/coding-specialist \
  --graph examples/coding-specialist.graph.yaml --provider mock
# Trained router + scripted mock workers, no network:
uv run mg eval e2e --suite evals/e2e/hello-trained \
  --graph training/graphs/hello-triage-trained.graph.yaml --provider mixed \
  --registry registry/models --registry training/models
```

Reports are written to `runs/evals/<report_id>/` (ignored by git) as `report.json`,
`report.md`, `manifest.json`, and one JSON file per case under `cases/`. An e2e report also
writes `e2e.json`, `oracle-labels.json`, and the runtime `runs/<run_id>/run.json` manifest
for every primary case.

## Suite layout

```text
evals/
  mock/
    js-specialist/
      suite.yaml
      cases.jsonl
    triage/
      suite.yaml
      cases.jsonl
    structured-contract/
      suite.yaml
      cases.jsonl
  e2e/
    composed-assistant/
      suite.yaml
      cases.jsonl
    coding-specialist/
      suite.yaml
      cases.jsonl
  fixtures/
    structured.graph.yaml
  baselines/
    policy.json               # the single reviewed gate policy
    node.json routing.json e2e.json mock.json
    update.py
```

`schemas/eval-suite.schema.json` is the canonical authoring contract for `suite.yaml` and
`#/$defs/case` for each JSONL line. The suite id, version, and a content hash over the
normalized suite document plus the bytes of `suite.yaml`, the cases file, and the rubric file
are recorded in every report. Changing any of those files changes the hash and invalidates
comparisons.

## `suite.yaml`

Minimal node suite:

```yaml
apiVersion: modelgraph/eval/v0
kind: EvalSuite
metadata:
  id: mock/my-node
  name: My node suite
  version: 1.0.0
  description: What this measures.
spec:
  level: node
  provider: mock
  claim: The claim this suite tests.
  target:
    graph: examples/coding-specialist.graph.yaml
    node: js-specialist
  metric: task_success
  thresholds:
    task_success: 0.8
    validity: 1.0
    protocol_error_rate: 0.0
  budget:
    max_cost_usd: 0.25
    max_tokens: 500000
  cases: cases.jsonl
  provenance:
    source: Hand-written prompts.
    contamination: checked
```

Routing suites use `level: routing` and a routing target:

```yaml
spec:
  level: routing
  target:
    graph: examples/hello-triage.graph.yaml
    router: triage   # optional; defaults to the graph entry node
  metric: oracle_agreement
  oracle:
    labeling_method: synthetic-authored
    labeling_cost_usd: 0.0
    ambiguity_rule: When both child menus plausibly fit.
  calibration:
    bins: 10
    threshold_steps: 20
```

Fields:

| Field | Meaning |
| --- | --- |
| `level` | `node` or `routing`. |
| `provider` | `mock` (default), `trained-router` (offline), or a live provider (`auto`, `ollama`, `openai-compatible`, `llamacpp`). Live runs are gated. |
| `claim` | Copied into the report; state what the evidence supports. |
| `target` | Node target (`graph`, `node`) or routing target (`graph`, optional `router`). |
| `metric` | Primary metric used for the gate and paired comparison. |
| `thresholds` | Metric to gate value. Higher-is-better metrics use `>=`; cost, latency, protocol error, and ECE use `<=`. |
| `budget` | `max_cost_usd` and `max_tokens` stop the suite; `max_case_*` override per-case runtime budgets. |
| `selection` | Deterministic subset: `case_ids`, `tags`, `tag_mode`, `limit`, `reason`. |
| `judge` | Pinned judge configuration when rubric checks are used. |
| `oracle` | Required for routing: labeling method, cost, sample source, ambiguity rule. |
| `calibration` | Reliability bin count and threshold-sweep steps. |
| `provenance` | Source, contamination status, notes, private flag. |
| `cases` | JSONL path relative to the suite directory. |
| `rubric` | Optional rubric path relative to the suite directory. |

## `cases.jsonl`

One JSON object per line. Node case:

```json
{"id":"js-001","tags":["domain:js","difficulty:easy"],"input":{"task":"Write a debounce function."},"mock":"export function debounce(fn, ms) { ... }","checks":[{"type":"artifact","artifact":"answer"},{"type":"contains","artifact":"answer","value":"debounce"}]}
```

`input.artifacts` supplies additional scoped inputs for a joiner or a worker with extra
reads. `metadata` is free-form and is passed to a judge hook. `weight` is used for weighted
node success; routing uses it together with `oracle.cost_usd` for cost-weighted accuracy.

Routing case:

```json
{"id":"tri-001","tags":["domain:coding"],"input":{"task":"Write a TypeScript function."},"oracle":{"route":"coding","quality":0.9,"quality_by_route":{"coding":0.9,"story":0.6},"cost_usd":0.001,"ambiguous":false},"mock":{"route":"coding","confidence":0.95,"reason":"TypeScript implementation."}}
```

`mock` is the offline script:

- a string is a raw provider response (use it to test malformed output);
- an object is serialized as JSON (use it for router decisions);
- an array is served in order, with the last response repeating for a repair attempt.

If a case has no `mock`, the mock provider's deterministic default applies. Suites that
claim reproducible quality should script every case.

## Checks

| `type` | Fields | Meaning |
| --- | --- | --- |
| `artifact` | `artifact`, `non_empty`, `min_bytes` | The declared artifact exists and is non-empty. |
| `exact_match` | `value`, `normalize` | Artifact content equals the value after `none`, `strip`, `casefold`, or `collapse_whitespace`. |
| `contains` | `value`, `normalize` | Substring check. |
| `regex` | `pattern`, `flags` (`i`, `m`, `s`) | Regular-expression search. |
| `json_schema` | `schema` or `schema_path` | Artifact parses as JSON and validates against Draft 2020-12. |
| `command` | `command`, `timeout_seconds`, `expect_exit_code`, `stdout_contains` | Runs a local trusted process. Placeholders: `{python}`, `{artifact_path}`, `{artifact}`, `{case_id}`. The artifact is written to a temp directory; a restricted environment is passed. |
| `rubric` | `min_score` | Calls the configured judge hook once per case and compares the score. Fails closed when no judge is configured or the judge output is unparseable/out of range. |

`artifact` defaults to the node's sole declared write. Multi-write nodes must name the
artifact on every check that needs one.

Command checks are local code from the suite author. Review eval suites like test code. They
can be disabled for a run (`allow_commands=False`); disabled checks fail rather than silently
pass.

## Metrics

Node metrics:

- `task_success`: cases where artifacts are valid and every check passes.
- `validity`: artifact contract satisfied (present, non-empty, JSON parses when required,
  output not truncated). A validity failure is still recorded with its raw provider
  responses and any partial artifacts.
- `run_success_rate`: runtime/provider calls completed without an exception.
- `check_pass_rate`: individual checks passed / checks total.
- `quality_score`: mean rubric score over judged cases (null when no judge is used).
- `cost_total_usd`, `cost_per_task_usd`, token counts.
- `latency_*`: wall latency (includes harness overhead).
- `protocol_error_rate`: fraction of cases with an `MG-2xxx` error; plus
  `protocol_errors_per_100_calls`.
- Slices by tag and a failure list with per-case reasons.

Routing metrics:

- `oracle_agreement`: correct child / non-ambiguous labeled cases. Ambiguous cases have no
  oracle-optimal child, so they are excluded and reported separately. `oracle_agreement_all_cases`
  includes them.
- `cost_weighted_accuracy`: sum of weights on correct non-ambiguous cases / total weight.
  Weight is the case weight times `oracle.cost_usd` when provided.
- `cost_weighted_error`: mean quality loss where `quality_by_route` is supplied.
- `oracle_gap`: mean oracle-route quality minus mean routed quality, when quality labels
  exist for every routed non-ambiguous case.
- Confusion matrix over child ids plus `abstain` and `error`.
- `abstain_precision`, `abstain_recall`, `abstain_rate` against the `ambiguous` label.
  A low-confidence fallback counts as an abstain.
- `ece` and reliability bins over routed non-ambiguous decisions; `threshold_sweep` reports
  coverage, selective accuracy, and abstain rate for each confidence threshold.

Intervals: Wilson 95% intervals for proportions; seeded percentile bootstrap for means and
p95 latency. The bootstrap seed and resample count are runner arguments and are recorded in
the report's determinism section.

## Oracle labels and their cost

Routing cases need an authored `oracle.route`, and may include `ambiguous`, `quality`,
`quality_by_route`, `cost_usd`, `misroute_loss`, and labeling metadata. The v0 fixtures use
synthetic labels: a human wrote the best route and ambiguity flag without running every child
for every case. That is cheap but weak evidence, and the suite says so in `spec.oracle` and
in the report limitations.

Measured oracle labeling follows `docs/evaluation.md` §3.1: render each router view, run every
candidate child or a stratified sample, score with the same metric, and choose the best
quality-per-cost child. It is expensive, so:

- label a stratified sample (the design guidance is 200–500 requests for routing claims);
- keep the labels in a versioned file with a hash (changing labels changes the suite hash);
- record the labeling method and total cost in `spec.oracle` and per case if useful;
- run a subset while developing with `selection.limit` or `selection.tags`, and state the
  subset reason in reports;
- do not fit and evaluate calibration on the same cases. v0 reports descriptive bins only;
  a held-out calibration split and temperature scaling belong to router training/calibration
  work (T-0024).

## Determinism and reproducibility

The mock provider uses case scripts and the runtime's token estimator. For scripted suites:

- quality, validity, cost, token counts, decisions, and artifact hashes are reproducible;
- wall latency, timestamps, and report ids vary;
- the report's `determinism.fingerprint` hashes the deterministic subset (suite hash, graph
  hash, selected case ids, metrics excluding latency, and case outcomes). Two runs with the
  same script and suite hash produce the same fingerprint.

## Live provider gate

Live providers are refused unless `MG_EVAL_LIVE=1` (or `allow_live=True` on the API). The
harness does not start servers or load weights. A gated smoke run is documented as a manual
step; it is not part of CI. Live output is non-deterministic, so repeat runs and record model
versions, provider, and hardware with the report.

## Reports

`report.json` validates against `src/modelgraph/evaluation/eval-report.schema.json`. It
contains the suite and graph hashes, the recorded CLI command and a runnable Python reproduce
command, the eval run manifest path, environment and git commit, metrics with
intervals, threshold checks, per-tag slices, failures, per-case prompts/responses/artifacts,
and routing-specific oracle/confusion/abstain/calibration sections. `report.md` is a rendering
of the same data; it is not a substitute for the JSON.

`compare_reports(candidate, baseline)` requires matching suite hashes, pairs cases by id, and
reports the aggregate delta, a seeded bootstrap interval, win/loss/tie counts, cost and
latency deltas, and the cases where the candidate is worse.

## Contribution gates

`evals/baselines/policy.json` is the single source for tolerances and hard gates. The offline
CI path is:

```bash
make eval-gates
```

That regenerates the mock candidate reports under `runs/evals/ci/`, compares each level
against `evals/baselines/`, writes the comparison reports, and exits non-zero once if any
level fails. A candidate whose report is missing a policy metric fails closed; a missing
baseline or policy is an explicit error (exit 3), not an empty pass. See
[`baselines/README.md`](baselines/README.md) for the update and waiver process and
[`docs/evaluation.md`](../docs/evaluation.md#6-contribution-gates) for the rationale.

## Known gaps

- Node evaluation executes the node executor directly; it writes an eval run manifest, not a
  runtime run directory for each case. E2E runs do write runtime manifests for primary cases.
- The judge path is a Python hook; there is no CLI flag that constructs a provider-backed
  judge, and the configured judge model is recorded by name rather than pinned by manifest
  hash.
- Calibration is descriptive; no calibration map is fit on a held-out split.
- The routing fixture's oracle labels are synthetic, not measured child quality. E2E oracle
  labels are measured by first-hop route probes but do not enumerate nested paths.
- E2E v0 passes the task text to the runtime; suite input artifacts are recorded but not
  injected by the current `run_graph` API.
