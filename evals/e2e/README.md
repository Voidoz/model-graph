# End-to-end evaluation suites

`mg eval e2e` runs a compiled graph (and optional run profile) over a task suite, grades the
graph's exit artifacts, and compares the graph with four measured baselines plus an incumbent
report. The deterministic mock path runs offline in CI; a live provider is gated by
`MG_EVAL_LIVE=1` or `--live`.

```bash
# Offline, deterministic
uv run mg eval e2e \
  --suite evals/e2e/composed-assistant \
  --graph examples/composed-assistant.graph.yaml \
  --provider mock \
  --profile local-8gb

# Gated local model run
MG_EVAL_LIVE=1 uv run mg eval e2e \
  --suite evals/e2e/composed-assistant \
  --graph examples/composed-assistant.graph.yaml \
  --provider ollama \
  --profile local-8gb
```

Reports are written to `runs/evals/<report_id>/`:

| file | contents |
| --- | --- |
| `report.json` | machine-readable `modelgraph/eval-report/v0` report |
| `report.md` | human summary including baselines, Pareto table, failure slices |
| `manifest.json` | eval run manifest; references the report, `e2e.json`, and oracle labels |
| `e2e.json` | the full e2e block: baselines, labels, Pareto, judge, run manifests |
| `oracle-labels.json` | measured/per-case oracle labels with ambiguity flags |
| `cases/` | one JSON file per case, including raw responses and check results |
| `runs/` | the runtime `run.json`/trace for each primary graph case |

## Suite shape

An e2e suite is a normal `modelgraph/eval/v0` suite with `level: e2e`, a `target.graph`
(optional `target.profile`), and at least one check per case. Cases use `mock_nodes` to script
one response sequence per node id; `mock` remains the node/routing suite field and is ignored
by e2e.

```yaml
spec:
  level: e2e
  target:
    graph: examples/composed-assistant.graph.yaml
    profile: local-8gb        # optional; --profile overrides
  metric: task_success
  thresholds: {task_success: 0.6, validity: 0.8}
  baselines:
    best_single: true
    cheapest_capable: true
    oracle_routing: true
    random_routing: true
    measure_oracle: true
    oracle_ambiguity_delta: 0.05
    random_seed: 7
```

A case can also carry authored oracle labels (`oracle.route`, `oracle.quality_by_route`,
`oracle.ambiguous`) for documentation or for `measure_oracle: false`. Measured labels win when
enabled.

## Baselines

- **Best single**: runs each worker/joiner node directly on every task, grades the same checks,
  and keeps the best eligible node. Eligibility is the graph/suite cost envelope. It measures
  whether composition beats one worker; it deliberately excludes routing and handoff.
- **Cheapest capable**: among nodes with non-zero task success, picks the lowest cost per task,
  breaking ties by success then name.
- **Oracle routing**: forces the entry router through every candidate branch using a provider
  wrapper that only answers the router call, then grades the same exit checks. The best branch
  is the oracle label; near-ties become ambiguous. The oracle gap uses only non-ambiguous cases.
- **Random routing**: seeded uniform choice over the measured branches. It is a sanity check,
  not a credible policy.
- **Incumbent**: pass a previous e2e `report.json` (or its directory) with `--baseline`. The
  report must use the same suite hash; otherwise the comparison is reported as unavailable.

Add a baseline by adding a suite flag, implementing a branch in `_single_node_baselines` or
`_measure_oracle`, and including it in `_apply_pareto`'s entries. Do not change an existing
baseline after looking at results without reporting both versions.

## Reading the Pareto report

`e2e.pareto.configurations` is sorted by quality (task success) then cost. `frontier: true`
marks configurations not dominated by another configuration with at least as much quality and
no more cost. `bands` names the highest-quality configuration at each cost ceiling. All example
model manifests are free, so every configuration sits in the same `$0` band; that is why the
oracle is the only frontier point in the mock fixture.

Never compare configurations at different budgets and call the difference quality. Quality and
cost must appear together, with the suite hash and case count.

## Judge and statistics

Rubric checks use the suite's configured judge hook and fail closed. `score_blinded_pair`
scores two outputs with neutral `A`/`B` slots and calls the hook twice (AB then BA), with the
target output in slot `A` both times. That gives each output the same slot, so the pair delta
is not confounded by a fixed A/B position preference; it is not a two-sided position-bias
estimate. The judge model, rubric hash, temperature, and output cap are recorded in the
report. Calibration is reported only when cases carry a numeric `metadata.human_score` (or
`human_quality`); otherwise the report says calibration is unavailable.

Paired comparisons over the same cases report the mean quality delta, a seeded bootstrap 95%
interval, wins/losses/ties, cost delta with interval, latency delta with interval, and Cohen's
d.

## Limitations

- Mock metrics validate the harness and suite, not model quality.
- Best-single baselines run each node on the full task; they are not full single-model graphs.
- Oracle labels force only the first router. Nested routing follows the suite's mock scripts or
  the live provider.
- The first-hop oracle baseline requires measured branch runs; authored labels alone compute an
  oracle gap but not an oracle baseline.
- E2E v0 passes the task text to the runtime; suite input artifacts are recorded but not
  injected by the current `run_graph` API.
- Wall-clock latency includes harness overhead and is excluded from the determinism fingerprint.
