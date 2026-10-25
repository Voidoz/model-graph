# CI gate baselines and waivers

This directory is the committed reference point for the offline contribution gates. It is
reviewed like test data: a normal feature PR must not move a baseline as a side effect.

## What is committed

| File | Contents |
| --- | --- |
| `policy.json` | The single source of truth for tolerances and hard gates. |
| `node.json` | Mock `evals/mock/js-specialist` node report. |
| `routing.json` | Mock `evals/mock/triage` routing report. |
| `e2e.json` | Mock `evals/e2e/composed-assistant` report with five cases. |
| `safety.json` | Mock `evals/safety/model-safety` safety fixture report (zero violations). |
| `mock.json` | Alias of `e2e.json`, used by the ticket smoke command. |
| `update.py` | Regenerates the four baseline reports and the e2e alias. |

`e2e.json` intentionally excludes the `missing-artifact` fixture case. That case exists to make
the harness report a validity failure, but the release hard gate requires 100% validity. The CI
e2e candidate uses the same five case ids so the unchanged graph passes and a new validity
failure still fails immediately.

The baselines are full `modelgraph/eval-report/v0` reports: they carry the suite hash, graph
hash, case results, metrics, decisions/responses, and determinism fingerprint. They are
generated with `persist=False` deliberately so they do not embed absolute run-directory paths
from the generating machine; generated candidate reports are the ones that also carry runtime
run-manifest paths. `compare.py` refuses a candidate whose suite hash differs; a graph-hash
change is expected for a contribution and is reported, not rejected.

## Thresholds

`policy.json` is the only place tolerances live. It is loaded by
`src/modelgraph/evaluation/compare.py` and is linked from
[`docs/evaluation.md` section 6](../../docs/evaluation.md#6-contribution-gates). The defaults
implement the initial documented tolerances:

- node: task success, run success, and protocol-error rate may each regress by at most 2 points
  (`0.02`);
- routing: oracle agreement, run success, and protocol-error rate may each regress by at most
  2 points; cost per task may rise by at most 10% relative to the baseline;
- e2e: task success, run success, and protocol-error rate may each regress by at most 2 points;
- safety: a suite labelled `level: safety` uses the `safety` level; `safety_violations` may not
  rise (tolerance `0`), and its task-success, run-success, and protocol-error metrics keep the
  same 2-point tolerances as the other levels. The safety hard gate is *required* for that
  level: a safety report that omits `metrics.safety_violations` and the per-case
  `safety_violations` field fails rather than reporting `not_measured`;
- hard gates: validity must be 100% and must not fall below the baseline, safety violations must
  be zero when the suite reports them, no run may stop on its budget, the candidate report must
  validate, and the candidate report is scanned for credential-shaped strings.

Each metric delta is reported with a seeded, paired-bootstrap 95% confidence interval. The
decision uses the point estimate against the tolerance, as the docs describe; the interval is
evidence about noise, not a second acceptance test. If the interval is wider than the tolerance,
the correct fix is a larger or less noisy suite, not a smaller gate.

## Running the gates

Offline (no model server and no network):

```text
make eval-gates
```

That target regenerates the three mock candidate reports under `runs/evals/ci/`, then runs:

```text
uv run mg eval compare --baseline evals/baselines/node.json \
  --candidate runs/evals/ci/node.json --reports-dir runs/evals/ci/comparisons
uv run mg eval compare --baseline evals/baselines/routing.json \
  --candidate runs/evals/ci/routing.json --reports-dir runs/evals/ci/comparisons
uv run mg eval compare --baseline evals/baselines/e2e.json \
  --candidate runs/evals/ci/e2e.json --reports-dir runs/evals/ci/comparisons
```

The measured safety level is generated and compared with the same APIs (no model server):

```text
uv run python evals/safety/run.py --json > runs/evals/ci/safety.json
uv run mg eval compare --baseline evals/baselines/safety.json \
  --candidate runs/evals/ci/safety.json --reports-dir runs/evals/ci/comparisons
```

`evals/safety/run.py` exits 1 on a measured violation and 3 on an input/contract error. `make
eval-gates` runs this pair with the other comparisons, so the safety hard gate runs on every PR;
the safety report itself shows `hard.safety` as `pass`/`fail` instead of `not_measured`.

`mg eval compare` exits 0 when every gate passes and 1 when any gate fails. The comparison
report is written as `report.json` + `report.md`, and the Markdown contains the per-metric table,
hard-gate table, and any waivers. The CLI prints the same decision summary. A missing baseline
or policy, or a candidate with a different suite hash, is an explicit error (exit 3), never an
empty pass. A policy metric that is absent or null in either report fails its check: dropping a
gate metric is a gate failure, not an unmeasured pass.

## Updating a baseline

Baseline changes are a separate reviewed commit:

1. Change the suite/graph/code and make the ordinary tests pass.
2. Run `make baselines`. This runs the offline mock suites through the real compiler/runtime and
   rewrites `node.json`, `routing.json`, `e2e.json`, and `safety.json`.
3. Review the semantic diff: suite hash, graph hash, metric values, changed cases, and the
   determinism fingerprint. Ignore expected noise in `generated_at`, `report_id`, environment
   paths, and wall-clock latency fields. For `safety.json`, also review every case's
   `safety_rules_checked` and `safety_violations` list; a rule change changes the safety
   fingerprint and is a reviewed baseline move, not a silent fix.
4. Open a PR that contains **only** the baseline files (or labels clearly why the baseline moved
   with the code change). The baseline owner — the maintainer who owns the affected suite —
   reviews and approves it. A feature PR cannot approve its own baseline move.
5. If the new baseline is intentionally worse, say so in the PR description and record the
   accepted trade-off; a lowering baseline is a change to the contribution standard.

The committed reports are generated by the same APIs as normal evaluation; neither a dashboard
nor a hosted benchmark is involved.

## Waivers

A waiver is an explicit JSON file, not a CLI flag that silently skips a check:

```json
{
  "schema": "modelgraph/gate-waiver/v0",
  "approved_by": "maintainer-name",
  "reason": "Known tokenizer artifact; tracked in issue #123.",
  "checks": ["routing.cost_per_task_usd"],
  "expires": "2026-11-05",
  "suite_hash": "sha256:..."
}
```

`approved_by`, `reason`, `checks`, and `expires` are required. `expires` is an ISO date or the
literal `pr`; an expired waiver is rejected. `suite_hash`/`graph_hash` optionally scope the
waiver. Pass one or more files with `mg eval compare --waiver <file>`.

Only quality/reliability checks can be waived. Entries naming `hard.validity`,
`hard.schema`, `hard.safety`, `hard.budget`, `hard.secret_scan`, or a hard gate name are
recorded as rejected and the hard gate still fails. Applied and rejected waivers both appear in
`report.json` (`gate.waivers`) and in `report.md`; the gate passes only for applied soft checks.
A CI waiver is written from an explicit PR comment into a temporary file and is never committed,
so it expires with the PR by construction.

## CI and nightly real-model runs

`.github/workflows/ci.yml` runs `make check` and the offline `make eval-gates` job on every pull
request. The comparison reports and candidate reports are uploaded as workflow artifacts and the
Markdown is written to the job summary. No step needs a model server or network access beyond
the checkout and dependency setup.

Real-model suites are manual/nightly only. Set the repository variable `MG_LIVE_RUNNER` to a
self-hosted runner label to enable the scheduled live job; without that variable the job is
skipped. The live job uses `MG_EVAL_LIVE=1` and the same runner/report code, then uploads its
reports. It is not part of the default PR path because hosted runners do not have a model
server.
