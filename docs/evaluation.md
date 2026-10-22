# Evaluation

Evaluation is the connective tissue of Model Graph. It decides whether a route is working,
whether a specialist is better, whether a graph is worth its cost, and whether a
contribution can be merged. Without it, composition is guesswork with extra steps.

The evaluation system has one job: **produce reproducible, cost-aware, quality-aware
evidence at every level of the graph.**

## Contents

1. [Evaluation levels](#1-evaluation-levels)
2. [Metrics](#2-metrics)
3. [Routing evaluation](#3-routing-evaluation)
4. [Baselines](#4-baselines)
5. [Suites and data hygiene](#5-suites-and-data-hygiene)
6. [Contribution gates](#6-contribution-gates)
7. [Statistical discipline](#7-statistical-discipline)
8. [Reporting](#8-reporting)
9. [Judge-based evaluation](#9-judge-based-evaluation)
10. [Anti-gaming](#10-anti-gaming)
11. [What evaluation cannot tell you](#11-what-evaluation-cannot-tell-you)

## 1. Evaluation levels

| Level | Question | Inputs | Output |
| --- | --- | --- | --- |
| **Node** | Is this model/prompt good at its declared job? | Node test set, rendered node prompts | Quality, validity, cost, latency |
| **Routing** | Does the router send requests to the right child? | Labeled requests + rendered menus | Accuracy, confusion, calibration, abstain |
| **Path** | Was the chosen path better than alternatives? | Same request, candidate paths | Oracle gap, cost of misroute |
| **Graph** | Does the whole composition beat baselines on real tasks? | E2E task suite, run profiles | Task success, quality, cost, p95 latency |
| **Suite** | Which configuration wins overall? | Multiple graphs/models/profiles | Pareto frontier, regression report |

A node can pass while the graph fails. That is not a contradiction; it usually means
routing or handoff is the problem. Evaluation reports always state the level.

## 2. Metrics

### 2.1 Quality

| Metric | Definition | Use |
| --- | --- | --- |
| **Task success** | Fraction of tasks where an automated check passes (tests, schema, exact match). | Primary for code, JSON, structured tasks |
| **Quality score** | Rubric score in `[0,1]` from a calibrated judge or human. | Writing, explanation, planning |
| **Win rate** | Paired comparison against a baseline, including ties. | Relative improvements |
| **Validity** | Fraction of outputs satisfying the node contract (artifact present, JSON parses, no truncation). | Reliability, especially small models |
| **Safety violations** | Fraction of outputs violating the suite's safety rules. | Hard gate, not a trade-off |

### 2.2 Cost and latency

| Metric | Definition |
| --- | --- |
| **Input tokens / output tokens** | Summed from normalized provider usage. |
| **Cost per task** | Estimated from manifest prices; report local compute separately. |
| **Wall latency p50 / p95** | End-to-end; model time and runtime overhead split. |
| **Time to first token (TTFT)** | Request start to first output token; routing latency is visible here. |
| **Inter-token latency (ITL)** | Steady-state decode time per output token. |
| **Transition latency** | Last token of node A to first prefill token of node B, split into host, transfer, adapter switch, and prefill. |
| **Runtime overhead ratio** | Non-model wall time divided by total wall time. |
| **GPU idle ratio** | Fraction of wall time with no kernel executing (best effort). |
| **VRAM peak / cold load** | Peak memory and startup cost of a resident backend. |
| **Hop count** | Node invocations per task. |
| **Cache savings** | Tokens/cost avoided by exact-match cache hits. |
| **Prune ratio** | Fraction of nodes/edges removed by a profile. |
| **Quality per dollar** | Primary quality metric divided by cost per task; report with confidence intervals. |

Do not report quality without cost, or cost without quality. An improvement that triples
cost is a different claim than an improvement at parity.

### 2.3 Reliability

| Metric | Definition |
| --- | --- |
| **Run success rate** | Fraction of runs completing without runtime/provider failure. |
| **Protocol error rate** | Invalid router decisions + malformed artifacts per 100 calls. |
| **Abstain rate** | Fraction of router decisions that abstain. |
| **Fallback rate** | Fraction of runs that needed the graph fallback. |
| **Budget exhaustion rate** | Fraction of runs stopped by a budget. |

A high task success rate with a 20% protocol error rate is not production-ready.

## 3. Routing evaluation

Routing is evaluated on a fixed set of requests with known oracle routes.

### 3.1 Producing oracle labels

For each request in the routing set:

1. Render the router's view exactly as at runtime.
2. Run each candidate child (or a stratified sample for expensive candidates).
3. Score each result with the same metric used for the node.
4. Choose the child with the best quality-per-cost under the evaluation policy. Record
   ties and near-ties; a 1-point difference at high variance is not a real preference.
5. Mark the request **ambiguous** if no child is clearly better or if the menu lacks a good
   option. Ambiguity is a label, not a failure.

Oracle labeling is expensive. Use it on a stratified sample (200–500 requests) and reserve
larger labeled sets for calibration.

### 3.2 Metrics

```text
oracle agreement      = requests routed to an oracle-optimal child / all requests
cost-weighted error   = Σ (quality loss of chosen child) / N
abstain precision     = abstained requests that were genuinely ambiguous / abstained
abstain recall        = genuinely ambiguous requests that were abstained / ambiguous
ECE                   = expected calibration error of confidence vs correctness
```

Report a confusion matrix over child IDs, not just aggregate accuracy. The interesting
failure is usually a specific pair of domains.

### 3.3 Calibration

A calibrated confidence is required for threshold policies to mean anything.

- Split the routing set into calibration and test partitions.
- Fit temperature scaling (or isotonic regression if data is plentiful) on calibration.
- Choose the threshold on validation using a cost matrix:
  - cost of a misroute = quality loss + downstream cost;
  - cost of abstention = fallback quality loss + latency.
- Report the reliability diagram and the threshold used.

If calibration is poor, prefer a conservative threshold and a fallback. An often-correct
router with wildly overconfident errors is worse than a timid one with a good fallback.

### 3.4 The oracle gap

The most important routing number is not accuracy; it is **how much quality is lost versus
oracle**. Compute:

```text
oracle_quality   = mean quality of the best child per request
routed_quality   = mean quality of the router's chosen child
oracle_gap       = oracle_quality - routed_quality
```

If the oracle gap is small, the router is not the bottleneck — improve the specialists or
remove the branch. If the gap is large, train the router before adding more experts.

## 4. Baselines

Every graph report includes these baselines:

| Baseline | Purpose |
| --- | --- |
| **Best single model** (same cost envelope) | Tests whether composition beats a monolith. |
| **Cheapest capable model** | Tests whether the graph is paying for quality it does not need. |
| **Oracle routing** | Upper bound for the current set of nodes. |
| **Incumbent graph** (previous version) | Detects regressions. |
| **Random router** | Sanity check; a learned router must beat it substantially. |
| **Always-fallback** | Tests whether the fallback alone is sufficient. |

A graph that loses to the best single model at equal cost is not yet a success, even if it
is more modular. Modularity is a means, not the metric.

## 5. Suites and data hygiene

### 5.1 Suite structure

```text
evals/
  js-specialist/
    suite.yaml          # id, version, metric, thresholds, cost budget
    cases.jsonl         # request + expected artifacts/tests + tags
    rubric.md           # judge rubric (if judged)
  routing/
    triage/
      cases.jsonl       # request + oracle route + ambiguity label
  e2e/
    coding-graph/
      cases.jsonl
      checks/           # test runners or validators
```

A `suite.yaml` declares the metric, the pass threshold, the sample size, and the expected
cost. Suites are versioned and content-hashed; changing a suite invalidates comparisons
unless both versions are reported.

### 5.2 Data sources

| Source | Strength | Risk |
| --- | --- | --- |
| Public benchmarks (HumanEval+, MBPP, GSM8K, MATH, etc.) | Comparable, cheap | Contamination, saturation, poor fit to real use |
| Curated internal tasks | Reflects actual use | Small, biased toward authors |
| Trace-derived tasks | Real distribution | Privacy, feedback loops, leakage |
| Adversarial/boundary cases | Finds routing confusion | Expensive to author |
| Synthetic tasks | Scalable | Distribution mismatch |

Use public benchmarks for external comparability, but never as the only evidence. Keep a
private held-out suite for decisions; rotate it periodically and retire leaked cases.

### 5.3 Contamination

- Check training data against suite cases (exact and near-duplicate).
- Prefer suites with a publication date after the base model's training cutoff, where
  possible.
- Record the check in the eval report. "Unknown" is an honest result; "assumed clean" is not.
- Do not tune prompts against the test set. Use a validation partition, then confirm once.

### 5.4 Sample size

| Claim | Minimum | Preferred |
| --- | --- | --- |
| Node smoke test | 20 cases | 100 |
| Node quality claim | 100 | 300+ |
| Routing accuracy claim | 200 | 500+ |
| E2E quality claim | 100 per domain | 300+ |
| Regression gate | 100 per domain | 300+ |

Small suites are for development, not for public claims. Report the sample size with every
number.

## 6. Contribution gates

These are **initial defaults**, recorded so they can be changed deliberately rather than
argued case by case. A gate can be relaxed for a research branch, never silently for a
release.

| Change | Must pass | Tolerance |
| --- | --- | --- |
| New specialist model | Node suite + graph regression | +5 points on node suite or parity at ≥20% lower cost; e2e no drop >2 points |
| New/updated router | Routing suite + graph e2e | Oracle agreement ≥85% (target 90%); e2e no drop >2 points; cost increase ≤10% |
| Prompt/template change | Node suite + replay comparison | No e2e drop >2 points; behavior diff reviewed |
| Graph topology change | Graph suite + all node suites | No drop >2 points; cost/latency reported |
| New provider adapter | Contract tests + one graph e2e | 100% contract pass; no quality drop >2 points |
| New execution backend | Logical-parity suite + transition benchmark | Decisions/artifacts identical under scripted responses; overhead gates in T-0033 met; fallback explicit |
| Dataset or eval change | Provenance + contamination check | Suite version bump; both versions reported |
| Dependency update | Full CI + one e2e suite | No quality drop >2 points; security check |

The 2-point tolerance is a placeholder for measurement noise with the chosen suites. If the
confidence interval is wider than the tolerance, the suite is too small or too noisy; fix
the measurement before tightening the gate.

### 6.1 Hard gates

Some things cannot be traded away:

- **Validity**: 100% of successful runs produce the declared exit artifacts.
- **Schema**: all graphs validate; all examples compile.
- **Safety**: zero safety violations in the safety suite for a release.
- **Security**: no secrets in traces or prompts (automated scan).
- **Budget**: no run exceeds its declared budget.
- **Backend parity**: a backend must produce the declared artifact text and routing
  decisions under scripted responses, and must not silently fall back or page weights.

Backend performance is reported separately from quality. A faster backend does not excuse a
quality regression, and a quality gain does not excuse a silent latency-class change.

## 7. Statistical discipline

- **Report uncertainty.** Use bootstrap confidence intervals for quality, routing, and cost.
- **Paired comparisons.** Run the same cases for both configurations; compare per-case
  differences, not just means.
- **Seeds.** For training and sampling, report at least 3 seeds for a claimed improvement.
- **Multiple comparisons.** If you test many nodes and keep the best, the win is inflated.
  Hold out a confirmation set or apply a correction.
- **Effect size over p-values.** A 0.3-point gain with a huge n is not a product win.
- **Pre-register the decision rule.** Write the threshold and stopping rule before looking
  at the test results.
- **Report failures.** A table of cases where the new node is worse is more useful than a
  single average.

## 8. Reporting

Every evaluation produces a machine-readable JSON report and a human-readable Markdown
summary. Both reference the run manifest and suite hashes.

```json
{
  "report_id": "eval_01J9Z...",
  "suite": {"id": "routing/triage", "version": "1.2.0", "hash": "sha256:..."},
  "graph": {"id": "composed-assistant", "version": "1.0.0", "hash": "sha256:..."},
  "profile": "local-8gb",
  "metrics": {
    "task_success": 0.812,
    "task_success_ci95": [0.771, 0.849],
    "cost_per_task_usd": 0.0021,
    "latency_p50_s": 2.4,
    "latency_p95_s": 6.1,
    "oracle_agreement": 0.887,
    "abstain_rate": 0.04,
    "protocol_error_rate": 0.003
  },
  "baselines": {"best_single_model": 0.79, "oracle": 0.891},
  "comparisons": [{"name": "incumbent", "delta": 0.008, "ci95": [-0.01, 0.03]}],
  "slices": {"domain:js": 0.86, "domain:general": 0.78},
  "failures": [{"case_id": "c-014", "reason": "artifact missing"}],
  "environment": {"python": "3.12.8", "provider": "ollama", "hardware": "..."},
  "commands": ["mg eval --suite routing/triage --graph composed-assistant@1.0.0"]
}
```

The summary must include:

1. The claim being tested.
2. The baseline and the delta with uncertainty.
3. Cost and latency, not only quality.
4. The slice where the change is worst.
5. The exact command and commit.
6. Known confounds (contamination, small n, non-deterministic provider).

## 9. Judge-based evaluation

For writing, explanation, and planning, automated tests are insufficient. Use a rubric-based
judge, with guardrails:

- **Rubric first.** Define the criteria and anchors before choosing a judge model.
- **Calibrate the judge** against human labels on a sample. Report agreement.
- **Pin the judge.** The judge model version is part of the eval configuration; changing it
  changes the measurement.
- **Blind comparisons.** Do not reveal which system produced which output; randomize order.
- **Control position bias.** Swap A/B order and average.
- **Fail closed.** If the judge output is unparseable, mark the case invalid; do not count it
  as a win.
- **Never use the judge as the only gate** for a release. Pair it with human review of a
  sample and task-specific checks.

Judge-measured gains are provisional until a human sample confirms them.

## 10. Anti-gaming

- **Frozen test sets** with access controls; contributors do not see the private slice.
- **Suite versioning** and review. Relaxing a test requires a new version and an explanation.
- **Held-out confirmation** for public claims.
- **Cost accounting cannot be waived.** A model that "wins" by using 100× tokens loses on
  the primary metric.
- **No cherry-picked seeds.** Report all seeds run, including bad ones.
- **No changing the baseline** after seeing results without reporting both.
- **Reproduce before merge.** A report without a command is an anecdote.

## 11. What evaluation cannot tell you

- **Whether users want the output.** Task success is not product value.
- **Whether a domain will stay stable.** Data drift is not captured by a frozen suite.
- **Whether composition is the right architecture.** The oracle gap can rule a graph out,
  but a small oracle gap for a bad set of experts just means the alternatives are also bad.
- **Whether a small improvement generalizes.** The only defense is diverse held-out data
  and honest uncertainty.
- **Whether the cost estimate matches your bill.** Provider pricing changes; local compute
  cost is not fully captured by zero-marginal-cost manifests.

Evaluation reduces uncertainty; it does not eliminate judgment. The design documents state
hypotheses so that evaluation can falsify them rather than decorate them.
