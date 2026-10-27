# Training

The economic argument for Model Graph rests on one claim: **a node can be improved by a
person with a narrow dataset and ordinary hardware, and that improvement can be proven and
composed without retraining the system.** This document describes how that works and where
it does not.

Training is deliberately simple in v0. We do not train foundation models, we do not do
end-to-end differentiation through graphs, and we do not promise that composition replaces
large-scale pre-training. We provide recipes for routers and specialists and a strict
artifact contract so improvements are reproducible.

## Contents

1. [Ownership model](#1-ownership-model)
2. [What can be trained](#2-what-can-be-trained)
3. [Data](#3-data)
4. [Router training](#4-router-training)
5. [Specialist training](#5-specialist-training)
6. [Joiner training](#6-joiner-training)
7. [Hardware tiers](#7-hardware-tiers)
8. [Training artifacts](#8-training-artifacts)
9. [The data flywheel](#9-the-data-flywheel)
10. [Limits and honest failure modes](#10-limits-and-honest-failure-modes)
11. [Governance](#11-governance)

## 1. Ownership model

| Artifact | Owner | Versioned independently | Promotion gate |
| --- | --- | --- | --- |
| Graph topology and prompts | Graph author | Yes (semver) | Graph eval suite |
| Model (specialist or router) | Model contributor | Yes (`id@version`) | Node eval + graph regression |
| Dataset manifest | Data contributor | Yes (hash) | License/provenance review |
| Eval suite | Evaluator | Yes | Review; contamination check |
| Model card | Model contributor | With the model | Completeness check |
| Run trace | Operator | Immutable per run | Redaction policy |

A contributor can own a node without owning the graph. A graph author can upgrade a node
without retraining anything. The contracts in [`schemas/`](../schemas/) are what keep those
two people from having to coordinate.

## 2. What can be trained

| Target | Method | Typical size | Impact |
| --- | --- | --- | --- |
| Specialist worker | SFT, LoRA/QLoRA, distillation | 0.5B–14B | Higher quality in a narrow domain |
| Router | Fine-tune a small model or train a classifier | 0.3B–3B / embedding model | Fewer misroutes, better abstention |
| Joiner | SFT on merge/synthesis examples | 1B–8B | Better conflict resolution |
| Prompt/instruction | Manual or automated search | n/a | Cheap gains, must be evaluated |
| Graph topology | Human design + ablation | n/a | Routing quality; hardest to automate |

Not trainable in v0: cross-node weights, shared latent state, end-to-end graph objectives.
Those are research directions ([ADR-0001](adr/0001-language-as-the-wire.md)), not roadmap
commitments.

### 2.1 Why routers first

The router is the highest-leverage trainable component. A specialist can be excellent and
still useless if it never receives the right requests. Router errors compound: a misroute
sends the request to a model that cannot succeed, and the graph has no second chance by
default. Training a router is also cheaper than training a specialist (classification-scale
data, small models), so it is the best place to demonstrate the flywheel.

## 3. Data

Data quality is the main determinant of node quality. The project treats data as a
first-class artifact, not a folder of JSONL.

### 3.1 Dataset manifest

Every training run references a dataset manifest:

```yaml
apiVersion: modelgraph/v0
kind: Dataset
metadata:
  id: js-specialist-sft
  version: 0.2.0
spec:
  task: "JavaScript/TypeScript implementation"
  sources:
    - uri: "hf://datasets/example/js-sft"
      revision: "abc123"
      license: "apache-2.0"
      split: "train"
  preprocessing:
    dedupe: exact+minhash
    filters: ["min_tokens: 32", "max_tokens: 4096", "no_pii"]
    split_strategy: "group_by_repository"
  counts: {train: 8400, validation: 600, test: 600}
  hash: "sha256:..."
  eval_contamination: "checked against evals/js-specialist@1.0.0"
```

The manifest is what makes a training claim auditable: source, license, preprocessing,
counts, and contamination check.

### 3.2 Split hygiene

- **Split by source group, not by row.** Repositories, authors, conversations, and time
  buckets must not straddle train/test. Row-level splits leak near-duplicates and inflate
  results.
- **Freeze test sets.** The node's test split is immutable for a version; otherwise
  contributors can accidentally train on it.
- **Check contamination** against the graph's eval suite and any public benchmark used in
  reporting. If contamination is found, report it and exclude the affected items.
- **Keep a "hard" slice.** Examples the current node fails. Improvement on easy examples is
  not evidence.

### 3.3 Trace-derived data

Execution traces are the unique data asset of a graph system:

- A trace records the request, the route, the rendered prompt, the artifacts, usage, and
  outcome.
- With a quality signal (human accept, test pass, eval judge), it becomes a labeled example.
- Traces must be redacted and consented before export. Local-only by default; export is an
  explicit command with a manifest.

### 3.4 Dataset export

`mg data export` turns recorded runs into a reviewed dataset. It is an explicit, local command:
raw traces stay in the run directory, and nothing is uploaded. The export reads `run.json`,
`trace.jsonl`, and `envelope.json`, reconstructs the exact rendered prompts through the
runtime's own replay path, and writes a manifest next to the JSONL.

```text
uv run mg data export --run runs/<run-id> --out /tmp/dataset \
  --consent "operator approved local training" \
  --labels human-labels.json --checks node-checks.json \
  --eval-suite evals/mock/js-specialist
```

**Consent.** Export fails closed unless consent is recorded with `--consent`. The statement and
its scope go into the dataset manifest and every example. `--no-require-consent` is an explicit
opt-out for synthetic/debug data; the manifest records `required: false, recorded: false`.
Contribution terms remain an open question (`Q-010`), so an unknown dataset license stays
non-distributable.

**Selection.** Only runs whose outcome is `succeeded` are exported by default. Non-succeeded runs
are skipped; `--include-failures` includes them and labels every example from them `failure` with
`source: run_outcome`. A failure label is not inferred from a successful run: node labels default
to `unknown`, and routing labels are unknown without an oracle/human route.

**Examples.** A record is either `node` (worker/joiner rendered input, produced artifacts, finish
reason, usage, error) or `routing` (rendered menu, chosen route, decision, confidence, threshold,
outcome). Each record carries `source` (run, graph, node, trace event), `label`, `consent`,
`license`, redaction pattern names, and input/output/example hashes. Records include a `split`
field; `splits/{train,validation,test}.jsonl` are the same records partitioned.

**Labels and checks.** `--labels` accepts a YAML/JSON mapping with a `labels` list (or a JSONL
list). Entries may set `run_id`, `kind` (`node`/`routing`), `node`, `outcome`
(`success`/`failure`/`unknown`), `route` (routing oracle, or `abstain`), `ambiguous`, `reason`,
`labeler`, and `group`. `--checks` accepts a `CheckSpec`-style `checks` list per node and applies the
existing evaluation checks to the trace's artifacts (`--allow-command-checks` is required to
execute local command checks; rubric checks are skipped because export has no judge, and the
reason records the skip).
Explicit human labels take precedence over checks; a failed run takes precedence over both.

**Redaction.** `--redact default` applies built-in secret/PII patterns (private keys, common API
tokens, email, home paths); `--redact none` disables redaction; a YAML/JSON path supplies custom
patterns, actions (`hash`, `mask`, `drop`), and optional field rules. Redaction runs on the whole
record before hashes and dedupe. Artifact hashes are recomputed after redaction; when the content
changes, the original hash is retained as `source_hash`. `redaction-report.json` lists each
pattern's action, count, and field paths — never the matched text. A clean report is not proof
that no secret exists.

**Deduplication and splits.** Exact dedupe normalizes case and whitespace. Near-duplicate dedupe
uses 64-bit SimHash over token shingles and a configurable Hamming-distance threshold
(`--near-duplicate-distance`, default 3). Dedupe is global, so a near-duplicate cannot survive in
both train and test. Splits are assigned by group (`--group-by source|graph|run|license`, or a
label `group` override), never by row. The assignment is a deterministic hash of the group and
`--split-seed`; `--split-plan FILE` persists and reuses assignments, keeping the test groups
frozen across exports. Ratios default to 80/10/10 (`--split-ratio`).

**Contamination.** `--eval-suite` (repeatable) checks example task text against each eval case,
exact and near-duplicate. Matching examples are excluded and listed (suite, case id, method) in
`eval_contamination`; without a suite the status is honestly `not-checked`.

**Files and determinism.** The output contains `examples.jsonl`, `splits/{train,validation,test}.jsonl`,
`dataset.json`, `redaction-report.json`, and `splits.json`. The documented schema is
[`schemas/dataset.schema.json`](../schemas/dataset.schema.json): the top-level document is the
`kind: Dataset` manifest, and each JSONL line validates against `$defs.example`
(`modelgraph/dataset-example/v0`). The manifest records sources, per-source license and hashes,
consent, filters, counts (including per-node and per-route), split assignments and hashes,
preprocessing, contamination, and limitations. `by_node` groups examples by the node that
produced them, including routing examples under their router; it is a source-node count, so its
values can sum above the example total. `by_route` counts chosen routes and only covers routing
examples. Given the same runs, options, and `created_at`,
the JSONL, split files, redaction report, and manifest are byte-identical; only
`metadata.created_at` is time-dependent.

## 4. Router training

### 4.1 Labels

There are three ways to label which child a request should go to:

1. **Outcome labels (best).** Run the request through each candidate child (or a sample),
   score the result, and label the child with the best quality-per-cost. This is the
   oracle route. It is expensive but directly optimizes what matters.
2. **Trace labels (cheap).** Use requests that already succeeded at a child. This reflects
   the deployed router's behavior and can reinforce its mistakes; use with care.
3. **Human labels (small, high value).** Domain experts label a few hundred boundary cases.
   Use them for the test set and calibration, not only training.

For each router, build a dataset of `(rendered view, route menu, oracle label)` plus a
`should_abstain` label for ambiguous cases. Menus must be rendered exactly as at runtime;
training on a different menu format is a silent failure.

### 4.2 Methods

Start simple and only escalate when the data supports it:

| Method | When to use | Cost |
| --- | --- | --- |
| Rules / keyword baseline | Always. Establishes a floor and catches regressions. | Near zero |
| Embedding + classifier (logistic regression / kNN) | Domains are separable, latency matters, little data | Low |
| Fine-tuned small SLM (0.3B–1.5B) with JSON output | Nuanced `when` clauses, more data, tooling exists | Medium |
| Distilled router from a larger judge | You have judge labels and a latency budget | Medium |
| Learned confidence calibration | Any learned router | Low |

The router's output contract is fixed by [protocol.md](protocol.md#5-router-protocol). A
classifier adapter must produce the same JSON shape; that is the only interface the runtime
knows.

### 4.3 Calibration and abstention

A router's confidence is only useful if it is calibrated. After training:

1. Fit a calibration curve (temperature scaling or isotonic regression) on a held-out set.
2. Choose the abstain threshold by optimizing a cost-sensitive objective: a misroute costs
   more than an abstention when a fallback exists; an abstention costs more when it does not.
3. Report a reliability diagram, expected calibration error, and the confusion matrix in the
   model card.
4. Never tune the threshold on the test set. Use validation; confirm on test once.

### 4.4 Router evaluation

Routing quality is not one number. Report:

- **Oracle agreement:** fraction of requests routed to the oracle-optimal child.
- **Cost-weighted accuracy:** agreement weighted by the quality lost per misroute.
- **Abstain rate:** and what happens after abstention (fallback quality).
- **Confusion matrix:** which domains are confused, with example requests.
- **End-to-end delta:** run the full graph with the trained router versus the incumbent and
  versus oracle routing.

If oracle routing does not beat the best single model on a domain, do not train a router
for it. Remove the branch; one less node is a feature.

### 4.5 Local router recipe (implemented)

The v0 recipe is deliberately dependency-light: **pure Python, no torch, no numpy, no
scikit-learn**. It runs on a CPU and is small enough to audit. It implements the dataset
bridge, a rules floor, a conditional-logit classifier, temperature scaling, a cost-sensitive
abstain threshold, and a provider adapter that the existing router executor calls unchanged.

```text
T-0023 export ──► mg train dataset ──► router examples + summary
                                      │
                                      ├─ rules/keyword baseline (floor)
                                      └─ conditional-logit classifier
                                             │
                         validation-only temperature scaling
                         validation-only abstain threshold (cost matrix)
                                             │
                                      artifact JSON + report
                                             │
                               provider: trained-router
```

**Dataset bridge.** `mg train dataset` reads a T-0023 dataset directory (manifest +
`examples.jsonl`) or a bare `examples.jsonl`. It keeps the rendered messages, parses the menu,
and requires a positive route label (`oracle`, `human_label`, or a matching trace label) plus
`should_abstain`. Unknown labels are counted and excluded; contradictory labels (route not in
the menu, `route=abstain` with `should_abstain=false`, a route whose source is `unknown`, or
`should_abstain=false` with no route at all) are rejected. It preserves the dataset manifest
hash, example id, run/node/event, consent, license, redaction and hashes, reports class
balance/per-route counts, and rejects a group that straddles splits. The output contract is
documented in code
([`src/modelgraph/training/dataset.py`](../src/modelgraph/training/dataset.py)); it is not a new
graph contract.

**Trained model.** A multinomial conditional logit scores every candidate in the rendered menu
using hashed request words/bigrams, route-definition overlap (a shared "definition hit" feature
so an unseen request word that matches the menu can still help), candidate identity, and a few
scalars. Training is full-batch Adam with a fixed seed, deterministic example order, no random
initialization, and optional balanced class weights. Ambiguous training examples are excluded
from the route classifier by default; the abstain decision is a calibrated confidence threshold,
not a learned "abstain keyword". The class inventory comes from the **train split only**, so a
validation- or test-only route cannot add a class or weight to the artifact. The artifact schema
is [`router-artifact.schema.json`](../src/modelgraph/training/schemas/router-artifact.schema.json)
(`modelgraph/router-artifact/v0`) and records the feature version and `features.bigrams` switch,
classes, weights, temperature, threshold, cost matrix, training parameters, dataset identity,
and provenance.

**Calibration and abstention.** Temperature scaling is fitted on **validation route
predictions only** (non-ambiguous, routed). The test split is evaluated once. The abstain
threshold is selected by minimizing a documented cost matrix on validation; the default is
`correct_route=0`, `misroute=1.0`, `unnecessary_abstain=0.35`, `misroute_ambiguous=0.6`,
`correct_abstain=0`. Raise `unnecessary_abstain` for a graph with no fallback. Reports include
ECE, a reliability table, coverage/precision/recall, the cost matrix, and the confusion matrix.
The reliability table and the reported ECE use the same population: routed, non-ambiguous
predictions accepted at the abstain threshold; the before/after rows are the same cases.

**Runtime integration.** Additive provider kind `trained-router` (model manifest and run
manifest schemas, `build_provider`). The manifest's `spec.model` is the artifact path; an
optional `integrity.sha256` is verified before use. A relative path resolves from the current
directory then the project root. The provider parses the same rendered prompt, emits canonical
decision JSON, and is consumed by the unchanged router executor. [`training/models/router-triage.yaml`](../training/models/router-triage.yaml)
and [`training/graphs/hello-triage-trained.graph.yaml`](../training/graphs/hello-triage-trained.graph.yaml)
are a reference binding. The graph sets `confidence_threshold: 0.0` so the artifact's threshold
is the single policy; an artifact abstention still uses the graph's normal fallback path.

**Commands (from the repository root).** These are the exact offline commands; no network and
no real model are required.

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
export UV_PYTHON_INSTALL_DIR=$PWD/.uv-python

# 1. Synthetic fixture through the real T-0023 exporter (not a quality claim).
uv run python training/scripts/build_demo_dataset.py \
  --seed 20261025 --count 200 --force

# 2. T-0023 dataset -> router examples + class/split summary.
uv run mg train dataset \
  --dataset runs/training/router-dataset \
  --out runs/training/router-examples

# 3. Train + calibrate + select the abstain threshold + write artifact/report.
uv run mg train router --config training/configs/router-triage.yaml

# 4. Routing-only external check on the small hand-authored suite.
uv run mg eval routing \
  --suite evals/mock/triage \
  --graph training/graphs/hello-triage-trained.graph.yaml \
  --provider trained-router \
  --registry registry/models --registry training/models \
  --reports-dir runs/training/eval
```

For a real dataset, replace step 1 with `mg data export` and a label file that supplies oracle
routes and `ambiguous`, then point `dataset` in the config at that export. Use `--split-ratio`,
`--split-seed`, and `--split-plan` so the test groups stay frozen. The recipe config is
[`training/configs/router-triage.yaml`](../training/configs/router-triage.yaml); `mg train
router --dataset/--artifact/--report` can override paths without editing the config.

**Hardware and seeds.** Python 3.12; any CPU; roughly 0.1 GB RAM and under a few seconds for
the 125-example, 40-epoch fixture. No GPU, CUDA, torch, transformers, PEFT, numpy, or network.
Seeds: `20261025` for the demo generator and training; split seed `router-demo-20261025`;
`created_at` fixed to `2026-10-25T00:00:00Z`. The split plan is written to
`runs/training/demo-splits.json`. Weights and run data stay under git-ignored `runs/`; only the
config, graph, manifest, script, schema, and docs are committed.

**Measured result on the synthetic fixture (2026-10-25).** 200 routing examples
(train 125 / validation 39 / test 36, grouped by run). The classifier (40 epochs, 512 hashed
buckets) reached test oracle agreement **0.933**, coverage **0.778**, abstain precision
**0.750**, abstain recall **1.000**, ECE **0.003**, and expected cost **0.019/case**. Temperature
was **0.799**, fitted on 31 validation route predictions. The rules floor reached agreement
0.867, coverage 0.944, ECE 0.073, cost 0.175. The scripted incumbent reached agreement 0.900,
coverage 1.000, ECE 0.100, cost 0.183. On this fixture the trained router improved both oracle
agreement and cost against the scripted comparators; the gain is routing-only and the
incumbent is a stand-in. On the separate hand-authored `evals/mock/triage` suite it scored
agreement **0.90**, abstain precision **0.50**, and ECE **0.008**, passing that suite's gate;
the suite has 12 cases, so the interval is wide ([0.60, 0.98] for agreement). These are not
end-to-end quality claims; e2e needs live worker providers and a real suite.

**Known limits.** The demo data is synthetic and small; the incumbent is a scripted keyword
stand-in, not `router-small`; the confidence threshold does not separate ambiguous from
unfamiliar-but-clear requests; the menu/feature extractor is lexical; and there is no SLM
fine-tune path in this environment. Treat the recipe as a reproducible baseline and a
calibration harness, not a promotion.

## 5. Specialist training

### 5.1 Recipes

| Recipe | Data needed | Compute | Use when |
| --- | --- | --- | --- |
| Prompt/instruction tuning | 20–200 examples in-context | none | First attempt; often enough |
| LoRA SFT | 500–20,000 examples | 1 consumer GPU | The standard path |
| QLoRA SFT | 500–20,000 examples | 8–24GB VRAM | Larger base models |
| Full SFT | 10k+ examples | multi-GPU | Justified only by eval gains |
| Distillation from a teacher | Teacher outputs on your domain slice | inference + LoRA | You have a big model but need a small one |
| DPO/preference tuning | pairwise preferences | 1–2 GPUs | Style/failure modes that SFT cannot fix |

For v0, provide LoRA/QLoRA SFT and offline distillation. Preference tuning is a later
recipe, not a requirement.

### 5.1.1 Adapters as a fused deployment path

A LoRA adapter is not only a training convenience; it is the key to GPU-resident
composition. If a set of specialists share one base model and one tokenizer, they can be
served as adapters in a single engine, with the base weights loaded once and the graph's
handoffs kept on-device ([gpu-execution.md](gpu-execution.md), [ADR-0011](adr/0011-gpu-resident-backends.md)).

This introduces a trade-off the model card must state:

- **Benefit:** one resident base, low transition overhead, one KV pool, a realistic
  "single model feel."
- **Cost:** every node is constrained to the same base model; a best-in-class external model
  cannot join the fused backend; adapter rank and training data limit specialization.

B2 (T-0032) extends the manifest to describe an adapter; the training kit (T-0025) should
produce adapter artifacts that B2 can consume directly. The shared base is a deployment
choice, not a requirement of the graph contract.

### 5.2 Distillation

Distillation is the most direct expression of the project's thesis: use a large model to
teach a small specialist, then delete the large model from the serving path.

```text
teacher (large, slow, expensive)
   │  generate on domain slice + diverse prompts
   ▼
filtered traces (correct, deduped, licensed, hard cases preserved)
   │  SFT / LoRA
   ▼
student (small, local)
   │  node eval + graph regression
   ▼
manifest + card + eval report
```

Requirements:

- **Filter, do not blindly copy.** Teacher mistakes become student mistakes. Use tests,
  judges, or human review, and record the filter.
- **Preserve diversity.** Temperature sampling and prompt diversity matter more than volume.
- **Include failures.** Examples where the teacher recovered or explained an error are
  valuable; pure success data teaches brittle patterns.
- **Respect the teacher's license.** Distillation output may inherit restrictions; record
  the teacher's license in the dataset and model manifests.

### 5.3 Distilling the graph itself

A graph can also be a **teacher**. Collect representative inputs, run them through the graph
(B0/B1/B2), filter the outputs, and train a single student to imitate the graph's behavior.
This is the B4 deployment target ([performance.md](performance.md#58-b4-distilled-single-model),
[ADR-0012](adr/0012-compiled-deployment-targets.md)):

- It removes routing, handoff, and per-node prefill from the hot path.
- It trades runtime modularity for single-model speed.
- It can lose rare-route quality and can go stale when the graph changes.
- Promotion requires slice-level evaluation, a re-distillation policy, and provenance back to
  the source graph.

Use it only for stable, high-volume paths where a single model is the right runtime shape. The
graph remains the authoring, evaluation, and customization layer. Promotion and staleness follow
the revision workflow in [iteration.md](iteration.md).

### 5.4 Training loop and gates

1. Establish the incumbent: current model + node eval + graph e2e.
2. Train on a frozen dataset manifest.
3. Evaluate on the frozen node test set and a public suite if relevant.
4. Run the graph regression suite (routing + e2e) with the new node pinned.
5. Compare quality, cost, latency, and failure slices.
6. Promote only if it clears the node threshold and does not regress e2e beyond the agreed
   tolerance (see [evaluation.md](evaluation.md#6-contribution-gates)).
7. Publish manifest, card, eval report, and weights reference.

A model that is better on the node test but worse end-to-end is not promoted. The graph is
the product; the node is a part.

### 5.5 Local specialist kit (T-0025, review)

The v0 specialist kit has two paths. The **CPU rehearsal** runs here and proves the artifact
pipeline. The **real LoRA/QLoRA/distillation paths** are documented and fail closed; they are
**unverified in this repository's development environment**, which has no
torch/transformers/PEFT stack, CUDA driver, or GPU. T-0025 therefore remains in review until a
real 8GB and 24GB run is recorded on real hardware.

```text
T-0023 dataset (JSONL + manifest)
   │ consent/license/provenance preserved
   ▼
node filter + exact dedupe + (distill: recorded-output filter)
   ▼
deterministic CPU rehearsal ──► artifact.json      (synthetic: true, quality_claim: false)
   │                          ├─ manifest.yaml     (schema-valid modelgraph/v0)
   │                          ├─ card.md           (base/data/method/hyperparams/eval/limits/license)
   │                          ├─ eval-report.json  (node suite + graph regression, model pinned)
   │                          ├─ eval/             (report + run-manifest evidence, hashed)
   │                          ├─ reproducibility.json
   │                          └─ training-report.md
   ▼
candidate only. --promote refuses a synthetic artifact and refuses any run without a passing eval report.
```

**Dataset ingestion.** `load_specialist_dataset` reads a T-0023 directory (`dataset.json` +
`examples.jsonl`) or a bare JSONL file. It verifies the examples hash against the manifest,
rejects unrecorded required consent, filters to `kind: node` and the configured source node,
requires rendered messages and a non-empty output artifact, and records every exclusion. It
preserves dataset/source/run hashes, per-example license and consent, redaction status, and
contamination status. It never invents a label; node labels are reported as recorded.

**Deterministic rehearsal.** `mg train specialist` builds a self-describing
`modelgraph/specialist-artifact/v0` artifact whose pseudo-tensors are a deterministic SHA-256
function of the dataset digest and seed. No optimizer step is taken, no adapter is produced,
and the artifact is marked `synthetic: true`, `quality_claim: false`. The manifest, card, eval
hook, and reproducibility record are real artifacts with real hashes; only the weights are
rehearsal data.

**Offline distillation rehearsal.** `training/configs/distill-synthetic.yaml` treats the
recorded node outputs in the dataset as teacher samples. It applies the documented filter
(include/exclude failures, minimum bytes, exact-duplicate output), records what was dropped,
and runs the same student pipeline with `teacher.calls: 0`. No teacher model is called and no
teacher quality is inherited or claimed. A real teacher path is in
`training/distill.py`/`configs/distill-24gb.yaml`; it needs a licensed live teacher, network,
and real training hardware, and is unverified here.

**Eval hook.** The hook writes a copy of the target graph with the target node pinned to the
new `id@version`, then compiles it through the normal compiler/registry path and runs:

- the node suite (`evals/mock/js-specialist` by default), and
- the graph regression (`evals/e2e/coding-specialist` by default).

Both reports are copied next to the contribution and hashed. The registry eval report records
the model pin, suite hashes, graph hash, evidence report/run-manifest paths and hashes, seeds,
hardware, and gate results. It uses the deterministic mock provider: a pass proves the harness,
the pinned graph, and the evidence format, not the artifact's quality. A missing, failing,
wrong-model, or hash-mismatched report is rejected.

**Promotion gate.** `--promote` requires a passing eval report (supplied with `--eval-report`
or produced by the hook). If there is no passing report it fails. A synthetic artifact is then
refused regardless of the report. The rehearsal is a candidate, never a promoted model. A real
adapter must be trained on real hardware, served by a real provider, evaluated with real
responses, and submitted with a passing report.

**Reproducibility record.** `reproducibility.json` contains the git commit and dirty flag,
config hash, dataset id/version/hash/license/consent/counts/sources, artifact/manifest/card
hashes, training and evaluation seeds, the split seed, configured hardware plus observed
platform/Python/torch, the optional image digest, the exact command, and explicit limitations.
A real run must also record the container image digest and the actual GPU/driver.

**Exact commands (offline, from the repository root).**

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
export UV_PYTHON_INSTALL_DIR=$PWD/.uv-python

uv run python training/scripts/build_specialist_demo_dataset.py \
  --seed 20261025 --count 40 --force

uv run mg train specialist --config training/configs/specialist-synthetic.yaml --eval
uv run mg train specialist --config training/configs/distill-synthetic.yaml --eval

uv run pytest tests/training/test_specialist.py -q
```

Outputs are written under `runs/training/<model-id>/` (git-ignored). No weights, adapters, or
run data are committed.

**Reference hardware and data (real path, unverified here).**

| Tier | Practical recipe | Expected data | Public command |
| --- | --- | --- | --- |
| 8GB | 3B QLoRA, rank 16, seq 2048, effective batch 16 | 500–5,000 examples, ~1–20 MB JSONL | `uv run python training/sft_lora.py --config training/configs/specialist-8gb-qlora.yaml --execute` |
| 24GB | 7B BF16 LoRA, rank 32, seq 4096, effective batch 16 | 2,000–20,000 examples, ~5–100 MB JSONL | `uv run python training/sft_lora.py --config training/configs/specialist-24gb-lora.yaml --execute` |
| 24GB | 3B student distilled from a 32B teacher, then LoRA | 2,000–20,000 filtered teacher samples | `uv run python training/distill.py --config training/configs/distill-24gb.yaml --sample` then `--train` |

The hardware assumptions, base-model download sizes, VRAM peaks, and runtime estimates live in
the configs and [`training/README.md`](../training/README.md). `--dry-run` prints the resolved
plan and missing modules without training. `--execute` refuses when the stack or CUDA device is
absent. Real training writes an adapter and `training-result.json`; it does not promote or
evaluate the adapter by itself.

**Known limits.** The mock eval cannot validate an adapter; distillation inherits teacher
mistakes and license terms (record both); narrow SFT data regresses off-distribution; the
reviewed 8GB/24GB claim is a product promise that must be tested on real hardware. The model
card says all of this and the artifact states `quality_claim: false`.

## 6. Joiner training

Joiners benefit from examples of conflicting inputs and the desired resolution: which
source to trust, how to state uncertainty, when to pick one answer. They are lower
priority than routers and specialists because v0 rarely fans out. Train a joiner only
after fan-out exists and the single-input pass-through is no longer the norm.

## 7. Hardware tiers

Approximate, not guarantees; quantization and sequence length dominate.

| VRAM | Practical training | Notes |
| --- | --- | --- |
| 8GB | 1–3B QLoRA, small routers | Short sequences; gradient checkpointing |
| 12–16GB | 3–7B QLoRA | The hobbyist sweet spot |
| 24GB | 7B LoRA, 13B QLoRA | A serious independent contributor |
| 48GB+ | 13–32B LoRA | Workstation |
| Multi-GPU / cloud | 32B+ full or LoRA, distillation at scale | Not required to participate |

The project should publish reference recipes tested on at least one 8GB and one 24GB
configuration. "Runs on your machine" is a claim that must be tested, not assumed.

## 8. Training artifacts

Every trained node must ship:

| Artifact | Required content |
| --- | --- |
| **Model manifest** | ID, version, provider, resolved model identifier, context, limits, cost, resources, license, weights URI, SHA-256 |
| **Model card** | Base model, training data summary, method, hyperparameters, eval results, intended use, limitations, known failure modes |
| **Eval report** | Node suite results, graph regression results, cost/latency, seeds, command used |
| **Dataset manifest** | Sources, licenses, counts, preprocessing, contamination check |
| **Reproducibility info** | Code commit, container/image digest, random seeds, hardware |

The registry accepts a model only when these are present and the eval report is
reproducible. A weights file with no card is not a node; it is an unlabelled artifact. See
[`registry/README.md`](../registry/README.md) for the versioned layout, the exact evidence
format, and `mg registry verify`.

## 9. The data flywheel

```text
        deployed graph
              │
              ▼
        execution traces ──────────────┐
              │                        │
      failure slices                   │ accepted outcomes
              │                        │
              ▼                        ▼
   specialist training data      router training data
              │                        │
              ▼                        ▼
       better workers  ───────▶  better routing
              │                        │
              └──────────┬─────────────┘
                         ▼
                  better graph outcomes
                         │
                         ▼
                    more usage
```

The flywheel is the long-term advantage of a graph system: every run is potential training
data for a specific node. It only works if:

- traces are labeled with outcomes (tests, judges, user feedback);
- data is licensed and consented;
- training is reproducible and gated;
- improvements are attributed to nodes, so credit is not diluted.

Privacy is not an afterthought. The default is local-only, opt-in export, redaction, and a
manifest that records exactly what left the machine.

## 10. Limits and honest failure modes

| Limitation | Consequence | Mitigation |
| --- | --- | --- |
| No end-to-end gradients | Cannot jointly optimize router, workers, and joiner | Optimize each node against a shared eval; use oracle routing to bound gains |
| Credit assignment is indirect | A bad answer may be a routing, handoff, or worker failure | Trace-level failure attribution; per-node suites; ablation |
| Narrow data overfits | Specialist degrades off-distribution | Hard slices, held-out domains, e2e regression, confidence/abstain |
| Distillation inherits teacher bias | Student reproduces blind spots | Filtering, diverse prompts, human review of hard cases |
| Router labels reflect current behavior | Feedback loop reinforces mistakes | Oracle labels on a sample; exploration; periodic re-label |
| Small models are brittle at boundaries | Protocol violations, dropped instructions | Strict contracts, repair attempts, node-level validity metrics |
| Eval overfitting | Public numbers improve while real quality does not | Frozen held-out sets, rotating private slices, contamination checks |
| Data licensing | Cannot share weights or traces | Manifests, provenance, allowlists, clean-room training |

If a node cannot be evaluated reproducibly, it cannot be improved reliably. The evaluation
system is therefore not a later phase; it is a prerequisite for the training loop.

## 11. Governance

- **Consent and license** are recorded per dataset and per model. Unknown license means
  not distributable.
- **PII and secrets** are redacted before export; raw traces stay local by default.
- **Model cards** must state limitations, not only benchmark wins.
- **Contribution terms** (license of contributed weights, eval data, and cards) are an open
  question (`Q-001`, `Q-010`) and must be resolved before a public registry.
- **No benchmark gaming.** Eval suites are versioned; changes to a suite require review and
  a note in the eval report. A contributor cannot silently relax the test.
