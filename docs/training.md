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

### 5.3 Training loop and gates

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
reproducible. A weights file with no card is not a node; it is an unlabelled artifact.

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
