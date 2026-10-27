# Training recipes

This directory holds the offline reference recipes for the two trainable node roles:

- **Router (T-0024):** a dependency-light conditional-logit classifier with calibration and a
  cost-sensitive abstain threshold. It runs on a CPU with plain Python.
- **Specialist (T-0025):** LoRA/QLoRA SFT and offline distillation configs, plus a deterministic
  CPU rehearsal that emits the full artifact set (artifact, manifest, card, eval report,
  reproducibility record) without pretending to train a model.

The implementation lives in [`src/modelgraph/training/`](../src/modelgraph/training/). The
narrative is in [`docs/training.md`](../docs/training.md) §4.5 (router) and §5.5 (specialist).

## What is here

| Path | Purpose |
| --- | --- |
| `configs/router-triage.yaml` | router dataset/features/training/calibration/cost config |
| `graphs/hello-triage-trained.graph.yaml` | binds the triage router to `trained-router` |
| `models/router-triage.yaml` | model manifest pointing at the generated router artifact |
| `scripts/build_demo_dataset.py` | deterministic synthetic T-0023 router export |
| `configs/specialist-synthetic.yaml` | runnable specialist pipeline rehearsal (CPU) |
| `configs/specialist-8gb-qlora.yaml` | 8GB QLoRA SFT reference (unverified here) |
| `configs/specialist-24gb-lora.yaml` | 24GB LoRA SFT reference (unverified here) |
| `configs/distill-synthetic.yaml` | runnable recorded-output distillation rehearsal (CPU) |
| `configs/distill-24gb.yaml` | real teacher-sampling + student-training reference (unverified) |
| `scripts/build_specialist_demo_dataset.py` | deterministic synthetic T-0023 specialist export |
| `sft_lora.py` | real PEFT trainer driver; fails closed without the stack/GPU |
| `distill.py` | teacher-sampling/filter/student-training recipe driver |
| `../tests/training/` | router and specialist tests |

## Router quick start (T-0024)

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
export UV_PYTHON_INSTALL_DIR=$PWD/.uv-python

uv run python training/scripts/build_demo_dataset.py --seed 20261025 --count 200 --force
uv run mg train dataset --dataset runs/training/router-dataset --out runs/training/router-examples
uv run mg train router --config training/configs/router-triage.yaml
```

The artifact and report are written to `runs/training/router-triage/` (git-ignored). Use
`--dataset`, `--artifact`, and `--report` to override paths. `--no-write` trains and reports to
stdout without touching disk.

Router artifact and report:

- Artifact schema: `modelgraph/router-artifact/v0`
  (`src/modelgraph/training/schemas/router-artifact.schema.json`).
- Config schema: `modelgraph/router-training-config/v0`.
- Report: Markdown + JSON sibling next to the configured report path.
- The artifact contains weights, feature version, temperature, abstain threshold, cost matrix,
  training parameters, dataset hash/counts, and provenance. Weights and run data stay out of
  git.

Router eval:

```bash
uv run mg eval routing \
  --suite evals/mock/triage \
  --graph training/graphs/hello-triage-trained.graph.yaml \
  --provider trained-router \
  --registry registry/models --registry training/models \
  --reports-dir runs/training/eval
```

## Specialist quick start (T-0025, rehearsal)

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
export UV_PYTHON_INSTALL_DIR=$PWD/.uv-python

# 1. Synthetic fixture through the real T-0023 exporter (not a quality claim).
uv run python training/scripts/build_specialist_demo_dataset.py \
  --seed 20261025 --count 40 --force

# 2. Deterministic CPU rehearsal: artifact + manifest + card + eval report + repro record.
uv run mg train specialist --config training/configs/specialist-synthetic.yaml --eval

# 3. Same pipeline with recorded-output "distillation" (zero teacher calls).
uv run mg train specialist --config training/configs/distill-synthetic.yaml --eval

# 4. Optional: add the rehearsal contribution to a scratch registry.
uv run mg registry add runs/training/js-specialist-synthetic --root runs/registry-specialist
uv run mg registry verify --root runs/registry-specialist --write-index
```

Outputs under `runs/training/<model-id>/` (git-ignored):

| File | Meaning |
| --- | --- |
| `artifact.json` | versioned `modelgraph/specialist-artifact/v0`; `synthetic: true`, pseudo-weights |
| `manifest.yaml` | schema-valid `modelgraph/v0` model manifest |
| `card.md` | base model, data, method, hyperparameters, eval, limitations, license |
| `eval-report.json` | registry eval report over the pinned graph + node suite evidence |
| `eval/` | copied node/graph `report.json` and `manifest.json` evidence |
| `reproducibility.json` | commit, config/dataset/artifact hashes, seeds, hardware, command |
| `dataset-summary.json` | preserved T-0023 provenance, consent, license, counts, exclusions |
| `training-report.md` / `.json` | human and machine summary of the run |

The synthetic artifact is **never promotable**, even when the mock eval gate passes.
`--promote` first refuses without a passing eval report, then refuses a synthetic artifact.
A real adapter must be trained on real hardware and evaluated on a served provider.

## Specialist recipes: 8GB and 24GB

These are **unverified in this repository's development environment**. There is no
torch/transformers/PEFT stack, CUDA driver, or GPU here. The commands are exact, but do not
claim they ran until they have been executed on the stated hardware and the evidence is
recorded on T-0025.

| Tier | Base / method | Expected data | Hardware assumption | Exact command |
| --- | --- | --- | --- | --- |
| 8GB | 3B, QLoRA 4-bit, rank 16, seq 2048, effective batch 16 | 500–5,000 examples, ~1–20 MB JSONL | RTX 3060 Ti/4060 8GB, CUDA 12.1+, driver 535+, 32GB RAM | `uv run python training/sft_lora.py --config training/configs/specialist-8gb-qlora.yaml --execute` |
| 24GB | 7B, BF16 LoRA, rank 32, seq 4096, effective batch 16 | 2,000–20,000 examples, ~5–100 MB JSONL | RTX 3090/4090/A10G 24GB, CUDA 12.1+, driver 535+, 64GB RAM | `uv run python training/sft_lora.py --config training/configs/specialist-24gb-lora.yaml --execute` |
| 24GB | 3B student distilled from a 32B teacher, then LoRA | 2,000–20,000 filtered samples; raw sampling 2–4× larger | same 24GB student GPU; teacher endpoint/network | `uv run python training/distill.py --config training/configs/distill-24gb.yaml --sample` then `--train` |

`--dry-run` prints the resolved plan, missing modules, and the exact execute command without
training. `--execute` fails closed when the stack or CUDA device is absent; it never fabricates
a training result. Real training writes `training-result.json` and `adapter/` but does **not**
promote or evaluate the adapter; evaluation must run against a served model.

Install the real stack only on a machine that needs it:

```bash
uv pip install torch transformers peft datasets accelerate trl
# QLoRA only:
uv pip install bitsandbytes
```

The 8GB/24GB reference configs set the base model, rank/alpha/dropout, learning rate, epochs,
batch/gradient accumulation, sequence length, quantization, target modules, optimizer,
scheduler, warmup, hardware, expected data, expected runtime, and seeds. Treat the runtime
estimates as estimates, not measurements.

## Eval hook and promotion gate

`mg train specialist --eval` (the default) writes a graph pinned to the new
`id@version`, then runs:

1. the node suite (`evals/mock/js-specialist` by default), and
2. a graph regression (`evals/e2e/coding-specialist` by default).

The registry eval report records the model pin, suite hashes, graph hash, and content hashes
for both evidence report/run-manifest pairs. A missing, failing, mismatched, or
wrong-model report is rejected. The hook is offline and uses the deterministic mock provider:
a pass proves the harness and the pinned graph, **not** the artifact's quality.

## Tests

```bash
uv run pytest tests/training -q
uv run pytest tests/registry -q
```

No network, real model, torch, numpy, or scikit-learn is required.

## Limits

- The synthetic path is a pipeline rehearsal, not a model. Do not deploy or benchmark it.
- The mock eval hook does not evaluate a trained adapter.
- The 8GB/24GB and real distillation recipes are unverified here; T-0025 stays in review until
  they are run on real hardware and the evidence is recorded.
- Router data is synthetic and its incumbent is a scripted stand-in; see the router limits
  above.
- Model cards retain their upstream license declarations. This project's code is GPLv3; a model
  card's license is a declaration, not legal advice.
