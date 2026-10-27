# Local model registry

The registry is a local, file-based catalogue of model contributions. It records what a model
is, where it came from, what license applies, and what evidence supports it. It never stores or
downloads weights: it stores a manifest, a card, an eval report, and hashes.

- Runtime model resolution lives in [`src/modelgraph/registry.py`](../src/modelgraph/registry.py).
- The versioned store, index, and verification live in
  [`src/modelgraph/registry_store.py`](../src/modelgraph/registry_store.py).
- The normative contribution report contract is
  [`schemas/registry-eval-report.schema.json`](../schemas/registry-eval-report.schema.json); the
  deterministic index contract is [`schemas/registry-index.schema.json`](../schemas/registry-index.schema.json).
- Training/evidence background is in [`docs/training.md`](../docs/training.md) §5 and §8.

Everything is offline. No command in this workflow opens a network connection or downloads a
weight file.

## Layout

```text
registry/
  index.yaml                                  # deterministic index + content hash
  models/
    coder-7b.yaml                             # legacy flat starter manifest (still resolves)
    <id>/<version>/manifest.yaml              # versioned model manifest
    <id>/<version>/card.md                    # model card
    <id>/<version>/eval-report.json           # registry eval report
    <id>/<version>/eval/...                   # evidence files referenced by the report
    <id>/<version>/weights/...                # optional local weights file (never committed)
```

The existing flat starter manifests under `registry/models/*.yaml` are unchanged. A graph can
reference either layout; a versioned entry is resolved by `id@version` (or by bare `id` when
there is one version). A duplicate `id@version` with different bytes is an error: published
versions are immutable.

## Commands

From the repository root:

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
export UV_PYTHON_INSTALL_DIR=$PWD/.uv-python

# Add a verified contribution directory (manifest.yaml, card.md, eval-report.json, evidence).
uv run mg registry add /path/to/contribution

# Add into a scratch registry and choose the license policy.
uv run mg registry add /path/to/contribution \
  --root runs/registry-scratch --license-policy distributable

# List and inspect.
uv run mg registry list
uv run mg registry inspect js-specialist-3b@0.4.0
uv run mg registry list --json
uv run mg registry inspect js-specialist-3b@0.4.0 --json

# Verify everything, then write the deterministic index.
uv run mg registry verify
uv run mg registry verify --write-index
uv run mg registry verify --json
```

`--root` points at a different registry tree (useful for tests or a user-local registry). The
default is the repository `registry/`.

## What a contribution must contain

1. **`manifest.yaml`** — a schema-valid `modelgraph/v0` Model manifest. It must satisfy the
   manifest schema and the manifest trust policy in
   [`docs/security.md`](../docs/security.md). At minimum: id, name, version, description,
   provider, resolved model, context window, license, and resources. If it declares
   `spec.integrity.sha256`, that hash is a weights pin.
2. **`card.md`** — a model card with non-empty sections for:
   **Base model**, **Training data**, **Method**, **Evaluation**, **Limitations**, and
   **License**. A **Hyperparameters** section is strongly recommended (and `mg registry verify`
   warns when it is missing). State intended use, known failure modes, and that the license is
   a declaration, not legal advice.
3. **`eval-report.json`** — a `modelgraph/registry-eval-report/v0` document. It must:
   - name the same `model.id@version` as the manifest;
   - pin the graph to that exact model (`graph.pinned_model`);
   - record the node suite and graph-regression suite ids, versions, and content hashes;
   - record the graph content hash;
   - reference the node and graph `report.json` and `manifest.json` files with `path` **and**
     `sha256`, including `passed: true`;
   - record seeds, hardware, a reproduce command, and limitations.
4. **Evidence files** — the files referenced by the report, with paths relative to
   `eval-report.json`. `mg registry add` copies them. Missing or hash-mismatched evidence fails
   verification.
5. **Optional local weights** — if a contribution bundles a weights file, include it in the
   report as `weights: {path, hash}`, and make the manifest's `integrity.sha256` match it.
   Without a bundled file, a declared weights hash is recorded as unverified.

`mg train specialist --config training/configs/specialist-synthetic.yaml --eval` writes a
complete contribution directory (manifest, card, eval report, evidence, reproducibility record)
under `runs/training/...`. The synthetic artifact is a rehearsal and is never promotable; use
the same layout for a real artifact after training and a real evaluation.

## What "reproducible eval report" means

Another person must be able to rerun the evidence without guessing:

- the exact command is recorded (`command`, `reproduce`);
- the suite and graph are content-hashed, so edits invalidate the claim;
- the graph is pinned to the submitted `id@version`;
- the evidence files are bundled and hashed;
- seeds, hardware, and provider are recorded;
- a passing report covers both the node suite and the graph regression;
- any waiver, mock provider, small sample, or unmeasured slice is stated in the limitations.

A number in a chat message or a card-only claim is not evidence. A report that cannot be
rerun is a candidate, not a verified contribution.

The offline mock path in the training kit is a harness check. A real contribution that claims
quality must carry a real-provider report (the registry verifies structure and hashes; it does
not judge whether a model is good).

## Duplicates, index, and determinism

- Adding identical bytes is idempotent and does not rewrite the entry.
- Adding the same `id@version` with different bytes fails; bump the version instead.
- `index.yaml` sorts entries by id and version, contains no timestamps, and records
  `manifest`, `card`, `eval_report`, and optional `weights` hashes.
- `content_hash` is the SHA-256 of the canonical API version/kind/model list. Editing the index
  or changing any entry makes `mg registry verify` fail until `--write-index` recomputes it.
- `mg registry verify` is a release-checklist item in
  [`docs/security.md`](../docs/security.md).

## License policy

- The default `--license-policy distributable` rejects unknown/placeholder license names
  (`unknown`, `tbd`, `none`, `n/a`, …). `--allow-unknown-license` is an explicit operator
  opt-in; `--license-policy local` is a local-only posture.
- `--profile FILE|NAME` applies a profile's `licenses` allowlist. A license not in the list is
  rejected; an unknown license is allowed only when that profile explicitly lists it.
- A license field is a declaration, not a legal guarantee. The registry checks declarations and
  policy, not copyright ownership or redistribution rights. Confirm the upstream terms before
  redistributing weights or cards.
- Project code is GPL-3.0-or-later. Contributed model cards and weights retain their own
  upstream licenses; do not relicense a model by adding it to the registry.

## Limitations

- No cryptographic signing or transparency log yet; a content hash pins bytes, not authorship.
- No remote registry, marketplace, or weight download.
- The local CLI and GUI have no authentication; keep them on loopback.
- `owner/model` namespacing for public contributions is not decided (`Q-001`). Use distinct
  ids for now.
