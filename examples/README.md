# Examples

These examples validate against the schemas and run with the MVP runtime.

```text
mg validate examples/hello-triage.graph.yaml
mg run examples/hello-triage.graph.yaml --provider mock
```

See [`docs/quickstart.md`](../docs/quickstart.md) for the full walkthrough.

| File | What it demonstrates |
| --- | --- |
| [`hello-triage.graph.yaml`](hello-triage.graph.yaml) | Minimal router with two workers and a fallback |
| [`coding-specialist.graph.yaml`](coding-specialist.graph.yaml) | A nested coding team: JS/TS specialist vs. general coder |
| [`composed-assistant.graph.yaml`](composed-assistant.graph.yaml) | The worked example: domain router → coding subgraph or storyteller |
| [`envelope.json`](envelope.json) | A serialized runtime envelope with a path, note, and artifact |
| [`models/`](models/) | Local model manifests used by the graphs |

## Model manifests

The manifests are illustrative and point at local Ollama endpoints. They include upstream
license fields and a note to verify the license before redistribution. They do not download
anything. To run the examples locally, install the referenced models in Ollama, for
example:

```text
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:3b
ollama pull mistral:7b-instruct
```

`router-small` maps to `qwen2.5:1.5b-instruct`; the coder manifests map to the
`qwen2.5-coder` family; `story-7b` maps to `mistral:7b-instruct`. Confirm the actual model
names and licenses before publishing results.

## Commands

```text
mg validate examples/hello-triage.graph.yaml
mg validate examples/coding-specialist.graph.yaml
mg validate examples/composed-assistant.graph.yaml

mg run examples/hello-triage.graph.yaml --provider mock
mg run examples/hello-triage.graph.yaml --registry examples/models
mg run examples/composed-assistant.graph.yaml --provider mock --json
```

Profiles are a later ticket (T-0015); the MVP uses the provider declared in each manifest.

The hello, composed, and coding-specialist graphs have been run live against the local Ollama
models on CPU. See `.scratch/runs/2026-10-22-live-ollama.md` for timings and caveats.

## What to look at in `envelope.json`

- `run.path` shows the router decision and the subgraph entry.
- `run.budget` shows both limits and consumption.
- `inputs` and `artifacts` share a shape; `producer` distinguishes `$input` from a node.
- `notes` carries a router hint without replacing the original request.
- Hashes are content hashes, not identifiers of provenance. The run manifest records
  provenance.

The content is illustrative and not a real model output. Do not use it as an evaluation
example.
