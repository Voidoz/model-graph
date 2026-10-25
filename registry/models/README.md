# Tutorial model manifests

These manifests are resolved by the starter graphs in [`../../graphs/`](../../graphs/). They do
not download anything: install each tag in Ollama before a live run.

```text
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:3b
ollama pull mistral:7b-instruct
ollama pull qwen2.5:7b-instruct
```

| Manifest | Ollama tag | Role |
| --- | --- | --- |
| [`router-small.yaml`](router-small.yaml) | `qwen2.5:1.5b-instruct` | Routing decider |
| [`coder-7b.yaml`](coder-7b.yaml) | `qwen2.5-coder:7b` | General coding worker |
| [`js-specialist-3b.yaml`](js-specialist-3b.yaml) | `qwen2.5-coder:3b` | JavaScript/TypeScript worker |
| [`story-7b.yaml`](story-7b.yaml) | `mistral:7b-instruct` | Storyteller |
| [`writer-7b.yaml`](writer-7b.yaml) | `qwen2.5:7b-instruct` | Writing/research worker |

The model names, resources, and licenses are illustrative. Confirm the upstream license and the
exact tag before publishing results; change the tag here, never in the graph. The 4GB and 8GB
profiles use the declared `resources.vram_gb` values, so an edit to a resource fact changes what
the profile allows.

See [`../../docs/tutorial.md`](../../docs/tutorial.md) for the full walkthrough.
