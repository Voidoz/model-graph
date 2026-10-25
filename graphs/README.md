# Starter graph library

These graphs are the tested entry point for new users. They resolve against
[`../registry/models/`](../registry/models/) and are used by
[`docs/tutorial.md`](../docs/tutorial.md).

| Graph | What it demonstrates | Default mock path |
| --- | --- | --- |
| [`hello-triage.graph.yaml`](hello-triage.graph.yaml) | Router with a coding worker and a storytelling worker | `triage -> coding` |
| [`coding-specialist.graph.yaml`](coding-specialist.graph.yaml) | Second-level router: JavaScript/TypeScript vs. general coding | `coding-triage -> js-specialist` |
| [`composed-assistant.graph.yaml`](composed-assistant.graph.yaml) | Subgraph composition: domain router → nested coding team or storyteller | `domain-triage -> coding subgraph -> coding-triage -> js-specialist` |
| [`writing-studio.graph.yaml`](writing-studio.graph.yaml) | Writing/research router with no code execution | `brief-triage -> draft` |

```text
uv run mg validate graphs/*.graph.yaml
uv run mg run graphs/hello-triage.graph.yaml --provider mock
uv run mg run graphs/composed-assistant.graph.yaml --provider mock --profile profiles/local-4gb.yaml
```

The deterministic mock provider always selects the first route in the menu. It is for
plumbing and documentation checks; it does not measure model quality. For live runs, install
the models listed in the tutorial and omit `--provider mock`.
