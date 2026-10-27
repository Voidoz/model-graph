# ADR-0009: Local-first, provider-neutral model binding

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [architecture.md](../architecture.md), [protocol.md](../protocol.md), ADR-0002

## Context

The project's premise is that people can train and run their own components. That requires:

- running models locally, on hardware the user controls;
- mixing local and hosted models in one graph;
- switching providers without rewriting a graph;
- knowing what a model is, where it came from, and what license governs it.

Embedding endpoints and credentials in graph files would make graphs non-portable and leak
secrets. Requiring a hosted registry would recreate central control. Supporting only one
provider would make the "universal language bus" claim hollow.

## Decision

**Graphs reference models by manifest ID; manifests bind to providers; adapters implement
providers; weights never move through the graph runtime.**

1. **Model manifests** declare provider, resolved model identifier, context, limits, cost,
   resources (VRAM/CPU), license, weights URI, and integrity hash. They are the only place
   provider details live.
2. **Graphs are provider-neutral.** A node says `model: js-specialist-3b`; the registry
   resolves that to a manifest. Running the same graph against a different registry (e.g.,
   a hosted variant) is a registry change, not a graph change.
3. **Local-first.** Default examples and tutorials use local adapters (Ollama, llama.cpp).
   Hosted providers are allowed but explicit. No telemetry or trace upload by default.
4. **Weights stay where they are.** The runtime stores weights URIs and hashes, never
   copies weights into the repository or transfer them between nodes. Downloading a model is
   an explicit registry/CLI action, not a side effect of running a graph.
5. **Capability filters** use manifest facts (provider availability, VRAM, license, network
   requirement) to prune graphs for a profile.
6. **Adapters are thin.** They normalize chat requests/responses and expose capabilities and
   health. They do not know graphs, routing, or artifacts.
7. **Version pinning is mandatory in a run.** A floating model tag is resolved at compile
   time and recorded; the run manifest includes the resolved revision/hash where available.
8. **License and provenance are recorded**, not inferred. Unknown license means the model is
   excluded from profiles that require a known license.

## Consequences

**Positive**

- A graph runs unchanged on a laptop, a workstation, or a hosted endpoint with the
  appropriate registry.
- Contributors can test locally without an API key; CI uses the mock adapter.
- Secrets stay in environment/keychain, not in graph files.
- Pruning has factual inputs (VRAM, license, provider availability) rather than guesses.
- Users can audit what model actually ran and under what license.
- The system remains useful offline.

**Negative**

- Manifests add ceremony; every model needs one (a generator can help).
- Provider differences in tokenization, context limits, and JSON support leak into quality
  and cost; the manifest cannot hide all of them.
- Local performance varies widely; benchmarks must report hardware.
- Resolved versions are not always exposed by providers, weakening reproducibility.
- Keeping weights out of git means a run can be unreproducible if a weights URI disappears;
  integrity hashes detect drift but cannot resurrect a deleted model.

**Neutral**

- A future remote registry can host manifests and eval reports without hosting weights.
- Hosted serving may be added as another provider; it does not change the binding model.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Provider config inside graph YAML | Non-portable, leaks endpoints/secrets, duplicates per profile. |
| One required provider (e.g., OpenAI) | Contradicts local-first and independent ownership. |
| Download and bundle weights with graphs | Huge, license-problematic, and shifts distribution burden to the project. |
| Auto-detect models and infer properties | Unreliable; resource/license facts must be declared and reviewable. |
| A central model store with opaque IDs | Recreates centralization and makes offline use hard. |

## Follow-ups

- T-0002 locks the manifest schema.
- T-0012 and
  T-0013 implement adapters.
- T-0015 implements capability-based
  pruning.
- T-0026 adds registry verification and
  contribution cards.
- Resolve contribution licensing before a public registry (`Q-001`, `Q-010`).
