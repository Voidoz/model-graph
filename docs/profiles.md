# Run profiles and pruning

A **run profile** describes the machine or policy a graph runs under. The same authored
graph can run on a laptop or a workstation without editing YAML: the profile filters the
graph at runtime and the run records what was removed. The normative contracts are in
[graph-model.md §7.1](graph-model.md#71-pruning) and
[execution.md §6](execution.md#6-pruning-and-lazy-loading); this page documents the v0
implementation in `modelgraph.pruning`.

## Format

Profiles are YAML or JSON. Unknown fields are rejected so a typo cannot silently weaken a
policy.

```yaml
name: local-4gb
network: loopback            # none | loopback | internet (a ceiling)
gpu: true
max_vram_gb: 4
max_ram_gb: 16
max_disk_gb: 20
providers: [ollama, llamacpp, mock]
licenses: [apache-2.0, mit]
max_cost_usd: 0.0
budget_overrides: {max_steps: 8}
backend: auto
```

Example profiles live in [`examples/profiles/`](../examples/profiles/).

`--profile` accepts a path or a name. A bare name is searched in `./profiles`,
`./examples/profiles`, and the same directories under the repository root. An explicit
path is never searched elsewhere.

## Semantics

- **Network is a ceiling.** `none` < `loopback` < `internet`. A model whose manifest
  declares a higher `resources.network_access` is pruned. A model that does not declare
  the fact is treated as `internet` when a ceiling is set: an undeclared capability is not
  assumed safe.
- **Providers and licenses are allowlists.** `None` means no filter; an empty list allows
  nothing. License comparison is case-insensitive. An unknown license is simply not in the
  allowlist, so it is pruned.
- **VRAM/RAM/disk are ceilings.** A declared manifest value above the ceiling prunes the
  model. A missing resource fact is not treated as a failure (the manifest omitted a fact);
  this is a registry-quality gap, not a runtime permission.
- **`gpu: false`** prunes models that declare `cpu_only: false`.
- **Cost.** `max_cost_usd` prunes models whose minimum call cost exceeds the cap (including
  a zero cap against any positive per-token or per-request price) and models whose pricing
  is `unknown`, unless the profile sets `allow_unknown_cost: true`.
- **Budget overrides tighten by default.** An override that raises a graph default is
  ignored with a warning unless `mg run --allow-budget-override` is passed. Explicit CLI
  flags (`--max-steps`, `--max-tokens`, `--max-cost`) are applied after profile overrides,
  so a profile cannot silently raise a tighter user-specified budget. The effective budget
  is recorded in the run manifest.

## Pruning

`prune_graph(graph, profile)` returns a new `CompiledGraph` and a `PruneReport`; the
authored document and its graph hash are unchanged. Pruning:

1. Walks the compiled graph and nested subgraphs, marking nodes whose models are
   unsupported (provider, network, resources, license, cost, GPU).
2. Removes incident edges and every node that cannot reach an exit artifact from the entry.
   A subgraph whose entry or exit cannot survive becomes an unsupported node in its parent.
3. Filters router edges, so a router menu never contains a pruned target. A removed fallback
   is cleared and reported.
4. Fails with `MG-1005` when the entry disappears or any exit artifact loses every producer.
   The error carries the full report; the CLI prints it.

The report lists every removed node and edge with a reason, removed exit artifacts, and
warnings such as a router reduced to one route or an unreachable branch. The CLI prints the
report and records it under `prune` in `run.json`; `profile` records the effective profile
and the trace gets `run.started.profile` plus a `prune.applied` event.

Pruning is policy, not editing. To change the graph permanently, edit the YAML.

## Lazy loading

`LazyModelLoader` in `modelgraph.pruning` wraps a provider factory and loads a model only on
first use. It records `cold_load_ms` per model in `model_loads` (run manifest), and exposes
an optional LRU eviction hook (`max_loaded`) plus `unload()` and `close()`. Provider-specific
cold-load cost may be a no-op; for the portable backend it measures adapter construction.
`cold_load_ms` is deliberately separate from provider generation latency in
`provider.response`.

Two limitations are explicit in v0:

- The portable engine already creates providers on first node invocation, so the loader is
  mostly an accounting/interface layer. A resident backend may use the same abstraction to
  pin or evict whole plans ([gpu-execution.md](gpu-execution.md)).
- Automatic `max_loaded` eviction is not safe while the engine keeps its own provider
  cache: an evicted provider could still be referenced and would not be reloaded. Leave
  `max_loaded` unset for `mg run`; unload after the run or from a resident backend.
