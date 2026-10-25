# Tutorial: a running graph in ten minutes

This page takes you from a fresh clone to a mock run, a real local model run, a trace you can
read, a pruned deployment, and a node evaluation. The default path needs **no API key** and no
network beyond the loopback interface to Ollama. The mock path is tested in CI; the live path is
tested by hand (see [hardware and tested scope](#hardware-and-tested-scope)).

If you only have five minutes, stop after [step 4](#4-run-the-hello-graph-with-the-mock-provider).

## Hardware and tested scope

Model Graph itself is small; the models are not. These are the hardware assumptions behind the
starter profiles, not minimum guarantees:

| Profile | Assumes | What runs |
| --- | --- | --- |
| `profiles/local-8gb.yaml` | ~8GB usable VRAM, 32GB RAM, 50GB free disk | Every starter graph with the default 7B models |
| `profiles/local-4gb.yaml` | ~4GB usable VRAM, 16GB RAM, 20GB free disk | `coding-specialist` and the JavaScript branch of `composed-assistant`; it removes every 6GB model |
| `profiles/no-network.yaml` | policy only: no sockets at all | Prunes every starter node, because local Ollama is loopback network access |

- **CPU-only works.** It is slower: prior runs on the hardware below took roughly 7–16 seconds
  per graph on CPU versus 1–9 seconds warm on GPU. Your hardware will differ.
- **8GB is a ceiling, not a promise.** A quantized 7B model can still exceed 8GB with a long
  context or another model resident. If a model does not fit, use `local-4gb`, the JavaScript
  branch, or a smaller manifest.
- **Tested hardware:** AMD Ryzen 7 5800X3D, 23GiB RAM, CPU-only; and an NVIDIA RTX 3080 (10GiB)
  through Ollama Vulkan. The live matrix covered `hello-triage`, `coding-specialist`, and
  `composed-assistant`. `writing-studio` is covered by the offline mock path but has **not** been
  live-tested.
- **Linux** was tested. **macOS** should work with Ollama's Metal backend but was not verified in
  this session. Windows is via WSL2 (`Q-012`).
- **Disk:** the five tutorial models are roughly 17GB before Ollama overhead. Pull only the ones
  you need if disk is tight.

The starter library lives in `graphs/`, run profiles in `profiles/`, and model manifests in
`registry/models/`. `mg` resolves the registry and graph directories from the repository root by
default.

## 1. Install and check

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/). From the repository root:

```text
uv sync
uv run mg version
```

Expected: `0.4.0` (or the current version). The test suite is offline:

```text
uv run pytest -q
```

If that passes, you do not need a model server for the rest of the mock path.

## 2. Pull the models with Ollama

Install [Ollama](https://ollama.com/) separately; `mg` never downloads weights. Make sure the
server is running (`ollama serve`, or `sudo systemctl start ollama` on Linux), then pull:

```text
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:3b
ollama pull mistral:7b-instruct
ollama pull qwen2.5:7b-instruct
```

Verify the server and the installed tags:

```text
curl -s http://127.0.0.1:11434/api/tags
ollama list
```

The manifest IDs are not the same as the Ollama tags. `mg models list` shows the mapping:

```text
uv run mg models list
```

| Manifest | Ollama tag | Declared need | Used by |
| --- | --- | --- | --- |
| `router-small` | `qwen2.5:1.5b-instruct` | 2GB VRAM | All starter graphs |
| `coder-7b` | `qwen2.5-coder:7b` | 6GB VRAM | `hello-triage`, `coding-specialist` |
| `js-specialist-3b` | `qwen2.5-coder:3b` | 4GB VRAM | `coding-specialist`, `composed-assistant` |
| `story-7b` | `mistral:7b-instruct` | 6GB VRAM | `hello-triage`, `composed-assistant`, `writing-studio` |
| `writer-7b` | `qwen2.5:7b-instruct` | 6GB VRAM | `writing-studio` |

The manifests are illustrative. Confirm the upstream license and the exact tag before you
publish results. Change a tag in `registry/models/`, not in the graph.

## 3. Validate the starter library

```text
uv run mg validate graphs/*.graph.yaml
```

Expected: four `valid:` lines. `hello-triage` also prints a warning:

```text
valid: graphs/hello-triage.graph.yaml (hello-triage)
  warning: [MG-1016] ... exit artifact 'answer' has conflicting types: ['code', 'text']
```

That warning is intentional: the coding worker writes `code` and the storyteller writes `text`
to the same exit artifact. The graph is still valid because the runtime routes to exactly one
producer. Machine-readable output and a failing exit code work too:

```text
uv run mg validate graphs/hello-triage.graph.yaml --json
uv run mg validate graphs/does-not-exist.graph.yaml
```

`validate` checks structure, references, budgets, and reachability. It does not need a model
server. If a model or subgraph is missing, the default is a warning; `mg run` is strict.

## 4. Run the hello graph with the mock provider

```text
uv run mg run graphs/hello-triage.graph.yaml --provider mock \
  --task "Write a debounce function in TypeScript."
```

Expected shape:

```text
run:      run_...
outcome:  succeeded
usage:    steps=2 tokens=... cost=$0.000000 elapsed=...s
node:     triage (router) route -> coding
node:     coding (worker)

--- answer (code) ---
export function debounce...
```

The mock provider is deterministic and offline. It selects the **first menu entry**, which is
`coding` here, regardless of the task text. That is deliberate: the mock exercises the plumbing,
not model quality. Use `--mock-scenario PATH` when you need a specific route or a scripted
failure; see [docs/quickstart.md](quickstart.md#3-run-offline-with-the-mock-provider).

Now run the same graph against the local models (omit `--provider mock`; the default is `auto`,
which uses each manifest's provider):

```text
uv run mg run graphs/hello-triage.graph.yaml \
  --task "Write a debounce function in TypeScript."
```

The first call loads the model, so it is slower. If it fails, jump to
[troubleshooting](#troubleshooting).

## 5. Read the trace, decision, artifacts, and cost

`runs list` is newest-first. Capture the latest run directory in a shell variable:

```text
RUN=$(uv run mg runs list --json | uv run python -c 'import json,sys; print(json.load(sys.stdin)[0]["dir"])')
echo "$RUN"
```

`run.json` is the manifest: graph hash, model hashes, request, budget, usage, path, artifacts,
and (when used) the profile and prune report.

```text
uv run mg runs show "$RUN"
```

The `path` array is the routing history. For a router it includes `decision`, `to`, `confidence`,
and `reason`; a subgraph entry includes the nested graph ref and hash. The `usage` object is the
cost record: `input_tokens`, `output_tokens`, `total_tokens`, `cost_usd`, and `elapsed_seconds`.
With these manifests `cost_usd` is `0.0` because local inference is priced `free`. A paid
manifest uses its per-million prices; cost is computed from normalized usage, not guessed.

The trace is append-only JSONL. Read it as text or as JSON:

```text
uv run mg trace show "$RUN"
uv run mg trace show "$RUN" --json
```

The events to look for:

| Event | What it tells you |
| --- | --- |
| `run.started` | Graph hash, target budget, model/provider map, profile if any |
| `render.completed` | Prompt hash and estimated tokens for each node |
| `provider.request` / `provider.response` | Which model was called, attempt, text, usage, latency |
| `decision.made` | Router decision, target, confidence, reason, threshold |
| `artifact.written` | Artifact name, type, bytes |
| `budget.charged` | Tokens/cost consumed so far |
| `prune.applied` | Profile and the full `PruneReport` |
| `run.finished` / `run.failed` | Outcome, usage, artifact hashes, typed error |

Artifacts are content-addressed files under `artifacts/`; the manifest records the hash and the
producer. To print the recorded artifact (the same digest the manifest lists):

```text
DIGEST=$(uv run mg runs show "$RUN" | uv run python -c 'import json,sys; print(json.load(sys.stdin)["artifacts"][0]["hash"].split(":")[-1])')
ls "$RUN"/artifacts
cat "$RUN/artifacts/$DIGEST"
```

The `artifacts/` directory also holds the input blobs, so `cat "$RUN"/artifacts/*` is not a
reliable way to read only the answer. The `envelope.json` next to them is the request-scoped
state that was validated at runtime.

Replay serves the recorded provider responses. It calls no model and no network, and it writes
`replay.json`:

```text
uv run mg replay "$RUN"
```

Expected: `replay: OK`. Replay refuses if the graph hash changed (for example after you edit a
prompt or a manifest). `--allow-ir-mismatch` is for deliberate experiments only.

## 6. Add a JavaScript specialist and re-run

The starter library already contains the "after" graph so this path stays tested. The change
from the hello shape is:

1. a second router, `coding-triage`, that separates JavaScript/TypeScript from general coding;
2. a `js-specialist` worker on `js-specialist-3b`;
3. a `generic-coder` worker on `coder-7b` as the general branch and the fallback;
4. router edges with `when` clauses and examples for both branches.

The full reference is `graphs/coding-specialist.graph.yaml`. The essential added node looks like
this (copy the whole blocks from the file into your own copy of a graph):

```yaml
- id: coding-triage
  kind: router
  description: Distinguishes JavaScript/TypeScript requests from general programming requests.
  model: router-small
  reads: [task]
  routing:
    confidence_threshold: 0.6
    on_low_confidence: fallback
    max_retries: 1

- id: js-specialist
  kind: worker
  description: Writes modern JavaScript and TypeScript, including tests and type annotations.
  model: js-specialist-3b
  reads: [task]
  instruction: |
    You are a JavaScript and TypeScript specialist. Prefer standard library solutions.
    Write the implementation first, then a short list of assumptions and edge cases.
    Put the final deliverable in the artifact named "answer".
  writes:
    - name: answer
      type: code
      media_type: text/typescript
```

Add the matching outgoing edge (and keep the general branch as the fallback):

```yaml
- from: coding-triage
  to: js-specialist
  when: The request is specifically about JavaScript or TypeScript code, types, or tooling.
  examples:
    - Add types to this React hook.
  artifact: code
  priority: 20
```

If you start from the hello graph, you also replace its single `coding` worker: point the
`triage` edge at `coding-triage`, set `fallback: generic-coder`, and copy the `generic-coder`
worker and the second router edge from the reference file. Copying the whole `nodes` and `edges`
sections from `coding-specialist.graph.yaml` is less error-prone than editing them piecemeal.

Then validate and run. The mock picks the first route, so it always shows the JavaScript branch:

```text
uv run mg validate graphs/coding-specialist.graph.yaml
uv run mg run graphs/coding-specialist.graph.yaml --provider mock \
  --task "Add types to this React hook."
```

Expected path: `coding-triage -> js-specialist`. Run it live:

```text
uv run mg run graphs/coding-specialist.graph.yaml \
  --task "Add types to this React hook."
```

If you edit your own graph, run `mg validate` after every topology change. A router menu change
also invalidates routing comparisons; see [docs/evaluation.md](evaluation.md#3-routing-evaluation).

### Compose the specialist into an assistant

`graphs/composed-assistant.graph.yaml` wraps the coding team as a subgraph alongside a
storyteller:

```text
uv run mg run graphs/composed-assistant.graph.yaml --provider mock
```

Expected path:

```text
domain-triage -> coding (subgraph) -> coding-triage -> js-specialist
```

The inner graph's `answer` is mapped to the outer `answer`; unmapped inner artifacts stay in the
trace. Nesting shares the outer run's budget and depth counter.

The writing/research example routes without any code execution:

```text
uv run mg run graphs/writing-studio.graph.yaml --provider mock \
  --task "Tell me a story about a lighthouse keeper."
```

Expected path: `brief-triage -> draft`. Use a research request with a live model to exercise the
`research` branch; the mock always takes the first route.

## 7. Prune with a profile

A profile filters the graph at runtime; it never edits the YAML. Run the composed assistant on a
4GB machine:

```text
uv run mg run graphs/composed-assistant.graph.yaml --provider mock \
  --profile profiles/local-4gb.yaml
```

Expected output starts with the profile and a prune report:

```text
profile:  local-4gb (.../profiles/local-4gb.yaml)
prune:    profile=local-4gb removed_nodes=2 removed_edges=2 removed_exits=0
  removed node story (composed-assistant): model 'story-7b' requires 6GB VRAM; ...
  removed node generic-coder (coding-specialist): model 'coder-7b' requires 6GB VRAM; ...
  warning: router 'domain-triage' now has one route; consider bypassing it
  warning: fallback 'generic-coder' was pruned; ...
```

The run still succeeds on the JavaScript branch. The authored graph hash is unchanged; the
`run.json` records the effective profile and the report under `profile` and `prune`. The router
menu is filtered, so the model is not offered a branch it cannot run.

`--profile` accepts a path or a bare name. A bare name is searched in `./profiles`,
`./examples/profiles`, and the repository copies:

```text
uv run mg run graphs/composed-assistant.graph.yaml --provider mock --profile local-4gb
uv run mg run graphs/composed-assistant.graph.yaml --provider mock --profile profiles/local-8gb.yaml
```

The 8GB profile removes nothing from the starter graphs. The 4GB profile cannot run
`hello-triage` or `writing-studio` (both workers need 6GB), so it exits `3` with `MG-1005` and
the full report. That is a policy result, not a bug: the entry or every exit producer was pruned.

You may also see `warning: profile override max_steps=8 ignored because it raises the graph
default 6`. The starter profiles set `max_steps: 8`, but `hello-triage`, `coding-specialist`,
and `writing-studio` default to 6. Profiles tighten budgets by default, so the raise is ignored
and recorded; `composed-assistant` defaults to 8, so it does not warn. Pass
`--allow-budget-override` only when you intend to loosen a graph's hard default.

The air-gapped profile is an audit tool:

```text
uv run mg run graphs/hello-triage.graph.yaml --provider mock \
  --profile profiles/no-network.yaml
```

Expected: exit `3`, `MG-1005`, and a report showing that every model requires loopback network
access. v0 has no in-process model adapter, so a loopback HTTP server is still network access.
Use `local-8gb` for a real local run; use `no-network` to see what a graph would need before you
allow any socket.

## 8. Run a node evaluation

The node suite renders the exact node prompt and checks the artifact with deterministic checks.
Run the JavaScript specialist suite against the starter graph:

```text
uv run mg eval node --suite evals/mock/js-specialist \
  --graph graphs/coding-specialist.graph.yaml --provider mock
```

Expected: `gate: PASS`, `metric: task_success=1.0`, `failures: 0`, and a report directory under
`runs/evals/`. The suite uses scripted mock responses; it validates the evaluation instrument,
not the quality of the real model. Read the human summary and the machine report:

```text
EVAL=$(ls -td runs/evals/eval_* | head -1)
cat "$EVAL/report.md"
uv run python -m json.tool "$EVAL/report.json" | head -60
```

The report includes the suite hash, graph hash, selection, primary metric with a confidence
interval, validity, cost, latency, and per-case files. `--fail-on-gate` turns a threshold failure
into exit code `1`. For a live measurement, run the same suite against Ollama and record the
hardware:

```text
uv run mg eval node --suite evals/mock/js-specialist \
  --graph graphs/coding-specialist.graph.yaml --provider ollama --live
```

A live node suite is a smoke measurement until it has enough cases and a baseline
([docs/evaluation.md](evaluation.md#5-suites-and-data-hygiene)). Do not turn one run into a
quality claim.

## Troubleshooting

These are the failures a first run is most likely to hit. Every failure after the graph starts
still writes a trace and a run manifest, so `mg trace show "$RUN"` and `mg runs show "$RUN"` are
the first diagnostics. A compile or prune failure exits before a run exists; read the printed
error and report.

### 1. Connection refused — `MG-3001`

`Ollama connection failed: ...` means the adapter could not reach the endpoint in the manifest.

```text
curl -sS http://127.0.0.1:11434/api/tags
systemctl status ollama
ollama serve
```

Check `spec.endpoint` in `registry/models/*.yaml`; the default is `http://127.0.0.1:11434`. If
you use a remote or container endpoint, change the manifest, not the graph. After installing a
GPU backend, restart the service and confirm it is using it:

```text
systemctl restart ollama
journalctl -u ollama -n 30 --no-pager | grep -i 'inference compute'
ollama ps
```

### 2. Model missing — `MG-3005`

`model '...' is not available in Ollama` means the tag was not pulled or does not match the
manifest exactly.

```text
ollama list
ollama pull qwen2.5-coder:3b
uv run mg models list
```

Tags matter (`qwen2.5-coder:3b` is not `qwen2.5-coder:3b-instruct`). If you changed a manifest,
re-run `mg validate` and start a fresh run; an old run keeps its recorded hashes.

### 3. Context overflow or token budget — `MG-4002`

The runtime enforces the graph's hard token budget before each provider call. `MG-4002` means
the rendered prompt plus the configured output allowance would exceed `max_total_tokens`. The
short fix is a shorter task, fewer artifacts in `reads`, or a larger budget:

```text
uv run mg run graphs/hello-triage.graph.yaml --max-tokens 48000
```

Treat a raise as a deployment decision, not a default. v0 does not yet enforce the per-model
`MG-2203` context-overflow path from [docs/protocol.md](protocol.md#34-context-overflow); if the
prompt is under the run budget but over the model's window, Ollama may truncate or return an
error. Keep the first tasks short and check `render.completed.tokens` in the trace.

### 4. Invalid router JSON — `MG-2101`

Small routers sometimes return prose, markdown, a missing object, an invented route, or an
out-of-range confidence. The runtime makes **one** repair attempt with a stricter prompt, then
fails with `MG-2101`. The fallback does not rescue a malformed decision.

Look at `provider.response` for the router in the trace. Then:

- use a model with `json_mode` (the starter router does);
- keep temperature at `0` for routers;
- shorten the menu or the examples if the router is confused;
- run the routing suite before trusting a prompt change
  ([docs/evaluation.md](evaluation.md#3-routing-evaluation));
- use a mock scenario with a deliberate bad response to test your failure handling.

### 5. The profile removed the branch you expected — `MG-1005`

Read the `PruneReport` printed before the error. Common causes:

- the model declares more VRAM/RAM/disk than the profile allows (6GB models under `local-4gb`);
- the model's provider is not in `providers`;
- the model's license is not in `licenses`;
- `network_access` is `loopback` but the profile ceiling is `none`;
- pricing is `unknown` and `max_cost_usd` is set.

Fix the policy (use `local-8gb`), the model (a smaller manifest), or the graph (remove the
branch). Do not edit the run directory; re-run after the change. A profile that removes the
entry or the last producer of an exit fails closed with `MG-1005` rather than silently running a
different graph.

### Also likely after your first edit

- **Replay mismatch:** `graph hash changed` means the YAML, a manifest, the renderer, or the
  parser changed since the recording. Re-run instead of forcing `--allow-ir-mismatch`.
- **No models found:** the default registry is `registry/models/`. Pass `--registry PATH` if
  yours is elsewhere; `mg models list` must show the IDs used by the graph.
- **Expensive run:** local manifests are `free`, but a hosted manifest is not. Read
  `usage.cost_usd` and set `--max-cost`; budgets are hard.

## 9. What to read next

| You want to… | Read |
| --- | --- |
| Understand why the project exists | [vision.md](vision.md) and [README.md](../README.md) |
| Learn the graph contract | [graph-model.md](graph-model.md) and [schemas/graph.schema.json](../schemas/graph.schema.json) |
| Learn prompts, envelopes, and decisions | [protocol.md](protocol.md) |
| Understand budgets, traces, replay, pruning | [execution.md](execution.md) and [profiles.md](profiles.md) |
| Measure quality, cost, and routing | [evaluation.md](evaluation.md) and [../evals/README.md](../evals/README.md) |
| Change models safely | [iteration.md](iteration.md) |
| Work on the runtime | [architecture.md](architecture.md) and [development.md](development.md) |
| Use the local GUI | [gui.md](gui.md) |
| See what to build next | [../.scratch/tickets/BOARD.md](../.scratch/tickets/BOARD.md) and [roadmap.md](roadmap.md) |

Run the checks before sending a change:

```text
uv run mg validate graphs/*.graph.yaml
uv run pytest tests/tutorial -q
make check
```

If a tutorial command stops working, that is a bug in the starter library, not a reader problem.
Open a ticket with the command, the run directory, and the exact error code.
