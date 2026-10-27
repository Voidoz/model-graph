# Static inspector

`mg inspect --html RUN_DIR --graph GRAPH.yaml --out FILE.html` writes one read-only,
self-contained HTML file for a recorded run. It is the T-0028 spike: a static artifact, not
a new application stack and not a frontend build step.

T-0040 adds two optional, read-only iteration surfaces:

```text
# Impact panel, stale badges, and revision timeline for a stored change set.
uv run mg inspect --html runs/<run-id> --graph examples/composed-assistant.graph.yaml \
  --out /tmp/run-impact.html --change <change-id> --store .mg/revisions

# Side-by-side comparison of two recorded runs on the same (or different) inputs.
uv run mg inspect --html runs/<run-id> --graph examples/composed-assistant.graph.yaml \
  --out /tmp/run-compare.html --compare runs/<other-run-id>
```

Both flags can be combined. They load and render; they never promote, roll back, recompile,
re-distill, or rewrite a revision or run.

## What is read

The command consumes two inputs:

- `RUN_DIR` — a recorded run directory: `run.json`, `trace.jsonl`, `artifacts/`, and, when
  present, `envelope.json`. The run manifest stays authoritative for the outcome, path,
  budget, usage, profile, prune report, model pins, and artifact metadata.
- `GRAPH.yaml` — the authored graph. It is validated and compiled through the existing
  compiler/registry path (`compile_graph(..., strict_models=True)`), so the graph shown is
  the same compiled IR the runtime and GUI use. The recorded run may have used a pruned
  view; that is expected and displayed, not treated as a mismatch error.

Artifact bytes are resolved by the content hash in `run.json` under `artifacts/`, then
embedded in the output. The inspector never references the source run directory at view
time. A copied run directory renders identically because the page embeds content rather
than linking to files.

## The static boundary

The page is a single file with inline CSS and inline SVG:

- no CDN, fonts, images, stylesheets, scripts, or network requests;
- a restrictive `Content-Security-Policy` meta tag (`default-src 'none'`, no connect/img/
  font sources) enforces the no-network boundary in the browser;
- no `xmlns` fetch, `fetch()`, `XMLHttpRequest`, `eval`, `new Function`, or `innerHTML` with
  data;
- one fixed inline script implements node/outcome/cache filtering and expand/collapse. It
  only toggles `hidden`/`open` and writes a count with `textContent`.

Artifact contents are untrusted. Every value derived from the graph, manifest, trace,
artifacts, model names, reasons, or error text is HTML-escaped. C0/C1 control characters are
rendered as visible escapes (for example `\x00`) instead of raw control bytes. The embedded
JSON block escapes `<`, `>`, and `&` as JSON unicode escapes, so content such as
`</script>` cannot close the block.

Artifact payloads are capped at 1,000,000 bytes per artifact and 8,000,000 bytes total per
inspection document. A truncated artifact carries a visible notice and its original byte
count; the visible panel and the embedded JSON share the same capped value. The whole file
must still be treated as sensitive: it embeds run data, and a static file cannot hide what it
contains.

## The shared JSON document

`modelgraph.inspector.build_inspection()` returns the canonical document
(`schema: modelgraph/inspector/v0`) that `render_inspector_html()` renders. The same document
is embedded in the page as:

```html
<script type="application/json" id="mg-inspector-data">…</script>
```

Top-level fields:

| Field | Meaning |
| --- | --- |
| `schema` | `modelgraph/inspector/v0` |
| `summary` | run status, graph/model hashes, timestamps, profile/prune identity, budget, error |
| `totals` | cost, latency, tokens, and steps copied from the run manifest usage block |
| `cache` | trace-derived hit/miss/skipped and avoided-usage data, or unavailable |
| `path` | ordered manifest path entries with decisions, confidence, target, reason |
| `graph` | compiled authored nodes/edges, entry, exit artifact names, chosen/pruned flags |
| `nodes` | per-node trace projections: provider, model, tokens, cost, latency, retries, errors |
| `prune` | removed nodes/edges/exit artifacts and warnings from `run.json` |
| `artifacts` | name, type, producer, hash, exit marker, capped content, truncation metadata |
| `iteration` | optional, when `--change` is used: change, impact, stale badges, timeline, command text |
| `compare` | optional, when `--compare` is used: input/graph indicators, path/node/artifact/totals rows |

`iteration` and `compare` are omitted entirely for the original T-0028 invocation, so the
existing document and HTML stay unchanged when the new flags are absent.

The document is deliberately presentation-neutral. `render_inspector_html()` is one view of
it. A future TypeScript UI should consume this JSON (from a future `mg inspect --json` or a
GUI endpoint) and only implement presentation:

- the Python inspector decides chosen-path edges, prune marking, token/cost/latency totals,
  cache availability, and truncation; the UI must not re-derive those rules differently;
- the UI may lay out the graph differently, but must keep the JSON meaning of `chosen`,
  `pruned`, `exit`, `status`, `source`, and `truncated`;
- if a field is missing, render "unknown"; do not infer a replacement from the trace.

This keeps semantics in one place. The static file is a rendering of the document, not a
second inspector implementation.

## Iteration panels (`--change` and `--compare`)

### Impact before promotion (`--change`)

With `--change CHANGE_ID --store STORE`, the CLI loads the stored change set, its stored
`impact.json`, and a read-only revision timeline. It then renders:

- the change class, target, base and candidate revisions, and status;
- compatibility checks (base, tokenizer, context window, blocking message);
- required evaluation levels, satisfied/missing coverage, and any provided evidence;
- invalidations (compiled plan, prefix cache, route cache, distilled student) with scope and
  reason;
- VRAM before/after/delta and profile admission;
- rollback target, retention, and the impact notes/risks;
- stale badges for the plan, caches, and student;
- a revision timeline distinguishing `ACTIVE`, `PREVIOUS`, `CANDIDATE`, and hash-based `STALE`
  retired revisions, with `PROMOTED`/`ROLLED BACK` markers;
- safe follow-up commands (open the change, dry-run promotion, recompile the authored graph)
  as text.

The store is opened with `create=False`; the inspector never writes a revision, change set,
pointer, or run. If the store or change is absent, the command fails cleanly instead of
creating one.

**Honest not-applicable state.** v0 has no compiled/fused plan, prefix/route cache, or distilled
student artifact. The impact report marks those artifacts `not_applicable` (portable target) or
`invalidated` with `present: false`, and a distilled student is `stale` as ADR-0013 requires.
The inspector shows the badge and `present: none`, and keeps `plan_hash`/`student_hash` null.
It does not invent a hash, a rebuild, or a successful promotion. The "rebuild plan" and
"re-distill" actions are therefore marked not available; the available commands are ordinary
CLI text such as `mg compile <graph> --json` and `mg change explain <change-id>`. The page does
not execute any of them.

### Side-by-side comparison (`--compare`)

With `--compare RUN_B`, the CLI reads both manifests and traces without touching either run and
renders:

- a `Same inputs:` indicator derived from the hash of the recorded `request` document, plus a
  warning when that hash differs;
- a separate same/different/unknown graph-hash note and warnings from either trace;
- side-by-side totals, path/decisions, per-node provider/model/tokens/cost/latency, and
  artifacts by name/hash;
- differences highlighted in the row and cell (`data-diff`, `diff-cell`), and `—` for a node or
  artifact that exists on only one side.

A cross-input comparison is deliberately shown as such; it is not silently treated as a paired
replay. If one run has no recorded request, the indicator reads `unknown` rather than guessing.
Artifact content is not embedded a second time: the comparison uses names and hashes, keeping
the file bounded by the normal run artifact caps.

## Determinism and tests

Given the same run directory and compiled graph, the CLI writes byte-identical HTML. The
page contains recorded timestamps from the manifest, but no generation timestamp. Tests in
`tests/experience/test_inspector.py` cover determinism, a real mock run, graph/path/
artifact/totals content, hostile artifact and error text, no external assets, prune marking,
copied runs, per-artifact and total caps, cache filters, and clean CLI failure modes. The T-0040
tests in `tests/experience/test_iteration_ux.py` cover the impact panel/evidence/invalidations/
rollback, explicit not-applicable stale badges, active/previous/candidate timeline markers,
read-only `--change`, same/different input indicators, missing nodes/artifacts, hostile impact
and manifest escaping, no external assets, and comparison slices/uncertainty.

## Limits

- One recorded run; `--compare` adds exactly one second run. No live streaming, editing,
  replay control, authentication, or hosting.
- The inspection is static: the impact panel and timeline are read-only. Follow-up "actions"
  are command text, not links, buttons, or live mutations.
- A revision timeline reflects the store passed with `--store` (or `MG_REVISION_STORE`); it
  does not query a server or infer revisions that are not in that store.
- Compiled plans, prefix/route caches, and distilled students do not exist in v0. The panel
  shows `not_applicable`/`none`; it does not synthesize those artifacts.
- The SVG renders the authored outer graph. Inner subgraph nodes appear in the node table as
  trace-only rows, not as a nested graph.
- Trace events are read in full; only artifact content is capped.
- The trace projection is intentionally best-effort. The manifest usage block remains the
  authoritative totals source, and the page labels trace-derived sums separately.
- Redaction is whatever the recorded run already applied. The inspector does not add a new
  redaction pass; its embedded-data warning is part of the contract.
