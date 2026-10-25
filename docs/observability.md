# Observability

`modelgraph` has one event source: the append-only trace at `runs/<run-id>/trace.jsonl`.
Structured logs, metrics, local summaries, and spans are projections of that trace (or of
the same events while a run is in progress). There is no second event model and no metrics
database in v0.

This document covers:

1. [Structured logging](#1-structured-logging)
2. [Metrics derived from traces](#2-metrics-derived-from-traces)
3. [Local summaries with `mg stats`](#3-local-summaries-with-mg-stats)
4. [OpenTelemetry export](#4-opentelemetry-export)
5. [Privacy defaults and how to disable everything](#5-privacy-defaults-and-how-to-disable-everything)
6. [Limitations](#6-limitations)

The trace event schema itself is defined in [execution.md §9](execution.md#9-observability).
Run evidence (manifest, trace, environment) is described in
[execution.md §1](execution.md#1-from-yaml-to-a-run) and the
[run manifest schema](../schemas/run-manifest.schema.json).

## 1. Structured logging

### JSONL mode

`modelgraph.observability.StructuredLogger` writes one JSON object per line. Every record
has at least:

| Field | Meaning |
| --- | --- |
| `ts` | ISO-8601 UTC timestamp (`2026-10-25T12:34:56Z`) |
| `level` | `debug`, `info`, `warning`, `error`, or `critical` |
| `event` | Short event name, for example `run.started` |
| `run_id` | Run identifier, or `null` when the logger has no run context |
| `node` | Node identifier, present only when the event applies to a node |

Additional scalar fields (counts, durations, codes, model/provider identifiers) are allowed.

```python
from modelgraph.observability import StructuredLogger

logger = StructuredLogger(run_id="run_1234", path="logs/run_1234.jsonl")
logger.info("node.entered", node="triage", depth=0)
logger.warning("budget.charged", node="triage", tokens=512, cost_usd=0.0003)
logger.close()
```

Output:

```json
{"ts":"2026-10-25T12:34:56Z","level":"info","event":"node.entered","run_id":"run_1234","node":"triage","depth":0}
{"ts":"2026-10-25T12:34:57Z","level":"warning","event":"budget.charged","run_id":"run_1234","node":"triage","tokens":512,"cost_usd":0.0003}
```

### Console mode

`ConsoleLogger` renders the same fields as one human-readable line and never emits JSON:

```text
2026-10-25T12:34:56Z INFO    node.entered run=run_1234 node=triage depth=0
```

### Environment configuration

| Variable | Values | Default | Meaning |
| --- | --- | --- | --- |
| `MG_LOG_FORMAT` | `off`, `json`, `console` | `off` | Output mode. |
| `MG_LOG_FILE` | path | unset → stderr | Destination for JSONL mode. |
| `MG_LOG_LEVEL` | `debug`…`critical` | `info` | Minimum level. |
| `MG_LOG_RUN_ID` | string | unset | Default `run_id` for records. |

```python
from modelgraph.observability import build_logger_from_env

logger = build_logger_from_env()  # NullLogger unless MG_LOG_FORMAT is set
```

`build_logger_from_env` returns a `NullLogger` when logging is disabled, so callers do not
need a branch. When logging is enabled, an unknown format or level, and an `MG_LOG_FILE` that
cannot be opened, raise `ModelGraphError` (`MG-5000`) rather than a filesystem traceback.

`mg run` builds the logger and span exporter from the environment and passes them to
`run_graph`; library callers can pass their own `logger`/`span_exporter` or leave the defaults
(`NullLogger`/`NullSpanExporter`). Setting `MG_LOG_FORMAT` alone therefore adds JSONL log lines
to a CLI run while the trace remains the runtime's source of truth.

### Redaction

Field values are scrubbed before serialization. Keys that name prompts, messages, task
text, artifacts, content, bodies, raw payloads, or credentials are replaced with
`[redacted]`; a recursive pass covers nested dictionaries and lists. Every other string
value is then scrubbed with the shared default secret/PII pattern set (private keys,
provider tokens, bearer tokens, e-mail addresses, home paths) and replaced with
`[REDACTED:<pattern>:sha256:<digest>]`, so a secret echoed into an innocently named field
such as `reason` or `note` is still removed; matched text is never written. The rule is
conservative, so a safe field such as `prompt_hash` is also redacted. Callers should still
pass identifiers rather than content: redaction is a backstop, not permission.

## 2. Metrics derived from traces

### Projection API

```python
from modelgraph.observability import project_trace, project_events, collect_stats

snapshot = project_trace("runs/run_1234/trace.jsonl")   # one stored trace
snapshot = project_events(read_events(path))            # any stored event stream
report = collect_stats("runs", last=20)                 # recent-run summary
```

`MetricsAccumulator.observe(event)` consumes the raw event dictionaries from
`trace.jsonl` (`TraceRecorder` records) and `MetricsAccumulator.snapshot()` aggregates
them. Because the projection has no in-memory dependency, it can be recomputed from disk
after a process restart and is deterministic for a fixed trace.

### Event-to-metric mapping

| Metric | Source events | Derivation |
| --- | --- | --- |
| Runs started | `run.started` | One per distinct run id. |
| Runs finished | `run.finished`, `run.failed` | One terminal event per run. |
| Outcomes | `run.finished.outcome`, `run.failed.outcome` | Counter by `succeeded`, `failed`, `budget_exhausted`, `interrupted`, … |
| Node calls | `node.entered` | Counter per node id. |
| Protocol errors | `run.failed.code` in `MG-21xx`/`MG-22xx`, `protocol.error`, `node.failed` | Counter by code. |
| Provider errors | `provider.error.code`, `run.failed.code` in `MG-3xxx` | Counter by code. |
| Retries | `provider.request.attempt > 1` | Count per node. |
| Tokens by node | `budget.charged.tokens` | Charged tokens, attributed to the emitting node. |
| Tokens by model | `budget.charged.tokens` plus `provider.request.model` | Attributed through the node→model map. |
| Cost by node/model | `budget.charged.cost_usd` | Per-call cost, attributed the same way. |
| Latency by node | `provider.response.latency_seconds` (or `node.exited.duration_seconds` when present) | p50/p95 per node. |
| Latency by provider | `provider.response.latency_seconds` | p50/p95 per provider. |
| Run latency | `run.finished.usage.elapsed_seconds`, else terminal timestamps | p50/p95 over runs. |
| Cache hits | `provider.response.cached`, `provider.response` with `cached: true`, or `cache.hit` | Paired runtime events share a `cache_hit_id`, so one hit is counted once. |
| Cache misses | explicit `cache.miss` or `cached: false` | A plain `provider.response` is **not** a miss, so pre-cache traces stay at zero. |
| Budget exhaustion | terminal `MG-4xxx` code, `budget.exhausted` | Counter by type: `step`, `token`, `cost`, `wall_clock`, `depth`, `other`. |

`MG-4001` = step, `MG-4002` = token, `MG-4003` = cost, `MG-4004` = wall clock, `MG-4006` =
depth.

### Agreement with run manifests

For a completed run the manifest is the evidence record and the trace is its event log.
The projection uses the same values the manifest records:

- total charged tokens compare equal to `manifest.usage.total_tokens`;
- total cost compares equal to `manifest.usage.cost_usd`;
- terminal `outcome` compares equal to `manifest.outcome`.

The tests in `tests/observability/` assert this on a real mock run. The projection reads only
`trace.jsonl`; the manifest is not consulted, which is what makes it recomputable. A trace
whose `budget.charged` events are missing falls back to `run.finished.usage` for totals.

### Subgraphs

A run and its subgraphs share one trace file, so child node calls appear under the parent
trace and are counted once. The child manifests point at the same trace
(`"trace": "../../trace.jsonl"`), so they are not separate metric sources.

## 3. Local summaries with `mg stats`

`mg stats` reads the newest `runs/<id>/trace.jsonl` files and prints a summary. It never
constructs a provider, opens a network connection, or reads a model manifest.

```text
uv run mg stats --last 5
uv run mg stats --last 20 --runs-dir runs --json
```

Human output:

```text
runs dir: /repo/runs
runs:     5 of 41 (last 5)
outcomes: succeeded=4 failed=1
success:  80.0%
tokens:   input=12518 output=1290 total=13808
cost:     $0.000000
latency:  p50=0.006s p95=0.009s (n=5)
cache:    hits=0 misses=0
failures: MG-4001=1
```

`--last N` selects the N newest traces by file modification time. `--json` emits the
aggregate plus a per-run array:

```json
{
  "runs_dir": "runs",
  "available_runs": 41,
  "selected_runs": 5,
  "last": 5,
  "success_rate": 0.8,
  "outcomes": {"failed": 1, "succeeded": 4},
  "total_input_tokens": 12518,
  "total_output_tokens": 1290,
  "total_tokens": 13808,
  "total_cost_usd": 0.0,
  "latency": {"count": 5, "p50_seconds": 0.006, "p95_seconds": 0.009, "...": "..."},
  "top_failure_codes": [{"code": "MG-4001", "count": 1}],
  "cache": {"hits": 0, "misses": 0},
  "budget_exhaustion": {"step": 1},
  "metrics": {"...": "full projection"},
  "runs": [{"run_id": "...", "run_dir": "...", "outcome": "...", "tokens": {}, "cost_usd": 0.0}]
}
```

Success rate is `succeeded / terminal runs` over the selected traces (provisional runs
have no trace yet, so they are not in the denominator). `top_failure_codes` lists up to five
codes ordered by count and then code.

## 4. OpenTelemetry export

OpenTelemetry is **not a required dependency**. `modelgraph.observability` imports it
lazily and the core, CLI, and `mg stats` run with it absent. The exporter is off by default.

### Enabling it

| Variable | Values | Default | Meaning |
| --- | --- | --- | --- |
| `MG_OTEL` (or `MG_OTEL_ENABLED`) | `off`, `on`/`1`/`true`/`api`/`otlp`/`console` | `off` | Build the OTel adapter or the null exporter. |
| `MG_OTEL_SERVICE_NAME` | string | `modelgraph` | Tracer service name. |
| `OTEL_SERVICE_NAME` | string | unset | Standard fallback for the service name. |

```python
from modelgraph.observability import build_span_exporter, run_span, node_span

exporter = build_span_exporter()  # NullSpanExporter unless MG_OTEL is enabled
with run_span(exporter, run_id, graph_id="hello-triage", target="portable"):
    with node_span(exporter, run_id, "triage", kind="router", depth=0, step=1):
        ...  # graph semantics are unchanged; spans only observe
```

When `MG_OTEL` is enabled but the optional `opentelemetry-api` package is missing,
`build_span_exporter()` raises `ModelGraphError` (`MG-5000`) with an install hint instead of
silently dropping spans.

The runtime does not auto-instrument arbitrary scheduler internals. `run_graph` creates a
`modelgraph.run` span and a `modelgraph.node` span per dispatched node when a span exporter is
provided; `mg run` provides one only when `MG_OTEL` is enabled. This keeps execution semantics
unchanged and keeps the default path free of optional dependencies.

### What is exported

Span names: `modelgraph.run` and `modelgraph.node`.

Allowed attributes are an explicit allowlist of identifiers and timings:

`run_id`, `parent_run_id`, `graph_id`, `graph_hash`, `node`, `kind`, `node_kind`, `depth`,
`step`, `provider`, `model`, `model_hash`, `model_ref`, `profile`, `target`,
`deployment_revision`, `seed`, `outcome`, `error_code`, `child_run_id`.

Prompts, messages, task text, artifact content, raw model responses, and credentials are
**not** exported. Attributes are filtered again inside each exporter, so passing a
content-shaped keyword to `run_span`/`node_span` cannot leak it. Durations come from the
span lifetime; the recording exporter exposes `started_at`, `finished_at`, and
`duration_seconds` for tests.

### Adapter shape and the collector

`OpenTelemetrySpanExporter` adapts any tracer with
`start_as_current_span(name, attributes=...)`. It does not install an SDK, configure an
OTLP exporter, or start a collector. The host application owns the global tracer provider:
install `opentelemetry-sdk` plus an exporter such as `opentelemetry-exporter-otlp`, then set
the standard `OTEL_EXPORTER_OTLP_ENDPOINT` for that exporter. With only the API installed,
OpenTelemetry itself supplies a no-op tracer.

The repository tests cover the absent path, the null exporter, the in-memory recording
exporter, and the adapter against an OTel-shaped tracer. They do **not** prove delivery to a
real collector; that remains a manual/host integration step.

## 5. Privacy defaults and how to disable everything

Default posture:

- Structured logging is off (`MG_LOG_FORMAT=off`); when enabled, prompt/artifact/secret
  fields are redacted.
- The span exporter is off (`MG_OTEL=off`); when enabled, only allowlisted identifiers and
  timings are exported.
- `mg stats` prints aggregates and identifiers, not task text or artifact content.
- Metrics are derived from the trace; they do not add new content to it. The trace itself is
  governed by trace redaction (`TraceRecorder.write`), not by this module.

To disable observability completely:

```text
MG_LOG_FORMAT=off MG_OTEL=off uv run mg run examples/hello-triage.graph.yaml --provider mock
```

That is the default posture. `MG_LOG_FORMAT` and `MG_OTEL` configure the CLI run path; when
both are off, `mg run` emits neither log lines nor spans. Do not call `StructuredLogger`,
`build_span_exporter`, or `mg stats` in a pipeline that must not produce observability output.
There is no background thread, daemon, or telemetry beacon in this module.

## 6. Limitations

- **Node latency is provider-call latency.** The current runtime reports
  `provider.response.latency_seconds`; it does not add a per-node `node.exited.duration`
  field. A node that makes several provider calls contributes several samples and its
  rendering/parsing time is not included. The projection prefers an explicit
  `node.exited.duration_seconds` if a future trace provides one.
- **Run latency for failed runs may be coarse.** `run.finished.usage.elapsed_seconds` is
  precise; a failed run without usage falls back to whole-second trace timestamps.
- **A budget charge that raises is not a `budget.charged` event.** When `Budget.charge`
  increments usage and then raises `BudgetExceeded`, the event is never emitted, so the
  trace-only token/cost projection is slightly below the manifest for that failed run. The
  error code and budget type are still projected.
- **Cache hits are not provider latency.** A cached response emits
  `provider.response` with `cached: true` and zero latency; the projection counts the hit
  and skips the latency sample. Avoided tokens and cost are in `mg run --json` and the
  `provider.response.cached` event, not in the `mg stats` `cache` object.
- **Subgraph attribution is by event, not hierarchy.** Child node calls appear in the parent
  trace and are counted once, but per-subgraph rollups are not produced.
- **No distributed tracing, aggregation backend, dashboard, alerting, or profiling.** This
  is a local projection plus an optional adapter.
- **Logging and spans are wired for `mg run`/`run_graph`, not for every internal helper.**
  `mg run` emits JSONL logs and run/node spans when configured; direct `run_graph` callers
  default to null implementations and must pass `logger`/`span_exporter` to opt in.
- **No retention policy here.** `mg stats` reads whatever traces are in the runs directory;
  retention/GC is a separate revision-store policy.
- **`mg stats` discovers `runs/<id>/trace.jsonl`.** It does not recover provisional manifests
  or follow a manifest's custom `trace` path. A run that never finalized has no trace and is
  not summarized.
