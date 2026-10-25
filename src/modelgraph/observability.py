"""Observability: structured logs, trace-derived metrics, and optional spans.

The trace is the source of truth. Nothing in this module keeps a second event model:
metrics and summaries are projections over the raw ``trace.jsonl`` event dictionaries, so
they can be recomputed after a process restart. The optional OpenTelemetry adapter is
imported lazily and is off by default; the core runtime never depends on it.

Privacy defaults:

* structured logs redact prompt/artifact/secret-shaped fields before writing;
* span attributes are restricted to an allowlist of identifiers and timings;
* ``mg stats`` reads traces but prints only aggregates and identifiers.

See ``docs/observability.md`` for the event-to-metric table and the environment variables.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, TextIO, cast

from modelgraph.errors import ModelGraphError
from modelgraph.evaluation.metrics import latency_summary
from modelgraph.security import redact_text_default
from modelgraph.util import utc_now


class ObservabilityError(ModelGraphError):
    """Observability configuration is invalid or its optional dependency is missing."""

    code = "MG-5000"


# --------------------------------------------------------------------------------------
# Structured logging
# --------------------------------------------------------------------------------------

REDACTED = "[redacted]"

_LOG_LEVELS: dict[str, int] = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
}

# Keys that are content, prompts, artifacts, or credentials. The matching is intentionally
# conservative: observability must never become a side channel for user data.
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "artifact",
        "artifacts",
        "authorization",
        "body",
        "completion",
        "content",
        "credentials",
        "input",
        "message",
        "messages",
        "output",
        "password",
        "prompt",
        "prompts",
        "raw",
        "request",
        "response",
        "secret",
        "task",
        "text",
    }
)
_SENSITIVE_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "prompt",
    "secret",
)


def is_sensitive_key(key: str) -> bool:
    """Whether a log field name looks like content, a prompt/artifact, or a secret."""
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _SENSITIVE_KEYS:
        return True
    return any(fragment in normalized for fragment in _SENSITIVE_FRAGMENTS)


def redact_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of ``fields`` with content/secret-shaped values redacted.

    Redaction is recursive so a nested provider payload cannot smuggle a prompt through.
    Sensitive keys are kept in the output as ``[redacted]`` so the presence of a field is
    visible without its value. Every remaining string value is also scrubbed with the shared
    default secret/PII pattern set, so a secret or address in an innocently named field
    (``reason``, ``note``, ``error``) cannot reach the log.
    """
    return {str(key): _redact_value(str(key), value) for key, value in fields.items()}


def _redact_value(key: str, value: Any) -> Any:
    if is_sensitive_key(key):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_value(str(child_key), child)
            for child_key, child in value.items()
        }
    if isinstance(value, list | tuple | set | frozenset):
        return [_redact_value("", item) for item in value]
    if isinstance(value, str):
        return redact_text_default(value)
    if value is None or isinstance(value, int | float | bool):
        return value
    return redact_text_default(str(value))


class EventLogger(Protocol):
    """The small logging surface used by callers (JSONL, console, or null)."""

    def event(
        self,
        event: str,
        *,
        level: str = "info",
        node: str | None = None,
        run_id: str | None = None,
        **fields: Any,
    ) -> dict[str, Any] | None: ...

    def info(self, event: str, **fields: Any) -> dict[str, Any] | None: ...

    def warning(self, event: str, **fields: Any) -> dict[str, Any] | None: ...

    def error(self, event: str, **fields: Any) -> dict[str, Any] | None: ...


class NullLogger:
    """A logger that writes nothing; the default when logging is not configured."""

    def event(
        self,
        event: str,
        *,
        level: str = "info",
        node: str | None = None,
        run_id: str | None = None,
        **fields: Any,
    ) -> None:
        return None

    def info(self, event: str, **fields: Any) -> None:
        return None

    def warning(self, event: str, **fields: Any) -> None:
        return None

    def error(self, event: str, **fields: Any) -> None:
        return None


class StructuredLogger:
    """Write one JSON object per line to a file or stream.

    Every record has ``ts`` (ISO-8601 UTC), ``level``, ``event``, ``run_id``, and ``node``
    when one applies. Content/secret-shaped fields are redacted before serialization.
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        level: str = "info",
        path: str | Path | None = None,
        stream: TextIO | None = None,
        mode: str = "a",
    ) -> None:
        if level not in _LOG_LEVELS:
            raise ValueError(f"unknown log level {level!r}")
        self.run_id = run_id
        self.level = level
        self.path = Path(path) if path is not None else None
        self._owns_stream = False
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream: TextIO = cast(TextIO, self.path.open(mode, encoding="utf-8"))
            self._owns_stream = True
        else:
            self._stream = stream if stream is not None else sys.stderr

    def event(
        self,
        event: str,
        *,
        level: str = "info",
        node: str | None = None,
        run_id: str | None = None,
        **fields: Any,
    ) -> dict[str, Any] | None:
        normalized = level.lower()
        if normalized not in _LOG_LEVELS:
            raise ValueError(f"unknown log level {level!r}")
        if _LOG_LEVELS[normalized] < _LOG_LEVELS[self.level]:
            return None
        record: dict[str, Any] = {
            "ts": utc_now(),
            "level": normalized,
            "event": event,
            "run_id": run_id if run_id is not None else self.run_id,
        }
        if node is not None:
            record["node"] = node
        record.update(redact_fields(fields))
        self._stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._stream.flush()
        return record

    def info(self, event: str, **fields: Any) -> dict[str, Any] | None:
        return self.event(event, level="info", **fields)

    def warning(self, event: str, **fields: Any) -> dict[str, Any] | None:
        return self.event(event, level="warning", **fields)

    def error(self, event: str, **fields: Any) -> dict[str, Any] | None:
        return self.event(event, level="error", **fields)

    def close(self) -> None:
        if self._owns_stream:
            self._stream.close()

    def __enter__(self) -> StructuredLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class ConsoleLogger:
    """Human-readable rendering of the same events; never JSON."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        level: str = "info",
        stream: TextIO | None = None,
    ) -> None:
        if level not in _LOG_LEVELS:
            raise ValueError(f"unknown log level {level!r}")
        self.run_id = run_id
        self.level = level
        self.stream = stream if stream is not None else sys.stderr

    def event(
        self,
        event: str,
        *,
        level: str = "info",
        node: str | None = None,
        run_id: str | None = None,
        **fields: Any,
    ) -> dict[str, Any] | None:
        normalized = level.lower()
        if normalized not in _LOG_LEVELS:
            raise ValueError(f"unknown log level {level!r}")
        if _LOG_LEVELS[normalized] < _LOG_LEVELS[self.level]:
            return None
        record: dict[str, Any] = {
            "ts": utc_now(),
            "level": normalized,
            "event": event,
            "run_id": run_id if run_id is not None else self.run_id,
        }
        if node is not None:
            record["node"] = node
        record.update(redact_fields(fields))
        details = " ".join(
            f"{key}={_console_value(value)}"
            for key, value in record.items()
            if key not in {"ts", "level", "event", "run_id", "node"}
        )
        target = f" run={record['run_id']}" if record["run_id"] else ""
        node_text = f" node={record['node']}" if "node" in record else ""
        line = f"{record['ts']} {normalized.upper():<7} {event}{target}{node_text}"
        if details:
            line += f" {details}"
        self.stream.write(line + "\n")
        self.stream.flush()
        return record

    def info(self, event: str, **fields: Any) -> dict[str, Any] | None:
        return self.event(event, level="info", **fields)

    def warning(self, event: str, **fields: Any) -> dict[str, Any] | None:
        return self.event(event, level="warning", **fields)

    def error(self, event: str, **fields: Any) -> dict[str, Any] | None:
        return self.event(event, level="error", **fields)

    def close(self) -> None:
        return None


def _console_value(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return f'"{text}"' if " " in text else text


def build_logger_from_env(
    env: Mapping[str, str] | None = None,
    *,
    stream: TextIO | None = None,
    path: str | Path | None = None,
) -> StructuredLogger | ConsoleLogger | NullLogger:
    """Build a logger from environment variables.

    ``MG_LOG_FORMAT`` is ``off`` (default), ``json``, or ``console``. JSON mode writes to
    ``MG_LOG_FILE`` when set, otherwise stderr. ``MG_LOG_LEVEL`` (default ``info``) and
    ``MG_LOG_RUN_ID`` are optional.
    """
    values = os.environ if env is None else env
    fmt = values.get("MG_LOG_FORMAT", "off").strip().lower()
    run_id = values.get("MG_LOG_RUN_ID") or None
    level = values.get("MG_LOG_LEVEL", "info").strip().lower()
    destination = path if path is not None else values.get("MG_LOG_FILE") or None
    if fmt in {"", "off", "none", "0", "false"}:
        return NullLogger()
    if level not in _LOG_LEVELS:
        raise ObservabilityError(
            f"unknown MG_LOG_LEVEL {level!r}; expected debug, info, warning, error, or critical"
        )
    if fmt == "json":
        try:
            return StructuredLogger(run_id=run_id, level=level, path=destination, stream=stream)
        except OSError as exc:
            raise ObservabilityError(
                f"cannot open MG_LOG_FILE {str(destination)!r}: {exc}",
                details={"path": str(destination)},
            ) from exc
    if fmt == "console":
        return ConsoleLogger(run_id=run_id, level=level, stream=stream)
    raise ObservabilityError(f"unknown MG_LOG_FORMAT {fmt!r}; expected off, json, or console")


# --------------------------------------------------------------------------------------
# Trace-derived metrics
# --------------------------------------------------------------------------------------

_BUDGET_TYPES: dict[str, str] = {
    "MG-4001": "step",
    "MG-4002": "token",
    "MG-4003": "cost",
    "MG-4004": "wall_clock",
    "MG-4006": "depth",
}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def read_trace_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield stored trace events, skipping blank or corrupt lines.

    A recorded trace is written as a whole file by :class:`~modelgraph.runtime.trace.
    TraceRecorder`; skipping a corrupt line keeps ``mg stats`` usable instead of hiding
    every other run behind one damaged file.
    """
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield cast(dict[str, Any], event)


@dataclass
class LatencySeries:
    """Accumulate latency samples for one dimension."""

    values: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self.values.append(value)

    def summary(self) -> dict[str, Any]:
        return latency_summary(self.values)


def _sorted_counts(values: Mapping[str, int]) -> dict[str, int]:
    return {key: values[key] for key in sorted(values)}


def _sorted_numbers(values: Mapping[str, float]) -> dict[str, float]:
    return {key: round(values[key], 8) for key in sorted(values)}


@dataclass
class RunMetrics:
    """Metrics projected from one run's trace events."""

    run_id: str
    graph_id: str | None = None
    graph_hash: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    outcome: str | None = None
    elapsed_seconds: float | None = None
    started: bool = False
    terminated: bool = False
    node_calls: Counter[str] = field(default_factory=Counter)
    provider_requests: int = 0
    retries: int = 0
    retries_by_node: Counter[str] = field(default_factory=Counter)
    protocol_errors: Counter[str] = field(default_factory=Counter)
    provider_errors: Counter[str] = field(default_factory=Counter)
    failures: Counter[str] = field(default_factory=Counter)
    budget_exhaustion: Counter[str] = field(default_factory=Counter)
    tokens_input: int = 0
    tokens_output: int = 0
    charged_tokens: int = 0
    charged_cost_usd: float = 0.0
    final_used_cost_usd: float | None = None
    charged_events: int = 0
    reported_input_tokens: int = 0
    reported_output_tokens: int = 0
    reported_total_tokens: int = 0
    reported_cost_usd: float = 0.0
    tokens_by_node: Counter[str] = field(default_factory=Counter)
    tokens_by_model: Counter[str] = field(default_factory=Counter)
    cost_by_node: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    #: Hit ids already counted. The runtime emits both ``provider.response`` (with
    #: ``cached: true``) and ``provider.response.cached`` for one hit; the shared
    #: ``cache_hit_id`` keeps the projection from counting the pair twice.
    cache_hit_events: set[str] = field(default_factory=set)
    node_latency: dict[str, LatencySeries] = field(default_factory=dict)
    provider_latency: dict[str, LatencySeries] = field(default_factory=dict)
    node_latency_is_explicit: set[str] = field(default_factory=set)
    node_models: dict[str, str] = field(default_factory=dict)
    node_providers: dict[str, str] = field(default_factory=dict)

    def effective_input_tokens(self) -> int:
        return self.tokens_input or self.reported_input_tokens

    def effective_output_tokens(self) -> int:
        return self.tokens_output or self.reported_output_tokens

    def effective_total_tokens(self) -> int:
        if self.charged_tokens:
            return self.charged_tokens
        direct = self.tokens_input + self.tokens_output
        if direct:
            return direct
        return self.reported_total_tokens

    def effective_cost_usd(self) -> float:
        if self.final_used_cost_usd is not None:
            return self.final_used_cost_usd
        if self.charged_events:
            return self.charged_cost_usd
        return self.reported_cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "graph_hash": self.graph_hash,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": self.elapsed_seconds,
            "tokens": {
                "input_tokens": self.effective_input_tokens(),
                "output_tokens": self.effective_output_tokens(),
                "total_tokens": self.effective_total_tokens(),
            },
            "cost_usd": round(self.effective_cost_usd(), 8),
            "node_calls": _sorted_counts(self.node_calls),
            "provider_requests": self.provider_requests,
            "retries": self.retries,
            "retries_by_node": _sorted_counts(self.retries_by_node),
            "protocol_errors": _sorted_counts(self.protocol_errors),
            "provider_errors": _sorted_counts(self.provider_errors),
            "failures_by_code": _sorted_counts(self.failures),
            "budget_exhaustion": _sorted_counts(self.budget_exhaustion),
            "tokens_by_node": _sorted_counts(self.tokens_by_node),
            "tokens_by_model": _sorted_counts(self.tokens_by_model),
            "cost_by_node": _sorted_numbers(self.cost_by_node),
            "cost_by_model": _sorted_numbers(self.cost_by_model),
            "latency_by_node": {
                key: self.node_latency[key].summary() for key in sorted(self.node_latency)
            },
            "latency_by_provider": {
                key: self.provider_latency[key].summary() for key in sorted(self.provider_latency)
            },
            "cache": {"hits": self.cache_hits, "misses": self.cache_misses},
        }


@dataclass
class MetricsSnapshot:
    """Aggregated trace metrics for one or more runs."""

    runs_started: int
    runs_finished: int
    outcomes: dict[str, int]
    failures_by_code: dict[str, int]
    node_calls: dict[str, int]
    provider_requests: int
    retries: int
    retries_by_node: dict[str, int]
    protocol_errors: dict[str, int]
    provider_errors: dict[str, int]
    budget_exhaustion: dict[str, int]
    tokens_by_node: dict[str, int]
    tokens_by_model: dict[str, int]
    cost_by_node: dict[str, float]
    cost_by_model: dict[str, float]
    latency_by_node: dict[str, dict[str, Any]]
    latency_by_provider: dict[str, dict[str, Any]]
    latency: dict[str, Any]
    cache: dict[str, int]
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs_started": self.runs_started,
            "runs_finished": self.runs_finished,
            "outcomes": dict(self.outcomes),
            "failures_by_code": dict(self.failures_by_code),
            "node_calls": dict(self.node_calls),
            "provider_requests": self.provider_requests,
            "retries": self.retries,
            "retries_by_node": dict(self.retries_by_node),
            "protocol_errors": dict(self.protocol_errors),
            "provider_errors": dict(self.provider_errors),
            "budget_exhaustion": dict(self.budget_exhaustion),
            "tokens_by_node": dict(self.tokens_by_node),
            "tokens_by_model": dict(self.tokens_by_model),
            "cost_by_node": dict(self.cost_by_node),
            "cost_by_model": dict(self.cost_by_model),
            "latency_by_node": dict(self.latency_by_node),
            "latency_by_provider": dict(self.latency_by_provider),
            "latency": dict(self.latency),
            "cache": dict(self.cache),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }


class MetricsAccumulator:
    """In-process projection accumulator over raw trace events.

    The accumulator observes the stored event dictionaries directly. It is not a second
    event model: the only contract is the trace event schema in ``docs/execution.md`` §9.
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunMetrics] = {}
        self._order: list[str] = []

    def _run(self, run_id: str) -> RunMetrics:
        existing = self._runs.get(run_id)
        if existing is not None:
            return existing
        created = RunMetrics(run_id=run_id)
        self._runs[run_id] = created
        self._order.append(run_id)
        return created

    def runs(self) -> list[RunMetrics]:
        return [self._runs[run_id] for run_id in self._order]

    def observe(self, event: Mapping[str, Any]) -> None:
        name = _text(event.get("event"))
        if name is None:
            return
        run = self._run(_text(event.get("run_id")) or "unknown")
        if name == "run.started":
            self._observe_run_started(run, event)
        elif name == "run.finished":
            self._observe_run_finished(run, event)
        elif name == "run.failed":
            self._observe_run_failed(run, event)
        elif name == "node.entered":
            node = _text(event.get("node"))
            if node is not None:
                run.node_calls[node] += 1
        elif name == "node.exited":
            node = _text(event.get("node"))
            if node is not None and node not in run.node_latency_is_explicit:
                duration = _number(event.get("duration_seconds"))
                if duration is None:
                    duration = _number(event.get("duration"))
                if duration is not None:
                    run.node_latency.setdefault(node, LatencySeries()).values.clear()
                    run.node_latency[node].observe(duration)
                    run.node_latency_is_explicit.add(node)
        elif name == "provider.request":
            self._observe_provider_request(run, event)
        elif name in {"provider.response", "provider.response.cached"}:
            self._observe_provider_response(run, event, name)
        elif name == "provider.error":
            self._observe_failure(run, event, kind="provider")
        elif name in {"node.failed", "protocol.error"}:
            self._observe_failure(run, event, kind="protocol")
        elif name == "budget.charged":
            self._observe_budget_charged(run, event)
        elif name in {"budget.exhausted", "budget.exhaustion"}:
            self._observe_budget_exhaustion(run, event)
        elif name in {"cache.hit", "provider.cache.hit"}:
            run.cache_hits += 1
        elif name in {"cache.miss", "provider.cache.miss"}:
            run.cache_misses += 1

    def _observe_run_started(self, run: RunMetrics, event: Mapping[str, Any]) -> None:
        run.started = True
        run.started_at = _text(event.get("ts")) or run.started_at
        graph = _mapping(event.get("graph"))
        if graph is not None:
            run.graph_id = _text(graph.get("id")) or run.graph_id
            run.graph_hash = _text(graph.get("hash")) or run.graph_hash
        run.graph_id = _text(event.get("graph_id")) or run.graph_id
        run.graph_hash = _text(event.get("graph_hash")) or run.graph_hash
        models = _mapping(event.get("models"))
        if models is not None:
            for node, record in models.items():
                node_id = str(node)
                info = _mapping(record)
                if info is None:
                    continue
                model = _text(info.get("model")) or _text(info.get("ref"))
                if model is not None:
                    run.node_models.setdefault(node_id, model)

    def _observe_run_finished(self, run: RunMetrics, event: Mapping[str, Any]) -> None:
        outcome = _text(event.get("outcome")) or "succeeded"
        self._terminate(run, outcome, _text(event.get("ts")))
        usage = _mapping(event.get("usage"))
        if usage is not None:
            run.reported_input_tokens = _integer(usage.get("input_tokens")) or 0
            run.reported_output_tokens = _integer(usage.get("output_tokens")) or 0
            run.reported_total_tokens = _integer(usage.get("total_tokens")) or 0
            run.reported_cost_usd = _number(usage.get("cost_usd")) or 0.0
            elapsed = _number(usage.get("elapsed_seconds"))
            if elapsed is not None:
                run.elapsed_seconds = elapsed

    def _observe_run_failed(self, run: RunMetrics, event: Mapping[str, Any]) -> None:
        outcome = _text(event.get("outcome")) or "failed"
        self._terminate(run, outcome, _text(event.get("ts")))
        self._observe_failure(run, event, kind="auto")

    def _terminate(self, run: RunMetrics, outcome: str, timestamp: str | None) -> None:
        if run.terminated:
            return
        run.terminated = True
        run.outcome = outcome
        run.finished_at = timestamp
        if run.elapsed_seconds is None:
            run.elapsed_seconds = _elapsed_between(run.started_at, run.finished_at)

    def _observe_provider_request(self, run: RunMetrics, event: Mapping[str, Any]) -> None:
        run.provider_requests += 1
        node = _text(event.get("node"))
        model = _text(event.get("model"))
        provider = _text(event.get("provider"))
        if node is not None and model is not None:
            run.node_models[node] = model
        if node is not None and provider is not None:
            run.node_providers[node] = provider
        attempt = _integer(event.get("attempt"))
        if attempt is not None and attempt > 1:
            run.retries += 1
            if node is not None:
                run.retries_by_node[node] += 1

    def _observe_provider_response(
        self, run: RunMetrics, event: Mapping[str, Any], name: str
    ) -> None:
        node = _text(event.get("node"))
        provider = (
            _text(event.get("provider"))
            or (run.node_providers.get(node or "") if node is not None else None)
            or "unknown"
        )
        model = _text(event.get("model"))
        if node is not None and model is not None:
            run.node_models[node] = model
        cached = event.get("cached")
        hit = name == "provider.response.cached" or cached is True
        if hit:
            hit_id = _text(event.get("cache_hit_id"))
            if hit_id is not None and hit_id in run.cache_hit_events:
                return
            if hit_id is not None:
                run.cache_hit_events.add(hit_id)
            run.cache_hits += 1
            # A cache hit is not provider latency and its usage was already accounted
            # for when the response was produced; do not count either again.
            return
        latency = _number(event.get("latency_seconds"))
        if latency is not None:
            provider_series = run.provider_latency.setdefault(provider, LatencySeries())
            provider_series.observe(latency)
            if node is not None and node not in run.node_latency_is_explicit:
                run.node_latency.setdefault(node, LatencySeries()).observe(latency)
        if cached is False:
            run.cache_misses += 1
        usage = _mapping(event.get("usage"))
        if usage is not None:
            run.tokens_input += _integer(usage.get("input_tokens")) or 0
            run.tokens_output += _integer(usage.get("output_tokens")) or 0

    def _observe_budget_charged(self, run: RunMetrics, event: Mapping[str, Any]) -> None:
        node = _text(event.get("node"))
        tokens = _integer(event.get("tokens")) or 0
        cost = _number(event.get("cost_usd")) or 0.0
        run.charged_events += 1
        run.charged_tokens += tokens
        run.charged_cost_usd += cost
        used_cost = _number(event.get("used_cost_usd"))
        if used_cost is not None:
            run.final_used_cost_usd = used_cost
        key = node or "unknown"
        run.tokens_by_node[key] += tokens
        run.cost_by_node[key] = run.cost_by_node.get(key, 0.0) + cost
        model = run.node_models.get(node or "") or "unknown"
        run.tokens_by_model[model] += tokens
        run.cost_by_model[model] = run.cost_by_model.get(model, 0.0) + cost

    def _observe_budget_exhaustion(self, run: RunMetrics, event: Mapping[str, Any]) -> None:
        code = _text(event.get("code"))
        kind = _text(event.get("budget")) or _text(event.get("type"))
        if kind is None and code is not None:
            kind = _BUDGET_TYPES.get(code, "other")
        run.budget_exhaustion[kind or "unknown"] += 1

    def _observe_failure(self, run: RunMetrics, event: Mapping[str, Any], *, kind: str) -> None:
        code = _text(event.get("code")) or "unknown"
        run.failures[code] += 1
        if kind == "provider" or code.startswith("MG-3"):
            run.provider_errors[code] += 1
        elif kind == "protocol" or code.startswith(("MG-21", "MG-22")):
            run.protocol_errors[code] += 1
        if code.startswith("MG-4"):
            run.budget_exhaustion[_BUDGET_TYPES.get(code, "other")] += 1

    def snapshot(self) -> MetricsSnapshot:
        outcomes: Counter[str] = Counter()
        failures: Counter[str] = Counter()
        node_calls: Counter[str] = Counter()
        retries_by_node: Counter[str] = Counter()
        protocol_errors: Counter[str] = Counter()
        provider_errors: Counter[str] = Counter()
        budget_exhaustion: Counter[str] = Counter()
        tokens_by_node: Counter[str] = Counter()
        tokens_by_model: Counter[str] = Counter()
        cost_by_node: dict[str, float] = {}
        cost_by_model: dict[str, float] = {}
        latency_by_node: dict[str, LatencySeries] = {}
        latency_by_provider: dict[str, LatencySeries] = {}
        run_latencies: list[float] = []
        cache_hits = 0
        cache_misses = 0
        provider_requests = 0
        retries = 0
        total_input = 0
        total_output = 0
        total_tokens = 0
        total_cost = 0.0
        runs_started = 0
        runs_finished = 0

        for run_id in self._order:
            run = self._runs[run_id]
            if run.started:
                runs_started += 1
            if run.terminated:
                runs_finished += 1
                outcomes[run.outcome or "unknown"] += 1
            failures.update(run.failures)
            node_calls.update(run.node_calls)
            retries_by_node.update(run.retries_by_node)
            protocol_errors.update(run.protocol_errors)
            provider_errors.update(run.provider_errors)
            budget_exhaustion.update(run.budget_exhaustion)
            tokens_by_node.update(run.tokens_by_node)
            tokens_by_model.update(run.tokens_by_model)
            for key, value in run.cost_by_node.items():
                cost_by_node[key] = cost_by_node.get(key, 0.0) + value
            for key, value in run.cost_by_model.items():
                cost_by_model[key] = cost_by_model.get(key, 0.0) + value
            for key, series in run.node_latency.items():
                latency_by_node.setdefault(key, LatencySeries()).values.extend(series.values)
            for key, series in run.provider_latency.items():
                latency_by_provider.setdefault(key, LatencySeries()).values.extend(series.values)
            if run.elapsed_seconds is not None:
                run_latencies.append(run.elapsed_seconds)
            cache_hits += run.cache_hits
            cache_misses += run.cache_misses
            provider_requests += run.provider_requests
            retries += run.retries
            total_input += run.effective_input_tokens()
            total_output += run.effective_output_tokens()
            total_tokens += run.effective_total_tokens()
            total_cost += run.effective_cost_usd()

        return MetricsSnapshot(
            runs_started=runs_started,
            runs_finished=runs_finished,
            outcomes=_sorted_counts(outcomes),
            failures_by_code=_sorted_counts(failures),
            node_calls=_sorted_counts(node_calls),
            provider_requests=provider_requests,
            retries=retries,
            retries_by_node=_sorted_counts(retries_by_node),
            protocol_errors=_sorted_counts(protocol_errors),
            provider_errors=_sorted_counts(provider_errors),
            budget_exhaustion=_sorted_counts(budget_exhaustion),
            tokens_by_node=_sorted_counts(tokens_by_node),
            tokens_by_model=_sorted_counts(tokens_by_model),
            cost_by_node=_sorted_numbers(cost_by_node),
            cost_by_model=_sorted_numbers(cost_by_model),
            latency_by_node={
                key: latency_by_node[key].summary() for key in sorted(latency_by_node)
            },
            latency_by_provider={
                key: latency_by_provider[key].summary() for key in sorted(latency_by_provider)
            },
            latency=latency_summary(run_latencies),
            cache={"hits": cache_hits, "misses": cache_misses},
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 8),
        )


def _elapsed_between(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        started = datetime.fromisoformat(start.replace("Z", "+00:00"))
        finished = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=UTC)
    seconds = (finished - started).total_seconds()
    return round(seconds, 3) if seconds >= 0 else None


def project_events(events: Iterable[Mapping[str, Any]]) -> MetricsSnapshot:
    """Project a stream of stored trace events into aggregate metrics."""
    accumulator = MetricsAccumulator()
    for event in events:
        accumulator.observe(event)
    return accumulator.snapshot()


def project_trace(path: str | Path) -> MetricsSnapshot:
    """Project one stored ``trace.jsonl`` file into metrics, without in-memory state."""
    return project_events(read_trace_events(path))


def project_run(path: str | Path) -> RunMetrics:
    """Project one stored trace into per-run metrics."""
    accumulator = MetricsAccumulator()
    for event in read_trace_events(path):
        accumulator.observe(event)
    runs = accumulator.runs()
    return runs[0] if runs else RunMetrics(run_id=Path(path).parent.name)


# --------------------------------------------------------------------------------------
# Local run summaries (`mg stats`)
# --------------------------------------------------------------------------------------


def discover_run_dirs(runs_directory: str | Path) -> list[Path]:
    """Return run directories with a trace, newest first, without touching providers."""
    root = Path(runs_directory)
    if not root.is_dir():
        return []
    discovered: list[tuple[float, str, Path]] = []
    try:
        candidates = list(root.iterdir())
    except OSError:
        return []
    for candidate in candidates:
        trace = candidate / "trace.jsonl"
        if not candidate.is_dir() or not trace.is_file():
            continue
        try:
            modified = trace.stat().st_mtime
        except OSError:
            continue
        discovered.append((modified, candidate.name, candidate))
    return [path for _, _, path in sorted(discovered, reverse=True)]


@dataclass
class RunStats:
    """One run's trace-derived summary plus its location."""

    run_dir: str
    trace: str
    metrics: RunMetrics

    def to_dict(self) -> dict[str, Any]:
        return {"run_dir": self.run_dir, "trace": self.trace, **self.metrics.to_dict()}


@dataclass
class StatsReport:
    """Aggregate and per-run summary returned by :func:`collect_stats`."""

    runs_dir: str
    available_runs: int
    selected_runs: int
    last: int | None
    metrics: MetricsSnapshot
    runs: list[RunStats]

    @property
    def success_rate(self) -> float | None:
        if self.metrics.runs_finished <= 0:
            return None
        return self.metrics.outcomes.get("succeeded", 0) / self.metrics.runs_finished

    @property
    def top_failure_codes(self) -> list[dict[str, Any]]:
        ordered = sorted(
            self.metrics.failures_by_code.items(), key=lambda item: (-item[1], item[0])
        )
        return [{"code": code, "count": count} for code, count in ordered[:5]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs_dir": self.runs_dir,
            "available_runs": self.available_runs,
            "selected_runs": self.selected_runs,
            "last": self.last,
            "success_rate": self.success_rate,
            "outcomes": dict(self.metrics.outcomes),
            "total_input_tokens": self.metrics.total_input_tokens,
            "total_output_tokens": self.metrics.total_output_tokens,
            "total_tokens": self.metrics.total_tokens,
            "total_cost_usd": self.metrics.total_cost_usd,
            "latency": dict(self.metrics.latency),
            "top_failure_codes": self.top_failure_codes,
            "failures_by_code": dict(self.metrics.failures_by_code),
            "cache": dict(self.metrics.cache),
            "budget_exhaustion": dict(self.metrics.budget_exhaustion),
            "metrics": self.metrics.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
        }


def collect_stats(runs_directory: str | Path, *, last: int | None = None) -> StatsReport:
    """Summarize the most recent recorded runs in ``runs_directory``.

    Only ``trace.jsonl`` files are read; no provider, model registry, or network access is
    involved. ``last`` limits the sample to that many newest traces.
    """
    if last is not None and last < 1:
        raise ValueError("last must be at least 1")
    discovered = discover_run_dirs(runs_directory)
    selected = discovered[:last] if last is not None else discovered
    accumulator = MetricsAccumulator()
    runs: list[RunStats] = []
    for run_dir in selected:
        trace = run_dir / "trace.jsonl"
        events = list(read_trace_events(trace))
        local = MetricsAccumulator()
        for event in events:
            local.observe(event)
            accumulator.observe(event)
        local_runs = local.runs()
        metrics = local_runs[0] if local_runs else RunMetrics(run_id=run_dir.name)
        runs.append(RunStats(run_dir=str(run_dir), trace=str(trace), metrics=metrics))
    return StatsReport(
        runs_dir=str(runs_directory),
        available_runs=len(discovered),
        selected_runs=len(selected),
        last=last,
        metrics=accumulator.snapshot(),
        runs=runs,
    )


# --------------------------------------------------------------------------------------
# Optional OpenTelemetry-compatible spans
# --------------------------------------------------------------------------------------

RUN_SPAN_NAME = "modelgraph.run"
NODE_SPAN_NAME = "modelgraph.node"

# Only identifiers and timings are exported. Content-shaped attributes are dropped even if
# a caller passes them, so the default exporter cannot leak prompts or artifacts.
SPAN_ATTRIBUTE_KEYS = frozenset(
    {
        "run_id",
        "parent_run_id",
        "graph_id",
        "graph_hash",
        "node",
        "node_kind",
        "kind",
        "depth",
        "step",
        "provider",
        "model",
        "model_hash",
        "model_ref",
        "profile",
        "target",
        "deployment_revision",
        "seed",
        "outcome",
        "error_code",
        "child_run_id",
    }
)


def safe_span_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only allowlisted, primitive span attributes; drop prompts/artifacts/secrets."""
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        name = str(key)
        if name not in SPAN_ATTRIBUTE_KEYS or is_sensitive_key(name):
            continue
        if isinstance(value, bool | int | float | str):
            safe[name] = value
    return safe


def opentelemetry_available() -> bool:
    """Whether the optional ``opentelemetry-api`` package can be imported."""
    try:
        return importlib.util.find_spec("opentelemetry") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


class SpanExporter(Protocol):
    """Minimal span surface; the OTel adapter and the in-memory recorder both satisfy it."""

    def span(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> AbstractContextManager[Any]: ...


class _NullSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class NullSpanExporter:
    """The default exporter: spans are accepted and dropped."""

    def span(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> AbstractContextManager[Any]:
        return _NullSpan()


@dataclass
class SpanRecord:
    """A span captured by :class:`RecordingSpanExporter`."""

    name: str
    attributes: dict[str, Any]
    started_at: str
    started_monotonic: float
    finished_at: str | None = None
    duration_seconds: float | None = None
    status: str = "running"
    error_code: str | None = None


class _RecordingSpan:
    def __init__(self, record: SpanRecord) -> None:
        self.record = record

    def __enter__(self) -> SpanRecord:
        return self.record

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.record.finished_at = utc_now()
        self.record.duration_seconds = time.monotonic() - self.record.started_monotonic
        if exc is not None:
            self.record.status = "error"
            details = getattr(exc, "details", None)
            if isinstance(details, Mapping):
                code = details.get("code")
                if code is not None:
                    self.record.error_code = str(code)
        else:
            self.record.status = "ok"


class RecordingSpanExporter:
    """In-memory exporter used by tests and offline diagnostics; no network."""

    def __init__(self) -> None:
        self.spans: list[SpanRecord] = []

    def span(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> AbstractContextManager[SpanRecord]:
        record = SpanRecord(
            name=name,
            attributes=safe_span_attributes(attributes or {}),
            started_at=utc_now(),
            started_monotonic=time.monotonic(),
        )
        self.spans.append(record)
        return _RecordingSpan(record)


class OpenTelemetrySpanExporter:
    """Adapter over an OpenTelemetry-shaped tracer.

    The adapter only needs ``start_as_current_span(name, attributes=...)``. It does not
    install an SDK, configure an exporter, or start a collector: the host application owns
    the global tracer provider (normally ``opentelemetry-sdk`` plus an OTLP exporter).
    """

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def span(
        self, name: str, attributes: Mapping[str, Any] | None = None
    ) -> AbstractContextManager[Any]:
        return cast(
            AbstractContextManager[Any],
            self._tracer.start_as_current_span(
                name, attributes=safe_span_attributes(attributes or {})
            ),
        )

    @classmethod
    def from_environment(cls, *, service_name: str = "modelgraph") -> OpenTelemetrySpanExporter:
        """Build the adapter from the installed ``opentelemetry-api`` package.

        Raises :class:`~modelgraph.errors.ModelGraphError` when the package is absent. With
        only the API installed and no SDK, OpenTelemetry itself provides a no-op tracer;
        this method does not pretend a collector was configured.
        """
        try:
            module = importlib.import_module("opentelemetry.trace")
        except ImportError as exc:
            raise ObservabilityError(
                "OpenTelemetry is not installed; install `opentelemetry-api` (and an SDK "
                "exporter) or set MG_OTEL=off"
            ) from exc
        tracer_factory = cast(Any, module).get_tracer
        return cls(tracer_factory(service_name))


def build_span_exporter(env: Mapping[str, str] | None = None) -> SpanExporter:
    """Build the configured span exporter.

    ``MG_OTEL`` is ``off`` by default. ``on``/``1``/``true``/``api`` builds the
    OpenTelemetry adapter from the globally configured tracer and raises if the optional
    package is missing. ``MG_OTEL_SERVICE_NAME`` (or ``OTEL_SERVICE_NAME``) sets the tracer
    service name. A collector endpoint is configured with the standard OpenTelemetry
    environment variables of the installed SDK, not by this module.
    """
    values = os.environ if env is None else env
    mode = values.get("MG_OTEL", values.get("MG_OTEL_ENABLED", "off")).strip().lower()
    if mode in {"", "off", "none", "0", "false", "no"}:
        return NullSpanExporter()
    if mode in {"on", "1", "true", "yes", "api", "otlp", "console"}:
        service_name = (
            values.get("MG_OTEL_SERVICE_NAME") or values.get("OTEL_SERVICE_NAME") or "modelgraph"
        )
        return OpenTelemetrySpanExporter.from_environment(service_name=service_name)
    raise ObservabilityError(f"unknown MG_OTEL {mode!r}; expected off, on, api, otlp, or console")


def run_span_attributes(
    run_id: str,
    *,
    graph_id: str | None = None,
    graph_hash: str | None = None,
    parent_run_id: str | None = None,
    profile: str | None = None,
    target: str | None = None,
    deployment_revision: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Identifiers only; no prompt, task, artifact, or response content."""
    return safe_span_attributes(
        {
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "graph_id": graph_id,
            "graph_hash": graph_hash,
            "profile": profile,
            "target": target,
            "deployment_revision": deployment_revision,
            "seed": seed,
        }
    )


def node_span_attributes(
    run_id: str,
    node: str,
    *,
    kind: str | None = None,
    depth: int | None = None,
    step: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    model_hash: str | None = None,
    model_ref: str | None = None,
) -> dict[str, Any]:
    """Identifiers and timings context for one node; no prompt or artifact content."""
    return safe_span_attributes(
        {
            "run_id": run_id,
            "node": node,
            "kind": kind,
            "depth": depth,
            "step": step,
            "provider": provider,
            "model": model,
            "model_hash": model_hash,
            "model_ref": model_ref,
        }
    )


@contextmanager
def run_span(exporter: SpanExporter, run_id: str, **attributes: Any) -> Iterator[Any]:
    """Start a run span. Attributes are filtered to identifiers; content is dropped."""
    payload = safe_span_attributes({"run_id": run_id, **attributes})
    with exporter.span(RUN_SPAN_NAME, payload) as span:
        yield span


@contextmanager
def node_span(exporter: SpanExporter, run_id: str, node: str, **attributes: Any) -> Iterator[Any]:
    """Start a node span; the exporter nests it under the current run span if supported."""
    payload = safe_span_attributes({"run_id": run_id, "node": node, **attributes})
    with exporter.span(NODE_SPAN_NAME, payload) as span:
        yield span
