"""Trace-to-metric projection: counts, cost/tokens, cache, budget, and failures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from modelgraph.observability import (
    MetricsAccumulator,
    project_events,
    project_trace,
    read_trace_events,
)


def _events(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_metrics_match_trace_counts_and_manifest(mock_run_dir: Path) -> None:
    events = _events(mock_run_dir)
    manifest = json.loads((mock_run_dir / "run.json").read_text(encoding="utf-8"))
    metrics = project_trace(mock_run_dir / "trace.jsonl")

    assert metrics.runs_started == 1
    assert metrics.runs_finished == 1
    assert metrics.outcomes == {"succeeded": 1}
    assert metrics.failures_by_code == {}
    assert metrics.protocol_errors == {}
    assert metrics.provider_errors == {}
    assert metrics.retries == 0
    assert metrics.budget_exhaustion == {}
    assert metrics.cache == {"hits": 0, "misses": 0}

    expected_calls = Counter(event["node"] for event in events if event["event"] == "node.entered")
    assert metrics.node_calls == dict(expected_calls)
    assert metrics.provider_requests == sum(
        1 for event in events if event["event"] == "provider.request"
    )
    assert metrics.latency_by_provider["mock"]["count"] == sum(
        1 for event in events if event["event"] == "provider.response"
    )

    usage = manifest["usage"]
    assert metrics.total_tokens == usage["total_tokens"]
    assert metrics.total_input_tokens == usage["input_tokens"]
    assert metrics.total_output_tokens == usage["output_tokens"]
    assert metrics.total_cost_usd == pytest.approx(usage["cost_usd"])
    assert sum(metrics.tokens_by_node.values()) == usage["total_tokens"]
    assert sum(metrics.tokens_by_model.values()) == usage["total_tokens"]
    assert sum(metrics.cost_by_node.values()) == pytest.approx(usage["cost_usd"])
    assert sum(metrics.cost_by_model.values()) == pytest.approx(usage["cost_usd"])
    assert metrics.latency["count"] == 1


def test_trace_events_are_the_only_input(mock_run_dir: Path) -> None:
    """Deleting the manifest must not change the projection."""
    manifest_path = mock_run_dir / "run.json"
    manifest_path.unlink()
    metrics = project_trace(mock_run_dir / "trace.jsonl")
    assert metrics.runs_finished == 1
    assert metrics.outcomes == {"succeeded": 1}
    assert metrics.total_tokens > 0


def _synthetic_events() -> list[dict[str, Any]]:
    return [
        {
            "ts": "2026-01-01T00:00:00Z",
            "run_id": "run_synthetic",
            "event": "run.started",
            "graph": {"id": "demo", "hash": "sha256:" + "0" * 64},
            "models": {"worker": {"ref": "model-a@1.0.0", "hash": "sha256:" + "1" * 64}},
        },
        {
            "ts": "2026-01-01T00:00:01Z",
            "run_id": "run_synthetic",
            "event": "node.entered",
            "node": "router",
        },
        {
            "ts": "2026-01-01T00:00:01Z",
            "run_id": "run_synthetic",
            "event": "provider.request",
            "node": "router",
            "provider": "mock",
            "model": "router-model",
            "attempt": 1,
        },
        {
            "ts": "2026-01-01T00:00:01Z",
            "run_id": "run_synthetic",
            "event": "provider.error",
            "node": "router",
            "code": "MG-3002",
            "attempt": 1,
        },
        {
            "ts": "2026-01-01T00:00:02Z",
            "run_id": "run_synthetic",
            "event": "provider.request",
            "node": "router",
            "provider": "mock",
            "model": "router-model",
            "attempt": 2,
        },
        {
            "ts": "2026-01-01T00:00:02Z",
            "run_id": "run_synthetic",
            "event": "provider.response",
            "node": "router",
            "provider": "mock",
            "model": "router-model",
            "latency_seconds": 0.4,
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        },
        {
            "ts": "2026-01-01T00:00:02Z",
            "run_id": "run_synthetic",
            "event": "budget.charged",
            "node": "router",
            "tokens": 15,
            "cost_usd": 0.0003,
        },
        {
            "ts": "2026-01-01T00:00:03Z",
            "run_id": "run_synthetic",
            "event": "run.failed",
            "outcome": "failed",
            "code": "MG-4003",
            "message": "cost budget exceeded",
        },
    ]


def test_failure_retry_budget_and_provider_counters() -> None:
    metrics = project_events(_synthetic_events())
    assert metrics.runs_started == 1
    assert metrics.runs_finished == 1
    assert metrics.outcomes == {"failed": 1}
    assert metrics.provider_errors == {"MG-3002": 1}
    assert metrics.retries == 1
    assert metrics.retries_by_node == {"router": 1}
    assert metrics.budget_exhaustion == {"cost": 1}
    assert metrics.failures_by_code == {"MG-3002": 1, "MG-4003": 1}
    assert metrics.protocol_errors == {}
    assert metrics.total_tokens == 15
    assert metrics.total_cost_usd == pytest.approx(0.0003)
    assert metrics.latency_by_node["router"]["p50_seconds"] == pytest.approx(0.4)


def test_budget_exhaustion_by_type_and_protocol_errors() -> None:
    events = [
        {
            "ts": "2026-01-01T00:00:00Z",
            "run_id": "run_step",
            "event": "run.started",
            "graph": {"id": "demo", "hash": "sha256:" + "0" * 64},
        },
        {
            "ts": "2026-01-01T00:00:01Z",
            "run_id": "run_step",
            "event": "run.failed",
            "outcome": "failed",
            "code": "MG-4001",
        },
        {
            "ts": "2026-01-01T00:00:00Z",
            "run_id": "run_token",
            "event": "run.started",
            "graph": {"id": "demo", "hash": "sha256:" + "0" * 64},
        },
        {
            "ts": "2026-01-01T00:00:01Z",
            "run_id": "run_token",
            "event": "run.failed",
            "outcome": "budget_exhausted",
            "code": "MG-4002",
        },
        {
            "ts": "2026-01-01T00:00:00Z",
            "run_id": "run_protocol",
            "event": "run.started",
            "graph": {"id": "demo", "hash": "sha256:" + "0" * 64},
        },
        {
            "ts": "2026-01-01T00:00:01Z",
            "run_id": "run_protocol",
            "event": "run.failed",
            "outcome": "failed",
            "code": "MG-2202",
        },
    ]
    metrics = project_events(events)
    assert metrics.runs_started == 3
    assert metrics.runs_finished == 3
    assert metrics.outcomes == {"budget_exhausted": 1, "failed": 2}
    assert metrics.budget_exhaustion == {"step": 1, "token": 1}
    assert metrics.protocol_errors == {"MG-2202": 1}
    assert metrics.provider_errors == {}


def test_cache_metrics_are_zero_without_explicit_cache_events(mock_run_dir: Path) -> None:
    metrics = project_trace(mock_run_dir / "trace.jsonl")
    assert metrics.cache == {"hits": 0, "misses": 0}


def test_cache_metrics_use_explicit_events_only() -> None:
    events: list[dict[str, Any]] = [
        {"run_id": "run_cache", "event": "run.started"},
        {
            "run_id": "run_cache",
            "event": "provider.response",
            "node": "worker",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        {
            "run_id": "run_cache",
            "event": "provider.response.cached",
            "node": "worker",
            "cached": True,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        {
            "run_id": "run_cache",
            "event": "cache.miss",
            "node": "worker",
        },
        {
            "run_id": "run_cache",
            "event": "provider.response",
            "node": "worker",
            "cached": False,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        {"run_id": "run_cache", "event": "run.finished", "outcome": "succeeded"},
    ]
    metrics = project_events(events)
    assert metrics.cache == {"hits": 1, "misses": 2}


def test_accumulator_and_snapshot_agree() -> None:
    accumulator = MetricsAccumulator()
    for event in _synthetic_events():
        accumulator.observe(event)
    assert accumulator.snapshot().to_dict() == project_events(_synthetic_events()).to_dict()


def test_corrupt_trace_lines_are_skipped(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps({"run_id": "r", "event": "run.started"}),
                "{not json",
                "",
                json.dumps({"run_id": "r", "event": "run.finished", "outcome": "succeeded"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert len(list(read_trace_events(trace))) == 2
    metrics = project_trace(trace)
    assert metrics.outcomes == {"succeeded": 1}
