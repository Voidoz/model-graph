"""Projection reproducibility: same metrics after re-reading and in a fresh process."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from modelgraph.observability import project_events, project_trace

ROOT = Path(__file__).resolve().parents[2]


def test_projection_is_stable_when_read_twice(mock_run_dir: Path) -> None:
    trace = mock_run_dir / "trace.jsonl"
    first = project_trace(trace)
    second = project_trace(trace)
    assert first.to_dict() == second.to_dict()


def test_projection_is_reproducible_in_a_fresh_process(mock_run_dir: Path) -> None:
    trace = mock_run_dir / "trace.jsonl"
    in_process = project_trace(trace).to_dict()
    script = (
        "import json, sys\n"
        "from modelgraph.observability import project_trace\n"
        "print(json.dumps(project_trace(sys.argv[1]).to_dict(), sort_keys=True))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(trace)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    fresh = json.loads(completed.stdout)
    assert fresh == in_process


def test_run_latency_percentiles_from_stored_traces(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    for index, elapsed in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        run_id = f"run_{index}"
        events.extend(
            [
                {
                    "ts": "2026-01-01T00:00:00Z",
                    "run_id": run_id,
                    "event": "run.started",
                    "graph": {"id": "demo", "hash": "sha256:" + "0" * 64},
                },
                {
                    "ts": "2026-01-01T00:00:05Z",
                    "run_id": run_id,
                    "event": "run.finished",
                    "outcome": "succeeded",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                        "cost_usd": 0.001,
                        "elapsed_seconds": elapsed,
                    },
                },
            ]
        )
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    metrics = project_trace(trace)
    assert metrics.latency["count"] == 5
    assert metrics.latency["p50_seconds"] == 3.0
    assert metrics.latency["p95_seconds"] == 4.8
    assert metrics.total_tokens == 75
    assert metrics.total_cost_usd == 0.005


def test_events_without_run_started_still_project(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({"run_id": "r", "event": "node.entered", "node": "triage"}) + "\n",
        encoding="utf-8",
    )
    metrics = project_events([{"run_id": "r", "event": "node.entered", "node": "triage"}])
    assert metrics.node_calls == {"triage": 1}
    assert metrics.runs_started == 0
    assert project_trace(trace).node_calls == {"triage": 1}
