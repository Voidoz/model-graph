"""Runtime wiring: ``mg run``/``run_graph`` must actually emit logs and spans."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from support import CODING_DECISION, TASK, compile_example, factory
from typer.testing import CliRunner

from modelgraph.cli import app
from modelgraph.observability import RecordingSpanExporter, StructuredLogger
from modelgraph.runtime.engine import run_graph

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def _run(tmp_path: Path, **kwargs):
    from modelgraph.providers.mock import MockScenario

    return run_graph(
        compile_example(),
        TASK,
        factory(MockScenario(responses={"triage": [CODING_DECISION]})),
        runs_dir=tmp_path,
        **kwargs,
    )


def test_run_graph_emits_structured_lifecycle_logs(tmp_path: Path) -> None:
    stream = StringIO()
    logger = StructuredLogger(stream=stream)
    outcome = _run(tmp_path, logger=logger)
    assert outcome.ok

    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    events = [record["event"] for record in records]
    assert events[0] == "run.started"
    assert "node.entered" in events
    assert "node.exited" in events
    assert events[-1] == "run.finished"
    assert all(record["run_id"] for record in records)
    assert all("task" not in record for record in records)
    assert all("messages" not in record for record in records)


def test_run_graph_creates_run_and_node_spans_without_content(tmp_path: Path) -> None:
    spans = RecordingSpanExporter()
    outcome = _run(tmp_path, span_exporter=spans)
    assert outcome.ok

    names = [span.name for span in spans.spans]
    assert "modelgraph.run" in names
    assert "modelgraph.node" in names
    for span in spans.spans:
        assert "task" not in span.attributes
        assert "prompt" not in span.attributes
        assert "messages" not in span.attributes
        assert "artifact" not in span.attributes
    node_spans = [span for span in spans.spans if span.name == "modelgraph.node"]
    assert {span.attributes.get("node") for span in node_spans} >= {"triage", "coding"}
    assert all(span.status == "ok" for span in spans.spans)


def test_mg_run_wires_logging_from_environment(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "run.jsonl"
    monkeypatch.setenv("MG_LOG_FORMAT", "json")
    monkeypatch.setenv("MG_LOG_FILE", str(log_path))
    monkeypatch.delenv("MG_OTEL", raising=False)
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(EXAMPLES / "hello-triage.graph.yaml"),
            "--provider",
            "mock",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--no-cache",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["event"] for record in records][0] == "run.started"
    assert records[-1]["event"] == "run.finished"
    assert all(record["run_id"] for record in records)


def test_run_graph_library_defaults_are_null(tmp_path: Path, monkeypatch, capsys) -> None:
    """Direct run_graph callers opt in; env config must not leak into the library path."""

    monkeypatch.setenv("MG_LOG_FORMAT", "json")
    monkeypatch.setenv("MG_OTEL", "on")  # would raise if run_graph built exporters itself
    outcome = _run(tmp_path)
    assert outcome.ok
    captured = capsys.readouterr()
    assert captured.err == ""


def test_invalid_observability_environment_fails_cleanly(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MG_LOG_FORMAT", "not-a-format")
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(EXAMPLES / "hello-triage.graph.yaml"),
            "--provider",
            "mock",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--no-cache",
        ],
    )
    assert result.exit_code == 3
    assert "MG-5000" in result.stderr


def test_invalid_log_destination_fails_cleanly(tmp_path: Path, monkeypatch) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("MG_LOG_FORMAT", "json")
    monkeypatch.setenv("MG_LOG_FILE", str(blocker / "log.jsonl"))
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(EXAMPLES / "hello-triage.graph.yaml"),
            "--provider",
            "mock",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--no-cache",
        ],
    )
    assert result.exit_code == 3
    assert "MG-5000" in result.stderr
    assert "MG_LOG_FILE" in result.stderr
