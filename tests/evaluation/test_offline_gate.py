from pathlib import Path

import pytest

from modelgraph.evaluation import EvalError, run_node_suite
from modelgraph.evaluation.providers import live_allowed, require_live
from modelgraph.evaluation.support import commands_for

ROOT = Path(__file__).resolve().parents[2]
JS_SUITE = ROOT / "evals" / "mock" / "js-specialist"
JS_GRAPH = ROOT / "examples" / "coding-specialist.graph.yaml"


def test_live_provider_is_gated_before_any_network(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MG_EVAL_LIVE", raising=False)
    with pytest.raises(EvalError, match="gated"):
        run_node_suite(JS_SUITE, provider="ollama", reports_dir=tmp_path, persist=False)


def test_live_gate_can_be_opened_explicitly(monkeypatch) -> None:
    monkeypatch.delenv("MG_EVAL_LIVE", raising=False)
    assert not live_allowed()
    require_live("ollama", allow_live=True)
    monkeypatch.setenv("MG_EVAL_LIVE", "1")
    assert live_allowed()
    require_live("ollama", allow_live=False)


def test_mock_provider_never_requires_the_live_gate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MG_EVAL_LIVE", raising=False)
    report = run_node_suite(JS_SUITE, reports_dir=tmp_path, limit=1)
    assert report.provider["kind"] == "mock"
    assert report.metrics["task_success"] == 1.0


def test_live_reproduction_command_records_the_gate(tmp_path: Path) -> None:
    live_command, commands, reproduce = commands_for(
        level="node",
        suite_path=JS_SUITE / "suite.yaml",
        graph_path=JS_GRAPH,
        provider="ollama",
        reports_dir=tmp_path,
    )
    assert "--live" in live_command
    assert "--reports-dir" in live_command
    assert "allow_live=True" in reproduce
    assert commands == [live_command, reproduce]

    mock_command, _, mock_reproduce = commands_for(
        level="node",
        suite_path=JS_SUITE / "suite.yaml",
        graph_path=JS_GRAPH,
        provider="mock",
        reports_dir=tmp_path,
    )
    assert "--live" not in mock_command
    assert "allow_live" not in mock_reproduce
