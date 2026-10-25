"""`mg stats`: recent-run summaries from traces, with no provider dependency."""

from __future__ import annotations

import json
from pathlib import Path

from support import run_mock
from typer.testing import CliRunner

from modelgraph.cli import app

runner = CliRunner()


def _stats(tmp_path: Path, *args: str):
    return runner.invoke(app, ["stats", "--runs-dir", str(tmp_path), *args])


def test_stats_json_summarizes_recent_runs(mock_runs_dir: Path) -> None:
    result = _stats(mock_runs_dir, "--last", "3", "--json")
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)

    assert payload["available_runs"] == 3
    assert payload["selected_runs"] == 3
    assert payload["last"] == 3
    assert payload["outcomes"] == {"failed": 1, "succeeded": 2}
    assert payload["success_rate"] == 2 / 3
    assert payload["total_tokens"] > 0
    assert payload["total_input_tokens"] > 0
    assert payload["total_output_tokens"] > 0
    assert payload["total_cost_usd"] >= 0.0
    assert payload["latency"]["count"] == 3
    assert payload["latency"]["p50_seconds"] is not None
    assert payload["latency"]["p95_seconds"] is not None
    assert payload["top_failure_codes"] == [{"code": "MG-4001", "count": 1}]
    assert payload["budget_exhaustion"] == {"step": 1}
    assert len(payload["runs"]) == 3
    for run in payload["runs"]:
        assert Path(run["run_dir"]).is_dir()
        assert run["trace"].endswith("trace.jsonl")
        assert "tokens" in run
        assert "cache" in run


def test_stats_last_selects_newest_runs(mock_runs_dir: Path) -> None:
    result = _stats(mock_runs_dir, "--last", "1", "--json")
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["available_runs"] == 3
    assert payload["selected_runs"] == 1
    assert len(payload["runs"]) == 1


def test_stats_human_output(mock_runs_dir: Path) -> None:
    result = _stats(mock_runs_dir, "--last", "3")
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "runs:" in result.stdout
    assert "outcomes:" in result.stdout
    assert "success:" in result.stdout
    assert "tokens:" in result.stdout
    assert "cost:" in result.stdout
    assert "latency:" in result.stdout
    assert "failures:" in result.stdout
    assert "MG-4001=1" in result.stdout


def test_stats_empty_directory(tmp_path: Path) -> None:
    result = _stats(tmp_path)
    assert result.exit_code == 0
    assert "no runs recorded" in result.stdout


def test_stats_rejects_non_positive_last(mock_runs_dir: Path) -> None:
    result = _stats(mock_runs_dir, "--last", "0")
    assert result.exit_code == 2
    assert "at least 1" in result.stderr


def test_stats_never_calls_a_provider(mock_runs_dir: Path, monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("mg stats must not construct or call a provider")

    import modelgraph.providers
    import modelgraph.providers.mock

    monkeypatch.setattr(modelgraph.providers, "build_provider", explode)
    monkeypatch.setattr(modelgraph.providers.mock.MockProvider, "chat", explode)
    monkeypatch.setattr(modelgraph.providers.mock.MockProvider, "__init__", explode)
    monkeypatch.setattr("modelgraph.cli.build_provider", explode)

    result = _stats(mock_runs_dir, "--last", "3", "--json")
    assert result.exit_code == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["selected_runs"] == 3


def test_stats_after_a_fresh_mock_run(tmp_path: Path) -> None:
    run_mock(tmp_path)
    result = _stats(tmp_path, "--last", "5", "--json")
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["success_rate"] == 1.0
    assert payload["top_failure_codes"] == []
    assert payload["cache"] == {"hits": 0, "misses": 0}
