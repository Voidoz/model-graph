import json
from pathlib import Path

from typer.testing import CliRunner

from modelgraph.cli import app

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.6.0"


def test_validate_json() -> None:
    result = runner.invoke(
        app,
        ["validate", str(EXAMPLES / "hello-triage.graph.yaml"), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["graphs"][0]["graph"]["id"] == "hello-triage"
    assert payload["graphs"][0]["warnings"]


def test_run_mock_and_replay(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES / "hello-triage.graph.yaml"),
            "--provider",
            "mock",
            "--runs-dir",
            str(tmp_path),
            "--no-cache",
            "--seed",
            "17",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["outcome"] == "succeeded"
    assert payload["artifacts"][0]["content"]
    run_dir = Path(payload["run_dir"])
    assert (run_dir / "run.json").exists()
    assert (run_dir / "trace.jsonl").exists()
    assert json.loads((run_dir / "run.json").read_text())["seed"] == 17

    replay = runner.invoke(app, ["replay", str(run_dir), "--json"])
    assert replay.exit_code == 0, replay.stdout + replay.stderr
    assert json.loads(replay.stdout)["ok"] is True


def test_trace_show_json(tmp_path: Path) -> None:
    run = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES / "hello-triage.graph.yaml"),
            "--provider",
            "mock",
            "--runs-dir",
            str(tmp_path),
            "--no-cache",
            "--json",
        ],
    )
    run_dir = Path(json.loads(run.stdout)["run_dir"])
    trace = runner.invoke(app, ["trace", "show", str(run_dir), "--json"])
    assert trace.exit_code == 0
    events = json.loads(trace.stdout)
    assert events[0]["event"] == "run.started"
    assert any(event["event"] == "decision.made" for event in events)


def test_models_list() -> None:
    result = runner.invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert {row["id"] for row in rows} >= {"router-small", "coder-7b"}


def test_validation_failure_exit_code(tmp_path: Path) -> None:
    bad = tmp_path / "bad.graph.yaml"
    bad.write_text("apiVersion: modelgraph/v0\nkind: Graph\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
    assert "INVALID" in result.stdout


def test_budget_override_fails(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES / "hello-triage.graph.yaml"),
            "--provider",
            "mock",
            "--runs-dir",
            str(tmp_path),
            "--no-cache",
            "--max-steps",
            "1",
            "--json",
        ],
    )
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["outcome"] == "failed"
    assert payload["error"]["code"] == "MG-4001"


def test_compile_json() -> None:
    result = runner.invoke(
        app,
        ["compile", str(EXAMPLES / "composed-assistant.graph.yaml"), "--json"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["id"] == "composed-assistant"
    assert payload["nodes"]
    assert payload["subgraphs"]
