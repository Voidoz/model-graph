"""Offline checks for the starter library, profiles, and tutorial (T-0027).

These tests are the CI half of the promise in ``docs/tutorial.md``: every starter graph
validates, every profile loads and is referenced by a documented command, and the mock
tutorial path runs end to end. The live model path is intentionally not exercised here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from modelgraph.cli import app
from modelgraph.compiler import compile_graph
from modelgraph.contracts import load_document, validate_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.pruning import load_profile
from modelgraph.registry import Registry, default_registry

ROOT = Path(__file__).resolve().parents[2]
GRAPHS = ROOT / "graphs"
PROFILES = ROOT / "profiles"
MODELS = ROOT / "registry" / "models"
TUTORIAL = ROOT / "docs" / "tutorial.md"
QUICKSTART = ROOT / "docs" / "quickstart.md"
README = ROOT / "README.md"

STARTER_GRAPH_NAMES = {
    "hello-triage",
    "coding-specialist",
    "composed-assistant",
    "writing-studio",
}
STARTER_PROFILE_NAMES = {"local-4gb", "local-8gb", "no-network"}

runner = CliRunner()


def _graph_paths() -> list[Path]:
    return sorted(GRAPHS.glob("*.graph.yaml"))


def _profile_paths() -> list[Path]:
    return sorted(PROFILES.glob("*.yaml"))


def _model_paths() -> list[Path]:
    return sorted(MODELS.glob("*.yaml"))


def _invoke(args: list[str], *, expect: int = 0) -> Any:
    result = runner.invoke(app, args)
    assert result.exit_code == expect, (
        f"expected exit {expect}, got {result.exit_code} for {args}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


def _json(result: Any) -> dict[str, Any]:
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# Library, manifests, and profiles
# ---------------------------------------------------------------------------


def test_starter_library_has_the_documented_entries() -> None:
    assert {path.name.removesuffix(".graph.yaml") for path in _graph_paths()} == (
        STARTER_GRAPH_NAMES
    )
    assert {path.stem for path in _profile_paths()} == STARTER_PROFILE_NAMES
    assert {path.stem for path in _model_paths()} == {
        "coder-7b",
        "js-specialist-3b",
        "router-small",
        "story-7b",
        "writer-7b",
    }


def test_every_starter_graph_validates_and_resolves_all_references() -> None:
    for path in _graph_paths():
        result = compile_graph(path, strict_models=True)
        assert result.ok, (path, result.errors)
        assert result.graph is not None
        assert result.graph.graph_hash.startswith("sha256:")

    result = _invoke(["validate", *(str(path) for path in _graph_paths())])
    assert result.stdout.count("valid:") == len(_graph_paths())
    assert "INVALID" not in result.stdout


def test_every_model_manifest_validates_and_resolves_from_the_default_registry() -> None:
    registry = default_registry()
    for path in _model_paths():
        document = load_document(path)
        assert not validate_document(document, "manifest", path=str(path))
        manifest = ModelManifest.model_validate(document)
        assert manifest.spec.resources.network_access is not None
        assert manifest.spec.license.name
        assert manifest.spec.notes
        assert registry.resolve_model(manifest.metadata.id) is not None

    for name in STARTER_GRAPH_NAMES:
        assert registry.resolve_graph(name) is not None

    # The starter graphs and the registry resolve each other without an extra flag.
    strict = Registry.from_paths([MODELS], [GRAPHS])
    for path in _graph_paths():
        result = compile_graph(path, registry=strict, strict_models=True)
        assert result.ok, (path, result.errors)


def test_every_starter_profile_loads_and_is_used_by_a_documented_command() -> None:
    tutorial = TUTORIAL.read_text(encoding="utf-8")
    for path in _profile_paths():
        profile = load_profile(path)
        assert profile.name == path.stem
        assert f"profiles/{path.name}" in tutorial
        assert f"--profile profiles/{path.name}" in tutorial

    assert load_profile("local-4gb").max_vram_gb == 4
    assert load_profile("local-8gb").max_vram_gb == 8
    assert load_profile("no-network").network == "none"


def test_tutorial_documents_every_graph_model_and_hardware_assumption() -> None:
    tutorial = TUTORIAL.read_text(encoding="utf-8")
    graph_readme = (GRAPHS / "README.md").read_text(encoding="utf-8")

    for path in _graph_paths():
        assert f"graphs/{path.name}" in tutorial
        assert path.name in graph_readme

    for path in _model_paths():
        manifest = ModelManifest.model_validate(load_document(path))
        assert f"ollama pull {manifest.spec.model}" in tutorial
        assert manifest.spec.model in tutorial

    assert "no api key" in tutorial.lower()
    assert "8GB usable VRAM" in tutorial
    assert "CPU-only" in tutorial
    assert "macOS" in tutorial
    assert "tutorial.md" in README.read_text(encoding="utf-8")
    assert "tutorial.md" in QUICKSTART.read_text(encoding="utf-8")


def test_tutorial_local_links_resolve() -> None:
    tutorial = TUTORIAL.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", tutorial):
        if target.startswith(("http://", "https://", "#")):
            continue
        relative = target.split("#", 1)[0]
        if not relative:
            continue
        assert (TUTORIAL.parent / relative).exists(), target


# ---------------------------------------------------------------------------
# The mock tutorial path
# ---------------------------------------------------------------------------


def test_mock_hello_run_trace_artifacts_and_replay(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    task = "Write a debounce function in TypeScript."

    created = _json(
        _invoke(
            [
                "run",
                str(GRAPHS / "hello-triage.graph.yaml"),
                "--provider",
                "mock",
                "--task",
                task,
                "--runs-dir",
                str(runs),
                "--json",
            ]
        )
    )
    assert created["outcome"] == "succeeded"
    assert [entry["node"] for entry in created["path"]] == ["triage", "coding"]
    assert created["path"][0]["decision"] == "route"
    assert created["path"][0]["to"] == "coding"
    assert "debounce" in created["artifacts"][0]["content"]

    run_dir = Path(created["run_dir"])
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["graph"]["id"] == "hello-triage"
    assert manifest["request"]["task"] == task
    assert manifest["usage"]["cost_usd"] == 0.0
    assert manifest["usage"]["total_tokens"] > 0
    assert manifest["path"][0]["reason"]
    assert (run_dir / "envelope.json").exists()

    for artifact in manifest["artifacts"]:
        digest = artifact["hash"].split(":", 1)[-1]
        assert (run_dir / "artifacts" / digest).read_text(encoding="utf-8") == (
            created["artifacts"][0]["content"]
        )

    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["event"] == "run.started"
    decisions = [event for event in events if event["event"] == "decision.made"]
    assert decisions and decisions[0]["to"] == "coding"
    assert decisions[0]["confidence"] > 0
    assert any(event["event"] == "artifact.written" for event in events)

    listed = json.loads(_invoke(["runs", "list", "--runs-dir", str(runs), "--json"]).stdout)
    assert isinstance(listed, list)
    assert listed[0]["dir"] == str(run_dir)
    shown = _json(_invoke(["runs", "show", str(run_dir)]))
    assert shown["run_id"] == created["run_id"]
    traced = json.loads(_invoke(["trace", "show", str(run_dir), "--json"]).stdout)
    assert isinstance(traced, list)
    assert any(event["event"] == "decision.made" for event in traced)

    replay = _invoke(["replay", str(run_dir)])
    assert "replay: OK" in replay.stdout
    assert (run_dir / "replay.json").exists()


def test_mock_specialist_composition_and_writing_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    specialist = _json(
        _invoke(
            [
                "run",
                str(GRAPHS / "coding-specialist.graph.yaml"),
                "--provider",
                "mock",
                "--task",
                "Add types to this React hook.",
                "--runs-dir",
                str(runs),
                "--json",
            ]
        )
    )
    assert specialist["outcome"] == "succeeded"
    assert [entry["node"] for entry in specialist["path"]] == [
        "coding-triage",
        "js-specialist",
    ]

    composed = _json(
        _invoke(
            [
                "run",
                str(GRAPHS / "composed-assistant.graph.yaml"),
                "--provider",
                "mock",
                "--runs-dir",
                str(runs),
                "--json",
            ]
        )
    )
    assert composed["outcome"] == "succeeded"
    assert [entry["node"] for entry in composed["path"]] == ["domain-triage", "coding"]
    assert composed["path"][1]["graph"] == "coding-specialist@1.0.0"

    writing = _json(
        _invoke(
            [
                "run",
                str(GRAPHS / "writing-studio.graph.yaml"),
                "--provider",
                "mock",
                "--task",
                "Tell me a story about a lighthouse keeper.",
                "--runs-dir",
                str(runs),
                "--json",
            ]
        )
    )
    assert writing["outcome"] == "succeeded"
    assert [entry["node"] for entry in writing["path"]] == ["brief-triage", "draft"]
    assert "lighthouse" in writing["artifacts"][0]["content"].lower()


def test_profile_pruning_is_recorded_and_does_not_edit_the_graph(tmp_path: Path) -> None:
    source = GRAPHS / "composed-assistant.graph.yaml"
    before = source.read_text(encoding="utf-8")
    runs = tmp_path / "runs"

    pruned = _json(
        _invoke(
            [
                "run",
                str(source),
                "--provider",
                "mock",
                "--profile",
                str(PROFILES / "local-4gb.yaml"),
                "--runs-dir",
                str(runs),
                "--json",
            ]
        )
    )
    assert pruned["outcome"] == "succeeded"
    assert pruned["profile"]["name"] == "local-4gb"
    assert {item["id"] for item in pruned["prune"]["removed_nodes"]} == {
        "generic-coder",
        "story",
    }
    assert "coding" in [entry["node"] for entry in pruned["path"]]

    manifest = json.loads((Path(pruned["run_dir"]) / "run.json").read_text(encoding="utf-8"))
    assert manifest["profile"]["name"] == "local-4gb"
    assert manifest["prune"] == pruned["prune"]
    assert [entry["model"] for entry in manifest["model_loads"]] == [
        "router-small",
        "js-specialist-3b",
    ]

    full = _json(
        _invoke(
            [
                "run",
                str(source),
                "--provider",
                "mock",
                "--profile",
                "local-8gb",
                "--runs-dir",
                str(runs),
                "--json",
            ]
        )
    )
    assert full["outcome"] == "succeeded"
    assert full["prune"]["removed_nodes"] == []
    assert full["prune"]["removed_edges"] == []

    assert source.read_text(encoding="utf-8") == before


def test_no_network_profile_is_an_audited_failure(tmp_path: Path) -> None:
    result = _invoke(
        [
            "run",
            str(GRAPHS / "hello-triage.graph.yaml"),
            "--provider",
            "mock",
            "--profile",
            str(PROFILES / "no-network.yaml"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--json",
        ],
        expect=3,
    )
    payload = _json(result)
    assert payload["outcome"] == "failed"
    assert payload["error"]["code"] == "MG-1005"
    assert payload["prune"]["removed_nodes"]
    assert all("network access" in item["reason"] for item in payload["prune"]["removed_nodes"])
    assert payload["prune"]["removed_exit_artifacts"][0]["name"] == "answer"


def test_node_eval_runs_offline_and_writes_a_report(tmp_path: Path) -> None:
    reports = tmp_path / "evals"
    result = _invoke(
        [
            "eval",
            "node",
            "--suite",
            str(ROOT / "evals" / "mock" / "js-specialist"),
            "--graph",
            str(GRAPHS / "coding-specialist.graph.yaml"),
            "--provider",
            "mock",
            "--reports-dir",
            str(reports),
            "--json",
        ]
    )
    payload = _json(result)
    assert payload["level"] == "node"
    assert payload["graph"]["id"] == "coding-specialist"
    assert payload["gate"]["passed"] is True
    assert payload["primary_metric"]["name"] == "task_success"
    assert payload["primary_metric"]["value"] == 1.0
    assert payload["failures"] == []

    report_dir = Path(payload["report_dir"])
    assert (report_dir / "report.json").exists()
    assert (report_dir / "report.md").exists()
    assert (report_dir / "manifest.json").exists()
