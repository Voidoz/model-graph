"""Run manifest evidence: provisional writes, atomic finalization, recovery, linkage."""

from __future__ import annotations

import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from modelgraph import __version__
from modelgraph.cli import app
from modelgraph.compiler import compile_graph
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.engine import run_graph
from modelgraph.runtime.manifest import (
    MANIFEST_FILENAME,
    STATUS_FINAL,
    STATUS_INTERRUPTED,
    STATUS_PROVISIONAL,
    read_run_manifest,
    recover_interrupted_run,
    recover_interrupted_runs,
    write_manifest_atomic,
)
from modelgraph.runtime.replay import replay_run
from modelgraph.util import canonical_json, sha256_text
from modelgraph.version import PARSER_VERSION, RENDERER_VERSION

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
SCHEMA = json.loads((ROOT / "schemas" / "run-manifest.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)

TASK = "Write a debounce function in TypeScript."
CODING_DECISION = json.dumps(
    {
        "decision": "route",
        "to": "coding",
        "confidence": 0.95,
        "reason": "the request asks for TypeScript code",
    }
)
JS_DECISION = json.dumps(
    {
        "decision": "route",
        "to": "js-specialist",
        "confidence": 0.95,
        "reason": "the request asks for TypeScript code",
    }
)


def _compile(name: str = "hello-triage.graph.yaml"):
    result = compile_graph(EXAMPLES / name, strict_models=True)
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def _factory(scenario: MockScenario):
    def factory(model: ModelManifest) -> MockProvider:
        return MockProvider(scenario, max_context=model.spec.context_window)

    return factory


def _manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def _assert_valid(manifest: dict[str, Any]) -> None:
    errors = sorted(VALIDATOR.iter_errors(manifest), key=lambda error: list(error.path))
    assert not errors, f"{[(list(error.path), error.message) for error in errors]}"


def _successful_run(tmp_path: Path, **kwargs: Any):
    scenario = MockScenario(responses={"triage": [CODING_DECISION]})
    return run_graph(_compile(), TASK, _factory(scenario), runs_dir=tmp_path, **kwargs)


def test_successful_run_writes_schema_valid_manifest_with_identity(tmp_path: Path) -> None:
    graph = _compile()
    outcome = _successful_run(tmp_path, seed=42)
    assert outcome.run_dir is not None
    manifest = _manifest(outcome.run_dir)

    _assert_valid(manifest)
    assert manifest["status"] == STATUS_FINAL
    assert manifest["outcome"] == "succeeded"
    assert manifest["seed"] == 42
    assert manifest["parent_run_id"] is None
    assert manifest["finished_at"]
    assert manifest["started_at"]
    assert manifest["graph"] == {
        "id": graph.id,
        "version": graph.version,
        "hash": graph.graph_hash,
        "source": graph.source,
    }
    assert manifest["renderer_version"] == RENDERER_VERSION
    assert manifest["parser_version"] == PARSER_VERSION
    assert manifest["ir_version"] == "modelgraph/ir/v0"
    assert manifest["profile_hash"] is None
    assert manifest["prune_hash"] is None
    assert manifest["deployment_revision"].startswith("rev_")
    assert manifest["trace"] == "trace.jsonl"
    assert (outcome.run_dir / "trace.jsonl").is_file()

    models = {record["node"]: record for record in manifest["models"]}
    for node in graph.nodes.values():
        if node.model_ref is None or node.model is None:
            continue
        record = models[node.id]
        assert record["ref"] == node.model_ref
        assert record["hash"] == node.model_hash
        assert record["provider"] == node.model.spec.provider.value
        assert record["model"] == node.model.spec.model
        assert record["version"] == node.model.metadata.version
        assert record["resolved_revision"] == node.model.spec.integrity.revision
        if node.model.spec.integrity.revision is None:
            assert record["revision_reason"]
        else:
            assert record["revision_reason"] is None

    assert manifest["environment"]["python_version"]
    assert manifest["environment"]["python_implementation"]
    assert manifest["environment"]["platform"]
    assert manifest["environment"]["pid"] == os.getpid()
    assert manifest["environment"]["hostname"] == socket.gethostname()
    assert manifest["environment"]["providers"] == [{"name": "mock", "version": __version__}]
    assert manifest["usage"]["steps"] == 2
    assert manifest["artifacts"][0]["name"] == "answer"
    assert manifest["artifacts"][0]["exit"] is True
    assert manifest["exit_artifacts"] == [{"name": "answer", "hash": outcome.artifacts[0].hash}]


def test_seed_is_forwarded_to_every_provider_request(tmp_path: Path) -> None:
    scenario = MockScenario(responses={"triage": [CODING_DECISION]})
    providers: list[MockProvider] = []

    def factory(model: ModelManifest) -> MockProvider:
        provider = MockProvider(scenario, max_context=model.spec.context_window)
        providers.append(provider)
        return provider

    outcome = run_graph(_compile(), TASK, factory, runs_dir=tmp_path, seed=7)
    assert outcome.ok
    requests = [request for provider in providers for request in provider.calls]
    assert requests
    assert all(request.seed == 7 for request in requests)


def test_provisional_manifest_is_visible_before_execution_begins(tmp_path: Path) -> None:
    observed: list[dict[str, Any]] = []

    def factory(model: ModelManifest) -> MockProvider:
        paths = sorted(tmp_path.glob(f"*/{MANIFEST_FILENAME}"))
        assert len(paths) == 1
        observed.append(json.loads(paths[0].read_text(encoding="utf-8")))
        return MockProvider(
            MockScenario(responses={"triage": [CODING_DECISION]}),
            max_context=model.spec.context_window,
        )

    outcome = run_graph(_compile(), TASK, factory, runs_dir=tmp_path)
    assert outcome.ok
    assert observed, "the provider factory should run during the run"
    assert observed[0]["status"] == STATUS_PROVISIONAL
    assert observed[0]["outcome"] == "running"
    assert observed[0]["finished_at"] is None
    assert observed[0]["artifacts"] == []
    _assert_valid(observed[0])
    assert _manifest(outcome.run_dir)["status"] == STATUS_FINAL


def test_failed_run_still_writes_a_final_manifest(tmp_path: Path) -> None:
    scenario = MockScenario(failures={"triage": "MG-3006"})
    outcome = run_graph(_compile(), TASK, _factory(scenario), runs_dir=tmp_path)
    assert outcome.outcome == "failed"
    assert outcome.run_dir is not None
    manifest = _manifest(outcome.run_dir)
    _assert_valid(manifest)
    assert manifest["status"] == STATUS_FINAL
    assert manifest["outcome"] == "failed"
    assert manifest["finished_at"]
    assert manifest["error"]["code"] == "MG-3006"
    assert manifest["exit_artifacts"] == []


def test_budget_exhausted_run_still_writes_a_final_manifest(tmp_path: Path) -> None:
    budget = Budget(
        max_steps=1,
        max_depth=3,
        max_total_tokens=32_000,
        max_cost_usd=0.25,
        max_wall_seconds=300,
        on_budget_exhausted="best_effort",
    )
    scenario = MockScenario(responses={"triage": [CODING_DECISION]})
    outcome = run_graph(_compile(), TASK, _factory(scenario), runs_dir=tmp_path, budget=budget)
    assert outcome.outcome == "budget_exhausted"
    manifest = _manifest(outcome.run_dir)
    _assert_valid(manifest)
    assert manifest["status"] == STATUS_FINAL
    assert manifest["outcome"] == "budget_exhausted"
    assert manifest["error"]["code"] == "MG-4001"


def test_writes_are_atomic_and_never_leave_a_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = _successful_run(tmp_path)
    original = _manifest(outcome.run_dir)
    expected = (outcome.run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")

    def fail_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr("modelgraph.runtime.manifest.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        write_manifest_atomic(outcome.run_dir, {**original, "run_id": "run_replacement"})
    assert (outcome.run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8") == expected
    assert list(outcome.run_dir.glob("*.tmp")) == []
    assert list(outcome.run_dir.glob(f".{MANIFEST_FILENAME}.*")) == []

    monkeypatch.undo()
    write_manifest_atomic(outcome.run_dir, {**original, "run_id": "run_replacement"})
    assert _manifest(outcome.run_dir)["run_id"] == "run_replacement"
    assert list(outcome.run_dir.glob("*.tmp")) == []


def test_recovery_marks_a_dead_provisional_run_interrupted(tmp_path: Path) -> None:
    outcome = _successful_run(tmp_path)
    path = outcome.run_dir / MANIFEST_FILENAME
    manifest = _manifest(outcome.run_dir)
    manifest["status"] = STATUS_PROVISIONAL
    manifest["outcome"] = "running"
    manifest["finished_at"] = None
    manifest["environment"] = {
        **manifest["environment"],
        "hostname": socket.gethostname(),
        "pid": 99_999_999,
    }
    write_manifest_atomic(outcome.run_dir, manifest)

    recovered = recover_interrupted_run(outcome.run_dir)
    assert recovered is not None
    assert recovered["status"] == STATUS_INTERRUPTED
    assert recovered["outcome"] == "interrupted"
    assert recovered["finished_at"]
    assert recovered["error"]["code"] == "MG-2001"
    _assert_valid(recovered)
    assert _manifest(outcome.run_dir)["status"] == STATUS_INTERRUPTED
    assert path.is_file()


def test_recovery_leaves_a_live_provisional_run_alone(tmp_path: Path) -> None:
    outcome = _successful_run(tmp_path)
    manifest = _manifest(outcome.run_dir)
    manifest["status"] = STATUS_PROVISIONAL
    manifest["outcome"] = "running"
    manifest["finished_at"] = None
    manifest["environment"] = {
        **manifest["environment"],
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    write_manifest_atomic(outcome.run_dir, manifest)

    assert recover_interrupted_run(outcome.run_dir) is None
    assert _manifest(outcome.run_dir)["status"] == STATUS_PROVISIONAL


def test_recovery_needs_staleness_when_process_identity_is_unknown(tmp_path: Path) -> None:
    outcome = _successful_run(tmp_path)
    manifest = _manifest(outcome.run_dir)
    manifest["status"] = STATUS_PROVISIONAL
    manifest["outcome"] = "running"
    manifest["finished_at"] = None
    manifest["environment"] = {**manifest["environment"], "hostname": None, "pid": None}
    write_manifest_atomic(outcome.run_dir, manifest)

    assert recover_interrupted_run(outcome.run_dir) is None
    recovered = recover_interrupted_run(outcome.run_dir, stale_after_seconds=0)
    assert recovered is not None
    assert recovered["status"] == STATUS_INTERRUPTED


def test_recover_all_scans_top_level_and_child_manifests(tmp_path: Path) -> None:
    outcome = _successful_run(tmp_path)
    parent = _manifest(outcome.run_dir)
    parent["status"] = STATUS_PROVISIONAL
    parent["outcome"] = "running"
    parent["finished_at"] = None
    parent["environment"] = {
        **parent["environment"],
        "hostname": socket.gethostname(),
        "pid": 99_999_999,
    }
    write_manifest_atomic(outcome.run_dir, parent)

    child_dir = outcome.run_dir / "subgraphs" / "run_child_fixture"
    child = json.loads(json.dumps(parent))
    child["run_id"] = "run_child_fixture"
    child["parent_run_id"] = parent["run_id"]
    write_manifest_atomic(child_dir, child)

    recovered = recover_interrupted_runs(tmp_path)
    assert sorted(path.name for path in recovered) == [MANIFEST_FILENAME, MANIFEST_FILENAME]
    assert _manifest(outcome.run_dir)["status"] == STATUS_INTERRUPTED
    assert _manifest(child_dir)["status"] == STATUS_INTERRUPTED


def test_subgraph_child_manifest_records_parent_run_id(tmp_path: Path) -> None:
    graph = _compile("composed-assistant.graph.yaml")
    scenario = MockScenario(
        responses={
            "domain-triage": [CODING_DECISION],
            "coding-triage": [JS_DECISION],
            "js-specialist": ["export const answer = 42;\n"],
        }
    )
    outcome = run_graph(graph, TASK, _factory(scenario), runs_dir=tmp_path)
    assert outcome.ok, outcome.error
    parent = _manifest(outcome.run_dir)
    _assert_valid(parent)
    assert parent["parent_run_id"] is None
    assert len(parent["subgraphs"]) == 1

    record = parent["subgraphs"][0]
    child_dir = outcome.run_dir / record["manifest"]
    assert child_dir.is_file()
    child = json.loads(child_dir.read_text(encoding="utf-8"))
    _assert_valid(child)
    assert child["parent_run_id"] == outcome.run_id
    assert child["run_id"] == record["run_id"]
    assert child["graph"]["hash"] == record["hash"]
    assert child["outcome"] == "succeeded"
    assert child["trace"] == "../../trace.jsonl"
    assert (outcome.run_dir / "trace.jsonl").is_file()


def test_manifest_graph_hash_pins_replay_and_records_the_override(tmp_path: Path) -> None:
    outcome = _successful_run(tmp_path)
    manifest = _manifest(outcome.run_dir)
    manifest["graph"]["hash"] = "sha256:" + "0" * 64
    write_manifest_atomic(outcome.run_dir, manifest)

    refused = replay_run(outcome.run_dir)
    assert not refused.ok
    assert any("graph hash changed" in mismatch for mismatch in refused.mismatches)

    allowed = replay_run(outcome.run_dir, allow_ir_mismatch=True)
    assert any("allowed by flag" in mismatch for mismatch in allowed.mismatches)
    report = json.loads((outcome.run_dir / "replay.json").read_text(encoding="utf-8"))
    assert "graph hash mismatch allowed by flag" in report["mismatches"]


def test_edited_graph_produces_a_different_manifest_graph_hash(tmp_path: Path) -> None:
    source = tmp_path / "hello.graph.yaml"
    shutil.copy(EXAMPLES / "hello-triage.graph.yaml", source)
    first = compile_graph(source, strict_models=True)
    assert first.graph is not None
    first_run = run_graph(
        first.graph,
        TASK,
        _factory(MockScenario(responses={"triage": [CODING_DECISION]})),
        runs_dir=tmp_path / "runs",
    )

    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    document["metadata"]["description"] += " Changed after the first run."
    source.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    second = compile_graph(source, strict_models=True)
    assert second.graph is not None
    second_run = run_graph(
        second.graph,
        TASK,
        _factory(MockScenario(responses={"triage": [CODING_DECISION]})),
        runs_dir=tmp_path / "runs",
    )

    first_manifest = read_run_manifest(first_run.run_dir)
    second_manifest = read_run_manifest(second_run.run_dir)
    assert first_manifest["graph"]["hash"] == first.graph.graph_hash
    assert second_manifest["graph"]["hash"] == second.graph.graph_hash
    assert first_manifest["graph"]["hash"] != second_manifest["graph"]["hash"]
    assert first_manifest["deployment_revision"] != second_manifest["deployment_revision"]


def test_profile_and_prune_hashes_pin_the_effective_configuration(tmp_path: Path) -> None:
    profile = {
        "name": "fixture-profile",
        "providers": ["mock"],
        "budget_overrides": {"max_steps": 8},
    }
    prune = {
        "profile": "fixture-profile",
        "removed_nodes": [],
        "removed_edges": [],
        "removed_exit_artifacts": [],
        "warnings": [],
    }
    scenario = MockScenario(responses={"triage": [CODING_DECISION]})
    outcome = run_graph(
        _compile(),
        TASK,
        _factory(scenario),
        runs_dir=tmp_path,
        profile=profile,
        prune_report=prune,
    )
    manifest = _manifest(outcome.run_dir)
    _assert_valid(manifest)
    assert manifest["profile"] == profile
    assert manifest["prune"] == prune
    assert manifest["profile_hash"] == sha256_text(canonical_json(profile))
    assert manifest["prune_hash"] == sha256_text(canonical_json(prune))


def _seed_dead_provisional(run_dir: Path) -> None:
    """Rewrite a finished run as the provisional file a crashed process would leave."""
    manifest = _manifest(run_dir)
    manifest["status"] = STATUS_PROVISIONAL
    manifest["outcome"] = "running"
    manifest["finished_at"] = None
    manifest["environment"] = {
        **manifest["environment"],
        "hostname": socket.gethostname(),
        "pid": 99_999_999,
    }
    write_manifest_atomic(run_dir, manifest)


def test_cli_runs_show_recovers_a_dead_provisional_manifest(tmp_path: Path) -> None:
    outcome = _successful_run(tmp_path)
    assert outcome.run_dir is not None
    _seed_dead_provisional(outcome.run_dir)

    result = CliRunner().invoke(app, ["runs", "show", str(outcome.run_dir)])
    assert result.exit_code == 0, result.stderr
    shown = json.loads(result.stdout)
    assert shown["status"] == STATUS_INTERRUPTED
    assert shown["outcome"] == "interrupted"
    assert shown["error"]["code"] == "MG-2001"
    _assert_valid(shown)
    assert _manifest(outcome.run_dir)["status"] == STATUS_INTERRUPTED


def test_cli_runs_list_recovers_dead_provisional_manifests(tmp_path: Path) -> None:
    outcome = _successful_run(tmp_path)
    assert outcome.run_dir is not None
    _seed_dead_provisional(outcome.run_dir)

    result = CliRunner().invoke(app, ["runs", "list", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stderr
    assert "interrupted" in result.stdout
    assert _manifest(outcome.run_dir)["status"] == STATUS_INTERRUPTED
