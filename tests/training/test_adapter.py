"""Trained-router adapter: canonical JSON, provider failures, and graph binding."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from training_support import synthetic_dataset

from modelgraph.compiler import compile_graph
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.evaluation.providers import ProviderPool, provider_summary, require_live
from modelgraph.providers import build_provider
from modelgraph.providers.base import ProviderFailure
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.registry import Registry
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.engine import run_graph
from modelgraph.runtime.envelope import RuntimeState
from modelgraph.runtime.executors import Executor, RouterExecutor
from modelgraph.runtime.trace import TraceRecorder
from modelgraph.training.config import RouterTrainingConfig
from modelgraph.training.model import train_router

ROOT = Path(__file__).resolve().parents[2]
TRIAGE = ROOT / "examples" / "hello-triage.graph.yaml"


def _trained(tmp_path: Path) -> tuple[Any, ModelManifest, Path]:
    config = RouterTrainingConfig(
        dataset="unused",
        artifact=str(tmp_path / "router" / "artifact.json"),
        report=None,
        seed=11,
        features={"dimension": 64},
        training={"epochs": 80, "learning_rate": 0.03},
    )
    result = train_router(synthetic_dataset(), config, write=True)
    assert result.artifact_path is not None
    manifest = ModelManifest.model_validate(
        {
            "apiVersion": "modelgraph/v0",
            "kind": "Model",
            "metadata": {
                "id": "router-trained",
                "name": "Trained router",
                "version": "0.1.0",
                "description": "Synthetic conditional-logit router used by the integration test.",
            },
            "spec": {
                "provider": "trained-router",
                "model": str(result.artifact_path),
                "context_window": 32768,
                "license": {"name": "CC0-1.0"},
                "integrity": {"sha256": result.predictor.artifact.sha256()},
            },
        }
    )
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "router-trained.yaml").write_text(
        yaml.safe_dump(manifest.to_dict()), encoding="utf-8"
    )
    return result, manifest, model_dir


def _trained_graph(tmp_path: Path, model_dir: Path) -> Path:
    document = yaml.safe_load(TRIAGE.read_text(encoding="utf-8"))
    document["metadata"]["id"] = "hello-triage-trained"
    for node in document["spec"]["nodes"]:
        if node["id"] == "triage":
            node["model"] = "router-trained"
            node["routing"]["confidence_threshold"] = 0.0
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    path = graph_dir / "hello-triage-trained.graph.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _compile(tmp_path: Path, model_dir: Path) -> Any:
    graph_path = _trained_graph(tmp_path, model_dir)
    registry = Registry.from_paths([ROOT / "registry/models", model_dir], [graph_path.parent])
    compiled = compile_graph(
        graph_path, registry=registry, strict_models=True, graph_paths=[graph_path.parent]
    )
    assert compiled.ok, compiled.errors
    assert compiled.graph is not None
    return compiled.graph


def test_provider_emits_canonical_json_and_router_executor_accepts_it(tmp_path: Path) -> None:
    result, manifest, model_dir = _trained(tmp_path)
    graph = _compile(tmp_path, model_dir)
    provider = build_provider(manifest)
    state = RuntimeState.create(graph, "Write a Python function that sorts records.")
    executor = Executor(
        provider, budget=Budget.from_defaults(graph.defaults), trace=TraceRecorder("adapter")
    )
    decision = RouterExecutor(executor).run(graph.node("triage"), graph, state)
    assert decision.action == "route"
    assert decision.to == "coding"
    assert 0.0 <= decision.confidence <= 1.0
    raw = json.loads(decision.raw)
    assert raw["decision"] == "route"
    assert raw["to"] == "coding"
    assert raw["reason"]
    assert result.predictor.artifact.sha256() == manifest.spec.integrity.sha256


def test_graph_binding_runs_without_runtime_changes(tmp_path: Path) -> None:
    _, _, model_dir = _trained(tmp_path)
    graph = _compile(tmp_path, model_dir)
    # Only the router uses the trained-router provider; the worker calls must stay offline
    # and deterministic. Building every manifest would otherwise reach the local Ollama
    # endpoint (or fail) and make this a hidden live test.
    scenario = MockScenario(
        responses={
            "coding": ["export const answer = 42;\n"],
            "story": ["The lighthouse keeper waited.\n"],
        }
    )
    seen_providers: list[str] = []

    def factory(manifest):
        if manifest.spec.provider == "trained-router":
            seen_providers.append(manifest.spec.provider)
            return build_provider(manifest)
        return MockProvider(scenario, max_context=manifest.spec.context_window)

    outcome = run_graph(
        graph,
        "Tell a story about a lighthouse keeper.",
        factory,
        runs_dir=tmp_path / "runs",
    )
    assert outcome.outcome == "succeeded", outcome.error
    assert outcome.run_dir is not None
    assert seen_providers == ["trained-router"]
    router_entries = [entry for entry in outcome.path if entry.kind == "router"]
    assert router_entries
    assert router_entries[0].decision == "route"
    assert router_entries[0].to == "story"


def test_missing_artifact_is_a_typed_provider_failure(tmp_path: Path) -> None:
    manifest = ModelManifest.model_validate(
        {
            "apiVersion": "modelgraph/v0",
            "kind": "Model",
            "metadata": {
                "id": "router-missing",
                "name": "Missing router",
                "version": "0.1.0",
                "description": "A manifest pointing at a missing router artifact.",
            },
            "spec": {
                "provider": "trained-router",
                "model": str(tmp_path / "does-not-exist.json"),
                "context_window": 4096,
                "license": {"name": "CC0-1.0"},
            },
        }
    )
    with pytest.raises(ProviderFailure) as exc:
        build_provider(manifest)
    assert exc.value.code == "MG-3005"


def test_hash_mismatch_is_a_typed_provider_failure(tmp_path: Path) -> None:
    result, manifest, _ = _trained(tmp_path)
    document = manifest.to_dict()
    document["spec"]["integrity"] = {"sha256": "0" * 64}
    tampered = ModelManifest.model_validate(document)
    with pytest.raises(ProviderFailure) as exc:
        build_provider(tampered)
    assert exc.value.code == "MG-3006"
    assert result.artifact_path.exists()


def test_unparseable_prompt_is_a_typed_provider_failure(tmp_path: Path) -> None:
    _, manifest, _ = _trained(tmp_path)
    provider = build_provider(manifest)
    from modelgraph.providers.base import ChatRequest, Message

    with pytest.raises(ProviderFailure) as exc:
        provider.chat(ChatRequest(model="trained-router", messages=[Message("user", "hello")]))
    assert exc.value.code == "MG-3006"
    assert provider.health().ok is True
    assert provider.capabilities().json_mode is True


def test_reference_training_binding_compiles() -> None:
    graph_path = ROOT / "training" / "graphs" / "hello-triage-trained.graph.yaml"
    registry = Registry.from_paths(
        [ROOT / "registry/models", ROOT / "training/models"], [graph_path.parent]
    )
    compiled = compile_graph(
        graph_path, registry=registry, strict_models=True, graph_paths=[graph_path.parent]
    )
    assert compiled.ok, compiled.errors
    assert compiled.graph is not None
    assert compiled.graph.node("triage").model is not None
    assert compiled.graph.node("triage").model.spec.provider == "trained-router"
    assert compiled.graph.node("triage").routing.confidence_threshold == 0.0


def test_trained_router_is_an_offline_eval_provider() -> None:
    require_live("trained-router", allow_live=False)
    pool = ProviderPool("trained-router", allow_live=False)
    assert pool.kind == "trained-router"
    summary = provider_summary(
        "trained-router", cast(Any, SimpleNamespace(provider="trained-router"))
    )
    assert summary["deterministic_expected"] is True
    assert summary["gated_live"] is False
