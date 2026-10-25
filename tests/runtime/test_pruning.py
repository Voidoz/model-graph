"""Profile loading, capability filtering, pruning, and lazy-loading tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from modelgraph.cli import app
from modelgraph.compiler import compile_graph
from modelgraph.compiler.ir import CompiledGraph
from modelgraph.contracts.graph import Defaults
from modelgraph.contracts.loader import load_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.providers.base import (
    Capabilities,
    ChatRequest,
    ChatResponse,
    Health,
    Usage,
)
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.pruning import (
    LazyModelLoader,
    Profile,
    ProfileError,
    PruneError,
    apply_budget_overrides,
    load_profile,
    prune_graph,
    resolve_profile_path,
)
from modelgraph.registry import Registry
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.engine import run_graph
from modelgraph.runtime.envelope import RuntimeState
from modelgraph.runtime.renderer import render_prompt

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
PROFILES = EXAMPLES / "profiles"
runner = CliRunner()

CODING = json.dumps(
    {"decision": "route", "to": "coding", "confidence": 0.95, "reason": "code request"}
)
JS = json.dumps(
    {"decision": "route", "to": "js-specialist", "confidence": 0.95, "reason": "JS request"}
)


def _profile(name: str = "test", **values: Any) -> Profile:
    return Profile.model_validate({"name": name, **values})


def _manifest_dict(
    model_id: str,
    *,
    provider: str = "ollama",
    vram: float | None = 2,
    ram: float | None = 4,
    disk: float | None = 1,
    license_name: str = "apache-2.0",
    network: str | None = "loopback",
    cpu_only: bool | None = True,
    pricing: str = "free",
    input_price: float = 0.0,
    output_price: float = 0.0,
    per_request: float = 0.0,
) -> dict[str, Any]:
    resources: dict[str, Any] = {
        "vram_gb": vram,
        "ram_gb": ram,
        "disk_gb": disk,
        "cpu_only": cpu_only,
    }
    if network is not None:
        resources["network_access"] = network
    return {
        "apiVersion": "modelgraph/v0",
        "kind": "Model",
        "metadata": {
            "id": model_id,
            "name": f"Model {model_id}",
            "version": "1.0.0",
            "description": f"Test manifest for {model_id}.",
        },
        "spec": {
            "provider": provider,
            "model": f"{model_id}:latest",
            "context_window": 8192,
            "limits": {"default_max_output_tokens": 128},
            "cost": {
                "pricing": pricing,
                "input_per_million_usd": input_price,
                "output_per_million_usd": output_price,
                "per_request_usd": per_request,
            },
            "resources": resources,
            "license": {"name": license_name},
        },
    }


def _write_registry(tmp_path: Path, manifests: list[dict[str, Any]]) -> Registry:
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    for manifest in manifests:
        path = model_dir / f"{manifest['metadata']['id']}.yaml"
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return Registry.from_paths([model_dir])


def _write_graph(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / f"{data['metadata']['id']}.graph.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _base_graph(
    *,
    graph_id: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    exits: list[str],
    fallback: str | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "entry": nodes[0]["id"],
        "inputs": [{"name": "task", "type": "text"}],
        "exit": exits,
        "defaults": {"max_steps": 10, "max_depth": 2},
        "nodes": nodes,
        "edges": edges,
    }
    if fallback is not None:
        spec["fallback"] = fallback
    return {
        "apiVersion": "modelgraph/v0",
        "kind": "Graph",
        "metadata": {
            "id": graph_id,
            "name": graph_id,
            "version": "1.0.0",
            "description": f"Test graph {graph_id}.",
        },
        "spec": spec,
    }


def _compile(
    tmp_path: Path, data: dict[str, Any], manifests: list[dict[str, Any]]
) -> CompiledGraph:
    graph_path = _write_graph(tmp_path, data)
    result = compile_graph(
        graph_path, registry=_write_registry(tmp_path, manifests), strict_models=True
    )
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def _worker(model: str, artifact: str = "answer") -> dict[str, Any]:
    return {
        "id": model,
        "kind": "worker",
        "description": f"Worker {model}.",
        "model": model,
        "reads": ["task"],
        "instruction": "Answer the request.",
        "writes": [{"name": artifact, "type": "text"}],
    }


def _router(model: str = "gate") -> dict[str, Any]:
    return {
        "id": "gate",
        "kind": "router",
        "description": "Choose a worker.",
        "model": model,
        "reads": ["task"],
    }


def _edge(source: str, target: str) -> dict[str, Any]:
    return {"from": source, "to": target, "when": f"Route to {target}."}


def _example_graph(name: str = "composed-assistant.graph.yaml") -> CompiledGraph:
    result = compile_graph(EXAMPLES / name, strict_models=True)
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def _fail_scenario() -> MockScenario:
    return MockScenario(
        responses={
            "domain-triage": [CODING],
            "coding-triage": [JS],
            "js-specialist": ["const js = true;\n"],
        }
    )


class _FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.closed = False

    def capabilities(self) -> Capabilities:
        return Capabilities(max_context=1024)

    def health(self) -> Health:
        return Health(ok=True)

    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(text="ok", usage=Usage(1, 1))

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Profile loading and validation
# ---------------------------------------------------------------------------


def test_load_example_profile_by_path_and_name() -> None:
    by_path = load_profile(PROFILES / "local-4gb.yaml")
    by_name = load_profile("local-4gb")
    assert by_path == by_name
    assert by_path.name == "local-4gb"
    assert by_path.network == "loopback"
    assert by_path.max_vram_gb == 4
    assert by_path.providers == ["ollama", "llamacpp", "mock"]
    assert by_path.budget_overrides == {"max_steps": 8}
    assert resolve_profile_path("local-8gb").name == "local-8gb.yaml"


def test_load_json_profile(tmp_path: Path) -> None:
    path = tmp_path / "tiny.json"
    path.write_text(
        json.dumps(
            {
                "name": "tiny",
                "network": "none",
                "max_vram_gb": 2,
                "providers": ["mock"],
                "licenses": [],
                "max_cost_usd": 0,
            }
        ),
        encoding="utf-8",
    )
    profile = load_profile(path)
    assert profile.network == "none"
    assert profile.to_dict()["name"] == "tiny"


def test_profile_rejects_unknown_fields_and_bad_overrides(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\ntypo_field: 1\n", encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile(path)

    bad_override = tmp_path / "bad-override.yaml"
    bad_override.write_text("name: bad\nbudget_overrides:\n  max_steps: 0\n", encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile(bad_override)

    unknown_key = tmp_path / "unknown-key.yaml"
    unknown_key.write_text("name: bad\nbudget_overrides:\n  max_widgets: 3\n", encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile(unknown_key)


def test_missing_profile_is_a_clear_error() -> None:
    with pytest.raises(ProfileError):
        load_profile("does-not-exist")


# ---------------------------------------------------------------------------
# Capability filtering
# ---------------------------------------------------------------------------


def test_license_network_provider_resource_and_cost_exclusions(tmp_path: Path) -> None:
    manifests = [
        _manifest_dict("gate"),
        _manifest_dict("good"),
        _manifest_dict("remote", provider="openai-compatible"),
        _manifest_dict("web", network="internet"),
        _manifest_dict("gpl", license_name="gpl-3.0"),
        _manifest_dict("too-big", vram=12),
        _manifest_dict("too-rammy", ram=64),
        _manifest_dict("too-disky", disk=100),
        _manifest_dict("paid", pricing="per_token", input_price=1.0, output_price=2.0),
        _manifest_dict("unknown-cost", pricing="unknown"),
    ]
    nodes = [_router(), *(_worker(item["metadata"]["id"]) for item in manifests[1:])]
    graph = _compile(
        tmp_path,
        _base_graph(
            graph_id="capability-filter",
            nodes=nodes,
            edges=[_edge("gate", node["id"]) for node in nodes[1:]],
            exits=["answer"],
        ),
        manifests,
    )
    profile = _profile(
        "strict",
        network="loopback",
        max_vram_gb=8,
        max_ram_gb=16,
        max_disk_gb=20,
        providers=["ollama"],
        licenses=["apache-2.0"],
        max_cost_usd=0.0,
    )
    result = prune_graph(graph, profile)
    removed = {item.id: item.reason for item in result.report.removed_nodes}

    assert set(removed) == {
        "remote",
        "web",
        "gpl",
        "too-big",
        "too-rammy",
        "too-disky",
        "paid",
        "unknown-cost",
    }
    assert "provider 'openai-compatible'" in removed["remote"]
    assert "internet" in removed["web"]
    assert "gpl-3.0" in removed["gpl"]
    assert "12GB VRAM" in removed["too-big"]
    assert "64GB RAM" in removed["too-rammy"]
    assert "100GB disk" in removed["too-disky"]
    assert "minimum call cost" in removed["paid"]
    assert "pricing" in removed["unknown-cost"]
    assert set(result.graph.nodes) == {"gate", "good"}
    # Every removed edge has a reason and names the pruned target.
    assert len(result.report.removed_edges) == len(removed)
    assert all(edge.reason for edge in result.report.removed_edges)
    assert {edge.to for edge in result.report.removed_edges} == set(removed)


def test_network_missing_fact_is_excluded_under_a_ceiling(tmp_path: Path) -> None:
    manifests = [_manifest_dict("gate"), _manifest_dict("mystery", network=None)]
    graph = _compile(
        tmp_path,
        _base_graph(
            graph_id="network-unknown",
            nodes=[_router(), _worker("mystery")],
            edges=[_edge("gate", "mystery")],
            exits=["answer"],
        ),
        manifests,
    )
    with pytest.raises(PruneError) as caught:
        prune_graph(graph, _profile("loopback-only", network="loopback"))
    reason = next(item.reason for item in caught.value.report.removed_nodes if item.id == "mystery")
    assert "network_access" in reason


def test_internet_ceiling_allows_an_undeclared_network_fact(tmp_path: Path) -> None:
    manifests = [_manifest_dict("gate"), _manifest_dict("mystery", network=None)]
    graph = _compile(
        tmp_path,
        _base_graph(
            graph_id="internet-unknown",
            nodes=[_router(), _worker("mystery")],
            edges=[_edge("gate", "mystery")],
            exits=["answer"],
        ),
        manifests,
    )
    result = prune_graph(graph, _profile("internet-ok", network="internet"))
    assert "mystery" in result.graph.nodes


def test_unknown_cost_can_be_explicitly_allowed(tmp_path: Path) -> None:
    manifests = [_manifest_dict("gate"), _manifest_dict("mystery", pricing="unknown")]
    graph = _compile(
        tmp_path,
        _base_graph(
            graph_id="unknown-cost",
            nodes=[_router(), _worker("mystery")],
            edges=[_edge("gate", "mystery")],
            exits=["answer"],
        ),
        manifests,
    )
    with pytest.raises(PruneError):
        prune_graph(graph, _profile("strict", max_cost_usd=0))
    result = prune_graph(graph, _profile("lenient", max_cost_usd=0, allow_unknown_cost=True))
    assert "mystery" in result.graph.nodes


def test_gpu_disabled_excludes_gpu_only_models(tmp_path: Path) -> None:
    manifests = [
        _manifest_dict("gate"),
        _manifest_dict("cpu", cpu_only=True),
        _manifest_dict("gpu", cpu_only=False),
    ]
    graph = _compile(
        tmp_path,
        _base_graph(
            graph_id="gpu-filter",
            nodes=[_router(), _worker("cpu"), _worker("gpu")],
            edges=[_edge("gate", "cpu"), _edge("gate", "gpu")],
            exits=["answer"],
        ),
        manifests,
    )
    result = prune_graph(graph, _profile("cpu-only", gpu=False))
    assert "gpu" not in result.graph.nodes
    assert "cpu" in result.graph.nodes


# ---------------------------------------------------------------------------
# Pruning the real examples at 2/4/8GB
# ---------------------------------------------------------------------------


def test_2gb_profile_cannot_run_composed_example() -> None:
    with pytest.raises(PruneError) as caught:
        prune_graph(_example_graph(), _profile("tiny", max_vram_gb=2))
    error = caught.value
    assert error.code == "MG-1005"
    report = error.report.to_dict()
    assert report["removed_exit_artifacts"]
    assert any(item["id"] == "coding" for item in report["removed_nodes"])
    assert any(item["id"] == "story" for item in report["removed_nodes"])


def test_4gb_profile_prunes_6gb_coder_and_keeps_js_branch(tmp_path: Path) -> None:
    graph = _example_graph()
    result = prune_graph(graph, load_profile("local-4gb"))
    pruned = result.graph
    removed = {item.id for item in result.report.removed_nodes}

    assert {"generic-coder", "story"} <= removed
    assert "coder-7b" in next(
        item.reason for item in result.report.removed_nodes if item.id == "generic-coder"
    )
    assert set(pruned.nodes) == {"domain-triage", "coding"}
    inner = pruned.subgraphs["coding-specialist@1.0.0"]
    assert set(inner.nodes) == {"coding-triage", "js-specialist"}
    assert [(edge.from_, edge.to) for edge in inner.edges] == [("coding-triage", "js-specialist")]
    assert sorted(pruned.exits) == ["answer"]

    scenario = _fail_scenario()
    outcome = run_graph(
        pruned,
        "Write a TypeScript function.",
        lambda model: MockProvider(scenario, max_context=model.spec.context_window),
        runs_dir=tmp_path,
    )
    assert outcome.ok, outcome.error
    assert outcome.artifacts[0].content == "const js = true;\n"
    assert "generic-coder" not in [entry.node for entry in outcome.path]


def test_8gb_profile_keeps_every_example_node() -> None:
    graph = _example_graph()
    result = prune_graph(graph, load_profile("local-8gb"))
    assert set(result.graph.nodes) == set(graph.nodes)
    inner = result.graph.subgraphs["coding-specialist@1.0.0"]
    assert len(inner.nodes) == len(graph.subgraphs["coding-specialist@1.0.0"].nodes)
    assert inner.edges == graph.subgraphs["coding-specialist@1.0.0"].edges
    assert result.report.removed_nodes == []
    assert result.report.removed_edges == []


def test_router_menu_never_offers_a_pruned_target() -> None:
    graph = _example_graph()
    pruned = prune_graph(graph, load_profile("local-4gb")).graph
    inner = pruned.subgraphs["coding-specialist@1.0.0"]
    state = RuntimeState.create(inner, "Write TypeScript.")
    prompt = render_prompt(inner.node("coding-triage"), inner, state)
    text = prompt.text()
    assert 'id="js-specialist"' in text
    assert "generic-coder" not in text

    outer_prompt = render_prompt(
        pruned.node("domain-triage"), pruned, RuntimeState.create(pruned, "Write code.")
    )
    assert 'id="coding"' in outer_prompt.text()
    assert "story" not in outer_prompt.text()


def test_report_warns_when_router_is_reduced_to_one_route() -> None:
    report = prune_graph(_example_graph(), load_profile("local-4gb")).report
    warnings = "\n".join(report.warnings)
    assert "router 'domain-triage' now has one route" in warnings
    assert "router 'coding-triage' now has one route" in warnings
    assert "branch 'domain-triage' -> 'story' is unreachable" in warnings


def test_fallback_is_cleared_when_its_target_is_pruned() -> None:
    graph = _example_graph()
    pruned = prune_graph(graph, load_profile("local-4gb")).graph
    assert graph.subgraphs["coding-specialist@1.0.0"].fallback == "generic-coder"
    assert pruned.subgraphs["coding-specialist@1.0.0"].fallback is None


# ---------------------------------------------------------------------------
# Entry and exit survival
# ---------------------------------------------------------------------------


def test_removing_the_entry_fails_with_mg1005_and_a_report() -> None:
    with pytest.raises(PruneError) as caught:
        prune_graph(_example_graph("hello-triage.graph.yaml"), load_profile("local-4gb"))
    error = caught.value
    assert error.code == "MG-1005"
    assert "entry" in error.message
    report = error.report.to_dict()
    assert any(item["id"] == "triage" for item in report["removed_nodes"])
    assert report["removed_exit_artifacts"][0]["name"] == "answer"
    assert error.details["code"] == "MG-1005"


def test_removing_every_producer_of_one_exit_fails_with_report(tmp_path: Path) -> None:
    manifests = [
        _manifest_dict("gate"),
        _manifest_dict("small"),
        _manifest_dict("big", vram=12),
    ]
    nodes = [_router(), _worker("small", "answer"), _worker("big", "extra")]
    graph = _compile(
        tmp_path,
        _base_graph(
            graph_id="two-exits",
            nodes=nodes,
            edges=[_edge("gate", "small"), _edge("gate", "big")],
            exits=["answer", "extra"],
        ),
        manifests,
    )
    with pytest.raises(PruneError) as caught:
        prune_graph(graph, _profile("4gb", max_vram_gb=4))
    report = caught.value.report
    assert "gate" not in {item.id for item in report.removed_nodes}
    assert [item.name for item in report.removed_exit_artifacts] == ["extra"]
    assert "big" in report.removed_exit_artifacts[0].reason
    assert caught.value.details["entry_lost"] is False


def test_unsupported_subgraph_removes_its_parent_node(tmp_path: Path) -> None:
    # The composed example's coding subgraph cannot run at 2GB, so the top-level
    # "coding" node is pruned even though it has no model of its own.
    with pytest.raises(PruneError) as caught:
        prune_graph(_example_graph(), _profile("tiny", max_vram_gb=2))
    report = caught.value.report.to_dict()
    coding = next(item for item in report["removed_nodes"] if item["id"] == "coding")
    assert "subgraph" in coding["reason"]


# ---------------------------------------------------------------------------
# Lazy loading and budget overrides
# ---------------------------------------------------------------------------


def test_lazy_loader_loads_on_first_use_and_unloads() -> None:
    manifests = [
        ModelManifest.model_validate(load_document(EXAMPLES / "models" / f"{name}.yaml"))
        for name in ("router-small", "coder-7b")
    ]
    made: list[_FakeProvider] = []

    def factory(manifest: ModelManifest) -> _FakeProvider:
        provider = _FakeProvider()
        made.append(provider)
        return provider

    loader = LazyModelLoader(factory)
    assert loader.loaded_models == []
    assert loader.records == []

    first = loader.provider_for(manifests[0])
    assert loader.loaded_models == ["router-small"]
    assert loader.records[0]["model"] == "router-small"
    assert loader.records[0]["cold_load_ms"] >= 0

    assert loader.provider_for(manifests[0]) is first
    assert len(made) == 1

    loader.provider_for(manifests[1])
    assert loader.loaded_models == ["router-small", "coder-7b"]
    assert loader.unload("router-small") is True
    assert made[0].closed is True
    assert loader.loaded_models == ["coder-7b"]
    assert loader.unload("missing") is False


def test_lazy_loader_lru_hook_evicts_the_oldest_model() -> None:
    manifest = ModelManifest.model_validate(
        load_document(EXAMPLES / "models" / "router-small.yaml")
    )
    other = manifest.model_copy(
        update={"metadata": manifest.metadata.model_copy(update={"id": "router-other"})}
    )
    loader = LazyModelLoader(lambda item: _FakeProvider(), max_loaded=1)
    loader.provider_for(manifest)
    loader.provider_for(other)
    assert loader.loaded_models == ["router-other"]
    assert any(record["status"] == "evicted" for record in loader.records)


def test_pruned_models_are_never_loaded_and_cold_loads_are_recorded(
    tmp_path: Path,
) -> None:
    graph = prune_graph(_example_graph(), load_profile("local-4gb")).graph
    scenario = _fail_scenario()
    loaded: list[str] = []

    def factory(manifest: ModelManifest) -> MockProvider:
        loaded.append(manifest.metadata.id)
        return MockProvider(scenario, max_context=manifest.spec.context_window)

    loader = LazyModelLoader(factory)
    outcome = run_graph(
        graph,
        "Write a TypeScript function.",
        loader.provider_for,
        runs_dir=tmp_path,
        model_loads=loader.records,
    )
    assert outcome.ok
    assert "coder-7b" not in loaded
    assert "story-7b" not in loaded
    assert loaded == ["router-small", "js-specialist-3b"]

    manifest = json.loads((outcome.run_dir / "run.json").read_text())
    assert [item["model"] for item in manifest["model_loads"]] == loaded
    assert all("cold_load_ms" in item for item in manifest["model_loads"])


def test_budget_overrides_tighten_and_raise_only_with_permission() -> None:
    defaults = Defaults(max_steps=6, max_total_tokens=1000)
    budget = Budget.from_defaults(defaults)
    warnings = apply_budget_overrides(
        budget, {"max_steps": 8, "max_total_tokens": 500}, defaults=defaults
    )
    assert budget.max_steps == 6
    assert budget.max_total_tokens == 500
    assert warnings and "max_steps" in warnings[0]

    relaxed = Budget.from_defaults(defaults)
    assert (
        apply_budget_overrides(
            relaxed,
            {"max_steps": 8},
            defaults=defaults,
            allow_raise=True,
        )
        == []
    )
    assert relaxed.max_steps == 8


def test_cli_budget_flags_apply_after_profile_overrides(tmp_path: Path) -> None:
    # local-4gb sets max_steps: 8; an explicit tighter CLI flag must win instead of being
    # silently raised back to the profile value.
    result = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES / "composed-assistant.graph.yaml"),
            "--profile",
            "local-4gb",
            "--provider",
            "mock",
            "--max-steps",
            "4",
            "--runs-dir",
            str(tmp_path),
            "--no-cache",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    manifest = json.loads((Path(payload["run_dir"]) / "run.json").read_text())
    assert manifest["budget"]["max_steps"] == 4
    assert manifest["profile"]["budget_overrides"]["max_steps"] == 8


# ---------------------------------------------------------------------------
# CLI wiring and recorded evidence
# ---------------------------------------------------------------------------


def test_cli_records_profile_and_prune_report_without_editing_yaml(tmp_path: Path) -> None:
    source = EXAMPLES / "composed-assistant.graph.yaml"
    before = source.read_text(encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "run",
            str(source),
            "--profile",
            "local-4gb",
            "--provider",
            "mock",
            "--runs-dir",
            str(tmp_path),
            "--no-cache",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["outcome"] == "succeeded"
    assert payload["profile"]["name"] == "local-4gb"
    assert {item["id"] for item in payload["prune"]["removed_nodes"]} == {
        "generic-coder",
        "story",
    }
    run_dir = Path(payload["run_dir"])
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["profile"]["name"] == "local-4gb"
    assert manifest["prune"] == payload["prune"]
    assert {item["model"] for item in manifest["model_loads"]} == {
        "router-small",
        "js-specialist-3b",
    }
    events = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    started = next(event for event in events if event["event"] == "run.started")
    assert started["profile"]["name"] == "local-4gb"
    applied = next(event for event in events if event["event"] == "prune.applied")
    assert applied["profile"] == "local-4gb"
    assert applied["report"]["removed_nodes"]
    assert source.read_text(encoding="utf-8") == before


def test_cli_prune_failure_prints_report_and_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES / "hello-triage.graph.yaml"),
            "--profile",
            str(PROFILES / "local-4gb.yaml"),
            "--provider",
            "mock",
            "--runs-dir",
            str(tmp_path),
            "--no-cache",
            "--json",
        ],
    )
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["outcome"] == "failed"
    assert payload["error"]["code"] == "MG-1005"
    assert payload["prune"]["removed_exit_artifacts"][0]["name"] == "answer"


def test_cli_run_without_profile_still_works(tmp_path: Path) -> None:
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
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["outcome"] == "succeeded"
    assert "profile" not in payload
    manifest = json.loads((Path(payload["run_dir"]) / "run.json").read_text())
    assert "profile" not in manifest
    assert "prune" not in manifest
