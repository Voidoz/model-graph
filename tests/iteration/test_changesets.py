"""Tests for change sets, impact analysis, and evidence-gated promotion (T-0038)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jsonschema
import pytest
import yaml

from modelgraph.compiler import compile_graph
from modelgraph.evaluation.e2e import run_e2e_suite
from modelgraph.evaluation.node import run_node_suite
from modelgraph.evaluation.routing import run_routing_suite
from modelgraph.registry import Registry
from modelgraph.revision import (
    CHANGE_REQUIRED_EVIDENCE,
    CompatibilityError,
    EvidenceError,
    RevisionError,
    RevisionStore,
    analyse_impact,
    attach_evidence,
    diff_graphs,
    mark_promoted,
    promotion_plan,
    propose_graph_change,
    propose_swap,
    required_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
MODELS = EXAMPLES / "models"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CANDIDATE_MODEL = FIXTURES / "models" / "js-specialist-3b-0.4.0.yaml"
CODING_GRAPH = EXAMPLES / "coding-specialist.graph.yaml"
SCHEMA = json.loads((ROOT / "schemas" / "change-set.schema.json").read_text(encoding="utf-8"))


def _registry() -> Registry:
    return Registry.from_paths([MODELS], [EXAMPLES])


def _compile(path: Path, registry: Registry):
    result = compile_graph(path, registry=registry, strict_models=True, graph_paths=[path.parent])
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def _bootstrap(
    store: RevisionStore, *, target: str = "portable", graph: Path = CODING_GRAPH
) -> tuple[str, object, Registry]:
    registry = _registry()
    compiled = _compile(graph, registry)
    revision, _created = store.create_revision(compiled, graph_source=graph, target=target)
    revision_id = str(revision["revision_id"])
    store.promote(revision_id)
    return revision_id, compiled, registry


def _pool_target_swap(store: RevisionStore, registry: Registry, **kwargs: object) -> dict:
    return propose_swap(
        store,
        node="js-specialist",
        to_model=str(CANDIDATE_MODEL),
        graph_path=CODING_GRAPH,
        registry=registry,
        **kwargs,
    )


def test_swap_creates_candidate_and_leaves_active_unchanged(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "store")
    active, _graph, registry = _bootstrap(store)
    result = _pool_target_swap(store, registry)

    change = result["change"]
    impact = result["impact"]
    assert result["dry_run"] is False
    assert store.active_revision_id() == active
    assert change["base_revision"] == active
    assert change["candidate_revision"] != active
    assert store.revision_exists(change["candidate_revision"])
    assert change["change_class"] == "model_swap"
    assert change["bindings"][0]["node"] == "js-specialist"
    assert change["bindings"][0]["to_ref"] == "js-specialist-3b@0.4.0"
    assert change["candidate_graph"]["source"] == str(
        store.revision_dir(change["candidate_revision"]) / "graph.yaml"
    )
    assert Path(change["candidate_graph"]["source"]).is_file()

    assert impact["compatibility"]["status"] == "compatible"
    assert impact["compatibility"]["blocking"] is False
    assert impact["change_class"] == "model_swap"
    assert impact["eval"]["required"] == ["node", "routing", "e2e"]
    assert impact["vram"]["delta_gb"] == pytest.approx(0.5)
    assert impact["rollback"]["target"] == active

    plans = {item["artifact"]: item for item in impact["invalidations"]}
    assert plans["compiled_plan"]["status"] == "not_applicable"
    assert plans["compiled_plan"]["applicable"] is False
    assert plans["prefix_cache"]["status"] == "not_applicable"
    assert plans["distilled_student"]["status"] == "stale"

    # The change-set contract is a real schema, not just an internal dict.
    jsonschema.validate(change, SCHEMA)
    assert json.loads(store.change_dir(change["change_id"]).joinpath("impact.json").read_text())


def test_candidate_model_swap_keeps_explicit_identity_compatible(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "store")
    _active, _graph, registry = _bootstrap(store)
    result = _pool_target_swap(store, registry)
    checks = result["impact"]["compatibility"]["checks"]
    assert checks[0]["base"]["status"] == "same"
    assert checks[0]["tokenizer"]["status"] == "same"
    assert checks[0]["base"]["source"].endswith("metadata.labels.base_model")


def test_b2_base_tokenizer_incompatible_swap_is_rejected(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "store")
    active, _graph, registry = _bootstrap(store)
    before = len(store.list_changes())
    with pytest.raises(CompatibilityError) as exc:
        propose_swap(
            store,
            node="js-specialist",
            to_model="story-7b",
            graph_path=CODING_GRAPH,
            target="fused",
            registry=registry,
        )
    message = exc.value.message
    assert "tokenizer" in message or "base model" in message
    assert "portable" in message
    assert "migrat" in message
    assert store.active_revision_id() == active
    assert len(store.list_changes()) == before
    assert {item["revision_id"] for item in store.list_revisions()} == {active}


def test_prompt_change_invalidates_compiled_artifacts_only_when_applicable(
    tmp_path: Path,
) -> None:
    store = RevisionStore(tmp_path / "store")
    active, graph, registry = _bootstrap(store)
    edited = tmp_path / "prompt-v2.graph.yaml"
    document = yaml.safe_load(CODING_GRAPH.read_text(encoding="utf-8"))
    for node in document["spec"]["nodes"]:
        if node["id"] == "js-specialist":
            node["instruction"] = "You are a JavaScript specialist. Prefer explicit types."
    edited.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    portable = propose_graph_change(
        store,
        candidate_graph_path=edited,
        base_revision=active,
        target="portable",
        registry=registry,
    )
    impact = portable["impact"]
    assert impact["change_class"] == "prompt"
    plans = {item["artifact"]: item for item in impact["invalidations"]}
    assert plans["compiled_plan"]["status"] == "not_applicable"
    assert plans["prefix_cache"]["status"] == "not_applicable"
    assert plans["distilled_student"]["status"] == "stale"
    assert impact["eval"]["required"] == ["node"]

    fused_store = RevisionStore(tmp_path / "fused-store")
    fused_active, _fused_graph, fused_registry = _bootstrap(fused_store, target="fused")
    fused = propose_graph_change(
        fused_store,
        candidate_graph_path=edited,
        base_revision=fused_active,
        target="fused",
        registry=fused_registry,
    )
    fused_impact = fused["impact"]
    assert fused_impact["change_class"] == "prompt"
    fused_plans = {item["artifact"]: item for item in fused_impact["invalidations"]}
    assert fused_plans["compiled_plan"]["status"] == "invalidated"
    assert fused_plans["prefix_cache"]["status"] == "invalidated"
    assert fused_plans["compiled_plan"]["present"] is False

    # The diff itself reports the prompt edit, not just the derived invalidation.
    diff = diff_graphs(graph, fused["candidate_graph"])
    assert diff["nodes_changed"][0]["fields"].keys() == {"instruction"}


def test_required_evidence_mapping_is_one_place() -> None:
    assert required_evidence("model_swap", routing_affected=True) == [
        "node",
        "routing",
        "e2e",
    ]
    assert required_evidence("model_swap", routing_affected=False) == ["node", "e2e"]
    assert required_evidence("prompt") == ["node"]
    assert required_evidence("base_swap", target="fused") == [
        "node",
        "routing",
        "e2e",
        "safety",
    ]
    assert CHANGE_REQUIRED_EVIDENCE["topology"] == ("e2e",)


def _reports_for_candidate(store: RevisionStore, revision_id: str, destination: Path) -> list[Path]:
    directory = store.revision_dir(revision_id)
    registry = Registry.from_paths([directory / "models"], [directory / "graphs", directory])
    graph_path = directory / "graph.yaml"
    node_report = run_node_suite(
        ROOT / "evals" / "mock" / "js-specialist",
        graph_path=graph_path,
        provider="mock",
        registry=registry,
        reports_dir=destination / "node",
    )
    routing_report = run_routing_suite(
        FIXTURES / "evals" / "routing-coding",
        graph_path=graph_path,
        provider="mock",
        registry=registry,
        reports_dir=destination / "routing",
    )
    e2e_report = run_e2e_suite(
        FIXTURES / "evals" / "e2e-coding",
        graph_path=graph_path,
        provider="mock",
        registry=registry,
        reports_dir=destination / "e2e",
    )
    assert node_report.gate is not None and node_report.gate["passed"] is True
    assert routing_report.gate is not None and routing_report.gate["passed"] is True
    assert e2e_report.gate is not None and e2e_report.gate["passed"] is True
    paths: list[Path] = []
    for report in (node_report, routing_report, e2e_report):
        assert report.report_dir is not None
        paths.append(report.report_dir / "report.json")
    return paths


def test_promotion_requires_complete_evidence_and_switches_atomically(
    tmp_path: Path,
) -> None:
    store = RevisionStore(tmp_path / "store")
    active, _graph, registry = _bootstrap(store)
    result = _pool_target_swap(store, registry)
    change = result["change"]
    candidate = change["candidate_revision"]

    with pytest.raises(EvidenceError) as missing:
        promotion_plan(store, candidate)
    assert "node, routing, e2e" in missing.value.message
    assert store.active_revision_id() == active

    reports = _reports_for_candidate(store, candidate, tmp_path / "reports")
    with pytest.raises(EvidenceError) as partial:
        promotion_plan(store, candidate, evidence_paths=reports[:1])
    assert "routing" in partial.value.message

    revision = store.load_revision(candidate)
    change = store.load_change(change["change_id"])
    _records, coverage = attach_evidence(
        store,
        change,
        candidate_graph_hash=str(revision["graph"]["hash"]),
        candidate_graph_id=str(revision["graph"]["id"]),
        evidence_paths=reports[:1],
    )
    assert coverage["complete"] is False
    assert store.load_change(change["change_id"])["status"] == "evaluated"
    _records, coverage = attach_evidence(
        store,
        change,
        candidate_graph_hash=str(revision["graph"]["hash"]),
        candidate_graph_id=str(revision["graph"]["id"]),
        evidence_paths=reports,
    )
    assert coverage["complete"] is True
    assert store.load_change(change["change_id"])["status"] == "promotable"

    plan = promotion_plan(store, candidate, evidence_paths=reports)
    assert plan["allowed"] is True
    assert plan["coverage"]["complete"] is True
    assert sorted(plan["coverage"]["satisfied"]) == ["e2e", "node", "routing"]
    assert store.active_revision_id() == active

    pointer = store.promote(candidate, evidence=plan["evidence"])
    mark_promoted(store, plan, reason="test promotion")
    assert pointer["revision_id"] == candidate
    assert pointer["previous_revision"] == active
    assert store.active_revision_id() == candidate
    promoted = store.load_change(change["change_id"])
    assert promoted["status"] == "promoted"
    assert promoted["promotion"]["rollback_target"] == active
    assert len(promoted["evidence"]) == 3
    jsonschema.validate(promoted, SCHEMA)

    store.rollback(active)
    assert store.active_revision_id() == active
    assert store.active_pointer()["previous_revision"] == candidate


def test_gc_refuses_open_change_and_active_revision(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "store")
    active, _graph, registry = _bootstrap(store)
    candidate = _pool_target_swap(store, registry)["change"]["candidate_revision"]
    with pytest.raises(RevisionError):
        store.delete_revision(candidate, now=None)
    future = datetime.now(UTC) + timedelta(days=30)
    with pytest.raises(RevisionError):
        store.delete_revision(candidate, now=future)
    report = store.gc(dry_run=True, now=future)
    blocked = {item["revision"]: item["reason"] for item in report["blocked"]}
    assert active in blocked
    assert blocked[active] == "active"
    assert candidate in blocked
    assert "open change" in blocked[candidate]


def test_impact_marks_route_cache_and_vram_admission(tmp_path: Path) -> None:
    from modelgraph.pruning import load_profile

    store = RevisionStore(tmp_path / "store")
    active, graph, registry = _bootstrap(store)
    candidate = _pool_target_swap(store, registry)["candidate_graph"]
    impact = analyse_impact(
        graph,
        candidate,
        target="portable",
        base_revision=active,
        profile=load_profile(ROOT / "profiles" / "local-4gb.yaml"),
        store=store,
    )
    invalidations = {item["artifact"]: item for item in impact["invalidations"]}
    assert invalidations["route_cache"]["status"] == "not_applicable"
    assert invalidations["route_cache"]["applicable"] is False
    assert impact["vram"]["admission"]["status"] == "exceeds"
    assert impact["vram"]["admission"]["max_vram_gb"] == 4
    assert impact["affected"]["routers"][0]["router"] == "coding-triage"
