"""Tests for immutable deployment revisions and atomic pointers (T-0038)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from modelgraph.compiler import compile_graph
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.revision import (
    RetentionError,
    RevisionStore,
    RevisionStoreError,
    annotate_run_trace,
    bootstrap_revision,
    promotion_plan,
    status_report,
)
from modelgraph.runtime.engine import run_graph
from modelgraph.runtime.trace import read_events

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
MODELS = EXAMPLES / "models"


def _compile(name: str):
    result = compile_graph(EXAMPLES / name, strict_models=True)
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def _promote_first(store: RevisionStore, graph_name: str = "hello-triage.graph.yaml") -> str:
    graph = _compile(graph_name)
    revision, _created = store.create_revision(
        graph, graph_source=EXAMPLES / graph_name, target="portable"
    )
    revision_id = str(revision["revision_id"])
    store.promote(revision_id)
    return revision_id


def test_revision_is_immutable_and_pointer_switch_is_atomic(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "store")
    graph = _compile("hello-triage.graph.yaml")
    revision, created = store.create_revision(
        graph, graph_source=EXAMPLES / "hello-triage.graph.yaml", target="portable"
    )
    assert created is True
    assert revision["schema"] == "modelgraph/revision/v0"
    assert revision["components"]["plan_hash"] is None
    assert revision["components"]["plan_status"] == "not_applicable"
    assert revision["models"]
    assert revision["graph"]["source"] == str(
        store.revision_dir(str(revision["revision_id"])) / "graph.yaml"
    )
    assert Path(revision["graph"]["source"]).is_file()
    assert revision["graph"]["original_source"] == str(EXAMPLES / "hello-triage.graph.yaml")

    duplicate, created_again = store.create_revision(
        graph, graph_source=EXAMPLES / "hello-triage.graph.yaml", target="portable"
    )
    assert created_again is False
    assert duplicate == revision

    revision_id = str(revision["revision_id"])
    assert (store.revision_dir(revision_id) / "graph.yaml").is_file()
    assert (store.revision_dir(revision_id) / "ir.json").is_file()
    assert (store.revision_dir(revision_id) / "models" / "coder-7b.yaml").is_file()

    pointer = store.promote(revision_id)
    assert pointer["revision_id"] == revision_id
    assert pointer["previous_revision"] is None
    assert store.active_revision_id() == revision_id

    with pytest.raises(RevisionStoreError):
        store.load_revision("rev_missing")

    # A corrupted artifact set is detected rather than silently used.
    graph_path = store.revision_dir(revision_id) / "graph.yaml"
    graph_path.write_text(graph_path.read_text().replace("hello-triage", "other-graph"))
    with pytest.raises(RevisionStoreError):
        store.compile_revision(revision_id)


def test_promote_then_rollback_restores_previous_revision(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "store")
    first = _promote_first(store, "hello-triage.graph.yaml")
    second = _promote_first(store, "coding-specialist.graph.yaml")
    assert first != second
    assert store.active_revision_id() == second
    pointer = store.active_pointer()
    assert pointer is not None
    assert pointer["previous_revision"] == first

    store.rollback(first)
    assert store.active_revision_id() == first
    restored = store.active_pointer()
    assert restored is not None
    assert restored["previous_revision"] == second
    assert (store.revision_dir(first) / "ir.json").is_file()
    assert store.previous_revisions()[0] == second

    # Rollback is a pointer switch, not a rebuild: it works offline against stored artifacts.
    assert store.compile_revision(first).graph_hash == store.load_revision(first)["graph"]["hash"]


def test_delete_respects_grace_period_and_active_pointer(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "store")
    first = _promote_first(store, "hello-triage.graph.yaml")
    second = _promote_first(store, "coding-specialist.graph.yaml")

    with pytest.raises(RetentionError):
        store.delete_revision(second)

    with pytest.raises(RetentionError) as inside_grace:
        store.delete_revision(first)
    assert "grace period" in inside_grace.value.message

    future = datetime.now(UTC) + timedelta(days=30)
    deleted = store.delete_revision(first, now=future)
    assert deleted["deleted"] == first
    assert not store.revision_dir(first).exists()
    assert store.revision_exists(second) is True


def test_gc_reports_reasons_and_refuses_active(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "store")
    first = _promote_first(store, "hello-triage.graph.yaml")
    second = _promote_first(store, "coding-specialist.graph.yaml")
    future = datetime.now(UTC) + timedelta(days=30)
    report = store.gc(dry_run=True, now=future)
    assert first in report["deleted"]
    blocked = {item["revision"]: item["reason"] for item in report["blocked"]}
    assert second in blocked
    assert blocked[second] == "active"


def test_run_manifest_and_trace_identify_deployment_revision(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "store")
    graph = _compile("hello-triage.graph.yaml")
    revision, _created = store.create_revision(
        graph, graph_source=EXAMPLES / "hello-triage.graph.yaml", target="portable"
    )
    decision = '{"decision": "route", "to": "coding", "confidence": 0.95, "reason": "code"}'
    outcome = run_graph(
        graph,
        "Write a debounce function in TypeScript.",
        lambda model: MockProvider(
            MockScenario(responses={"triage": [decision], "coding": ["answer"]}),
            max_context=model.spec.context_window,
        ),
        runs_dir=tmp_path / "runs",
    )
    assert outcome.ok
    assert outcome.manifest is not None
    assert outcome.run_dir is not None
    assert outcome.manifest["deployment_revision"] == revision["revision_id"]

    assert annotate_run_trace(outcome.run_dir, outcome.manifest) is True
    assert annotate_run_trace(outcome.run_dir, outcome.manifest) is False
    events = read_events(outcome.run_dir / "trace.jsonl")
    pinned = [event for event in events if event["event"] == "revision.pinned"]
    assert len(pinned) == 1
    assert pinned[0]["deployment_revision"] == revision["revision_id"]
    assert pinned[0]["target"] == "portable"


def test_status_report_lists_active_and_retained(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "store")
    active = _promote_first(store, "hello-triage.graph.yaml")
    report = status_report(store, include_revisions=True)
    assert report["active_revision"] == active
    assert report["target"] == "portable"
    assert report["revisions"][0]["active"] is True
    assert report["distilled_student_status"] == "not_tracked"


def test_unbuilt_compiled_target_cannot_be_promoted(tmp_path: Path) -> None:
    store = RevisionStore(tmp_path / "store")
    with pytest.raises(RevisionStoreError) as bootstrap_error:
        bootstrap_revision(
            store,
            graph_path=EXAMPLES / "hello-triage.graph.yaml",
            target="fused",
        )
    assert "not implemented" in bootstrap_error.value.message
    assert store.active_revision_id() is None
    assert store.list_revisions() == []

    graph = _compile("hello-triage.graph.yaml")
    revision, _created = store.create_revision(
        graph,
        graph_source=EXAMPLES / "hello-triage.graph.yaml",
        target="fused",
        bootstrap=True,
    )
    assert revision["components"]["plan_status"] == "not_built"
    with pytest.raises(RevisionStoreError) as promote_error:
        promotion_plan(store, str(revision["revision_id"]))
    assert "no built plan" in promote_error.value.message
    assert store.active_revision_id() is None
