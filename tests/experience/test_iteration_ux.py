"""T-0040: watch loop, impact panels, revision timeline, and side-by-side comparison."""

from __future__ import annotations

import copy
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from modelgraph import inspector
from modelgraph.cli import app
from modelgraph.compiler import compile_graph
from modelgraph.compiler.ir import CompiledGraph
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.evaluation.compare import compare_against_baseline
from modelgraph.evaluation.comparison import compare_reports
from modelgraph.evaluation.metrics import mean_or_none, percentile
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.registry import default_registry
from modelgraph.revision import (
    RevisionStore,
    bootstrap_revision,
    propose_graph_change,
    propose_swap,
    revision_timeline,
)
from modelgraph.runtime.engine import run_graph
from modelgraph.watch import WatchSession, watch_loop

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
GRAPHS = ROOT / "graphs"
HELLO = EXAMPLES / "hello-triage.graph.yaml"
CODING = GRAPHS / "coding-specialist.graph.yaml"
FIXTURE_MODELS = Path(__file__).resolve().parents[1] / "iteration" / "fixtures" / "models"
TASK = "Write a debounce function in TypeScript."
runner = CliRunner()


class _Audit(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[dict[str, str | None]] = []
        self.inline_handlers: list[str] = []
        self.asset_refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for name, _ in attrs:
            if name.startswith("on"):
                self.inline_handlers.append(f"{tag}[{name}]")
        if tag == "script":
            self.scripts.append(values)
        for name in ("src", "href"):
            if values.get(name):
                self.asset_refs.append(f"{tag}[{name}]={values[name]}")
        if tag in {"img", "link", "iframe", "object", "embed"}:
            self.asset_refs.append(tag)


def _compile(path: Path = HELLO) -> CompiledGraph:
    result = compile_graph(path, strict_models=True, graph_paths=[path.parent])
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def _factory(scenario: MockScenario | None = None):
    scenario = scenario or MockScenario()

    def factory(manifest: ModelManifest) -> MockProvider:
        return MockProvider(scenario, max_context=manifest.spec.context_window)

    return factory


def _run(
    graph_path: Path = HELLO,
    *,
    task: str = TASK,
    scenario: MockScenario | None = None,
    runs_dir: Path,
):
    return run_graph(_compile(graph_path), task, _factory(scenario), runs_dir=runs_dir)


def _inspect(
    run_dir: Path,
    *,
    graph_path: Path = HELLO,
    change: dict[str, Any] | None = None,
    impact: dict[str, Any] | None = None,
    timeline: dict[str, Any] | None = None,
    compare_run: Path | None = None,
) -> tuple[dict[str, Any], str]:
    model = inspector.build_inspection(
        run_dir,
        _compile(graph_path),
        change=change,
        impact=impact,
        timeline=timeline,
        compare_run=compare_run,
    )
    return model, inspector.render_inspector_html(model)


def _embedded_data(html: str) -> dict[str, Any]:
    match = re.search(
        r'<script type="application/json" id="mg-inspector-data">(.*?)</script>',
        html,
        flags=re.S,
    )
    assert match is not None
    payload = json.loads(match.group(1))
    assert isinstance(payload, dict)
    return payload


def _change_store(
    tmp_path: Path,
) -> tuple[RevisionStore, dict[str, Any], dict[str, Any], dict[str, Any]]:
    store = RevisionStore(tmp_path / "store")
    boot = bootstrap_revision(store, graph_path=CODING)
    store.promote(str(boot["revision"]["revision_id"]))
    swap = propose_swap(
        store,
        node="js-specialist",
        to_model="js-specialist-3b@0.4.0",
        model_dirs=[FIXTURE_MODELS],
        registry=default_registry(),
    )
    change = store.load_change(str(swap["change"]["change_id"]))
    impact = store.load_impact(str(change["change_id"]))
    timeline = revision_timeline(store, change=change)
    return store, change, impact, timeline


def _two_revision_store(tmp_path: Path) -> tuple[RevisionStore, str, str, dict[str, Any]]:
    """Create active + previous revisions and a distinct candidate change."""
    store = RevisionStore(tmp_path / "store")
    base = _compile(CODING)
    rev1, _ = store.create_revision(base, graph_source=CODING)
    store.promote(str(rev1["revision_id"]))

    document = yaml.safe_load(CODING.read_text(encoding="utf-8"))
    document["spec"]["nodes"][1]["instruction"] = "Prefer small, auditable functions."
    changed_path = tmp_path / "changed.graph.yaml"
    changed_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    changed = _compile(changed_path)
    rev2, _ = store.create_revision(changed, graph_source=changed_path)
    store.promote(str(rev2["revision_id"]))

    document["spec"]["nodes"][1]["instruction"] = "Prefer small functions and tests."
    candidate_path = tmp_path / "candidate.graph.yaml"
    candidate_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    proposed = propose_graph_change(
        store,
        candidate_graph_path=candidate_path,
        base_revision=str(rev2["revision_id"]),
        registry=default_registry(),
    )
    change = store.load_change(str(proposed["change"]["change_id"]))
    timeline = revision_timeline(store, change=change)
    return store, str(rev1["revision_id"]), str(rev2["revision_id"]), timeline


# --------------------------------------------------------------------------- watch


def test_watch_once_valid_graph_runs_mock_cycle(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    result = runner.invoke(
        app,
        ["watch", str(HELLO), "--once", "--runs-dir", str(runs), "--json"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["schema"] == "modelgraph/watch-cycle/v0"
    assert report["status"] == "ok"
    assert report["provider"] == "mock"
    assert report["graph"]["hash"].startswith("sha256:")
    assert [entry["node"] for entry in report["path"]] == ["triage", "coding"]
    assert report["artifacts"][0]["name"] == "answer"
    assert report["artifacts"][0]["hash"].startswith("sha256:")
    assert report["totals"]["total_tokens"] > 0
    assert report["totals"]["cost_usd"] == 0
    assert report["changed_from_previous"] is None
    assert (runs / report["run"]["run_id"] / "run.json").is_file()


def test_watch_once_never_selects_a_real_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        calls.append("real provider or loader selected")
        raise AssertionError("watch must not construct a real provider")

    monkeypatch.setattr("modelgraph.cli.build_provider", forbidden)
    monkeypatch.setattr("modelgraph.providers.build_provider", forbidden)
    monkeypatch.setattr("modelgraph.pruning.LazyModelLoader", forbidden)
    result = runner.invoke(
        app,
        ["watch", str(HELLO), "--once", "--runs-dir", str(tmp_path / "runs"), "--json"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["provider"] == "mock"
    assert calls == []


def test_watch_once_invalid_graph_exits_non_zero_with_last_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.graph.yaml"
    bad.write_text("apiVersion: modelgraph/v0\nkind: Graph\n", encoding="utf-8")
    runs = tmp_path / "runs"
    result = runner.invoke(app, ["watch", str(bad), "--once", "--runs-dir", str(runs), "--json"])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["status"] == "invalid"
    assert report["errors"]
    assert report["provider"] == "mock"
    assert not list(runs.glob("*/run.json"))


def test_watch_session_detects_rewritten_graph_and_compares_previous_success(
    tmp_path: Path,
) -> None:
    graph = tmp_path / "watch.graph.yaml"
    graph.write_text(HELLO.read_text(encoding="utf-8"), encoding="utf-8")
    session = WatchSession(graph, registry=default_registry(), runs_dir=tmp_path / "runs")
    first = session.poll_once()
    assert first is not None
    assert first["status"] == "ok"
    assert first["changed_from_previous"] is None
    assert session.poll_once() is None

    document = yaml.safe_load(graph.read_text(encoding="utf-8"))
    document["spec"]["nodes"][1]["instruction"] = "You are a very careful engineer."
    graph.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    second = session.poll_once()
    assert second is not None
    assert second["status"] == "ok"
    assert second["changed_from_previous"] is True
    assert second["change_flags"]["graph"] is True
    assert any("graph hash" in line for line in second["changes"])
    assert any("usage" in line or "path:" in line for line in second["changes"])


def test_watch_session_reruns_on_an_identical_save(tmp_path: Path) -> None:
    graph = tmp_path / "watch.graph.yaml"
    graph.write_text(HELLO.read_text(encoding="utf-8"), encoding="utf-8")
    session = WatchSession(graph, registry=default_registry(), runs_dir=tmp_path / "runs")
    first = session.poll_once()
    assert first is not None and first["status"] == "ok"
    assert session.poll_once() is None

    # Same bytes, new save timestamp: an editor save should still rerun.
    stat = graph.stat()
    graph.write_text(graph.read_text(encoding="utf-8"), encoding="utf-8")
    os.utime(graph, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = session.poll_once()
    assert second is not None
    assert second["status"] == "ok"
    assert second["changed_from_previous"] is False


def test_watch_once_output_is_deterministic(tmp_path: Path) -> None:
    args = ["watch", str(HELLO), "--once", "--runs-dir", str(tmp_path / "runs"), "--json"]
    first = runner.invoke(app, args)
    second = runner.invoke(app, args)
    assert first.exit_code == 0, first.stdout + first.stderr
    assert second.exit_code == 0, second.stdout + second.stderr
    assert first.stdout == second.stdout
    # The deterministic run id means the same graph does not accumulate duplicate runs.
    assert len([path for path in (tmp_path / "runs").iterdir() if path.is_dir()]) == 1


def test_watch_loop_detects_save_and_propagates_interrupt(tmp_path: Path) -> None:
    graph = tmp_path / "watch.graph.yaml"
    graph.write_text(HELLO.read_text(encoding="utf-8"), encoding="utf-8")
    session = WatchSession(graph, registry=default_registry(), runs_dir=tmp_path / "runs")
    reports: list[dict[str, Any]] = []
    sleeps = {"count": 0}

    def sleep(interval: float) -> None:
        sleeps["count"] += 1
        if sleeps["count"] == 1:
            document = yaml.safe_load(graph.read_text(encoding="utf-8"))
            document["spec"]["nodes"][1]["instruction"] = "A changed instruction."
            graph.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            return
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        watch_loop(session, interval=0.01, emit=reports.append, sleep=sleep)
    assert len(reports) == 2
    assert reports[0]["changed_from_previous"] is None
    assert reports[1]["changed_from_previous"] is True


def test_watch_loop_cli_exits_cleanly_on_ctrl_c(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_loop(session: WatchSession, *, interval: float, emit: Any) -> None:
        report = session.poll_once()
        assert report is not None
        emit(report)
        raise KeyboardInterrupt

    monkeypatch.setattr("modelgraph.cli.watch_loop", fake_loop)
    result = runner.invoke(
        app,
        ["watch", str(HELLO), "--runs-dir", str(tmp_path / "runs")],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "watch:    OK" in result.stdout
    assert "watch: stopped" in result.stdout


# --------------------------------------------------------------------------- impact panel


def test_inspector_impact_panel_shows_evidence_invalidations_and_rollback(
    tmp_path: Path,
) -> None:
    store, change, impact, timeline = _change_store(tmp_path)
    run = _run(graph_path=CODING, runs_dir=tmp_path / "runs")
    assert run.run_dir is not None
    model, html = _inspect(
        run.run_dir, graph_path=CODING, change=change, impact=impact, timeline=timeline
    )

    iteration = model["iteration"]
    assert iteration["change"]["change_id"] == change["change_id"]
    assert iteration["impact"]["change_class"] == "model_swap"
    assert iteration["impact"]["eval"]["required"] == ["node", "routing", "e2e"]
    assert iteration["impact"]["eval"]["missing"] == ["node", "routing", "e2e"]
    assert iteration["impact"]["invalidations"]
    assert iteration["impact"]["rollback"]["target"] == change["base_revision"]

    for needle in (
        "Impact before promotion",
        "Evaluation required before promotion",
        "Invalidations",
        "Rollback",
        "node",
        "routing",
        "e2e",
        change["candidate_revision"],
        change["base_revision"],
    ):
        assert needle in html
    assert 'data-artifact="compiled_plan"' in html
    assert 'data-artifact="distilled_student"' in html
    assert "NOT APPLICABLE" in html
    assert "STALE" in html


def test_stale_badges_are_explicit_when_no_artifact_exists(tmp_path: Path) -> None:
    _, change, impact, timeline = _change_store(tmp_path)
    section = inspector._iteration_section(change, impact, timeline)
    by_artifact = {item["artifact"]: item for item in section["stale_artifacts"]}
    # Portable v0 builds no plan/cache/student artifacts. The panel must say so.
    assert by_artifact["compiled_plan"]["badge"] == "not_applicable"
    assert by_artifact["prefix_cache"]["badge"] == "not_applicable"
    assert by_artifact["route_cache"]["badge"] == "not_applicable"
    assert by_artifact["compiled_plan"]["present"] is False
    assert by_artifact["compiled_plan"]["present_label"] == "none"
    assert by_artifact["compiled_plan"]["artifact_hash"] is None
    # A student is stale by ADR-0013 even though no student artifact exists yet.
    assert by_artifact["distilled_student"]["badge"] == "stale"
    assert by_artifact["distilled_student"]["present"] is False
    assert by_artifact["distilled_student"]["artifact_hash"] is None
    candidate_hash = next(
        entry["graph_hash"]
        for entry in timeline["entries"]
        if entry["revision_id"] == impact["candidate_revision"]
    )
    for item in by_artifact.values():
        assert item["source_hash"] == candidate_hash
        assert "hash" in item["hash_basis"]


def test_revision_timeline_distinguishes_active_previous_candidate_and_stale(
    tmp_path: Path,
) -> None:
    _, rev1, rev2, timeline = _two_revision_store(tmp_path)
    roles = {entry["revision_id"]: entry["roles"] for entry in timeline["entries"]}
    assert "active" in roles[rev2]
    assert "previous" in roles[rev1]
    assert "candidate" in roles[timeline["candidate_revision"]]
    assert "stale" in roles[rev1]
    # A candidate is proposed, not stale; stale applies to retired retained revisions.
    assert "stale" not in roles[timeline["candidate_revision"]]

    run = _run(runs_dir=tmp_path / "runs")
    assert run.run_dir is not None
    # Build with the same timeline and a minimal change for rendering.
    change_record = {
        "change_id": "chg_timeline",
        "status": "compiled",
        "change_class": "prompt",
        "target": "portable",
        "base_revision": rev2,
        "candidate_revision": timeline["candidate_revision"],
        "summary": "prompt change",
        "candidate_graph": {},
        "evidence": [],
    }
    model, html = _inspect(run.run_dir, change=change_record, impact=None, timeline=timeline)
    assert "Revision timeline" in html
    assert "ACTIVE" in html
    assert "PREVIOUS" in html
    assert "CANDIDATE" in html
    assert "STALE" in html
    assert 'data-role="active"' in html
    assert 'data-role="previous"' in html
    assert 'data-role="candidate"' in html
    assert 'data-marker="promote"' in html
    assert "PROMOTED" in html
    assert model["iteration"]["timeline"]["entries"]


def test_inspect_change_mode_is_read_only(tmp_path: Path) -> None:
    store, change, _, _ = _change_store(tmp_path)
    run = _run(runs_dir=tmp_path / "runs")
    assert run.run_dir is not None

    def snapshot(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    store_before = snapshot(store.root)
    run_before = snapshot(run.run_dir)
    out = tmp_path / "inspector.html"
    result = runner.invoke(
        app,
        [
            "inspect",
            str(run.run_dir),
            "--html",
            "--graph",
            str(HELLO),
            "--out",
            str(out),
            "--change",
            str(change["change_id"]),
            "--store",
            str(store.root),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert out.is_file()
    assert snapshot(store.root) == store_before
    assert snapshot(run.run_dir) == run_before
    assert "Impact before promotion" in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- compare


def test_compare_same_inputs_indicator_and_side_by_side_rows(tmp_path: Path) -> None:
    left = _run(runs_dir=tmp_path / "runs")
    right = _run(runs_dir=tmp_path / "runs")
    assert left.run_dir is not None and right.run_dir is not None
    model, html = _inspect(left.run_dir, compare_run=right.run_dir)

    compare = model["compare"]
    assert compare["same_inputs"] is True
    assert compare["same_graph"] is True
    assert 'data-same-inputs="true"' in html
    assert "Same inputs:" in html
    assert "Per-node provider and usage" in html
    assert "Artifacts by name and hash" in html
    assert "Path and decisions" in html


def test_compare_different_inputs_warns_and_highlights_differences(tmp_path: Path) -> None:
    left = _run(runs_dir=tmp_path / "runs", task=TASK)
    right = _run(runs_dir=tmp_path / "runs", task="Write a story about a lighthouse.")
    assert left.run_dir is not None and right.run_dir is not None
    model, html = _inspect(left.run_dir, compare_run=right.run_dir)

    assert model["compare"]["same_inputs"] is False
    assert 'data-same-inputs="false"' in html
    assert "Same inputs:" in html
    assert "not a same-input comparison" in html
    assert 'data-diff="true"' in html
    assert "diff-cell" in html


def test_compare_handles_nodes_and_artifacts_missing_on_one_side(tmp_path: Path) -> None:
    scenario = MockScenario(
        responses={
            "triage": [
                json.dumps(
                    {
                        "decision": "route",
                        "to": "story",
                        "confidence": 0.91,
                        "reason": "asks for a story",
                    }
                )
            ]
        }
    )
    left = _run(runs_dir=tmp_path / "runs", task="Tell me a story.", scenario=scenario)
    document = yaml.safe_load(HELLO.read_text(encoding="utf-8"))
    document["spec"]["nodes"] = [
        node for node in document["spec"]["nodes"] if node["id"] != "story"
    ]
    document["spec"]["edges"] = [
        edge for edge in document["spec"]["edges"] if edge["to"] != "story"
    ]
    pruned = tmp_path / "pruned.graph.yaml"
    pruned.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    right = _run(graph_path=pruned, runs_dir=tmp_path / "runs")
    assert left.run_dir is not None and right.run_dir is not None
    model, html = _inspect(left.run_dir, compare_run=right.run_dir)

    rows = {row["node"]: row for row in model["compare"]["nodes"]["rows"]}
    assert rows["story"]["missing_right"] is True
    assert rows["coding"]["missing_left"] is False
    assert "missing-cell" in html
    assert "—" in html
    artifact_rows = model["compare"]["artifacts"]["rows"]
    assert artifact_rows and artifact_rows[0]["name"] == "answer"
    assert artifact_rows[0]["same"] is False
    assert "diff-cell" in html


def test_compare_handles_a_failed_run_without_artifacts(tmp_path: Path) -> None:
    failing = MockScenario(failures={"coding": "MG-3001"})
    left = _run(runs_dir=tmp_path / "runs")
    right = _run(runs_dir=tmp_path / "runs", scenario=failing)
    assert left.run_dir is not None and right.run_dir is not None
    model, html = _inspect(left.run_dir, compare_run=right.run_dir)
    rows = model["compare"]["artifacts"]["rows"]
    assert rows
    assert rows[0]["missing_right"] is True
    assert "missing-cell" in html


# --------------------------------------------------------------------------- hostile content


def test_hostile_impact_and_timeline_content_is_escaped(tmp_path: Path) -> None:
    hostile = '<img src=x onerror=alert(1)> & "</script><script>alert(2)</script>\x00'
    run = _run(runs_dir=tmp_path / "runs")
    assert run.run_dir is not None
    change = {
        "change_id": "chg_hostile",
        "status": hostile,
        "change_class": hostile,
        "target": hostile,
        "base_revision": hostile,
        "candidate_revision": hostile,
        "summary": hostile,
        "candidate_graph": {"id": hostile, "version": hostile, "hash": hostile, "source": hostile},
        "evidence": [
            {
                "path": hostile,
                "level": hostile,
                "valid": False,
                "gate_passed": False,
                "errors": [hostile],
            }
        ],
    }
    impact = {
        "change_class": hostile,
        "change_class_description": hostile,
        "target": hostile,
        "base_revision": hostile,
        "candidate_revision": hostile,
        "compatibility": {
            "status": hostile,
            "blocking": True,
            "message": hostile,
            "checks": [
                {
                    "node": hostile,
                    "from_ref": hostile,
                    "to_ref": hostile,
                    "base": {"from": hostile, "to": hostile, "status": hostile},
                    "tokenizer": {"from": hostile, "to": hostile, "status": hostile},
                    "blocking": True,
                    "message": hostile,
                }
            ],
        },
        "eval": {
            "required": [hostile],
            "coverage": {"satisfied": [], "missing": [hostile], "complete": False},
        },
        "invalidations": [
            {
                "artifact": hostile,
                "status": hostile,
                "applicable": True,
                "present": False,
                "reason": hostile,
                "scope": [hostile],
            }
        ],
        "vram": {
            "before_gb": 1.0,
            "after_gb": 2.0,
            "delta_gb": 1.0,
            "complete": True,
            "unknown_models": [hostile],
            "admission": {"status": hostile, "reason": hostile},
        },
        "rollback": {"target": hostile, "retained": False, "grace_days": 7, "note": hostile},
        "affected": {
            "nodes": [hostile],
            "routers": [{"router": hostile, "reasons": [hostile]}],
            "routing_cards": [hostile],
            "route_menu_changed": True,
        },
        "risk": [hostile],
        "notes": [hostile],
    }
    timeline = {
        "store": hostile,
        "active_revision": hostile,
        "previous_revision": hostile,
        "candidate_revision": hostile,
        "entries": [
            {
                "revision_id": hostile,
                "graph_id": hostile,
                "graph_hash": hostile,
                "target": hostile,
                "created_at": hostile,
                "roles": ["active", "stale"],
                "components": {"plan_status": hostile, "plan_hash": None, "student_hash": None},
                "markers": [{"event": "promote", "ts": hostile, "reason": hostile}],
            }
        ],
        "notes": [hostile],
    }
    model, html = _inspect(run.run_dir, change=change, impact=impact, timeline=timeline)

    audit = _Audit()
    audit.feed(html)
    assert audit.inline_handlers == []
    assert len(audit.scripts) == 2
    assert all(script.get("src") is None for script in audit.scripts)
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
    assert "&lt;img src=x onerror=" in html
    assert html.count("</script>") == 2
    assert "\x00" not in html
    assert r"\x00" in html
    data = _embedded_data(html)
    assert data["iteration"]["change"]["summary"] == hostile
    assert data["iteration"]["impact"]["compatibility"]["message"] == hostile


def test_hostile_compare_manifest_content_is_escaped(tmp_path: Path) -> None:
    hostile = '<img src=x onerror=alert(3)> & "</script><script>alert(4)</script>\x00'
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_manifest_run(
        left, run_id="run_left", task="left", hostile=hostile, graph_hash="sha256:" + "1" * 64
    )
    _write_manifest_run(
        right, run_id="run_right", task="right", hostile=hostile, graph_hash="sha256:" + "2" * 64
    )
    model, html = _inspect(left, compare_run=right)

    audit = _Audit()
    audit.feed(html)
    assert audit.inline_handlers == []
    assert len(audit.scripts) == 2
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html
    assert "&lt;img src=x onerror=" in html
    assert html.count("</script>") == 2
    assert "\x00" not in html
    assert r"\x00" in html
    data = _embedded_data(html)
    assert data["compare"]["left"]["run_id"] == "run_left"
    assert data["compare"]["same_graph"] is False


def test_compare_missing_graph_hash_is_unknown_not_same(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    hostile = "plain"
    _write_manifest_run(
        left, run_id="run_left", task="same", hostile=hostile, graph_hash="sha256:" + "1" * 64
    )
    _write_manifest_run(
        right, run_id="run_right", task="same", hostile=hostile, graph_hash="sha256:" + "1" * 64
    )
    for directory in (left, right):
        path = directory / "run.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["graph"].pop("hash", None)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    model, html = _inspect(left, compare_run=right)
    assert model["compare"]["same_graph"] is None
    assert 'data-same-graph="unknown"' in html
    assert "Recorded graph hash: unknown" in html
    assert "same-graph status is unknown" in html


def test_iteration_and_compare_panels_have_no_external_assets(tmp_path: Path) -> None:
    _, change, impact, timeline = _change_store(tmp_path)
    left = _run(runs_dir=tmp_path / "runs")
    right = _run(runs_dir=tmp_path / "runs")
    assert left.run_dir is not None and right.run_dir is not None
    _, html = _inspect(
        left.run_dir,
        change=change,
        impact=impact,
        timeline=timeline,
        compare_run=right.run_dir,
    )
    audit = _Audit()
    audit.feed(html)
    assert audit.asset_refs == []
    assert not re.search(r"https?://", html)
    assert not re.search(r"@import", html)
    assert not re.search(r"url\(", html)
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html
    assert "innerHTML" not in html
    assert "new Function" not in html
    assert "eval(" not in html


def _write_manifest_run(
    directory: Path,
    *,
    run_id: str,
    task: str,
    hostile: str,
    graph_hash: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "status": "final",
        "outcome": "succeeded",
        "target": "portable",
        "graph": {"id": hostile, "version": hostile, "hash": graph_hash},
        "request": {"task": task},
        "models": [
            {
                "node": hostile,
                "provider": hostile,
                "model": hostile,
                "hash": "sha256:" + "2" * 64,
            }
        ],
        "path": [{"node": hostile, "kind": "worker", "decision": hostile, "to": hostile}],
        "artifacts": [
            {"name": hostile, "type": hostile, "hash": "sha256:" + "3" * 64, "producer": hostile}
        ],
        "usage": {"steps": 1, "total_tokens": 2, "cost_usd": 0.0, "elapsed_seconds": 0.1},
        "trace": "trace.jsonl",
    }
    (directory / "run.json").write_text(json.dumps(manifest), encoding="utf-8")


# --------------------------------------------------------------------------- diff / eval compare


def test_graph_diff_highlights_routing_menu_changes(tmp_path: Path) -> None:
    document = yaml.safe_load(HELLO.read_text(encoding="utf-8"))
    document["spec"]["edges"][0]["when"] = "The request is specifically TypeScript."
    candidate = tmp_path / "candidate.graph.yaml"
    candidate.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    result = runner.invoke(
        app,
        ["graph", "diff", str(HELLO), str(candidate), "--json"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["routing_menu"]["changed"] is True
    assert payload["routing_menu"]["routers"] == ["triage"]
    assert payload["topology_changed"] is True

    human = runner.invoke(app, ["graph", "diff", str(HELLO), str(candidate)])
    assert human.exit_code == 0
    assert "routing menu:" in human.stdout


def test_graph_diff_shows_deployment_target_changes() -> None:
    result = runner.invoke(
        app,
        [
            "graph",
            "diff",
            str(HELLO),
            str(HELLO),
            "--before-target",
            "portable",
            "--after-target",
            "resident",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["target"] == {
        "before": "portable",
        "after": "resident",
        "changed": True,
    }
    assert payload["graph_changed"] is False

    human = runner.invoke(
        app,
        [
            "graph",
            "diff",
            str(HELLO),
            str(HELLO),
            "--before-target",
            "portable",
            "--after-target",
            "resident",
        ],
    )
    assert human.exit_code == 0
    assert "target:     portable -> resident" in human.stdout
    assert "target changed" in human.stdout


def test_eval_compare_prints_candidate_baseline_and_oracle_reference(
    tmp_path: Path,
) -> None:
    baseline = ROOT / "evals" / "baselines" / "e2e.json"
    result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(baseline),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    reference = payload["reference"]
    assert reference["candidate"]["task_success"] == pytest.approx(0.8)
    assert reference["baseline"]["task_success"] == pytest.approx(0.8)
    assert reference["oracle_routing"]["task_success"] == pytest.approx(1.0)
    assert reference["best_single"]["task_success"] == pytest.approx(0.4)

    human = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(baseline),
            "--reports-dir",
            str(tmp_path / "reports-human"),
        ],
    )
    assert human.exit_code == 0, human.stdout + human.stderr
    assert "reference: candidate vs incumbent vs oracle" in human.stdout
    assert "oracle_routing" in human.stdout
    assert "best_single" in human.stdout


def test_eval_compare_omits_reference_without_oracle_baselines(tmp_path: Path) -> None:
    document = json.loads((ROOT / "evals" / "baselines" / "e2e.json").read_text(encoding="utf-8"))
    document["report_id"] = "eval_e2e_without_baselines"
    document["e2e"]["baselines"] = {}
    candidate = tmp_path / "e2e-no-baselines.json"
    candidate.write_text(json.dumps(document), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline",
            str(candidate),
            "--candidate",
            str(candidate),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "reference" not in json.loads(result.stdout)

    human = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline",
            str(candidate),
            "--candidate",
            str(candidate),
            "--reports-dir",
            str(tmp_path / "reports-human"),
        ],
    )
    assert human.exit_code == 0, human.stdout + human.stderr
    assert "reference:" not in human.stdout


def test_eval_compare_prints_slices_and_uncertainty(tmp_path: Path) -> None:
    baseline = ROOT / "evals" / "baselines" / "node.json"
    report = compare_against_baseline(baseline, baseline, persist=False)
    assert report.slices
    for entry in report.slices.values():
        assert "primary_delta_ci95" in entry
        assert "cost_delta_ci95" in entry
        assert "latency_delta_ci95" in entry
    assert report.latency.get("delta_p50_ci95") is not None

    result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline",
            str(baseline),
            "--candidate",
            str(baseline),
            "--reports-dir",
            str(tmp_path / "reports"),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "slices:" in result.stdout
    assert "cost CI:" in result.stdout
    assert "latency CI:" in result.stdout
    assert "ci=[" in result.stdout


def test_watch_once_run_failure_exits_non_zero_and_keeps_last_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modelgraph.providers.base import ProviderFailure
    from modelgraph.providers.mock import MockProvider

    def fail_chat(self: MockProvider, request: Any) -> Any:
        raise ProviderFailure("mock cycle failure", code="MG-3001", retryable=False)

    monkeypatch.setattr(MockProvider, "chat", fail_chat)
    result = runner.invoke(
        app,
        ["watch", str(HELLO), "--once", "--runs-dir", str(tmp_path / "runs"), "--json"],
    )
    assert result.exit_code == 3
    report = json.loads(result.stdout)
    assert report["status"] == "failed"
    assert any("MG-3001" in error for error in report["errors"])
    assert report["run"]["outcome"] == "failed"
    human = runner.invoke(
        app,
        ["watch", str(HELLO), "--once", "--runs-dir", str(tmp_path / "runs")],
    )
    assert human.exit_code == 3
    assert "FAILED" in human.stdout
    assert "MG-3001" in human.stdout


def test_revision_timeline_marks_rollback_and_previous_after_rollback(tmp_path: Path) -> None:
    store, rev1, rev2, timeline = _two_revision_store(tmp_path)
    candidate = str(timeline["candidate_revision"])
    store.rollback(rev1)
    # A rollback keeps the candidate change readable but changes active/previous roles.
    change_record = {
        "change_id": "chg_after_rollback",
        "status": "compiled",
        "change_class": "prompt",
        "target": "portable",
        "base_revision": rev2,
        "candidate_revision": candidate,
        "summary": "prompt change",
        "candidate_graph": {},
        "evidence": [],
    }
    rolled = revision_timeline(store)
    roles = {entry["revision_id"]: entry["roles"] for entry in rolled["entries"]}
    assert "active" in roles[rev1]
    assert "previous" in roles[rev2]
    run = _run(runs_dir=tmp_path / "runs")
    assert run.run_dir is not None
    _, html = _inspect(run.run_dir, change=change_record, impact=None, timeline=rolled)
    assert "ROLLED BACK" in html
    assert 'data-marker="rollback"' in html
    assert "PROMOTED" in html


def test_eval_compare_latency_p50_is_a_median_not_a_mean() -> None:
    """A mislabeled mean would hide a single slow case behind a five-case median of zero."""
    baseline = json.loads((ROOT / "evals" / "baselines" / "node.json").read_text(encoding="utf-8"))
    candidate = copy.deepcopy(baseline)
    candidate["report_id"] = "eval_skewed_latency"
    for index, case in enumerate(candidate["cases"]):
        case["latency_seconds"] = 10.0 if index == 0 else 0.0

    report = compare_reports(candidate, baseline, persist=False)
    deltas = [float(case["latency_delta_seconds"]) for case in report.cases]
    median = percentile(deltas, 0.5)
    mean = mean_or_none(deltas)
    assert median is not None and mean is not None
    assert median < mean

    assert report.latency["delta_p50_seconds"] == pytest.approx(median)
    assert report.metrics["latency_delta_p50_seconds"] == pytest.approx(median)
    assert report.latency["delta_p50_seconds"] != pytest.approx(mean)
    interval = report.latency["delta_p50_ci95"]
    assert interval is not None and interval[0] <= median <= interval[1]
    # The all-case slice has the same paired cases, so its p50 and interval must agree.
    slice_entry = report.slices["domain:js"]
    assert slice_entry["latency_delta_p50_seconds"] == pytest.approx(median)
    assert slice_entry["latency_delta_ci95"] == interval
