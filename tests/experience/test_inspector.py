"""T-0028: offline static HTML graph and trace inspector tests."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
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
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.runtime.engine import run_graph

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
GRAPH_PATH = EXAMPLES / "hello-triage.graph.yaml"
TASK = "Write a debounce function in TypeScript."
RUN_ID = "run_inspector_fixture"
runner = CliRunner()

CODING_DECISION = json.dumps(
    {
        "decision": "route",
        "to": "coding",
        "confidence": 0.93,
        "reason": "the request asks for TypeScript code",
    }
)


class _Audit(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.scripts: list[dict[str, str | None]] = []
        self.inline_handlers: list[str] = []
        self.asset_refs: list[str] = []
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(str(values["id"]))
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


def _compile_graph(graph_path: Path = GRAPH_PATH) -> CompiledGraph:
    result = compile_graph(graph_path, strict_models=True)
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def _digest(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _event(event_name: str, **fields: Any) -> dict[str, Any]:
    return {"ts": "2026-10-25T10:00:00Z", "run_id": RUN_ID, "event": event_name, **fields}


def _write_run(
    tmp_path: Path,
    *,
    artifact_content: str = "export function debounce() {}\n",
    extra_artifacts: list[tuple[str, str]] | None = None,
    cached: bool = False,
    hostile: bool = False,
    prune: bool = True,
    graph_path: Path = GRAPH_PATH,
) -> Path:
    graph = _compile_graph(graph_path)
    run_dir = tmp_path / "run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)

    artifact_specs: list[tuple[str, str, str, str, bool]] = [
        ("answer", artifact_content, "coding", "code", True)
    ]
    for index, (name, content) in enumerate(extra_artifacts or []):
        artifact_specs.append((name, content, f"worker-{index}", "text", False))

    artifact_records: list[dict[str, Any]] = []
    for name, content, producer, artifact_type, is_exit in artifact_specs:
        digest = _digest(content)
        (artifacts_dir / digest.split(":", 1)[1]).write_text(content, encoding="utf-8")
        artifact_records.append(
            {
                "name": name,
                "type": artifact_type,
                "hash": digest,
                "producer": producer,
                "bytes": len(content.encode("utf-8")),
                "exit": is_exit,
            }
        )

    if cached:
        usage = {
            "steps": 2,
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "cost_usd": 0.001,
            "elapsed_seconds": 0.2,
        }
        budget_used = (120, 0.001, 0.2)
    else:
        usage = {
            "steps": 2,
            "input_tokens": 1000,
            "output_tokens": 234,
            "total_tokens": 1234,
            "cost_usd": 0.012345,
            "elapsed_seconds": 1.25,
        }
        budget_used = (1234, 0.012345, 1.25)

    hostile_error = (
        '<img src=x onerror=alert(4)> & "</script><script>alert(5)</script>\x00'
        if hostile
        else None
    )
    provider_name = '"><svg onload=alert(6)>' if hostile else "mock"
    model_name = "mock-router" if not hostile else "</script><script>alert(7)</script>"
    coder_model = "mock-coder" if not hostile else '"><img src=x onerror=alert(8)>'
    reason = (
        '"><script>alert("reason")</script>' if hostile else "the request asks for TypeScript code"
    )

    manifest: dict[str, Any] = {
        "run_id": RUN_ID,
        "deployment_revision": "rev_inspector_fixture",
        "target": "portable",
        "status": "final",
        "outcome": "failed" if hostile else "succeeded",
        "graph": {
            "id": graph.id,
            "version": graph.version,
            "hash": graph.graph_hash,
            "source": str(GRAPH_PATH),
        },
        "request": {"task": TASK},
        "models": [
            {
                "node": "triage",
                "ref": "router-small",
                "version": "1.0.0",
                "hash": "sha256:" + "1" * 64,
                "provider": provider_name,
                "model": model_name,
                "resolved_revision": None,
                "revision_reason": "fixture does not pin a revision",
            },
            {
                "node": "coding",
                "ref": "coder-7b",
                "version": "1.0.0",
                "hash": "sha256:" + "2" * 64,
                "provider": provider_name,
                "model": coder_model,
                "resolved_revision": "fixture-revision",
                "revision_reason": None,
            },
        ],
        "budget": {
            "max_steps": 6,
            "used_steps": 2,
            "max_depth": 2,
            "max_total_tokens": 24000,
            "used_tokens": budget_used[0],
            "max_cost_usd": 0.05,
            "used_cost_usd": budget_used[1],
            "max_wall_seconds": 120.0,
            "elapsed_seconds": budget_used[2],
            "on_budget_exhausted": "fail",
        },
        "seed": 7,
        "environment": {
            "modelgraph_version": "0.5.0",
            "python_version": "3.12.0",
            "platform": "test-platform",
        },
        "renderer_version": "3",
        "parser_version": "2",
        "ir_version": "modelgraph/ir/v0",
        "profile": {"name": "local-8gb", "providers": ["mock"]},
        "profile_hash": "sha256:" + "3" * 64,
        "started_at": "2026-10-25T10:00:00Z",
        "finished_at": "2026-10-25T10:00:01Z",
        "usage": usage,
        "path": [
            {
                "node": "triage",
                "kind": "router",
                "decision": "route",
                "to": "coding",
                "confidence": 0.93,
                "reason": reason,
            },
            {"node": "coding", "kind": "worker"},
        ],
        "artifacts": artifact_records,
        "exit_artifacts": [{"name": "answer", "hash": artifact_records[0]["hash"]}],
        "trace": "trace.jsonl",
        "error": (
            {
                "code": "MG-9999",
                "message": hostile_error,
                "path": None,
                "details": {"message": hostile_error},
            }
            if hostile
            else None
        ),
        "parent_run_id": None,
        "subgraphs": [],
    }
    if prune:
        manifest["prune"] = {
            "profile": "local-8gb",
            "removed_nodes": [
                {"id": "story", "reason": "model requires 14GB VRAM; profile allows 8GB"}
            ],
            "removed_edges": [
                {"from": "triage", "to": "story", "reason": "target node was removed"}
            ],
            "removed_exit_artifacts": [],
            "warnings": ["router triage now has one route; consider bypassing it"],
        }
        manifest["prune_hash"] = "sha256:" + "4" * 64

    events = [
        _event("run.started", graph={"id": graph.id, "hash": graph.graph_hash}),
        _event("node.entered", node="triage", kind="router", depth=0, step=1),
        _event(
            "provider.request",
            node="triage",
            provider=provider_name,
            model=model_name,
            attempt=1,
        ),
        _event(
            "provider.response",
            node="triage",
            model=model_name,
            usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            latency_seconds=0.2,
            attempt=1,
        ),
        _event(
            "budget.charged",
            node="triage",
            tokens=120,
            cost_usd=0.001,
            used_tokens=120,
            used_cost_usd=0.001,
        ),
        _event(
            "decision.made",
            node="triage",
            decision="route",
            to="coding",
            confidence=0.93,
            reason=reason,
            threshold=0.6,
        ),
        _event("node.exited", node="triage", kind="router", depth=0),
        _event("node.entered", node="coding", kind="worker", depth=0, step=2),
        _event(
            "provider.request",
            node="coding",
            provider=provider_name,
            model=coder_model,
            attempt=1,
        ),
    ]
    if cached:
        events.extend(
            [
                _event(
                    "provider.response",
                    node="coding",
                    model=coder_model,
                    usage={"input_tokens": 900, "output_tokens": 214, "total_tokens": 1114},
                    latency_seconds=0.0,
                    attempt=1,
                    cached=True,
                    cache_hit_id="cachehit_fixture",
                ),
                _event(
                    "provider.response.cached",
                    node="coding",
                    model=coder_model,
                    usage={"input_tokens": 900, "output_tokens": 214, "total_tokens": 1114},
                    avoided_tokens=1114,
                    avoided_cost_usd=0.011345,
                    cached=True,
                    cache_hit_id="cachehit_fixture",
                ),
            ]
        )
    else:
        events.extend(
            [
                _event(
                    "provider.response",
                    node="coding",
                    model=coder_model,
                    usage={"input_tokens": 900, "output_tokens": 214, "total_tokens": 1114},
                    latency_seconds=1.05,
                    attempt=1,
                ),
                _event(
                    "budget.charged",
                    node="coding",
                    tokens=1114,
                    cost_usd=0.011345,
                    used_tokens=1234,
                    used_cost_usd=0.012345,
                ),
            ]
        )
    if hostile:
        events.append(
            _event(
                "provider.error",
                node="coding",
                code="MG-3000",
                message=hostile_error,
                retryable=False,
                attempt=1,
            )
        )
    events.extend(
        [
            _event(
                "artifact.written",
                node="coding",
                name="answer",
                type="code",
                bytes=len(artifact_content.encode("utf-8")),
            ),
            _event("node.exited", node="coding", kind="worker", depth=0),
            _event("run.finished", outcome=manifest["outcome"], usage=usage),
        ]
    )

    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return run_dir


def _inspect(run_dir: Path, graph_path: Path = GRAPH_PATH) -> tuple[dict[str, Any], str]:
    graph = _compile_graph(graph_path)
    model = inspector.build_inspection(run_dir, graph)
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


# --------------------------------------------------------------------------- core content


def test_mock_run_is_deterministic_and_contains_graph_path_artifacts_totals(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path)
    model, html = _inspect(run_dir)
    second_model, second_html = _inspect(run_dir)

    assert model == second_model
    assert html == second_html
    assert model["schema"] == inspector.INSPECTOR_SCHEMA
    assert model["totals"]["total_tokens"] == 1234
    assert model["totals"]["cost_usd"] == pytest.approx(0.012345)
    assert model["totals"]["elapsed_seconds"] == pytest.approx(1.25)

    assert 'id="graph-svg"' in html
    assert 'data-node="triage"' in html
    assert 'data-node="coding"' in html
    assert "chosen" in html
    assert "ENTRY" in html
    assert "EXIT: answer" in html
    assert "export function debounce() {}" in html
    assert "$0.012345" in html
    assert "1.250s" in html
    assert "1,234" in html
    assert "Chosen path" in html
    assert "Run manifest summary" in html

    data = _embedded_data(html)
    assert data["totals"]["total_tokens"] == 1234
    assert data["path"][0]["node"] == "triage"
    assert data["artifacts"][0]["name"] == "answer"


def test_real_mock_run_renders_and_is_reproducible(tmp_path: Path) -> None:
    graph = _compile_graph()
    scenario = MockScenario(responses={"triage": [CODING_DECISION]})

    def factory(manifest: ModelManifest) -> MockProvider:
        return MockProvider(scenario, max_context=manifest.spec.context_window)

    outcome = run_graph(graph, TASK, factory, runs_dir=tmp_path / "runs")
    assert outcome.run_dir is not None
    assert outcome.ok

    model, first = _inspect(outcome.run_dir)
    _, second = _inspect(outcome.run_dir)
    assert first == second
    assert model["summary"]["outcome"] == "succeeded"
    assert "export function debounce" in first
    assert "Chosen path" in first
    data = _embedded_data(first)
    assert data["artifacts"][0]["content"] == outcome.artifacts[0].content


def test_totals_match_run_json_and_node_rows(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    model, html = _inspect(run_dir)

    for key in (
        "steps",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "elapsed_seconds",
    ):
        assert model["totals"][key] == manifest["usage"][key]
        assert model["summary"]["totals"][key] == manifest["usage"][key]

    nodes = [node for node in model["nodes"] if node["in_graph"]]
    assert sum(node["tokens"] for node in nodes) == manifest["usage"]["total_tokens"]
    assert sum(node["cost_usd"] for node in nodes) == pytest.approx(manifest["usage"]["cost_usd"])
    assert sum(node["latency_seconds"] for node in nodes) == pytest.approx(
        manifest["usage"]["elapsed_seconds"]
    )
    assert "$0.012345" in html
    assert "1,234" in html


# --------------------------------------------------------------------------- safety


def test_hostile_artifact_and_error_content_is_escaped_not_executable(
    tmp_path: Path,
) -> None:
    hostile = (
        "before <script>alert('artifact')</script> "
        '<img src=x onerror="alert(2)"> '
        "</script><script>alert(3)</script> "
        "\x00\x07 end"
    )
    run_dir = _write_run(tmp_path, artifact_content=hostile, hostile=True)
    model, html = _inspect(run_dir)

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
    assert "\x07" not in html
    assert r"\x00" in html
    assert r"\x07" in html
    assert "<img src=x onerror=alert(4)>" not in html
    assert "&lt;img src=x onerror=alert(4)&gt;" in html

    data = _embedded_data(html)
    assert data["artifacts"][0]["content"] == hostile
    assert data["summary"]["error"]["message"] == (
        '<img src=x onerror=alert(4)> & "</script><script>alert(5)</script>\x00'
    )
    assert data["path"][0]["reason"] == '"><script>alert("reason")</script>'
    assert data["nodes"][0]["provider"] == '"><svg onload=alert(6)>'


def test_no_external_assets_or_network_references(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    _, html = _inspect(run_dir)

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
    assert 'http-equiv="Content-Security-Policy"' in html
    assert "connect-src 'none'" in html


# --------------------------------------------------------------------------- pruning


def test_pruned_nodes_and_edges_are_visually_distinct_and_labeled(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    model, html = _inspect(run_dir)

    story = next(node for node in model["graph"]["nodes"] if node["id"] == "story")
    assert story["pruned"] is True
    assert story["prune_reason"] == "model requires 14GB VRAM; profile allows 8GB"
    edge = next(
        edge
        for edge in model["graph"]["edges"]
        if edge["from"] == "triage" and edge["to"] == "story"
    )
    assert edge["pruned"] is True
    assert edge["prune_reason"] == "target node was removed"

    assert re.search(r'<g class="node kind-worker pruned" data-node="story"', html)
    assert re.search(r'class="edge pruned" .*data-from="triage" data-to="story"', html)
    assert "PRUNED" in html
    assert "Removed nodes" in html
    assert "Removed edges" in html
    assert "model requires 14GB VRAM; profile allows 8GB" in html


def test_authored_graph_may_already_have_pruned_nodes_removed(tmp_path: Path) -> None:
    document = yaml.safe_load(GRAPH_PATH.read_text(encoding="utf-8"))
    document["spec"]["nodes"] = [
        node for node in document["spec"]["nodes"] if node["id"] != "story"
    ]
    document["spec"]["edges"] = [
        edge for edge in document["spec"]["edges"] if edge["to"] != "story"
    ]
    pruned_graph = tmp_path / "pruned.graph.yaml"
    pruned_graph.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    run_dir = _write_run(tmp_path, graph_path=pruned_graph)
    model, html = _inspect(run_dir, pruned_graph)

    assert {node["id"] for node in model["graph"]["nodes"]} == {"triage", "coding"}
    assert "story" in {node["id"] for node in model["prune"]["removed_nodes"]}
    assert "no (not in compiled graph)" in html
    assert "PRUNED" in html
    assert 'data-node="story"' not in html
    assert 'data-to="story"' not in html


# --------------------------------------------------------------------------- portability


def test_copied_run_directory_still_renders_with_embedded_content(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, artifact_content="portable artifact body\n")
    copied = tmp_path / "copied" / "run"
    shutil.copytree(run_dir, copied)
    shutil.rmtree(run_dir)

    model, html = _inspect(copied)
    assert model["artifacts"][0]["content"] == "portable artifact body\n"
    assert "portable artifact body" in html
    assert str(run_dir) not in html
    assert str(GRAPH_PATH) not in html
    assert "CONTENT MISSING" not in html


# --------------------------------------------------------------------------- size caps


def test_per_artifact_cap_truncates_with_visible_notice(tmp_path: Path) -> None:
    content = "A" * (inspector.MAX_ARTIFACT_BYTES + 1000) + "TAIL"
    run_dir = _write_run(tmp_path, artifact_content=content)
    model, html = _inspect(run_dir)

    artifact = model["artifacts"][0]
    assert artifact["truncated"] is True
    assert artifact["truncation_reason"] == "per-artifact cap"
    assert artifact["original_bytes"] == len(content.encode("utf-8"))
    assert artifact["embedded_bytes"] <= inspector.MAX_ARTIFACT_BYTES
    assert len(artifact["content"].encode("utf-8")) <= inspector.MAX_ARTIFACT_BYTES
    assert artifact["content"].endswith("A")
    assert "TAIL" not in artifact["content"]
    assert 'data-truncated="true"' in html
    assert "Truncated (per-artifact cap)" in html
    assert "bytes." in html


def test_total_artifact_cap_bounds_all_embedded_content(tmp_path: Path) -> None:
    extra = [(f"artifact-{index}", f"{index}-" + "B" * 900_000) for index in range(10)]
    run_dir = _write_run(tmp_path, extra_artifacts=extra)
    model = inspector.build_inspection(run_dir, _compile_graph())

    embedded = sum(len(item["content"].encode("utf-8")) for item in model["artifacts"])
    assert embedded <= inspector.MAX_TOTAL_ARTIFACT_BYTES
    assert model["artifact_limits"]["embedded_bytes"] == embedded
    assert model["artifact_limits"]["truncated"] is True
    assert any(item["truncation_reason"] == "total artifact cap" for item in model["artifacts"])


def test_large_artifact_cut_by_the_total_cap_is_labeled_total_not_per_artifact(
    tmp_path: Path,
) -> None:
    extra = [(f"filler-{index}", "C" * 900_000) for index in range(8)]
    extra.append(("large", "D" * (inspector.MAX_ARTIFACT_BYTES + 1000)))
    run_dir = _write_run(tmp_path, extra_artifacts=extra)
    model = inspector.build_inspection(run_dir, _compile_graph())

    large = next(item for item in model["artifacts"] if item["name"] == "large")
    assert large["truncated"] is True
    # The total cap, not the larger per-artifact cap, is what cut this artifact; naming the
    # wrong cap would send users to the wrong remedy.
    assert large["truncation_reason"] == "total artifact cap"
    assert large["embedded_bytes"] <= inspector.MAX_TOTAL_ARTIFACT_BYTES


# --------------------------------------------------------------------------- cache filter


def test_cache_hits_support_a_cache_filter(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, cached=True)
    model, html = _inspect(run_dir)

    assert model["cache"]["hits"] == 1
    assert model["cache"]["avoided_tokens"] == 1114
    assert 'id="cache-filter"' in html
    assert 'data-cache="hit"' in html
    assert "Cache hits only" in html


def test_cache_filter_is_absent_when_no_cache_data_exists(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    _, html = _inspect(run_dir)
    assert 'id="cache-filter"' not in html
    assert "No cache events were recorded for this run." in html


# --------------------------------------------------------------------------- CLI


def test_cli_success_writes_self_contained_file(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    out = tmp_path / "inspector.html"
    result = runner.invoke(
        app,
        [
            "inspect",
            str(run_dir),
            "--html",
            "--graph",
            str(GRAPH_PATH),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert out.is_file()
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "Run manifest summary" in html
    assert "Embedded data warning" in html


def test_cli_missing_run_fails_cleanly(tmp_path: Path) -> None:
    out = tmp_path / "inspector.html"
    result = runner.invoke(
        app,
        [
            "inspect",
            str(tmp_path / "missing-run"),
            "--html",
            "--graph",
            str(GRAPH_PATH),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert "run directory not found" in result.output
    assert "Traceback" not in result.output
    assert not out.exists()


def test_cli_run_dir_without_manifest_fails_cleanly(tmp_path: Path) -> None:
    empty = tmp_path / "empty-run"
    empty.mkdir()
    out = tmp_path / "inspector.html"
    result = runner.invoke(
        app,
        [
            "inspect",
            str(empty),
            "--html",
            "--graph",
            str(GRAPH_PATH),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 1
    assert "run manifest not found" in result.output
    assert "Traceback" not in result.output
    assert not out.exists()


def test_cli_invalid_graph_fails_cleanly(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    bad_graph = tmp_path / "bad.graph.yaml"
    bad_graph.write_text("apiVersion: modelgraph/v0\nkind: Graph\n", encoding="utf-8")
    out = tmp_path / "inspector.html"
    result = runner.invoke(
        app,
        [
            "inspect",
            str(run_dir),
            "--html",
            "--graph",
            str(bad_graph),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert not out.exists()
    assert result.output.strip()


def test_cli_missing_output_parent_fails_cleanly(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    out = tmp_path / "missing-parent" / "inspector.html"
    result = runner.invoke(
        app,
        [
            "inspect",
            str(run_dir),
            "--html",
            "--graph",
            str(GRAPH_PATH),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 2
    assert "output directory not found" in result.output
    assert "Traceback" not in result.output
    assert not out.exists()
