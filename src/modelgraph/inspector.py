"""Offline, self-contained HTML inspection of a recorded run.

The inspector is deliberately small in scope: it compiles an authored graph with the
existing compiler/registry path, reads a recorded run directory (``run.json``,
``trace.jsonl``, and ``artifacts/``), builds one JSON inspection document, and renders it
as a single HTML file with inline CSS/SVG and a fixed inline filter script.

Two boundaries matter:

* The JSON document (``modelgraph/inspector/v0``) is the contract a future UI consumes.
  HTML rendering is one view of that document, not a second semantics implementation.
* Every graph, manifest, trace, artifact, model, and error value is untrusted text. It is
  HTML-escaped for the page, JSON-escaped inside the embedded data block, and control
  characters are made visible. Nothing from the run is used as markup or JavaScript.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any, cast

from modelgraph.compiler.ir import CompiledGraph
from modelgraph.runtime.manifest import read_run_manifest
from modelgraph.runtime.trace import read_events
from modelgraph.util import canonical_json, sha256_text

INSPECTOR_SCHEMA = "modelgraph/inspector/v0"
INSPECTOR_ITERATION_SCHEMA = "modelgraph/inspector-iteration/v0"
INSPECTOR_COMPARE_SCHEMA = "modelgraph/inspector-compare/v0"

#: Per-artifact embedded-content limit. Content is truncated, never silently dropped.
MAX_ARTIFACT_BYTES = 1_000_000
#: Total embedded-artifact limit across all artifacts in one inspector.
MAX_TOTAL_ARTIFACT_BYTES = 8_000_000

#: Fixed geometry for the dependency-free SVG layout.
_NODE_WIDTH = 208
_NODE_HEIGHT = 78
_COLUMN_GAP = 96
_ROW_GAP = 70
_MARGIN = 44
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_C1_CONTROLS = re.compile(r"[\x80-\x9f]")
_LINE_SEPARATORS = re.compile(r"[\u2028\u2029]")


# --------------------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------------------


def build_inspection(
    run_dir: str | Path,
    graph: CompiledGraph,
    *,
    change: Mapping[str, Any] | None = None,
    impact: Mapping[str, Any] | None = None,
    timeline: Mapping[str, Any] | None = None,
    compare_run: str | Path | None = None,
) -> dict[str, Any]:
    """Build the canonical JSON inspection document for one recorded run.

    The run directory is read only. The authored ``graph`` must already have been compiled
    and validated by the caller (the CLI uses :func:`modelgraph.compiler.compile_graph`).

    ``change``/``impact``/``timeline`` and ``compare_run`` are optional read-only iteration
    additions. They are passed in already loaded (the CLI uses
    :class:`modelgraph.revision.RevisionStore` in read-only mode) so this module never mutates a
    revision, a change set, or a run.
    """

    directory = Path(run_dir)
    manifest = read_run_manifest(directory)
    events, trace_note = _read_trace(directory, manifest)
    path = _path_entries(manifest)
    chosen_nodes = {entry["node"] for entry in path}
    chosen_edges = _chosen_edges(path, graph)

    prune = _prune_report(manifest)
    # A merged prune report can contain removals from nested subgraphs. Only removals that
    # belong to the rendered outer graph (or carry no graph label) mark nodes/edges here; the
    # full report is still shown in the pruning section.
    removed_nodes = {
        item["id"] for item in prune["removed_nodes"] if item["graph"] in (None, graph.id)
    }
    removed_edges = {
        (item["from"], item["to"])
        for item in prune["removed_edges"]
        if item["graph"] in (None, graph.id)
    }
    models = _model_records(manifest, graph)
    models_by_node = {item["node"]: item for item in models}

    stats = _trace_stats(events)
    failed_node = _failed_node(manifest, path)
    nodes = _node_records(
        graph,
        stats,
        path,
        chosen_nodes,
        models_by_node,
        prune,
        removed_nodes,
        failed_node,
        manifest,
    )
    edges = _edge_records(graph, chosen_edges, removed_edges, prune)
    artifacts, artifact_limits = _artifact_records(directory, manifest, graph)
    totals = _totals(manifest, nodes)
    cache = _cache_summary(events, manifest)
    budget = _budget_summary(manifest, totals)
    summary = _summary(manifest, graph, prune, models, budget, cache, totals, trace_note)

    document: dict[str, Any] = {
        "schema": INSPECTOR_SCHEMA,
        "summary": summary,
        "totals": totals,
        "cache": cache,
        "path": path,
        "graph": {
            "id": graph.id,
            "version": graph.version,
            "entry": graph.entry,
            "exits": list(graph.exits),
            "hash": graph.graph_hash,
            "nodes": _graph_node_records(graph, removed_nodes, chosen_nodes, prune),
            "edges": edges,
        },
        "nodes": nodes,
        "prune": prune,
        "artifacts": artifacts,
        "artifact_limits": artifact_limits,
    }
    if change is not None or impact is not None or timeline is not None:
        document["iteration"] = _iteration_section(change, impact, timeline)
    if compare_run is not None:
        document["compare"] = _compare_section(directory, Path(compare_run))
    return document


def render_inspector_html(model: Mapping[str, Any]) -> str:
    """Render an inspection document as one self-contained HTML file.

    The output has no external references. The embedded JSON block is escaped so data
    cannot close it, and the only inline script is the fixed filter/collapse controller
    below; it never assigns data to ``innerHTML``.
    """

    summary = cast(dict[str, Any], model["summary"])
    graph = cast(dict[str, Any], model["graph"])
    initials = _html_document_start(summary)
    body = [
        _warning_banner(),
        _render_summary(summary),
        _render_compare(model),
        _render_iteration(model),
        _render_totals(model),
        _render_path(model),
        _render_graph(graph),
        _render_prune(model),
        _render_nodes(model),
        _render_artifacts(model),
        _json_data_block(model),
        _filter_script(),
    ]
    # Optional iteration/compare sections return "" when absent. Filter them so the original
    # T-0028 invocation still produces byte-identical HTML.
    return "\n".join(
        [*initials, "<body>", *(section for section in body if section), "</body>", "</html>", ""]
    )


# --------------------------------------------------------------------------------------
# Input reading and normalisation
# --------------------------------------------------------------------------------------


def _read_trace(
    directory: Path, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str | None]:
    trace_name = manifest.get("trace", "trace.jsonl")
    if not isinstance(trace_name, str) or not trace_name:
        return [], "run manifest has no trace path"
    trace_path = _safe_child(directory, trace_name)
    if trace_path is None:
        return [], "run manifest trace path is unsafe; no trace was read"
    if not trace_path.is_file():
        return [], f"trace not found: {trace_name}"
    try:
        events = read_events(trace_path)
    except (OSError, ValueError) as exc:
        return [], f"trace is unreadable: {exc}"
    return events, None


def _path_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    raw_path = manifest.get("path")
    if not isinstance(raw_path, list):
        return entries
    for item in raw_path:
        if not isinstance(item, dict):
            continue
        entries.append(
            {
                "node": _text(item.get("node"), "(unknown node)"),
                "kind": _text(item.get("kind"), "unknown"),
                "decision": _optional_text(item.get("decision")),
                "to": _optional_text(item.get("to")),
                "confidence": _optional_float(item.get("confidence")),
                "reason": _optional_text(item.get("reason")),
                "graph": _optional_text(item.get("graph")),
            }
        )
    return entries


def _chosen_edges(path: Sequence[Mapping[str, Any]], graph: CompiledGraph) -> set[tuple[str, str]]:
    """Derive chosen edges from consecutive path entries, never guessing beyond them."""

    available = {(edge.from_, edge.to) for edge in graph.edges}
    chosen: set[tuple[str, str]] = set()
    for previous, current in zip(path, path[1:], strict=False):
        source = previous.get("node")
        target = current.get("node")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        recorded_target = previous.get("to")
        if recorded_target is not None and recorded_target != target:
            continue
        if (source, target) in available:
            chosen.add((source, target))
    return chosen


def _prune_report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("prune")
    document: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    removed_nodes: list[dict[str, Any]] = []
    for item in _mapping_list(document.get("removed_nodes")):
        removed_nodes.append(
            {
                "id": _text(item.get("id"), "(unknown node)"),
                "reason": _text(item.get("reason"), ""),
                "graph": _optional_text(item.get("graph")),
            }
        )
    removed_edges: list[dict[str, Any]] = []
    for item in _mapping_list(document.get("removed_edges")):
        removed_edges.append(
            {
                "from": _text(item.get("from"), "(unknown node)"),
                "to": _text(item.get("to"), "(unknown node)"),
                "reason": _text(item.get("reason"), ""),
                "graph": _optional_text(item.get("graph")),
            }
        )
    removed_exits: list[dict[str, Any]] = []
    for item in _mapping_list(document.get("removed_exit_artifacts")):
        removed_exits.append(
            {
                "name": _text(item.get("name"), "(unknown artifact)"),
                "reason": _text(item.get("reason"), ""),
                "graph": _optional_text(item.get("graph")),
            }
        )
    warnings = [
        _text(item, "") for item in _list(document.get("warnings")) if isinstance(item, str)
    ]
    return {
        "profile": _optional_text(document.get("profile")),
        "hash": _optional_text(manifest.get("prune_hash")),
        "removed_nodes": removed_nodes,
        "removed_edges": removed_edges,
        "removed_exit_artifacts": removed_exits,
        "warnings": warnings,
    }


def _model_records(manifest: Mapping[str, Any], graph: CompiledGraph) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw_models = manifest.get("models")
    if isinstance(raw_models, list):
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            records.append(
                {
                    "node": _text(item.get("node"), "(unknown node)"),
                    "ref": _optional_text(item.get("ref")),
                    "version": _optional_text(item.get("version")),
                    "hash": _optional_text(item.get("hash")),
                    "provider": _optional_text(item.get("provider")),
                    "model": _optional_text(item.get("model")),
                    "resolved_revision": _optional_text(item.get("resolved_revision")),
                    "revision_reason": _optional_text(item.get("revision_reason")),
                }
            )
    if not records:
        for node in sorted(graph.nodes.values(), key=lambda item: item.id):
            if node.model_ref is None:
                continue
            records.append(
                {
                    "node": node.id,
                    "ref": node.model_ref,
                    "version": node.model.metadata.version if node.model else None,
                    "hash": node.model_hash,
                    "provider": node.model.spec.provider.value if node.model else None,
                    "model": node.model.spec.model if node.model else None,
                    "resolved_revision": (
                        node.model.spec.integrity.revision if node.model else None
                    ),
                    "revision_reason": None,
                }
            )
    records.sort(key=lambda item: (str(item["node"]), str(item["ref"])))
    return records


def _graph_node_records(
    graph: CompiledGraph,
    removed_nodes: set[str],
    chosen_nodes: set[str],
    prune: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reasons = {
        item["id"]: item["reason"] for item in cast(list[dict[str, Any]], prune["removed_nodes"])
    }
    exits = set(graph.exits)
    records: list[dict[str, Any]] = []
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        wrote = {write.name for write in node.writes}
        records.append(
            {
                "id": node.id,
                "kind": node.kind.value,
                "description": node.description,
                "model_ref": node.model_ref,
                "model_hash": node.model_hash,
                "writes": sorted(wrote),
                "exit": bool(wrote & exits),
                "exit_artifacts": sorted(wrote & exits),
                "entry": node_id == graph.entry,
                "chosen": node_id in chosen_nodes,
                "pruned": node_id in removed_nodes,
                "prune_reason": reasons.get(node_id),
                "in_graph": True,
            }
        )
    return records


def _trace_stats(events: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for event in events:
        node = event.get("node")
        if not isinstance(node, str) or not node:
            continue
        entry = stats.setdefault(node, _empty_stats())
        kind = event.get("event")
        if kind == "node.entered":
            entry["entered"] += 1
            entry["kind"] = _optional_text(event.get("kind")) or entry["kind"]
        elif kind == "node.exited":
            entry["exited"] += 1
        elif kind == "decision.made":
            entry["decision"] = _optional_text(event.get("decision"))
            entry["confidence"] = _optional_float(event.get("confidence"))
            entry["reason"] = _optional_text(event.get("reason"))
            entry["target"] = _optional_text(event.get("to"))
        elif kind == "provider.request":
            entry["provider_calls"] += 1
            attempt = _as_int(event.get("attempt"))
            if attempt > 1:
                entry["retries"] += 1
            entry["provider"] = _optional_text(event.get("provider")) or entry["provider"]
            entry["model"] = _optional_text(event.get("model")) or entry["model"]
        elif kind == "provider.response":
            usage = event.get("usage")
            usage = usage if isinstance(usage, Mapping) else {}
            usage_tokens = _usage_tokens(usage)
            cached = bool(event.get("cached"))
            if cached:
                entry["cache_hits"] += 1
                entry["cache_tokens"] += usage_tokens
            else:
                entry["tokens"] += usage_tokens
                entry["latency_seconds"] += _as_float(event.get("latency_seconds"))
            entry["provider"] = _optional_text(event.get("provider")) or entry["provider"]
            entry["model"] = _optional_text(event.get("model")) or entry["model"]
        elif kind == "provider.response.cached":
            entry["cache_avoided_tokens"] += _as_int(event.get("avoided_tokens"))
            entry["cache_avoided_cost_usd"] += _as_float(event.get("avoided_cost_usd"))
        elif kind == "budget.charged":
            entry["charged_tokens"] += _as_int(event.get("tokens"))
            entry["cost_usd"] += _as_float(event.get("cost_usd"))
        elif kind == "provider.error":
            entry["errors"].append(
                {
                    "code": _optional_text(event.get("code")),
                    "message": _optional_text(event.get("message")),
                    "retryable": bool(event.get("retryable")),
                    "attempt": _as_int(event.get("attempt")),
                }
            )
    return stats


def _empty_stats() -> dict[str, Any]:
    return {
        "kind": None,
        "entered": 0,
        "exited": 0,
        "decision": None,
        "confidence": None,
        "reason": None,
        "target": None,
        "provider": None,
        "model": None,
        "provider_calls": 0,
        "retries": 0,
        "tokens": 0,
        "charged_tokens": 0,
        "cost_usd": 0.0,
        "latency_seconds": 0.0,
        "errors": [],
        "cache_hits": 0,
        "cache_tokens": 0,
        "cache_avoided_tokens": 0,
        "cache_avoided_cost_usd": 0.0,
    }


def _node_records(
    graph: CompiledGraph,
    stats: Mapping[str, dict[str, Any]],
    path: Sequence[Mapping[str, Any]],
    chosen_nodes: set[str],
    models_by_node: Mapping[str, Mapping[str, Any]],
    prune: Mapping[str, Any],
    removed_nodes: set[str],
    failed_node: str | None,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reasons = {
        item["id"]: item["reason"] for item in cast(list[dict[str, Any]], prune["removed_nodes"])
    }
    path_by_node: dict[str, Mapping[str, Any]] = {}
    for entry in path:
        path_by_node[str(entry["node"])] = entry
    run_error = _error_payload(manifest)
    records: list[dict[str, Any]] = []

    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        record = _node_record(
            node_id=node_id,
            kind=node.kind.value,
            description=node.description,
            in_graph=True,
            chosen=node_id in chosen_nodes,
            pruned=node_id in removed_nodes,
            prune_reason=reasons.get(node_id),
            stats=stats.get(node_id, _empty_stats()),
            path_entry=path_by_node.get(node_id),
            model_record=models_by_node.get(node_id),
            failed_node=failed_node,
            run_error=run_error,
            graph=graph,
        )
        records.append(record)

    known = set(graph.nodes)
    for node_id in sorted(set(stats) | {str(entry["node"]) for entry in path}):
        if node_id in known:
            continue
        entry = stats.get(node_id, _empty_stats())
        records.append(
            _node_record(
                node_id=node_id,
                kind=_optional_text(entry.get("kind")) or "trace",
                description="",
                in_graph=False,
                chosen=node_id in chosen_nodes,
                pruned=False,
                prune_reason=None,
                stats=entry,
                path_entry=path_by_node.get(node_id),
                model_record=None,
                failed_node=failed_node,
                run_error=run_error,
                graph=graph,
            )
        )
    return records


def _node_record(
    *,
    node_id: str,
    kind: str,
    description: str,
    in_graph: bool,
    chosen: bool,
    pruned: bool,
    prune_reason: str | None,
    stats: Mapping[str, Any],
    path_entry: Mapping[str, Any] | None,
    model_record: Mapping[str, Any] | None,
    failed_node: str | None,
    run_error: Mapping[str, Any] | None,
    graph: CompiledGraph,
) -> dict[str, Any]:
    executed = bool(stats.get("entered") or stats.get("provider_calls") or path_entry)
    errors = list(cast(list[dict[str, Any]], stats.get("errors") or []))
    if failed_node == node_id and run_error is not None:
        errors.append(
            {
                "code": run_error.get("code"),
                "message": run_error.get("message"),
                "retryable": False,
                "attempt": 0,
            }
        )
    if errors:
        status = "error"
    elif pruned and not executed:
        status = "pruned"
    elif executed:
        status = "ok"
    else:
        status = "not-run"

    decision = _optional_text(stats.get("decision"))
    reason = _optional_text(stats.get("reason"))
    target = _optional_text(stats.get("target"))
    confidence = _optional_float(stats.get("confidence"))
    if path_entry is not None:
        decision = decision or _optional_text(path_entry.get("decision"))
        reason = reason or _optional_text(path_entry.get("reason"))
        target = target or _optional_text(path_entry.get("to"))
        if confidence is None:
            confidence = _optional_float(path_entry.get("confidence"))

    model = model_record or {}
    if not model and node_id in graph.nodes:
        compiled = graph.nodes[node_id]
        model = {
            "provider": compiled.model.spec.provider.value if compiled.model else None,
            "model": compiled.model.spec.model if compiled.model else None,
            "ref": compiled.model_ref,
            "hash": compiled.model_hash,
        }

    tokens = _as_int(stats.get("charged_tokens")) or _as_int(stats.get("tokens"))
    return {
        "id": node_id,
        "kind": kind,
        "description": description,
        "in_graph": in_graph,
        "chosen": chosen,
        "pruned": pruned,
        "prune_reason": prune_reason,
        "executed": executed,
        "status": status,
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "target": target,
        "provider": _optional_text(stats.get("provider")) or _optional_text(model.get("provider")),
        "model": _optional_text(stats.get("model")) or _optional_text(model.get("model")),
        "model_ref": _optional_text(model.get("ref")),
        "model_hash": _optional_text(model.get("hash")),
        "tokens": tokens,
        "cost_usd": round(_as_float(stats.get("cost_usd")), 8),
        "latency_seconds": round(_as_float(stats.get("latency_seconds")), 4),
        "provider_calls": _as_int(stats.get("provider_calls")),
        "retries": _as_int(stats.get("retries")),
        "errors": errors,
        "cache_hits": _as_int(stats.get("cache_hits")),
        "cache_tokens": _as_int(stats.get("cache_tokens")),
        "cache_avoided_cost_usd": round(_as_float(stats.get("cache_avoided_cost_usd")), 8),
    }


def _edge_records(
    graph: CompiledGraph,
    chosen_edges: set[tuple[str, str]],
    removed_edges: set[tuple[str, str]],
    prune: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reasons: dict[tuple[str, str], str] = {}
    for item in cast(list[dict[str, Any]], prune["removed_edges"]):
        reasons[(item["from"], item["to"])] = item["reason"]
    records: list[dict[str, Any]] = []
    for edge in graph.edges:
        key = (edge.from_, edge.to)
        records.append(
            {
                "from": edge.from_,
                "to": edge.to,
                "when": edge.when,
                "priority": edge.priority,
                "chosen": key in chosen_edges,
                "pruned": key in removed_edges,
                "prune_reason": reasons.get(key),
            }
        )
    return records


def _artifact_records(
    directory: Path,
    manifest: Mapping[str, Any],
    graph: CompiledGraph,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_entries = _mapping_list(manifest.get("artifacts"))
    envelope: dict[str, Any] | None = None
    if raw_entries and any(not _artifact_file(directory, item).is_file() for item in raw_entries):
        envelope = _load_envelope(directory)
    envelope_by_hash, envelope_by_name = _envelope_contents(envelope)

    exits = set(graph.exits)
    artifacts: list[dict[str, Any]] = []
    used = 0
    for item in raw_entries:
        name = _text(item.get("name"), "(unnamed artifact)")
        artifact_hash = _optional_text(item.get("hash"))
        content_path = _artifact_file(directory, item)
        content_bytes: bytes | None = None
        original_bytes = _as_int(item.get("bytes"))
        if content_path.is_file():
            try:
                original_bytes = content_path.stat().st_size
                content_bytes = _read_prefix(content_path, MAX_ARTIFACT_BYTES + 4)
            except OSError:
                content_bytes = None
        if content_bytes is None and envelope is not None:
            fallback = None
            if artifact_hash:
                fallback = envelope_by_hash.get(artifact_hash)
            if fallback is None:
                fallback = envelope_by_name.get(name)
            if fallback is not None:
                content_bytes = fallback.encode("utf-8", "replace")
                original_bytes = max(original_bytes, len(content_bytes))

        missing = content_bytes is None
        raw = content_bytes or b""
        limit = min(MAX_ARTIFACT_BYTES, max(0, MAX_TOTAL_ARTIFACT_BYTES - used))
        truncated = not missing and (len(raw) > limit or original_bytes > len(raw))
        if missing:
            reason = None
        elif len(raw) > limit and limit <= 0:
            reason = "total artifact cap"
        elif len(raw) > limit:
            # The binding cap is whichever is smaller. A large artifact truncated after the
            # total budget was nearly exhausted is a total-cap truncation, not a per-artifact
            # one; naming the wrong cap sends users to the wrong remedy.
            reason = "per-artifact cap" if limit == MAX_ARTIFACT_BYTES else "total artifact cap"
        elif original_bytes > len(raw):
            reason = "per-artifact cap"
        else:
            reason = None
        shown = raw[:limit] if limit >= 0 else b""
        content = _decode_prefix(shown)
        embedded_bytes = len(content.encode("utf-8"))
        used += embedded_bytes
        artifacts.append(
            {
                "name": name,
                "type": _text(item.get("type"), "unknown"),
                "media_type": _optional_text(item.get("media_type")),
                "producer": _text(item.get("producer"), ""),
                "hash": artifact_hash,
                "exit": bool(item.get("exit")) or name in exits,
                "manifest_bytes": _as_int(item.get("bytes")),
                "original_bytes": original_bytes,
                "content": content,
                "embedded_bytes": embedded_bytes,
                "missing": missing,
                "truncated": truncated,
                "truncation_reason": reason,
            }
        )

    artifacts.sort(key=lambda item: (str(item["name"]), str(item["hash"]), str(item["producer"])))
    limits = {
        "per_artifact_bytes": MAX_ARTIFACT_BYTES,
        "total_bytes": MAX_TOTAL_ARTIFACT_BYTES,
        "embedded_bytes": used,
        "truncated": any(item["truncated"] for item in artifacts),
    }
    return artifacts, limits


def _artifact_file(directory: Path, item: Mapping[str, Any]) -> Path:
    value = item.get("hash")
    if not isinstance(value, str) or not value:
        return directory / "artifacts" / "__missing__"
    digest = value.split(":", 1)[-1]
    if not _SAFE_COMPONENT.match(digest):
        return directory / "artifacts" / "__unsafe__"
    return directory / "artifacts" / digest


def _read_prefix(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit)


def _load_envelope(directory: Path) -> dict[str, Any] | None:
    path = directory / "envelope.json"
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _envelope_contents(
    envelope: Mapping[str, Any] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    by_hash: dict[str, str] = {}
    by_name: dict[str, str] = {}
    if envelope is None:
        return by_hash, by_name
    for key in ("artifacts", "inputs"):
        entries = envelope.get(key)
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, str):
                continue
            name = _optional_text(item.get("name"))
            digest = _optional_text(item.get("hash"))
            if name:
                by_name.setdefault(name, content)
            if digest:
                by_hash.setdefault(digest, content)
    return by_hash, by_name


def _totals(manifest: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usage = manifest.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    trace_cost = sum(_as_float(node.get("cost_usd")) for node in nodes)
    trace_tokens = sum(_as_int(node.get("tokens")) for node in nodes)
    trace_latency = sum(_as_float(node.get("latency_seconds")) for node in nodes)
    trace_steps = sum(1 for node in nodes if node.get("executed"))
    input_tokens = _as_int(usage.get("input_tokens")) if "input_tokens" in usage else 0
    output_tokens = _as_int(usage.get("output_tokens")) if "output_tokens" in usage else 0
    total_tokens = (
        _as_int(usage.get("total_tokens"))
        if "total_tokens" in usage
        else input_tokens + output_tokens
    )
    if "total_tokens" not in usage and "input_tokens" not in usage and "output_tokens" not in usage:
        total_tokens = trace_tokens
    return {
        "steps": _as_int(usage.get("steps")) if "steps" in usage else trace_steps,
        "input_tokens": input_tokens or max(0, total_tokens - output_tokens),
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(
            _as_float(usage.get("cost_usd")) if "cost_usd" in usage else trace_cost, 8
        ),
        "elapsed_seconds": round(
            _as_float(usage.get("elapsed_seconds"))
            if "elapsed_seconds" in usage
            else trace_latency,
            3,
        ),
        "source": "manifest" if usage else "trace",
        "trace_sums": {
            "steps": trace_steps,
            "total_tokens": trace_tokens,
            "cost_usd": round(trace_cost, 8),
            "latency_seconds": round(trace_latency, 3),
        },
    }


def _cache_summary(
    events: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    hits = 0
    misses = 0
    skipped = 0
    avoided_tokens = 0
    avoided_cost = 0.0
    for event in events:
        kind = event.get("event")
        if kind == "provider.response" and bool(event.get("cached")):
            hits += 1
        elif kind == "provider.response.cached":
            avoided_tokens += _as_int(event.get("avoided_tokens"))
            avoided_cost += _as_float(event.get("avoided_cost_usd"))
        elif kind == "cache.miss":
            misses += 1
        elif kind == "cache.skipped":
            skipped += 1
    raw = manifest.get("cache")
    if isinstance(raw, Mapping):
        hits = max(hits, _as_int(raw.get("hits")))
        misses = max(misses, _as_int(raw.get("misses")))
        skipped = max(skipped, _as_int(raw.get("skipped")))
        avoided_tokens = max(avoided_tokens, _as_int(raw.get("avoided_tokens")))
        avoided_cost = max(avoided_cost, _as_float(raw.get("avoided_cost_usd")))
    return {
        "available": bool(hits or misses or skipped or avoided_tokens or avoided_cost),
        "hits": hits,
        "misses": misses,
        "skipped": skipped,
        "avoided_tokens": avoided_tokens,
        "avoided_cost_usd": round(avoided_cost, 8),
    }


def _budget_summary(manifest: Mapping[str, Any], totals: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("budget")
    raw = raw if isinstance(raw, Mapping) else {}
    return {
        "max_steps": _as_int(raw.get("max_steps")),
        "used_steps": _as_int(raw.get("used_steps")) if "used_steps" in raw else totals["steps"],
        "max_depth": _as_int(raw.get("max_depth")),
        "max_total_tokens": _as_int(raw.get("max_total_tokens")),
        "used_tokens": (
            _as_int(raw.get("used_tokens")) if "used_tokens" in raw else totals["total_tokens"]
        ),
        "max_cost_usd": round(_as_float(raw.get("max_cost_usd")), 8),
        "used_cost_usd": (
            round(_as_float(raw.get("used_cost_usd")), 8)
            if "used_cost_usd" in raw
            else totals["cost_usd"]
        ),
        "max_wall_seconds": round(_as_float(raw.get("max_wall_seconds")), 3),
        "elapsed_seconds": (
            round(_as_float(raw.get("elapsed_seconds")), 3)
            if "elapsed_seconds" in raw
            else totals["elapsed_seconds"]
        ),
        "on_budget_exhausted": _optional_text(raw.get("on_budget_exhausted")),
    }


def _summary(
    manifest: Mapping[str, Any],
    graph: CompiledGraph,
    prune: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    budget: Mapping[str, Any],
    cache: Mapping[str, Any],
    totals: Mapping[str, Any],
    trace_note: str | None,
) -> dict[str, Any]:
    graph_info = manifest.get("graph")
    graph_info = graph_info if isinstance(graph_info, Mapping) else {}
    profile = manifest.get("profile")
    profile = profile if isinstance(profile, Mapping) else None
    environment = manifest.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    manifest_hash = _optional_text(graph_info.get("hash"))
    return {
        "run_id": _text(manifest.get("run_id"), "(unknown run)"),
        "status": _text(manifest.get("status"), "unknown"),
        "outcome": _text(manifest.get("outcome"), "unknown"),
        "target": _optional_text(manifest.get("target")),
        "seed": manifest.get("seed"),
        "graph_id": _text(graph_info.get("id"), graph.id),
        "graph_version": _text(graph_info.get("version"), graph.version),
        "graph_hash": manifest_hash,
        "compiled_graph_hash": graph.graph_hash,
        "graph_hash_matches": manifest_hash == graph.graph_hash,
        "deployment_revision": _optional_text(manifest.get("deployment_revision")),
        "renderer_version": _optional_text(manifest.get("renderer_version")),
        "parser_version": _optional_text(manifest.get("parser_version")),
        "ir_version": _optional_text(manifest.get("ir_version")),
        "started_at": _optional_text(manifest.get("started_at")),
        "finished_at": _optional_text(manifest.get("finished_at")),
        "profile_name": _optional_text(profile.get("name")) if profile else None,
        "profile_hash": _optional_text(manifest.get("profile_hash")),
        "prune_profile": prune.get("profile"),
        "prune_hash": _optional_text(manifest.get("prune_hash")),
        "budget": dict(budget),
        "cache": dict(cache),
        "totals": dict(totals),
        "models": [dict(item) for item in models],
        "error": _error_payload(manifest),
        "trace_note": trace_note,
        "environment": {
            "modelgraph_version": _optional_text(environment.get("modelgraph_version")),
            "python_version": _optional_text(environment.get("python_version")),
            "platform": _optional_text(environment.get("platform")),
        },
        "warnings": _summary_warnings(manifest, graph, prune, trace_note),
    }


def _summary_warnings(
    manifest: Mapping[str, Any],
    graph: CompiledGraph,
    prune: Mapping[str, Any],
    trace_note: str | None,
) -> list[str]:
    warnings: list[str] = []
    if trace_note:
        warnings.append(trace_note)
    removed_nodes = len(cast(list[Any], prune["removed_nodes"]))
    if removed_nodes or cast(list[Any], prune["removed_edges"]):
        warnings.append(
            "The recorded run applied pruning; pruned nodes and edges are marked in the graph."
        )
    graph_info = manifest.get("graph")
    graph_info = graph_info if isinstance(graph_info, Mapping) else {}
    if graph_info.get("hash") != graph.graph_hash:
        warnings.append(
            "The authored graph hash differs from the recorded run graph hash. This is expected "
            "when the run used a pruned graph; pruning is shown from the run manifest."
        )
    if manifest.get("outcome") not in (None, "succeeded"):
        warnings.append("This run did not succeed; partial evidence is shown.")
    return warnings


def _failed_node(manifest: Mapping[str, Any], path: Sequence[Mapping[str, Any]]) -> str | None:
    if manifest.get("outcome") in (None, "succeeded"):
        return None
    if manifest.get("error") is None:
        return None
    if not path:
        return None
    node = path[-1].get("node")
    return node if isinstance(node, str) else None


def _error_payload(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = manifest.get("error")
    if not isinstance(raw, Mapping):
        return None
    details = raw.get("details")
    return {
        "code": _optional_text(raw.get("code")),
        "message": _optional_text(raw.get("message")),
        "path": _optional_text(raw.get("path")),
        "details": details if isinstance(details, Mapping) else None,
    }


# --------------------------------------------------------------------------------------
# Iteration additions: impact panel, stale badges, revision timeline, and run comparison
# --------------------------------------------------------------------------------------


def _iteration_section(
    change: Mapping[str, Any] | None,
    impact: Mapping[str, Any] | None,
    timeline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    change_record = _change_record(change or {})
    impact_record = _impact_record(impact or {})
    timeline_record = _timeline_record(timeline or {})
    return {
        "schema": INSPECTOR_ITERATION_SCHEMA,
        "change": change_record,
        "impact": impact_record,
        "stale_artifacts": _stale_artifacts(impact_record, timeline_record),
        "timeline": timeline_record,
        "safe_actions": _safe_actions(change_record, impact_record, timeline_record),
        "notes": [
            "This panel is a read-only projection of a stored impact report. It does not "
            "promote, roll back, recompile, or re-distill anything.",
            "Compiled plans, prefix/route caches, and distilled students do not exist in v0. "
            "They are marked not_applicable or none instead of pretending an artifact hash.",
            "Follow-up commands are rendered as text. The inspector never executes a command.",
        ],
    }


def _change_record(change: Mapping[str, Any]) -> dict[str, Any]:
    graph = change.get("candidate_graph")
    graph = cast(Mapping[str, Any], graph) if isinstance(graph, Mapping) else {}
    evidence = [
        {
            "path": _optional_text(item.get("path")),
            "level": _optional_text(item.get("level")),
            "valid": bool(item.get("valid")),
            "gate_passed": bool(item.get("gate_passed")),
            "errors": [
                _text(error, "") for error in _list(item.get("errors")) if isinstance(error, str)
            ],
        }
        for item in _mapping_list(change.get("evidence"))
    ]
    return {
        "change_id": _optional_text(change.get("change_id")),
        "status": _optional_text(change.get("status")),
        "change_class": _optional_text(change.get("change_class")),
        "target": _optional_text(change.get("target")),
        "base_target": _optional_text(change.get("base_target")),
        "base_revision": _optional_text(change.get("base_revision")),
        "candidate_revision": _optional_text(change.get("candidate_revision")),
        "summary": _optional_text(change.get("summary")),
        "candidate_graph": {
            "id": _optional_text(graph.get("id")),
            "version": _optional_text(graph.get("version")),
            "hash": _optional_text(graph.get("hash")),
            "source": _optional_text(graph.get("source")),
        },
        "evidence": evidence,
    }


def _impact_record(impact: Mapping[str, Any]) -> dict[str, Any]:
    compatibility = impact.get("compatibility")
    compatibility = compatibility if isinstance(compatibility, Mapping) else {}
    evaluation = impact.get("eval")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    coverage = evaluation.get("coverage")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    vram = impact.get("vram")
    vram = vram if isinstance(vram, Mapping) else {}
    admission = vram.get("admission")
    admission = admission if isinstance(admission, Mapping) else {}
    rollback = impact.get("rollback")
    rollback = rollback if isinstance(rollback, Mapping) else {}
    affected = impact.get("affected")
    affected = affected if isinstance(affected, Mapping) else {}
    checks = [
        {
            "node": _text(item.get("node"), ""),
            "from_ref": _optional_text(item.get("from_ref")),
            "to_ref": _optional_text(item.get("to_ref")),
            "base": _identity(item.get("base")),
            "tokenizer": _identity(item.get("tokenizer")),
            "context_window": _identity(item.get("context_window")),
            "blocking": bool(item.get("blocking")),
            "message": _optional_text(item.get("message")),
        }
        for item in _mapping_list(compatibility.get("checks"))
    ]
    invalidations = [
        {
            "artifact": _text(item.get("artifact"), "artifact"),
            "status": _text(item.get("status"), "unknown"),
            "applicable": bool(item.get("applicable")),
            "present": bool(item.get("present")),
            "reason": _text(item.get("reason"), ""),
            "scope": [
                _text(entry, "") for entry in _list(item.get("scope")) if isinstance(entry, str)
            ],
        }
        for item in _mapping_list(impact.get("invalidations"))
    ]
    routers = [
        {
            "router": _text(item.get("router"), ""),
            "reasons": [
                _text(reason, "")
                for reason in _list(item.get("reasons"))
                if isinstance(reason, str)
            ],
        }
        for item in _mapping_list(affected.get("routers"))
    ]
    return {
        "change_class": _optional_text(impact.get("change_class")),
        "change_class_description": _optional_text(impact.get("change_class_description")),
        "target": _optional_text(impact.get("target")),
        "base_target": _optional_text(impact.get("base_target")),
        "base_revision": _optional_text(impact.get("base_revision")),
        "candidate_revision": _optional_text(impact.get("candidate_revision")),
        "compatibility": {
            "status": _text(compatibility.get("status"), "unknown"),
            "blocking": bool(compatibility.get("blocking")),
            "message": _optional_text(compatibility.get("message")),
            "checks": checks,
        },
        "eval": {
            "required": [
                _text(item, "")
                for item in _list(evaluation.get("required"))
                if isinstance(item, str)
            ],
            "satisfied": [
                _text(item, "")
                for item in _list(coverage.get("satisfied"))
                if isinstance(item, str)
            ],
            "missing": [
                _text(item, "") for item in _list(coverage.get("missing")) if isinstance(item, str)
            ],
            "complete": bool(coverage.get("complete")),
        },
        "invalidations": invalidations,
        "vram": {
            "before_gb": _optional_float(vram.get("before_gb")),
            "after_gb": _optional_float(vram.get("after_gb")),
            "delta_gb": _optional_float(vram.get("delta_gb")),
            "complete": bool(vram.get("complete")),
            "unknown_models": [
                _text(item, "")
                for item in _list(vram.get("unknown_models"))
                if isinstance(item, str)
            ],
            "admission": {
                "status": _text(admission.get("status"), "unknown"),
                "profile": _optional_text(admission.get("profile")),
                "max_vram_gb": _optional_float(admission.get("max_vram_gb")),
                "required_gb": _optional_float(admission.get("required_gb")),
                "headroom_gb": _optional_float(admission.get("headroom_gb")),
                "reason": _optional_text(admission.get("reason")),
            },
        },
        "rollback": {
            "target": _optional_text(rollback.get("target")),
            "previous": _optional_text(rollback.get("previous")),
            "retained": (
                bool(rollback.get("retained")) if rollback.get("retained") is not None else None
            ),
            "grace_days": _optional_float(rollback.get("grace_days")),
            "note": _optional_text(rollback.get("note")),
        },
        "affected": {
            "nodes": [
                _text(item, "") for item in _list(affected.get("nodes")) if isinstance(item, str)
            ],
            "routers": routers,
            "routing_cards": [
                _text(item, "")
                for item in _list(affected.get("routing_cards"))
                if isinstance(item, str)
            ],
            "route_menu_changed": bool(affected.get("route_menu_changed")),
        },
        "risk": [_text(item, "") for item in _list(impact.get("risk")) if isinstance(item, str)],
        "notes": [_text(item, "") for item in _list(impact.get("notes")) if isinstance(item, str)],
    }


def _identity(value: object) -> dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    return {
        "from": _optional_text(raw.get("from")),
        "to": _optional_text(raw.get("to")),
        "status": _optional_text(raw.get("status")),
        "source": _optional_text(raw.get("source")),
    }


def _timeline_record(timeline: Mapping[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for item in _mapping_list(timeline.get("entries")):
        components = item.get("components")
        components = components if isinstance(components, Mapping) else {}
        markers = [
            {
                "event": _text(marker.get("event"), ""),
                "ts": _optional_text(marker.get("ts")),
                "reason": _optional_text(marker.get("reason")),
                "previous_revision": _optional_text(marker.get("previous_revision")),
            }
            for marker in _mapping_list(item.get("markers"))
        ]
        entries.append(
            {
                "revision_id": _text(item.get("revision_id"), ""),
                "graph_id": _optional_text(item.get("graph_id")),
                "graph_hash": _optional_text(item.get("graph_hash")),
                "target": _optional_text(item.get("target")),
                "created_at": _optional_text(item.get("created_at")),
                "bootstrap": bool(item.get("bootstrap")),
                "roles": [
                    _text(role, "") for role in _list(item.get("roles")) if isinstance(role, str)
                ],
                "active": bool(item.get("active")),
                "previous": bool(item.get("previous")),
                "candidate": bool(item.get("candidate")),
                "stale": bool(item.get("stale")),
                "missing": bool(item.get("missing")),
                "components": {
                    "plan_status": _optional_text(components.get("plan_status")),
                    "plan_hash": _optional_text(components.get("plan_hash")),
                    "student_hash": _optional_text(components.get("student_hash")),
                },
                "markers": markers,
            }
        )
    return {
        "schema": _optional_text(timeline.get("schema")),
        "store": _optional_text(timeline.get("store")),
        "active_revision": _optional_text(timeline.get("active_revision")),
        "previous_revision": _optional_text(timeline.get("previous_revision")),
        "candidate_revision": _optional_text(timeline.get("candidate_revision")),
        "entries": entries,
        "notes": [
            _text(item, "") for item in _list(timeline.get("notes")) if isinstance(item, str)
        ],
    }


def _stale_artifacts(
    impact: Mapping[str, Any], timeline: Mapping[str, Any]
) -> list[dict[str, Any]]:
    candidate_id = _optional_text(impact.get("candidate_revision"))
    components: Mapping[str, Any] = {}
    source_hash: str | None = None
    for entry in _mapping_list(timeline.get("entries")):
        if _optional_text(entry.get("revision_id")) == candidate_id:
            components = cast(Mapping[str, Any], entry.get("components") or {})
            source_hash = _optional_text(entry.get("graph_hash"))
            break
    artifacts: list[dict[str, Any]] = []
    for item in _mapping_list(impact.get("invalidations")):
        artifact = _text(item.get("artifact"), "artifact")
        status = _text(item.get("status"), "unknown")
        applicable = bool(item.get("applicable"))
        present = bool(item.get("present"))
        if not applicable:
            badge = "not_applicable"
        elif status == "stale":
            badge = "stale"
        elif status == "invalidated":
            badge = "invalidated"
        elif present:
            badge = "fresh"
        else:
            badge = "none"
        artifact_hash: str | None = None
        if artifact == "distilled_student":
            artifact_hash = _optional_text(components.get("student_hash"))
        elif artifact in {"compiled_plan", "prefix_cache", "route_cache"}:
            artifact_hash = _optional_text(components.get("plan_hash"))
        artifacts.append(
            {
                "artifact": artifact,
                "status": status,
                "applicable": applicable,
                "present": present,
                "present_label": "present" if present else "none",
                "badge": badge,
                "reason": _text(item.get("reason"), ""),
                "scope": [
                    _text(entry, "") for entry in _list(item.get("scope")) if isinstance(entry, str)
                ],
                "source_hash": source_hash,
                "artifact_hash": artifact_hash,
                "hash_basis": (
                    "stale/invalidated comes from graph/model/prompt hash keys in the impact "
                    "report; v0 records no plan or student artifact hash"
                ),
            }
        )
    return artifacts


def _safe_actions(
    change: Mapping[str, Any],
    impact: Mapping[str, Any],
    timeline: Mapping[str, Any],
) -> list[dict[str, Any]]:
    change_id = _optional_text(change.get("change_id"))
    candidate = _optional_text(change.get("candidate_revision")) or _optional_text(
        impact.get("candidate_revision")
    )
    store = _optional_text(timeline.get("store"))
    graph = change.get("candidate_graph")
    graph = graph if isinstance(graph, Mapping) else {}
    source = _optional_text(graph.get("source"))
    store_arg = f" --store {store}" if store else ""
    actions: list[dict[str, Any]] = []
    if change_id:
        actions.append(
            {
                "action": "open_change",
                "label": "Open change set (read-only)",
                "command": f"mg change explain {change_id}{store_arg}",
                "available": True,
                "mutates": False,
                "note": "Open the change set read-only; this does not promote or mutate it.",
            }
        )
    if candidate:
        actions.append(
            {
                "action": "promote_dry_run",
                "label": "Validate promotion without switching",
                "command": f"mg promote --revision {candidate}{store_arg} --dry-run",
                "available": True,
                "mutates": False,
                "note": "Checks compatibility and evidence coverage; --dry-run does not write.",
            }
        )
    actions.append(
        {
            "action": "recompile",
            "label": "Recompile authored graph to IR",
            "command": f"mg compile {source} --json" if source else None,
            "available": bool(source),
            "mutates": False,
            "note": (
                "v0 compiles the authored graph to canonical IR. It does not build a compiled "
                "plan, prefix cache, or route cache."
            ),
        }
    )
    actions.append(
        {
            "action": "rebuild_plan",
            "label": "Rebuild compiled plan",
            "command": None,
            "available": False,
            "mutates": False,
            "note": (
                "Not applicable in v0: no compiled/fused target builds a plan artifact. The "
                "impact panel marks the plan not_applicable or not_built."
            ),
        }
    )
    actions.append(
        {
            "action": "redistill",
            "label": "Re-distill student",
            "command": None,
            "available": False,
            "mutates": False,
            "note": (
                "Not applicable in v0: no distilled student artifact exists to rebuild. A "
                "future student target must add the command and artifact first."
            ),
        }
    )
    return actions


def _compare_section(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    left = _run_snapshot(left_dir)
    right = _run_snapshot(right_dir)
    same_inputs: bool | None
    if left["task_hash"] is None or right["task_hash"] is None:
        same_inputs = None
    else:
        same_inputs = left["task_hash"] == right["task_hash"]
    same_graph: bool | None
    if left["graph_hash"] is None or right["graph_hash"] is None:
        # Two missing hashes are not evidence that the graphs are the same. Report unknown
        # rather than claiming a same-graph comparison the manifests cannot support.
        same_graph = None
    else:
        same_graph = left["graph_hash"] == right["graph_hash"]
    warnings: list[str] = []
    if same_inputs is False:
        warnings.append(
            "The recorded task/request hash differs; this is not a same-input comparison."
        )
    elif same_inputs is None:
        warnings.append(
            "One or both runs do not record a task/request hash; same-input status is unknown."
        )
    if same_graph is False:
        warnings.append(
            "The recorded graph hashes differ; path and node differences may come from the "
            "graph, not only from the provider."
        )
    elif same_graph is None:
        warnings.append(
            "One or both runs do not record a graph hash; same-graph status is unknown."
        )
    for side, snapshot in (("left", left), ("right", right)):
        if snapshot["trace_note"]:
            warnings.append(f"{side}: {snapshot['trace_note']}")
    path_rows = _path_rows(left["path"], right["path"])
    node_rows = _node_compare_rows(left["nodes"], right["nodes"])
    artifact_rows = _artifact_compare_rows(left["artifacts"], right["artifacts"])
    totals_rows = _totals_compare_rows(left["totals"], right["totals"])
    return {
        "schema": INSPECTOR_COMPARE_SCHEMA,
        "left": _compare_side(left),
        "right": _compare_side(right),
        "same_inputs": same_inputs,
        "same_graph": same_graph,
        "warnings": warnings,
        "path": {
            "same": all(row["same"] for row in path_rows),
            "rows": path_rows,
        },
        "nodes": {
            "same": all(row["same"] for row in node_rows),
            "rows": node_rows,
        },
        "artifacts": {
            "same": all(row["same"] for row in artifact_rows),
            "rows": artifact_rows,
        },
        "totals": {
            "same": all(row["same"] for row in totals_rows),
            "rows": totals_rows,
        },
        "summary": {
            "path_same": all(row["same"] for row in path_rows),
            "node_rows": len(node_rows),
            "node_differences": sum(1 for row in node_rows if not row["same"]),
            "artifact_rows": len(artifact_rows),
            "artifact_differences": sum(1 for row in artifact_rows if not row["same"]),
        },
    }


def _compare_side(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": snapshot.get("run_id"),
        "outcome": snapshot.get("outcome"),
        "status": snapshot.get("status"),
        "graph_id": snapshot.get("graph_id"),
        "graph_hash": snapshot.get("graph_hash"),
        "target": snapshot.get("target"),
        "seed": snapshot.get("seed"),
        "task_hash": snapshot.get("task_hash"),
        "path": snapshot.get("path"),
        "totals": snapshot.get("totals"),
    }


def _run_snapshot(directory: Path) -> dict[str, Any]:
    manifest = read_run_manifest(directory)
    events, trace_note = _read_trace(directory, manifest)
    stats = _trace_stats(events)
    path = _path_entries(manifest)
    graph = manifest.get("graph")
    graph = graph if isinstance(graph, Mapping) else {}
    request = manifest.get("request")
    request_hash = (
        sha256_text(canonical_json(dict(request)))
        if isinstance(request, Mapping) and request
        else None
    )
    models: dict[str, Mapping[str, Any]] = {}
    for item in _mapping_list(manifest.get("models")):
        models[_text(item.get("node"), "")] = item
    path_by_node = {str(entry["node"]): entry for entry in path}
    node_ids = sorted(set(stats) | set(path_by_node) | set(models))
    nodes: list[dict[str, Any]] = []
    for node_id in node_ids:
        entry = stats.get(node_id, {})
        path_entry = path_by_node.get(node_id)
        model = models.get(node_id, {})
        executed = bool(entry.get("entered") or entry.get("provider_calls") or path_entry)
        errors = cast(list[dict[str, Any]], entry.get("errors") or [])
        status = "error" if errors else ("ok" if executed else "not-run")
        decision = _optional_text(entry.get("decision")) or (
            _optional_text(path_entry.get("decision")) if path_entry else None
        )
        target = _optional_text(entry.get("target")) or (
            _optional_text(path_entry.get("to")) if path_entry else None
        )
        nodes.append(
            {
                "node": node_id,
                "provider": _optional_text(entry.get("provider"))
                or _optional_text(model.get("provider")),
                "model": _optional_text(entry.get("model")) or _optional_text(model.get("model")),
                "tokens": (_as_int(entry.get("charged_tokens")) or _as_int(entry.get("tokens")))
                if executed
                else None,
                "cost_usd": round(_as_float(entry.get("cost_usd")), 8) if executed else None,
                "latency_seconds": (
                    round(_as_float(entry.get("latency_seconds")), 4) if executed else None
                ),
                "status": status,
                "decision": decision,
                "to": target,
                "cache_hits": _as_int(entry.get("cache_hits")),
            }
        )
    artifacts = [
        {
            "name": _text(item.get("name"), "(unnamed artifact)"),
            "hash": _optional_text(item.get("hash")),
            "type": _text(item.get("type"), "unknown"),
            "producer": _text(item.get("producer"), ""),
            "exit": bool(item.get("exit")),
            "bytes": _as_int(item.get("bytes")),
        }
        for item in _mapping_list(manifest.get("artifacts"))
    ]
    usage = manifest.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    return {
        "run_id": _text(manifest.get("run_id"), "(unknown run)"),
        "outcome": _text(manifest.get("outcome"), "unknown"),
        "status": _text(manifest.get("status"), "unknown"),
        "graph_id": _optional_text(graph.get("id")),
        "graph_hash": _optional_text(graph.get("hash")),
        "target": _optional_text(manifest.get("target")),
        "seed": manifest.get("seed"),
        "task_hash": request_hash,
        "trace_note": trace_note,
        "path": path,
        "nodes": nodes,
        "artifacts": artifacts,
        "totals": {
            "steps": _as_int(usage.get("steps")),
            "total_tokens": _as_int(usage.get("total_tokens")),
            "cost_usd": round(_as_float(usage.get("cost_usd")), 8),
            "elapsed_seconds": round(_as_float(usage.get("elapsed_seconds")), 3),
        },
    }


def _path_rows(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(max(len(left), len(right))):
        left_entry = dict(left[index]) if index < len(left) else None
        right_entry = dict(right[index]) if index < len(right) else None
        rows.append(
            {
                "index": index + 1,
                "left": left_entry,
                "right": right_entry,
                "same": left_entry == right_entry,
            }
        )
    return rows


def _node_compare_rows(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    left_by_id = {str(item.get("node")): item for item in left}
    right_by_id = {str(item.get("node")): item for item in right}
    fields = (
        "provider",
        "model",
        "tokens",
        "cost_usd",
        "latency_seconds",
        "status",
        "decision",
        "to",
        "cache_hits",
    )
    rows: list[dict[str, Any]] = []
    for node_id in sorted(set(left_by_id) | set(right_by_id)):
        left_node = left_by_id.get(node_id)
        right_node = right_by_id.get(node_id)
        differences = [
            field
            for field in fields
            if (left_node or {}).get(field) != (right_node or {}).get(field)
        ]
        rows.append(
            {
                "node": node_id,
                "left": dict(left_node) if left_node else None,
                "right": dict(right_node) if right_node else None,
                "missing_left": left_node is None,
                "missing_right": right_node is None,
                "same": not differences,
                "differences": differences,
            }
        )
    return rows


def _artifact_compare_rows(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    left_by_name: dict[str, list[Mapping[str, Any]]] = {}
    right_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for item in left:
        left_by_name.setdefault(str(item.get("name")), []).append(item)
    for item in right:
        right_by_name.setdefault(str(item.get("name")), []).append(item)
    rows: list[dict[str, Any]] = []
    for name in sorted(set(left_by_name) | set(right_by_name)):
        left_items = sorted(left_by_name.get(name, []), key=lambda item: str(item.get("hash")))
        right_items = sorted(right_by_name.get(name, []), key=lambda item: str(item.get("hash")))
        left_hashes = [item.get("hash") for item in left_items]
        right_hashes = [item.get("hash") for item in right_items]
        rows.append(
            {
                "name": name,
                "left": [dict(item) for item in left_items],
                "right": [dict(item) for item in right_items],
                "left_hash": ", ".join(str(value) for value in left_hashes) or None,
                "right_hash": ", ".join(str(value) for value in right_hashes) or None,
                "missing_left": not left_items,
                "missing_right": not right_items,
                "same": left_hashes == right_hashes and bool(left_items) and bool(right_items),
            }
        )
    return rows


def _totals_compare_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in ("steps", "total_tokens", "cost_usd", "elapsed_seconds"):
        rows.append(
            {
                "field": field,
                "left": left.get(field),
                "right": right.get(field),
                "same": left.get(field) == right.get(field),
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------------------


def _html_document_start(summary: Mapping[str, Any]) -> list[str]:
    title = f"Model Graph inspector — {_esc(summary.get('run_id'))}"
    return [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="referrer" content="no-referrer">',
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src 'none'; "
        "font-src 'none'; connect-src 'none'\">",
        f"<title>{title}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
    ]


def _warning_banner() -> str:
    return (
        '<div class="banner banner-warn" role="alert">'
        "<strong>Embedded data warning:</strong> this HTML file contains artifact contents, "
        "run metadata, and error text from the recorded run. Artifact content is capped and "
        "escaped, but the file can still hold sensitive data. Treat it as sensitive and do not "
        "publish it. The file is self-contained and makes no network requests."
        "</div>"
    )


def _render_summary(summary: Mapping[str, Any]) -> str:
    warnings = cast(list[str], summary.get("warnings") or [])
    warning_html = ""
    if warnings:
        warning_html = (
            '<div class="banner banner-note" role="status"><ul>'
            + "".join(f"<li>{_esc(item)}</li>" for item in warnings)
            + "</ul></div>"
        )
    rows = [
        ("Run id", _code(summary.get("run_id"))),
        ("Status", _esc(summary.get("status"))),
        ("Outcome", _esc(summary.get("outcome"))),
        ("Graph", _esc(summary.get("graph_id"))),
        ("Graph version", _esc(summary.get("graph_version"))),
        ("Recorded graph hash", _code(summary.get("graph_hash"))),
        ("Compiled authored graph hash", _code(summary.get("compiled_graph_hash"))),
        (
            "Graph hash match",
            "yes" if summary.get("graph_hash_matches") else "no (authored graph differs)",
        ),
        ("Deployment revision", _code(summary.get("deployment_revision"))),
        ("Renderer / parser / IR", _join_versions(summary)),
        ("Started", _esc(summary.get("started_at"))),
        ("Finished", _esc(summary.get("finished_at"))),
        ("Seed", _esc(summary.get("seed"))),
        ("Target", _esc(summary.get("target"))),
        ("Profile", _esc(summary.get("profile_name"))),
        ("Profile hash", _code(summary.get("profile_hash"))),
        ("Prune profile", _esc(summary.get("prune_profile"))),
        ("Prune hash", _code(summary.get("prune_hash"))),
    ]
    modelgraph_version = cast(Mapping[str, Any], summary.get("environment") or {}).get(
        "modelgraph_version"
    )
    if modelgraph_version:
        rows.append(("Model Graph version", _esc(modelgraph_version)))
    error = summary.get("error")
    error_html = ""
    if isinstance(error, Mapping):
        details = error.get("details")
        details_html = ""
        if details:
            details_html = f"<pre>{_esc(json.dumps(details, indent=2, ensure_ascii=True))}</pre>"
        error_html = (
            '<section class="panel panel-error" aria-labelledby="run-error-heading">'
            '<h2 id="run-error-heading">Run error</h2>'
            f"<p><code>{_esc(error.get('code'))}</code> {_esc(error.get('message'))}</p>"
            f"{details_html}</section>"
        )
    body = "".join(
        f'<tr><th scope="row">{_esc(label)}</th><td>{value}</td></tr>' for label, value in rows
    )
    return (
        '<section class="panel" aria-labelledby="summary-heading">'
        '<h2 id="summary-heading">Run manifest summary</h2>'
        f"{warning_html}"
        f'<table class="kv"><tbody>{body}</tbody></table>'
        f"{error_html}"
        f"{_render_models(cast(list[dict[str, Any]], summary.get('models') or []))}"
        "</section>"
    )


def _render_iteration(model: Mapping[str, Any]) -> str:
    iteration = model.get("iteration")
    if not isinstance(iteration, Mapping):
        return ""
    change = cast(Mapping[str, Any], iteration.get("change") or {})
    impact = cast(Mapping[str, Any], iteration.get("impact") or {})
    stale = cast(list[dict[str, Any]], iteration.get("stale_artifacts") or [])
    actions = cast(list[dict[str, Any]], iteration.get("safe_actions") or [])
    notes = cast(list[str], iteration.get("notes") or [])

    compatibility = cast(Mapping[str, Any], impact.get("compatibility") or {})
    evaluation = cast(Mapping[str, Any], impact.get("eval") or {})
    vram = cast(Mapping[str, Any], impact.get("vram") or {})
    admission = cast(Mapping[str, Any], vram.get("admission") or {})
    rollback = cast(Mapping[str, Any], impact.get("rollback") or {})
    affected = cast(Mapping[str, Any], impact.get("affected") or {})

    cards = "".join(
        '<div class="metric">'
        f'<div class="metric-label">{_esc(label)}</div>'
        f'<div class="metric-value">{value}</div>'
        "</div>"
        for label, value in (
            ("Change class", _esc(impact.get("change_class") or change.get("change_class"))),
            ("Compatibility", _compat_badge(str(compatibility.get("status") or "unknown"))),
            ("Target", _esc(impact.get("target") or change.get("target"))),
            ("Candidate revision", _code(change.get("candidate_revision"))),
            ("Base revision", _code(change.get("base_revision"))),
        )
    )
    summary = (
        f"<p><strong>Change:</strong> <code>{_esc(change.get('change_id'))}</code> "
        f"{_status_pill(change.get('status'))} {_esc(change.get('summary'))}</p>"
        f'<p class="muted">{_esc(impact.get("change_class_description"))}</p>'
    )
    checks = cast(list[dict[str, Any]], compatibility.get("checks") or [])
    check_rows = "".join(
        "<tr>"
        f"<td><code>{_esc(item.get('node'))}</code></td>"
        f"<td>{_esc(item.get('from_ref'))}</td>"
        f"<td>{_esc(item.get('to_ref'))}</td>"
        f"<td>{_esc(_identity_text(item.get('base')))}</td>"
        f"<td>{_esc(_identity_text(item.get('tokenizer')))}</td>"
        f"<td>{'yes' if item.get('blocking') else 'no'}</td>"
        f"<td>{_esc(item.get('message'))}</td>"
        "</tr>"
        for item in checks
    )
    compatibility_html = (
        f"<p><strong>Compatibility:</strong> {_compat_badge(str(compatibility.get('status')))} "
        f"blocking={'yes' if compatibility.get('blocking') else 'no'} — "
        f"{_esc(compatibility.get('message'))}</p>"
    )
    if check_rows:
        compatibility_html += (
            '<div class="table-wrap"><table aria-label="Compatibility checks">'
            '<thead><tr><th scope="col">Node</th><th scope="col">From ref</th>'
            '<th scope="col">To ref</th><th scope="col">Base</th>'
            '<th scope="col">Tokenizer</th><th scope="col">Blocking</th>'
            '<th scope="col">Message</th></tr></thead>'
            f"<tbody>{check_rows}</tbody></table></div>"
        )

    required = cast(list[str], evaluation.get("required") or [])
    satisfied = set(cast(list[str], evaluation.get("satisfied") or []))
    evidence_html = ""
    if required:
        items = "".join(
            f"<li>{_esc(level)} "
            + (
                '<span class="pill pill-ok">SATISFIED</span>'
                if level in satisfied
                else '<span class="pill pill-warn">MISSING</span>'
            )
            + "</li>"
            for level in required
        )
        evidence_html = f"<h3>Evaluation required before promotion</h3><ul>{items}</ul>"
    else:
        evidence_html = (
            "<h3>Evaluation required before promotion</h3>"
            '<p class="muted">No eval level is required for this change class.</p>'
        )
    evidence_records = cast(list[dict[str, Any]], change.get("evidence") or [])
    if evidence_records:
        evidence_rows = "".join(
            "<tr>"
            f"<td>{_code(item.get('path'))}</td>"
            f"<td>{_esc(item.get('level'))}</td>"
            f"<td>{'valid' if item.get('valid') else 'invalid'}</td>"
            f"<td>{'passed' if item.get('gate_passed') else 'failed'}</td>"
            f"<td>{_esc('; '.join(cast(list[str], item.get('errors') or [])))}</td>"
            "</tr>"
            for item in evidence_records
        )
        evidence_html += (
            '<div class="table-wrap"><table aria-label="Provided evidence">'
            '<thead><tr><th scope="col">Report</th><th scope="col">Level</th>'
            '<th scope="col">Report</th><th scope="col">Gate</th>'
            '<th scope="col">Errors</th></tr></thead>'
            f"<tbody>{evidence_rows}</tbody></table></div>"
        )

    invalidation_rows = "".join(
        "<tr>"
        f"<td><code>{_esc(item.get('artifact'))}</code></td>"
        f"<td>{_esc(item.get('status'))}</td>"
        f"<td>{'yes' if item.get('applicable') else 'no'}</td>"
        f"<td>{'yes' if item.get('present') else 'no'}</td>"
        f"<td>{_esc(', '.join(cast(list[str], item.get('scope') or [])) or '—')}</td>"
        f"<td>{_esc(item.get('reason'))}</td>"
        "</tr>"
        for item in cast(list[dict[str, Any]], impact.get("invalidations") or [])
    )
    invalidations_html = (
        "<h3>Invalidations</h3>"
        '<div class="table-wrap"><table aria-label="Invalidations">'
        '<thead><tr><th scope="col">Artifact</th><th scope="col">Status</th>'
        '<th scope="col">Applicable</th><th scope="col">Present</th>'
        '<th scope="col">Scope</th><th scope="col">Reason</th></tr></thead>'
        f"<tbody>{invalidation_rows}</tbody></table></div>"
    )

    stale_rows = "".join(
        '<tr data-stale-artifact="true" '
        f'data-artifact="{_attr(item.get("artifact"))}" '
        f'data-badge="{_attr(item.get("badge"))}">'
        f"<td><code>{_esc(item.get('artifact'))}</code></td>"
        f"<td>{_stale_badge(str(item.get('badge') or 'none'))}</td>"
        f"<td>{_esc(item.get('status'))}</td>"
        f"<td>{_esc(item.get('present_label'))}</td>"
        f"<td>{_code(item.get('source_hash'))}</td>"
        f"<td>{_code(item.get('artifact_hash'))}</td>"
        f"<td>{_esc(item.get('reason'))}</td>"
        "</tr>"
        for item in stale
    )
    stale_html = (
        "<h3>Plan, cache, and student artifacts</h3>"
        '<p class="muted">Badges come from the impact report\'s hash-based invalidation status. '
        "When an artifact does not exist in v0 the badge says not_applicable or none; no hash "
        "is invented.</p>"
        '<div class="table-wrap"><table aria-label="Stale artifact badges">'
        '<thead><tr><th scope="col">Artifact</th><th scope="col">Badge</th>'
        '<th scope="col">Status</th><th scope="col">Present</th>'
        '<th scope="col">Source graph hash</th><th scope="col">Artifact hash</th>'
        '<th scope="col">Reason</th></tr></thead>'
        f"<tbody>{stale_rows}</tbody></table></div>"
    )

    vram_html = (
        "<h3>VRAM</h3>"
        '<div class="table-wrap"><table aria-label="VRAM preview">'
        '<thead><tr><th scope="col">Before</th><th scope="col">After</th>'
        '<th scope="col">Delta</th><th scope="col">Admission</th>'
        '<th scope="col">Profile</th><th scope="col">Max</th>'
        '<th scope="col">Headroom</th><th scope="col">Reason</th></tr></thead><tbody><tr>'
        f"<td>{_gb(vram.get('before_gb'))}</td>"
        f"<td>{_gb(vram.get('after_gb'))}</td>"
        f"<td>{_gb_signed(vram.get('delta_gb'))}</td>"
        f"<td>{_esc(admission.get('status'))}</td>"
        f"<td>{_esc(admission.get('profile'))}</td>"
        f"<td>{_gb(admission.get('max_vram_gb'))}</td>"
        f"<td>{_gb(admission.get('headroom_gb'))}</td>"
        f"<td>{_esc(admission.get('reason'))}</td>"
        "</tr></tbody></table></div>"
    )

    rollback_html = (
        "<h3>Rollback</h3>"
        f"<p>Target: {_code(rollback.get('target'))} · "
        f"retained={'yes' if rollback.get('retained') else ('unknown' if rollback.get('retained') is None else 'no')} · "
        f"grace={_esc(rollback.get('grace_days'))} days</p>"
        f'<p class="muted">{_esc(rollback.get("note"))}</p>'
    )

    affected_nodes = cast(list[str], affected.get("nodes") or [])
    routers = cast(list[dict[str, Any]], affected.get("routers") or [])
    affected_html = (
        "<h3>Affected</h3>"
        f"<p>Nodes: {_esc(', '.join(affected_nodes) or 'none')}</p>"
        "<p>Routers: "
        + _esc(", ".join(str(item.get("router")) for item in routers) or "none")
        + "</p>"
        f"<p>Routing cards: {_esc(', '.join(cast(list[str], affected.get('routing_cards') or [])) or 'none')}</p>"
        f"<p>Route menu changed: {'yes' if affected.get('route_menu_changed') else 'no'}</p>"
    )

    risk_html = ""
    if cast(list[str], impact.get("risk") or []):
        risk_html = (
            "<h3>Risk</h3><ul>"
            + "".join(
                f"<li>{_esc(item)}</li>" for item in cast(list[str], impact.get("risk") or [])
            )
            + "</ul>"
        )
    notes_html = ""
    if cast(list[str], impact.get("notes") or []):
        notes_html = (
            "<h3>Impact notes</h3><ul>"
            + "".join(
                f"<li>{_esc(item)}</li>" for item in cast(list[str], impact.get("notes") or [])
            )
            + "</ul>"
        )

    action_items = []
    for item in actions:
        command = item.get("command")
        command_html = (
            f"<code>{_esc(command)}</code>"
            if command
            else '<span class="pill pill-na">NOT AVAILABLE</span>'
        )
        action_items.append(
            '<li data-action="' + _attr(item.get("action")) + '">'
            f"<strong>{_esc(item.get('label'))}</strong>: {command_html} "
            f'<span class="muted">{_esc(item.get("note"))}</span></li>'
        )
    actions_html = (
        "<h3>Follow-up commands (text only)</h3>"
        '<p class="muted">These are command strings, not links or buttons. The inspector does '
        "not execute them and does not mutate a revision, change set, or run.</p>"
        f'<ul class="commands">{"".join(action_items)}</ul>'
    )
    notes_panel = ""
    if notes:
        notes_panel = (
            "<h3>Read-only notes</h3><ul>"
            + "".join(f"<li>{_esc(item)}</li>" for item in notes)
            + "</ul>"
        )

    change_graph = cast(Mapping[str, Any], change.get("candidate_graph") or {})
    return (
        '<section class="panel panel-impact" aria-labelledby="impact-heading">'
        '<h2 id="impact-heading">Impact before promotion</h2>'
        '<p class="muted">Read-only impact report for the candidate change. Promotion, '
        "rollback, and artifact rebuilds are separate commands; this panel does not perform "
        "them.</p>"
        f"{summary}"
        f'<div class="metrics">{cards}</div>'
        f"<p>Candidate graph: {_esc(change_graph.get('id'))}@{_esc(change_graph.get('version'))} "
        f"{_code(change_graph.get('hash'))}</p>"
        f"{compatibility_html}{evidence_html}{invalidations_html}{stale_html}{vram_html}"
        f"{rollback_html}{affected_html}{risk_html}{notes_html}{actions_html}{notes_panel}"
        "</section>" + _render_timeline(iteration)
    )


def _render_timeline(iteration: Mapping[str, Any]) -> str:
    timeline = cast(Mapping[str, Any], iteration.get("timeline") or {})
    entries = cast(list[dict[str, Any]], timeline.get("entries") or [])
    if not entries:
        return (
            '<section class="panel" aria-labelledby="timeline-heading">'
            '<h2 id="timeline-heading">Revision timeline</h2>'
            '<p class="muted">No retained revisions are available for this store.</p>'
            "</section>"
        )
    items = []
    for entry in entries:
        roles = cast(list[str], entry.get("roles") or [])
        role_pills = "".join(_role_pill(role) for role in roles)
        markers = "".join(
            _marker_pill(str(marker.get("event")), marker)
            for marker in cast(list[dict[str, Any]], entry.get("markers") or [])
        )
        components = cast(Mapping[str, Any], entry.get("components") or {})
        plan = components.get("plan_status") or "not_applicable"
        plan_hash = components.get("plan_hash")
        student_hash = components.get("student_hash")
        missing = '<span class="pill pill-warn">MISSING</span>' if entry.get("missing") else ""
        items.append(
            '<li class="timeline-entry" '
            f'data-revision="{_attr(entry.get("revision_id"))}" '
            f'data-role="{_attr(" ".join(roles))}" '
            f'data-stale="{"true" if entry.get("stale") else "false"}">'
            f'<div class="timeline-head">{role_pills}{missing}'
            f"<code>{_esc(entry.get('revision_id'))}</code></div>"
            f'<div class="muted">{_esc(entry.get("graph_id"))} · '
            f"target={_esc(entry.get('target'))} · {_esc(entry.get('created_at'))}</div>"
            f"<div>graph hash: {_code(entry.get('graph_hash'))}</div>"
            f"<div>plan: {_esc(plan)} {_code(plan_hash)} · "
            f"student: {_code(student_hash) if student_hash else 'none'}</div>"
            f'<div class="timeline-markers">{markers}</div>'
            "</li>"
        )
    notes = "".join(
        f"<li>{_esc(item)}</li>" for item in cast(list[str], timeline.get("notes") or [])
    )
    return (
        '<section class="panel" aria-labelledby="timeline-heading">'
        '<h2 id="timeline-heading">Revision timeline</h2>'
        '<p class="muted">Active, previous, and candidate roles come from the stored active '
        "pointer and change set. STALE is hash-based: the retained revision's graph hash "
        "differs from the active revision. Compiled plan and student hashes are null in v0.</p>"
        f'<ol class="timeline">{"".join(items)}</ol>'
        f"<h3>Timeline notes</h3><ul>{notes}</ul>"
        "</section>"
    )


def _render_compare(model: Mapping[str, Any]) -> str:
    compare = model.get("compare")
    if not isinstance(compare, Mapping):
        return ""
    left = cast(Mapping[str, Any], compare.get("left") or {})
    right = cast(Mapping[str, Any], compare.get("right") or {})
    same_inputs = compare.get("same_inputs")
    same_graph = compare.get("same_graph")
    if same_inputs is True:
        indicator = (
            '<div class="banner banner-ok compare-inputs" data-same-inputs="true">'
            "<strong>Same inputs:</strong> yes — the recorded task/request hash is identical. "
            '<span class="muted">(same inputs)</span></div>'
        )
    elif same_inputs is False:
        indicator = (
            '<div class="banner banner-warn compare-inputs" data-same-inputs="false">'
            "<strong>Same inputs:</strong> no — the recorded task/request hash differs, so this "
            "is a cross-input comparison, not a paired replay. "
            '<span class="muted">(same inputs: no)</span></div>'
        )
    else:
        indicator = (
            '<div class="banner banner-note compare-inputs" data-same-inputs="unknown">'
            "<strong>Same inputs:</strong> unknown — one or both runs do not record a "
            'task/request hash. <span class="muted">(same inputs: unknown)</span></div>'
        )
    graph_status = (
        "same" if same_graph is True else ("different" if same_graph is False else "unknown")
    )
    graph_line = (
        f'<p class="muted" data-same-graph="{graph_status}">Recorded graph hash: {graph_status}</p>'
    )
    warnings = "".join(
        f"<li>{_esc(item)}</li>" for item in cast(list[str], compare.get("warnings") or [])
    )
    warnings_html = f'<ul class="warnings">{warnings}</ul>' if warnings else ""
    sides = "".join(
        "<tr>"
        f'<th scope="row">{_esc(label)}</th>'
        f"<td>{_esc(left.get(key))}</td>"
        f"<td>{_esc(right.get(key))}</td>"
        "</tr>"
        for label, key in (
            ("Run id", "run_id"),
            ("Outcome", "outcome"),
            ("Status", "status"),
            ("Graph", "graph_id"),
            ("Graph hash", "graph_hash"),
            ("Target", "target"),
            ("Task/request hash", "task_hash"),
        )
    )
    side_table = (
        '<h3>Run summary</h3><div class="table-wrap"><table aria-label="Run summary">'
        '<thead><tr><th scope="col">Field</th><th scope="col">Left run</th>'
        '<th scope="col">Right run</th></tr></thead>'
        f"<tbody>{sides}</tbody></table></div>"
    )
    totals_rows = "".join(
        '<tr class="' + ("diff-row" if not row.get("same") else "") + '">'
        f"<td>{_esc(row.get('field'))}</td>"
        f'<td class="number">{_compare_value(row.get("field"), row.get("left"))}</td>'
        f'<td class="number">{_compare_value(row.get("field"), row.get("right"))}</td>'
        f"<td>{_diff_marker(not row.get('same'))}</td>"
        "</tr>"
        for row in cast(
            list[dict[str, Any]],
            cast(Mapping[str, Any], compare.get("totals") or {}).get("rows") or [],
        )
    )
    totals_table = (
        '<h3>Recorded totals</h3><div class="table-wrap"><table aria-label="Totals comparison">'
        '<thead><tr><th scope="col">Field</th><th scope="col">Left</th>'
        '<th scope="col">Right</th><th scope="col">Same</th></tr></thead>'
        f"<tbody>{totals_rows}</tbody></table></div>"
    )

    path_rows = "".join(
        '<tr class="'
        + ("diff-row" if not row.get("same") else "")
        + '" data-diff="'
        + ("true" if not row.get("same") else "false")
        + '">'
        f'<td class="number">{_esc(row.get("index"))}</td>'
        f'<td class="{"diff-cell" if not row.get("same") else ""}">{_path_cell(row.get("left"))}</td>'
        f'<td class="{"diff-cell" if not row.get("same") else ""}">{_path_cell(row.get("right"))}</td>'
        f"<td>{_diff_marker(not row.get('same'))}</td>"
        "</tr>"
        for row in cast(
            list[dict[str, Any]],
            cast(Mapping[str, Any], compare.get("path") or {}).get("rows") or [],
        )
    )
    path_table = (
        '<h3>Path and decisions</h3><div class="table-wrap"><table aria-label="Path comparison">'
        '<thead><tr><th scope="col">#</th><th scope="col">Left run</th>'
        '<th scope="col">Right run</th><th scope="col">Same</th></tr></thead>'
        f"<tbody>{path_rows}</tbody></table></div>"
    )

    node_rows = "".join(
        '<tr class="'
        + ("diff-row" if not row.get("same") else "")
        + '" data-diff="'
        + ("true" if not row.get("same") else "false")
        + '">'
        f"<td><code>{_esc(row.get('node'))}</code></td>"
        f'<td class="{"missing-cell" if row.get("missing_left") else ("diff-cell" if _node_field_diff(row, "provider") or _node_field_diff(row, "model") else "")}">'
        f"{_node_provider_cell(row.get('left'))}</td>"
        f'<td class="{"missing-cell" if row.get("missing_right") else ("diff-cell" if _node_field_diff(row, "provider") or _node_field_diff(row, "model") else "")}">'
        f"{_node_provider_cell(row.get('right'))}</td>"
        f'<td class="number {"diff-cell" if _node_field_diff(row, "tokens") else ""}">'
        f"{_optional_integer(row.get('left'), 'tokens')}</td>"
        f'<td class="number {"diff-cell" if _node_field_diff(row, "tokens") else ""}">'
        f"{_optional_integer(row.get('right'), 'tokens')}</td>"
        f'<td class="number {"diff-cell" if _node_field_diff(row, "cost_usd") else ""}">'
        f"{_optional_money(row.get('left'), 'cost_usd')}</td>"
        f'<td class="number {"diff-cell" if _node_field_diff(row, "cost_usd") else ""}">'
        f"{_optional_money(row.get('right'), 'cost_usd')}</td>"
        f'<td class="number {"diff-cell" if _node_field_diff(row, "latency_seconds") else ""}">'
        f"{_optional_seconds(row.get('left'), 'latency_seconds')}</td>"
        f'<td class="number {"diff-cell" if _node_field_diff(row, "latency_seconds") else ""}">'
        f"{_optional_seconds(row.get('right'), 'latency_seconds')}</td>"
        f"<td>{_esc(', '.join(cast(list[str], row.get('differences') or [])) or 'same')}</td>"
        "</tr>"
        for row in cast(
            list[dict[str, Any]],
            cast(Mapping[str, Any], compare.get("nodes") or {}).get("rows") or [],
        )
    )
    node_table = (
        "<h3>Per-node provider and usage</h3>"
        '<p class="muted">Differences are highlighted. A node or artifact present on only one '
        "side is shown as missing, not invented.</p>"
        '<div class="table-wrap"><table aria-label="Node comparison">'
        '<thead><tr><th scope="col">Node</th>'
        '<th scope="col">Left provider / model</th><th scope="col">Right provider / model</th>'
        '<th scope="col">Left tokens</th><th scope="col">Right tokens</th>'
        '<th scope="col">Left cost</th><th scope="col">Right cost</th>'
        '<th scope="col">Left latency</th><th scope="col">Right latency</th>'
        '<th scope="col">Differences</th></tr></thead>'
        f"<tbody>{node_rows}</tbody></table></div>"
    )

    artifact_rows = "".join(
        '<tr class="' + ("diff-row" if not row.get("same") else "") + '">'
        f"<td><code>{_esc(row.get('name'))}</code></td>"
        f'<td class="{"missing-cell" if row.get("missing_left") else ("diff-cell" if not row.get("same") else "")}">'
        f"{_code(row.get('left_hash')) if row.get('left_hash') else '—'}</td>"
        f'<td class="{"missing-cell" if row.get("missing_right") else ("diff-cell" if not row.get("same") else "")}">'
        f"{_code(row.get('right_hash')) if row.get('right_hash') else '—'}</td>"
        f"<td>{_diff_marker(not row.get('same'))}</td>"
        "</tr>"
        for row in cast(
            list[dict[str, Any]],
            cast(Mapping[str, Any], compare.get("artifacts") or {}).get("rows") or [],
        )
    )
    artifact_table = (
        "<h3>Artifacts by name and hash</h3>"
        '<div class="table-wrap"><table aria-label="Artifact comparison">'
        '<thead><tr><th scope="col">Artifact</th><th scope="col">Left hash</th>'
        '<th scope="col">Right hash</th><th scope="col">Same</th></tr></thead>'
        f"<tbody>{artifact_rows}</tbody></table></div>"
    )
    return (
        '<section class="panel panel-compare" aria-labelledby="compare-heading">'
        '<h2 id="compare-heading">Compare runs</h2>'
        f"{indicator}{graph_line}{warnings_html}"
        f"{side_table}{totals_table}{path_table}{node_table}{artifact_table}"
        "</section>"
    )


def _render_models(models: Sequence[Mapping[str, Any]]) -> str:
    if not models:
        return '<p class="muted">No model records in the run manifest.</p>'
    rows = []
    for model in models:
        rows.append(
            "<tr>"
            f"<td>{_esc(model.get('node'))}</td>"
            f"<td>{_esc(model.get('ref'))}</td>"
            f"<td>{_esc(model.get('provider'))}</td>"
            f"<td>{_esc(model.get('model'))}</td>"
            f"<td>{_code(model.get('hash'))}</td>"
            f"<td>{_esc(model.get('version'))}</td>"
            f"<td>{_esc(model.get('resolved_revision'))}</td>"
            "</tr>"
        )
    return (
        '<h3 id="models-heading">Resolved models</h3>'
        '<div class="table-wrap"><table aria-labelledby="models-heading">'
        '<thead><tr><th scope="col">Node</th><th scope="col">Ref</th>'
        '<th scope="col">Provider</th><th scope="col">Model</th>'
        '<th scope="col">Manifest hash</th><th scope="col">Version</th>'
        '<th scope="col">Revision</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _render_totals(model: Mapping[str, Any]) -> str:
    totals = cast(Mapping[str, Any], model["totals"])
    trace_sums = totals.get("trace_sums")
    trace_sums = trace_sums if isinstance(trace_sums, Mapping) else {}
    budget = cast(Mapping[str, Any], model["summary"]).get("budget")
    budget = budget if isinstance(budget, Mapping) else {}
    cache = cast(Mapping[str, Any], model["cache"])
    source = _esc(totals.get("source"))
    cards = [
        ("Cost", _money(totals.get("cost_usd")), "total-cost", totals.get("cost_usd")),
        (
            "Latency",
            _seconds(totals.get("elapsed_seconds")),
            "total-latency",
            totals.get("elapsed_seconds"),
        ),
        (
            "Tokens",
            _integer(totals.get("total_tokens")),
            "total-tokens",
            totals.get("total_tokens"),
        ),
        (
            "Input / output",
            f"{_integer(totals.get('input_tokens'))} / {_integer(totals.get('output_tokens'))}",
            "total-io-tokens",
            totals.get("input_tokens"),
        ),
        ("Steps", _integer(totals.get("steps")), "total-steps", totals.get("steps")),
    ]
    card_html = "".join(
        '<div class="metric">'
        f'<div class="metric-label">{_esc(label)}</div>'
        f'<div class="metric-value" id="{ident}" data-value="{_attr(value)}">{text}</div>'
        "</div>"
        for label, text, ident, value in cards
    )
    budget_rows = [
        ("Steps", _integer(budget.get("used_steps")), _integer(budget.get("max_steps"))),
        ("Depth", "—", _integer(budget.get("max_depth"))),
        (
            "Tokens",
            _integer(budget.get("used_tokens")),
            _integer(budget.get("max_total_tokens")),
        ),
        (
            "Cost",
            _money(budget.get("used_cost_usd")),
            _money(budget.get("max_cost_usd")),
        ),
        (
            "Wall time",
            _seconds(budget.get("elapsed_seconds")),
            _seconds(budget.get("max_wall_seconds")),
        ),
    ]
    budget_html = "".join(
        f'<tr><th scope="row">{_esc(label)}</th><td>{used}</td><td>{limit}</td></tr>'
        for label, used, limit in budget_rows
    )
    cache_html = ""
    if cache.get("available"):
        cache_html = (
            '<p class="muted">Cache: '
            f"hits={_integer(cache.get('hits'))} "
            f"misses={_integer(cache.get('misses'))} "
            f"skipped={_integer(cache.get('skipped'))} "
            f"avoided_tokens={_integer(cache.get('avoided_tokens'))} "
            f"avoided_cost={_money(cache.get('avoided_cost_usd'))}</p>"
        )
    else:
        cache_html = '<p class="muted">No cache events were recorded for this run.</p>'
    return (
        '<section class="panel" aria-labelledby="totals-heading">'
        '<h2 id="totals-heading">Recorded totals</h2>'
        '<p class="muted">Totals are read from the run manifest usage block and match the '
        "recorded run JSON. Per-node values are derived from the trace; the manifest totals "
        "remain authoritative.</p>"
        f'<div class="metrics">{card_html}</div>'
        f'<p class="muted">Totals source: <strong>{source}</strong>. '
        f"Trace-derived sums: tokens={_integer(trace_sums.get('total_tokens'))}, "
        f"cost={_money(trace_sums.get('cost_usd'))}, "
        f"latency={_seconds(trace_sums.get('latency_seconds'))}.</p>"
        "<h3>Budget usage</h3>"
        '<div class="table-wrap"><table aria-label="Budget usage">'
        '<thead><tr><th scope="col">Budget</th><th scope="col">Used</th>'
        f'<th scope="col">Limit</th></tr></thead><tbody>{budget_html}</tbody></table></div>'
        f"{cache_html}"
        "</section>"
    )


def _render_path(model: Mapping[str, Any]) -> str:
    path = cast(list[dict[str, Any]], model["path"])
    if not path:
        return (
            '<section class="panel" aria-labelledby="path-heading">'
            '<h2 id="path-heading">Chosen path</h2>'
            '<p class="muted">No path entries were recorded in the run manifest.</p>'
            "</section>"
        )
    items = []
    for index, entry in enumerate(path, start=1):
        transition = ""
        if entry.get("to"):
            transition = f" → <strong>{_esc(entry.get('to'))}</strong>"
        confidence = ""
        if entry.get("confidence") is not None:
            confidence = f' <span class="muted">({_confidence(entry.get("confidence"))})</span>'
        reason = ""
        if entry.get("reason"):
            reason = f'<div class="reason">{_esc(entry.get("reason"))}</div>'
        decision = (
            f' <span class="pill">{_esc(entry.get("decision"))}</span>'
            if entry.get("decision")
            else ""
        )
        items.append(
            '<li class="path-step">'
            f'<span class="path-index">{index}</span>'
            f"<span><code>{_esc(entry.get('node'))}</code> "
            f'<span class="muted">{_esc(entry.get("kind"))}</span>'
            f"{decision}{transition}{confidence}{reason}</span>"
            "</li>"
        )
    return (
        '<section class="panel" aria-labelledby="path-heading">'
        '<h2 id="path-heading">Chosen path</h2>'
        '<p class="muted">The path and decisions come from the run manifest. Chosen nodes and '
        "edges are highlighted in the graph.</p>"
        f'<ol class="path">{"".join(items)}</ol>'
        "</section>"
    )


def _render_graph(graph: Mapping[str, Any]) -> str:
    nodes = cast(list[dict[str, Any]], graph["nodes"])
    edges = cast(list[dict[str, Any]], graph["edges"])
    if not nodes:
        return (
            '<section class="panel" aria-labelledby="graph-heading">'
            '<h2 id="graph-heading">Graph</h2>'
            '<p class="muted">The compiled authored graph has no nodes.</p>'
            "</section>"
        )
    positions = _layout(nodes, edges)
    width = max((item["x"] for item in positions.values()), default=0) + _NODE_WIDTH + _MARGIN
    height = max((item["y"] for item in positions.values()), default=0) + _NODE_HEIGHT + _MARGIN
    width = max(width, 420)
    height = max(height, 260)
    edge_html = _render_edges(edges, positions)
    node_html = _render_svg_nodes(nodes, positions)
    legend = (
        '<ul class="legend" aria-label="Graph legend">'
        '<li><span class="swatch swatch-router"></span>router</li>'
        '<li><span class="swatch swatch-worker"></span>worker</li>'
        '<li><span class="swatch swatch-joiner"></span>joiner</li>'
        '<li><span class="swatch swatch-subgraph"></span>subgraph</li>'
        '<li><span class="swatch swatch-chosen"></span>chosen path</li>'
        '<li><span class="swatch swatch-pruned"></span>pruned (from run)</li>'
        '<li><span class="swatch swatch-exit"></span>exit marker</li>'
        "</ul>"
    )
    return (
        '<section class="panel" aria-labelledby="graph-heading">'
        '<h2 id="graph-heading">Graph</h2>'
        '<p class="muted">Compiled authored graph. The recorded run may have used a pruned '
        "view; pruned nodes and edges are marked and labeled from the run manifest.</p>"
        f"{legend}"
        '<div class="svg-wrap">'
        f'<svg id="graph-svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="graph-title graph-desc" preserveAspectRatio="xMidYMid meet">'
        f'<title id="graph-title">Directed graph {_esc(graph.get("id"))}</title>'
        f'<desc id="graph-desc">entry {_esc(graph.get("entry"))}; '
        f"exits {_esc(', '.join(cast(list[str], graph.get('exits') or [])))}</desc>"
        f'<g class="edges">{edge_html}</g>'
        f'<g class="nodes">{node_html}</g>'
        "</svg></div></section>"
    )


def _layout(
    nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, float]]:
    by_id = {str(node["id"]): node for node in nodes}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    for edge in edges:
        source = str(edge["from"])
        target = str(edge["to"])
        if source in outgoing and target in by_id:
            outgoing[source].append(target)
    entry = next((str(node["id"]) for node in nodes if node.get("entry")), None)
    depth: dict[str, int] = {}
    queue: list[str] = []
    start = entry or (str(nodes[0]["id"]) if nodes else None)
    if start is not None:
        depth[str(start)] = 0
        queue.append(str(start))
    while queue:
        current = queue.pop(0)
        for target in outgoing.get(current, []):
            if target not in depth:
                depth[target] = depth[current] + 1
                queue.append(target)
    next_depth = max(depth.values(), default=-1) + 1
    for node_id in by_id:
        if node_id not in depth:
            depth[node_id] = next_depth
    layers: dict[int, list[str]] = {}
    for node_id in by_id:
        layers.setdefault(depth[node_id], []).append(node_id)
    positions: dict[str, dict[str, float]] = {}
    for layer in sorted(layers):
        for row, node_id in enumerate(sorted(layers[layer])):
            positions[node_id] = {
                "x": float(_MARGIN + layer * (_NODE_WIDTH + _COLUMN_GAP)),
                "y": float(_MARGIN + row * (_NODE_HEIGHT + _ROW_GAP)),
            }
    return positions


def _render_edges(
    edges: Sequence[Mapping[str, Any]],
    positions: Mapping[str, Mapping[str, float]],
) -> str:
    parts: list[str] = []
    for edge in edges:
        source = positions.get(str(edge["from"]))
        target = positions.get(str(edge["to"]))
        if source is None or target is None:
            continue
        classes = ["edge"]
        arrow_classes = ["edge-arrow"]
        if edge.get("chosen"):
            classes.append("chosen")
            arrow_classes.append("chosen")
        if edge.get("pruned"):
            classes.append("pruned")
            arrow_classes.append("pruned")
        if str(edge["from"]) == str(edge["to"]):
            x = source["x"] + _NODE_WIDTH * 0.35
            y = source["y"] + _NODE_HEIGHT
            parts.append(
                f'<path class="{" ".join(classes)}" d="M {x:.1f} {y:.1f} '
                f"C {x - 30:.1f} {y + 42:.1f}, {x + 60:.1f} {y + 42:.1f}, "
                f'{x + 42:.1f} {y:.1f}" data-from="{_attr(edge["from"])}" '
                f'data-to="{_attr(edge["to"])}"></path>'
            )
            continue
        start = _boundary_point(source, target)
        end = _boundary_point(target, source)
        parts.append(
            f'<line class="{" ".join(classes)}" x1="{start[0]:.1f}" y1="{start[1]:.1f}" '
            f'x2="{end[0]:.1f}" y2="{end[1]:.1f}" data-from="{_attr(edge["from"])}" '
            f'data-to="{_attr(edge["to"])}"></line>'
        )
        angle = _angle(start, end)
        arrow = _arrow_points(end[0], end[1], angle)
        if arrow:
            parts.append(
                f'<polygon class="{" ".join(arrow_classes)}" points="{arrow}" '
                f'data-from="{_attr(edge["from"])}" data-to="{_attr(edge["to"])}"></polygon>'
            )
        label = _edge_label(edge)
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2 - 7
            parts.append(
                f'<text class="edge-label" x="{mid_x:.1f}" y="{mid_y:.1f}" '
                f'text-anchor="middle">{_esc(label)}</text>'
            )
    return "".join(parts)


def _render_svg_nodes(
    nodes: Sequence[Mapping[str, Any]],
    positions: Mapping[str, Mapping[str, float]],
) -> str:
    parts: list[str] = []
    for node in nodes:
        node_id = str(node["id"])
        position = positions.get(node_id)
        if position is None:
            continue
        classes = ["node", f"kind-{_kind_class(node.get('kind'))}"]
        if node.get("chosen"):
            classes.append("chosen")
        if node.get("pruned"):
            classes.append("pruned")
        badges: list[str] = []
        if node.get("chosen"):
            badges.append("PATH")
        if node.get("pruned"):
            badges.append("PRUNED")
        if node.get("exit"):
            names = cast(list[str], node.get("exit_artifacts") or [])
            badges.append("EXIT: " + ", ".join(names) if names else "EXIT")
        entry = str(node.get("id")) == str(_graph_entry(nodes))
        if entry:
            badges.insert(0, "ENTRY")
        label = " · ".join(badges)
        model = _optional_text(node.get("model_ref")) or ""
        parts.append(
            f'<g class="{" ".join(classes)}" data-node="{_attr(node_id)}" '
            f'data-kind="{_attr(node.get("kind"))}" data-pruned="'
            f'{"true" if node.get("pruned") else "false"}" data-chosen="'
            f'{"true" if node.get("chosen") else "false"}" '
            f'transform="translate({position["x"]:.1f},{position["y"]:.1f})">'
            f"<title>{_esc(node_id)} ({_esc(node.get('kind'))})</title>"
            f'<rect width="{_NODE_WIDTH}" height="{_NODE_HEIGHT}" rx="9"></rect>'
            f'<text class="node-id" x="12" y="23">{_esc(node_id)}</text>'
            f'<text class="node-kind" x="12" y="42">{_esc(node.get("kind"))}</text>'
            f'<text class="node-model" x="12" y="60">{_esc(_short(model, 28))}</text>'
            f'<text class="node-badges" x="12" y="74">{_esc(label)}</text>'
            "</g>"
        )
    return "".join(parts)


def _render_prune(model: Mapping[str, Any]) -> str:
    prune = cast(Mapping[str, Any], model["prune"])
    removed_nodes = cast(list[dict[str, Any]], prune["removed_nodes"])
    removed_edges = cast(list[dict[str, Any]], prune["removed_edges"])
    removed_exits = cast(list[dict[str, Any]], prune["removed_exit_artifacts"])
    warnings = cast(list[str], prune["warnings"])
    if not (removed_nodes or removed_edges or removed_exits or warnings):
        return (
            '<section class="panel" aria-labelledby="prune-heading">'
            '<h2 id="prune-heading">Pruning</h2>'
            '<p class="muted">No prune report was recorded for this run.</p>'
            "</section>"
        )
    graph_nodes = {str(node["id"]) for node in cast(list[dict[str, Any]], model["graph"]["nodes"])}
    node_rows = "".join(
        f'<tr class="pruned-row"><td><code>{_esc(item.get("id"))}</code> '
        '<span class="pill pill-pruned">PRUNED</span></td>'
        f"<td>{'yes' if item.get('id') in graph_nodes else 'no (not in compiled graph)'}</td>"
        f"<td>{_esc(item.get('graph'))}</td><td>{_esc(item.get('reason'))}</td></tr>"
        for item in removed_nodes
    )
    edge_rows = "".join(
        f'<tr class="pruned-row"><td><span class="pill pill-pruned">PRUNED</span> '
        f"<code>{_esc(item.get('from'))}</code> → "
        f"<code>{_esc(item.get('to'))}</code></td>"
        f"<td>{_esc(item.get('graph'))}</td><td>{_esc(item.get('reason'))}</td></tr>"
        for item in removed_edges
    )
    exit_rows = "".join(
        f'<tr class="pruned-row"><td><span class="pill pill-pruned">PRUNED</span> '
        f"{_esc(item.get('name'))}</td><td>{_esc(item.get('graph'))}</td>"
        f"<td>{_esc(item.get('reason'))}</td></tr>"
        for item in removed_exits
    )
    tables = ""
    if removed_nodes:
        tables += (
            '<h3>Removed nodes</h3><div class="table-wrap"><table>'
            '<thead><tr><th scope="col">Node</th><th scope="col">In graph</th>'
            '<th scope="col">Graph</th><th scope="col">Reason</th></tr></thead>'
            f"<tbody>{node_rows}</tbody></table></div>"
        )
    if removed_edges:
        tables += (
            '<h3>Removed edges</h3><div class="table-wrap"><table>'
            '<thead><tr><th scope="col">Edge</th><th scope="col">Graph</th>'
            '<th scope="col">Reason</th></tr></thead>'
            f"<tbody>{edge_rows}</tbody></table></div>"
        )
    if removed_exits:
        tables += (
            '<h3>Removed exit artifacts</h3><div class="table-wrap"><table>'
            '<thead><tr><th scope="col">Artifact</th><th scope="col">Graph</th>'
            '<th scope="col">Reason</th></tr></thead>'
            f"<tbody>{exit_rows}</tbody></table></div>"
        )
    warning_html = ""
    if warnings:
        warning_html = (
            "<h3>Prune warnings</h3><ul>"
            + "".join(f"<li>{_esc(item)}</li>" for item in warnings)
            + "</ul>"
        )
    return (
        '<section class="panel" aria-labelledby="prune-heading">'
        '<h2 id="prune-heading">Pruning</h2>'
        '<p class="muted">Pruned nodes and edges are also marked in the graph. '
        "Reasons are copied from the run manifest prune report.</p>"
        f"{tables}{warning_html}</section>"
    )


def _render_nodes(model: Mapping[str, Any]) -> str:
    nodes = cast(list[dict[str, Any]], model["nodes"])
    cache = cast(Mapping[str, Any], model["cache"])
    cache_filter = ""
    if cache.get("hits"):
        cache_filter = (
            '<label for="cache-filter">Cache</label>'
            '<select id="cache-filter"><option value="all">All nodes</option>'
            '<option value="hits">Cache hits only</option>'
            '<option value="no-hits">No cache hits</option></select>'
        )
    controls = (
        '<div class="filters">'
        '<label for="node-filter">Node</label>'
        '<input id="node-filter" type="search" placeholder="id, kind, model, reason" '
        'autocomplete="off">'
        '<label for="outcome-filter">Outcome/error</label>'
        '<select id="outcome-filter"><option value="all">All nodes</option>'
        '<option value="errors">Errors only</option>'
        '<option value="ok">No errors</option></select>'
        f"{cache_filter}"
        '<button id="expand-all" type="button">Expand all details</button>'
        '<button id="collapse-all" type="button">Collapse all details</button>'
        '<span id="filter-count" class="muted" aria-live="polite"></span>'
        "</div>"
    )
    if not nodes:
        return (
            '<section class="panel" aria-labelledby="nodes-heading">'
            '<h2 id="nodes-heading">Node details</h2>'
            '<p class="muted">No nodes are available.</p></section>'
        )
    rows = "".join(_node_row(node) for node in nodes)
    return (
        '<section class="panel" aria-labelledby="nodes-heading">'
        '<h2 id="nodes-heading">Node details</h2>'
        '<p class="muted">Per-node values are derived from trace events; missing values are '
        "shown as an em dash. The node filter also matches model and reason text.</p>"
        f"{controls}"
        '<div class="table-wrap"><table aria-labelledby="nodes-heading">'
        '<thead><tr><th scope="col">Node</th><th scope="col">Kind</th>'
        '<th scope="col">Status</th><th scope="col">Decision</th>'
        '<th scope="col">Confidence</th><th scope="col">Reason</th>'
        '<th scope="col">Provider</th><th scope="col">Model</th>'
        '<th scope="col">Tokens</th><th scope="col">Cost</th>'
        '<th scope="col">Latency</th><th scope="col">Retries</th>'
        '<th scope="col">Errors</th><th scope="col">Cache</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div></section>"
    )


def _node_row(node: Mapping[str, Any]) -> str:
    status = str(node.get("status"))
    cache_hit = bool(node.get("cache_hits"))
    search = " ".join(
        str(node.get(key) or "")
        for key in ("id", "kind", "status", "decision", "reason", "provider", "model")
    ).lower()
    badges = ""
    if node.get("chosen"):
        badges += ' <span class="pill pill-chosen">PATH</span>'
    if node.get("pruned"):
        badges += ' <span class="pill pill-pruned">PRUNED</span>'
    if not node.get("in_graph"):
        badges += ' <span class="pill">trace only</span>'
    errors = cast(list[dict[str, Any]], node.get("errors") or [])
    if errors:
        error_text = "".join(
            f"<li><code>{_esc(item.get('code'))}</code> {_esc(item.get('message'))}</li>"
            for item in errors
        )
        errors_cell = (
            f'<details><summary>{len(errors)}</summary><ul class="errors">{error_text}</ul>'
            "</details>"
        )
    else:
        errors_cell = "—"
    return (
        f'<tr data-filter-row data-node="{_attr(node.get("id"))}" '
        f'data-status="{_attr(status)}" '
        f'data-cache="{"hit" if cache_hit else "none"}" '
        f'data-search="{_attr(search)}">'
        f'<th scope="row"><code>{_esc(node.get("id"))}</code>{badges}</th>'
        f"<td>{_esc(node.get('kind'))}</td>"
        f'<td><span class="status status-{_attr(status)}">{_esc(status)}</span></td>'
        f"<td>{_esc(node.get('decision'))}</td>"
        f"<td>{_confidence(node.get('confidence'))}</td>"
        f'<td class="reason-cell">{_esc(node.get("reason"))}</td>'
        f"<td>{_esc(node.get('provider'))}</td>"
        f"<td>{_esc(node.get('model'))}</td>"
        f'<td class="number">{_integer(node.get("tokens"))}</td>'
        f'<td class="number">{_money(node.get("cost_usd"))}</td>'
        f'<td class="number">{_seconds(node.get("latency_seconds"))}</td>'
        f'<td class="number">{_integer(node.get("retries"))}</td>'
        f"<td>{errors_cell}</td>"
        f"<td>{_integer(node.get('cache_hits'))}</td>"
        "</tr>"
    )


def _render_artifacts(model: Mapping[str, Any]) -> str:
    artifacts = cast(list[dict[str, Any]], model["artifacts"])
    limits = cast(Mapping[str, Any], model["artifact_limits"])
    intro = (
        '<p class="muted">Artifact payloads are embedded in this file and capped at '
        f"{_integer(limits.get('per_artifact_bytes'))} bytes per artifact and "
        f"{_integer(limits.get('total_bytes'))} bytes total in the inspection document; "
        "the visible panel and the embedded JSON share the capped value. "
        "Truncated content is labeled; treat the file as sensitive.</p>"
    )
    if not artifacts:
        return (
            '<section class="panel" aria-labelledby="artifacts-heading">'
            '<h2 id="artifacts-heading">Artifacts</h2>'
            f'{intro}<p class="muted">No artifacts were recorded in the run manifest.</p>'
            "</section>"
        )
    excluded = ""
    missing_exits = sorted(
        {str(name) for name in cast(list[str], model["graph"].get("exits") or [])}
        - {str(item.get("name")) for item in artifacts}
    )
    if missing_exits:
        excluded = (
            '<p class="notice">Exit artifact(s) not present in the manifest artifact list: '
            f"{_esc(', '.join(missing_exits))}</p>"
        )
    body = "".join(_artifact_html(item) for item in artifacts)
    return (
        '<section class="panel" aria-labelledby="artifacts-heading">'
        '<h2 id="artifacts-heading">Artifacts</h2>'
        f"{intro}{excluded}{body}</section>"
    )


def _artifact_html(artifact: Mapping[str, Any]) -> str:
    badges = ""
    if artifact.get("exit"):
        badges += '<span class="pill pill-exit">EXIT</span> '
    if artifact.get("truncated"):
        badges += '<span class="pill pill-warn">TRUNCATED</span> '
    if artifact.get("missing"):
        badges += '<span class="pill pill-warn">CONTENT MISSING</span> '
    meta = (
        f"type={_esc(artifact.get('type'))} · producer={_esc(artifact.get('producer'))} · "
        f"{_integer(artifact.get('original_bytes'))} bytes · hash="
        f"{_esc(artifact.get('hash'))}"
    )
    truncation = ""
    if artifact.get("truncated"):
        reason = artifact.get("truncation_reason") or "cap"
        truncation = (
            f'<p class="notice" data-truncated="true">Truncated ({_esc(reason)}): showing '
            f"{_integer(artifact.get('embedded_bytes'))} of "
            f"{_integer(artifact.get('original_bytes'))} bytes.</p>"
        )
    missing = ""
    if artifact.get("missing"):
        missing = '<p class="notice">Artifact content is not available in the run directory.</p>'
    content = _esc(artifact.get("content"))
    return (
        '<details class="artifact" data-expandable '
        f'data-artifact="{_attr(artifact.get("name"))}">'
        f'<summary><span class="artifact-name">{_esc(artifact.get("name"))}</span> '
        f'{badges}<span class="muted">{meta}</span></summary>'
        f'<div class="artifact-body">{truncation}{missing}'
        f"<pre><code>{content}</code></pre></div></details>"
    )


def _json_data_block(model: Mapping[str, Any]) -> str:
    return (
        f'<script type="application/json" id="mg-inspector-data">{_json_for_html(model)}</script>'
    )


def _filter_script() -> str:
    return "<script>\n" + _SCRIPT + "\n</script>"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _safe_child(root: Path, relative: str) -> Path | None:
    if not relative or relative.startswith(("/", "\\")):
        return None
    parts = Path(relative).parts
    if not parts or any(part in {"", ".."} for part in parts):
        return None
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def _mapping_list(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _usage_tokens(usage: Mapping[str, Any]) -> int:
    if "total_tokens" in usage:
        return _as_int(usage.get("total_tokens"))
    return _as_int(usage.get("input_tokens")) + _as_int(usage.get("output_tokens"))


def _decode_prefix(data: bytes) -> str:
    if not data:
        return ""
    for trim in range(4):
        try:
            return data[: len(data) - trim].decode("utf-8") if trim else data.decode("utf-8")
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = _text(value, "")
    return text if text else None


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else 0
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return 0
        return int(parsed) if math.isfinite(parsed) else 0
    return 0


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else 0.0
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return 0.0
        return parsed if math.isfinite(parsed) else 0.0
    return 0.0


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _visible_text(value: object) -> str:
    text = _text(value, "")
    text = _CONTROL_CHARACTERS.sub(lambda match: f"\\x{ord(match.group(0)):02x}", text)
    text = _C1_CONTROLS.sub(lambda match: f"\\u{ord(match.group(0)):04x}", text)
    return _LINE_SEPARATORS.sub(lambda match: f"\\u{ord(match.group(0)):04x}", text)


def _esc(value: object) -> str:
    return escape(_visible_text(value), quote=True)


def _attr(value: object) -> str:
    return escape(_visible_text(value), quote=True)


def _code(value: object) -> str:
    if value is None or value == "":
        return "—"
    return f"<code>{_esc(value)}</code>"


def _join_versions(summary: Mapping[str, Any]) -> str:
    values = [
        _esc(summary.get("renderer_version")),
        _esc(summary.get("parser_version")),
        _esc(summary.get("ir_version")),
    ]
    return " / ".join(values)


def _money(value: object) -> str:
    return f"${_as_float(value):.6f}"


def _seconds(value: object) -> str:
    return f"{_as_float(value):.3f}s"


def _confidence(value: object) -> str:
    number = _optional_float(value)
    return "—" if number is None else f"{number:.2f}"


def _compat_badge(status: str) -> str:
    css = {
        "compatible": "pill-ok",
        "migration": "pill-warn",
        "incompatible": "pill-error",
    }.get(status, "pill")
    return f'<span class="pill {css}">{_esc(status)}</span>'


def _status_pill(status: object) -> str:
    text = _text(status, "unknown")
    css = {
        "promoted": "pill-ok",
        "promotable": "pill-ok",
        "rejected": "pill-error",
        "draft": "pill-warn",
    }.get(text, "pill")
    return f'<span class="pill {css}">{_esc(text)}</span>'


def _stale_badge(badge: str) -> str:
    css = {
        "stale": "pill-stale",
        "invalidated": "pill-stale",
        "not_applicable": "pill-na",
        "none": "pill-na",
        "fresh": "pill-ok",
    }.get(badge, "pill")
    label = badge.replace("_", " ").upper()
    return f'<span class="pill {css}" data-badge="{_attr(badge)}">{_esc(label)}</span>'


def _role_pill(role: str) -> str:
    css = {
        "active": "pill-active",
        "previous": "pill-previous",
        "candidate": "pill-candidate",
        "stale": "pill-stale",
        "retained": "pill",
    }.get(role, "pill")
    return f'<span class="pill {css}" data-role="{_attr(role)}">{_esc(role.upper())}</span>'


def _marker_pill(event: str, marker: Mapping[str, Any]) -> str:
    label = "PROMOTED" if event == "promote" else "ROLLED BACK"
    css = "pill-promote" if event == "promote" else "pill-warn"
    detail = _esc(marker.get("ts") or "")
    reason = _esc(marker.get("reason") or "")
    title = f"{detail} {reason}".strip()
    return (
        f'<span class="pill {css}" data-marker="{_attr(event)}" '
        f'title="{_attr(title)}">{label}</span>'
    )


def _identity_text(value: object) -> str:
    if not isinstance(value, Mapping):
        return "—"
    return (
        f"{value.get('from') or '—'} -> {value.get('to') or '—'} "
        f"({value.get('status') or 'unknown'})"
    )


def _gb(value: object) -> str:
    number = _optional_float(value)
    return "—" if number is None else f"{number:g}GB"


def _gb_signed(value: object) -> str:
    number = _optional_float(value)
    return "—" if number is None else f"{number:+g}GB"


def _compare_value(field: object, value: object) -> str:
    if value is None:
        return "—"
    if field == "cost_usd":
        return _money(value)
    if field == "elapsed_seconds":
        return _seconds(value)
    return _esc(value)


def _diff_marker(changed: bool) -> str:
    if changed:
        return '<span class="diff-marker" data-diff="true">≠</span>'
    return '<span class="diff-marker same" data-diff="false">same</span>'


def _path_cell(entry: object) -> str:
    if not isinstance(entry, Mapping):
        return "—"
    node = _esc(entry.get("node"))
    decision = _optional_text(entry.get("decision"))
    target = _optional_text(entry.get("to"))
    if decision or target:
        return f"<code>{node}</code> [{_esc(decision or '?')} → {_esc(target or '?')}]"
    return f"<code>{node}</code>"


def _node_field_diff(row: Mapping[str, Any], field: str) -> bool:
    return field in cast(list[str], row.get("differences") or [])


def _node_provider_cell(node: object) -> str:
    if not isinstance(node, Mapping):
        return "—"
    provider = _optional_text(node.get("provider")) or "—"
    model = _optional_text(node.get("model")) or "—"
    return f"{_esc(provider)} / {_esc(model)}"


def _optional_integer(node: object, field: str) -> str:
    if not isinstance(node, Mapping) or node.get(field) is None:
        return "—"
    return _integer(node.get(field))


def _optional_money(node: object, field: str) -> str:
    if not isinstance(node, Mapping) or node.get(field) is None:
        return "—"
    return _money(node.get(field))


def _optional_seconds(node: object, field: str) -> str:
    if not isinstance(node, Mapping) or node.get(field) is None:
        return "—"
    return _seconds(node.get(field))


def _integer(value: object) -> str:
    return f"{_as_int(value):,}"


def _short(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _kind_class(value: object) -> str:
    kind = _text(value, "unknown").lower()
    return kind if kind in {"router", "worker", "joiner", "subgraph"} else "unknown"


def _graph_entry(nodes: Sequence[Mapping[str, Any]]) -> str:
    for node in nodes:
        if node.get("entry"):
            return str(node.get("id"))
    return ""


def _boundary_point(
    source: Mapping[str, float], target: Mapping[str, float]
) -> tuple[float, float]:
    source_x = source["x"] + _NODE_WIDTH / 2
    source_y = source["y"] + _NODE_HEIGHT / 2
    target_x = target["x"] + _NODE_WIDTH / 2
    target_y = target["y"] + _NODE_HEIGHT / 2
    dx = target_x - source_x
    dy = target_y - source_y
    if dx == 0 and dy == 0:
        return (source_x, source_y)
    scale_x = (_NODE_WIDTH / 2) / abs(dx) if dx else float("inf")
    scale_y = (_NODE_HEIGHT / 2) / abs(dy) if dy else float("inf")
    scale = min(scale_x, scale_y)
    return (source_x + dx * scale, source_y + dy * scale)


def _angle(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.atan2(end[1] - start[1], end[0] - start[0])


def _arrow_points(x: float, y: float, angle: float, size: float = 9.0) -> str:
    back = angle + math.pi
    left = (x + math.cos(back - 0.42) * size, y + math.sin(back - 0.42) * size)
    right = (x + math.cos(back + 0.42) * size, y + math.sin(back + 0.42) * size)
    return f"{x:.1f},{y:.1f} {left[0]:.1f},{left[1]:.1f} {right[0]:.1f},{right[1]:.1f}"


def _edge_label(edge: Mapping[str, Any]) -> str:
    if edge.get("pruned"):
        reason = _text(edge.get("prune_reason"), "")
        return _short("pruned" + (f": {reason}" if reason else ""), 58)
    when = _text(edge.get("when"), "")
    if edge.get("chosen"):
        return _short("chosen" + (f": {when}" if when else ""), 48)
    return _short(when, 44)


def _json_for_html(model: Mapping[str, Any]) -> str:
    """Serialize data so it cannot close the script element or become markup."""

    document = json.dumps(
        model,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return document.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


#: The fixed inline controller. It only toggles ``hidden``/``open`` and writes a count with
#: ``textContent``. No run-derived value enters it.
_SCRIPT = """\
(function () {
  "use strict";
  var rows = Array.prototype.slice.call(document.querySelectorAll("[data-filter-row]"));
  var nodeInput = document.getElementById("node-filter");
  var outcomeSelect = document.getElementById("outcome-filter");
  var cacheSelect = document.getElementById("cache-filter");
  var count = document.getElementById("filter-count");
  function value(element) { return element ? element.value : "all"; }
  function matches(row) {
    var needle = nodeInput ? nodeInput.value.trim().toLowerCase() : "";
    if (needle && (row.getAttribute("data-search") || "").indexOf(needle) === -1) {
      return false;
    }
    var outcome = value(outcomeSelect);
    if (outcome === "errors" && row.getAttribute("data-status") !== "error") {
      return false;
    }
    if (outcome === "ok" && row.getAttribute("data-status") === "error") {
      return false;
    }
    var cache = value(cacheSelect);
    var hit = row.getAttribute("data-cache") === "hit";
    if (cache === "hits" && !hit) { return false; }
    if (cache === "no-hits" && hit) { return false; }
    return true;
  }
  function apply() {
    var visible = 0;
    rows.forEach(function (row) {
      var show = matches(row);
      row.hidden = !show;
      if (show) { visible += 1; }
    });
    if (count) {
      count.textContent = visible + " of " + rows.length + " rows visible";
    }
  }
  if (nodeInput) { nodeInput.addEventListener("input", apply); }
  if (outcomeSelect) { outcomeSelect.addEventListener("change", apply); }
  if (cacheSelect) { cacheSelect.addEventListener("change", apply); }
  var details = Array.prototype.slice.call(document.querySelectorAll("details[data-expandable]"));
  var expand = document.getElementById("expand-all");
  var collapse = document.getElementById("collapse-all");
  if (expand) { expand.addEventListener("click", function () { details.forEach(function (item) { item.open = true; }); }); }
  if (collapse) { collapse.addEventListener("click", function () { details.forEach(function (item) { item.open = false; }); }); }
  apply();
})();
"""


_CSS = """\
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #172033;
  --muted: #5c667a;
  --line: #d8dde6;
  --accent: #0b5fff;
  --error: #b42318;
  --warn: #8a4b00;
  --router: #dbeafe;
  --router-line: #1d4ed8;
  --worker: #dcfce7;
  --worker-line: #15803d;
  --joiner: #fef3c7;
  --joiner-line: #b45309;
  --subgraph: #ede9fe;
  --subgraph-line: #6d28d9;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
h1 { font-size: 26px; margin: 0 0 6px; }
h2 { font-size: 19px; margin: 0 0 12px; }
h3 { font-size: 15px; margin: 18px 0 8px; }
p { margin: 8px 0; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
code { overflow-wrap: anywhere; }
pre {
  margin: 8px 0 0;
  padding: 12px;
  background: #0f172a;
  color: #e5edf9;
  border-radius: 8px;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  max-height: 420px;
}
.muted { color: var(--muted); }
.banner { padding: 12px 14px; border-radius: 9px; margin: 0 0 18px; border: 1px solid; }
.banner-warn { background: #fff7e6; border-color: #f0b429; color: #4a2c00; }
.banner-note { background: #eef6ff; border-color: #9cc4ff; }
.banner ul { margin: 0; padding-left: 20px; }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 18px;
  margin: 0 0 18px;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
}
.panel-error { border-color: #f2b8b5; background: #fff7f6; }
.kv { width: 100%; border-collapse: collapse; }
.kv th, .kv td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
.kv th { width: 240px; color: var(--muted); font-weight: 600; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 14px 0; }
.metric { border: 1px solid var(--line); border-radius: 10px; padding: 12px; background: #fbfcfe; }
.metric-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.metric-value { font-size: 20px; font-weight: 700; margin-top: 4px; }
.table-wrap { width: 100%; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 7px 9px; border-bottom: 1px solid var(--line); vertical-align: top; text-align: left; }
thead th { background: #f1f4f9; color: #344054; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }
tbody tr:hover { background: #fafcff; }
tbody tr.pruned-row { background: #fff7f6; }
.number { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.reason-cell { min-width: 180px; max-width: 340px; }
.status { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 12px; border: 1px solid; }
.status-ok { color: #05603a; background: #d1fadf; border-color: #75e0a7; }
.status-error { color: var(--error); background: #fee4e2; border-color: #fda29b; }
.status-pruned { color: #6941c6; background: #f4ebff; border-color: #d6bbfb; }
.status-not-run { color: #475467; background: #f2f4f7; border-color: #d0d5dd; }
.pill { display: inline-block; padding: 0 6px; border-radius: 999px; background: #eef2f7; border: 1px solid #d0d5dd; color: #344054; font-size: 11px; white-space: nowrap; }
.pill-chosen { background: #e0ecff; border-color: #9cc4ff; color: #0b4dd8; }
.pill-pruned { background: #fee4e2; border-color: #fda29b; color: var(--error); }
.pill-exit { background: #dcfce7; border-color: #86efac; color: #166534; font-weight: 700; }
.pill-warn { background: #fff7e6; border-color: #f0b429; color: var(--warn); }
.pill-ok { background: #d1fadf; border-color: #75e0a7; color: #05603a; }
.pill-error { background: #fee4e2; border-color: #fda29b; color: var(--error); font-weight: 700; }
.pill-active { background: #dbeafe; border-color: #60a5fa; color: #1e3a8a; font-weight: 700; }
.pill-previous { background: #f2f4f7; border-color: #98a2b3; color: #344054; font-weight: 700; }
.pill-candidate { background: #ede9fe; border-color: #a78bfa; color: #5b21b6; font-weight: 700; }
.pill-stale { background: #fee4e2; border-color: #fda29b; color: var(--error); font-weight: 700; }
.pill-na { background: #f2f4f7; border-color: #d0d5dd; color: #667085; }
.pill-promote { background: #d1fadf; border-color: #75e0a7; color: #05603a; font-weight: 700; }
.banner-ok { background: #ecfdf3; border-color: #75e0a7; color: #05603a; }
.commands { margin: 8px 0 0; padding-left: 20px; }
.commands li { margin: 8px 0; }
.commands code { background: #f1f4f9; border: 1px solid var(--line); border-radius: 4px; padding: 1px 5px; }
.timeline { list-style: none; margin: 8px 0 0; padding: 0; }
.timeline-entry { border-left: 4px solid var(--line); padding: 8px 12px; margin: 0 0 12px; background: #fbfcfe; }
.timeline-entry[data-role~="active"] { border-left-color: var(--accent); }
.timeline-entry[data-role~="candidate"] { border-left-color: #7c3aed; }
.timeline-entry[data-role~="previous"] { border-left-color: #667085; }
.timeline-entry[data-stale="true"] { border-left-color: var(--error); }
.timeline-head { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 4px; }
.timeline-markers { margin-top: 4px; display: flex; flex-wrap: wrap; gap: 6px; }
.diff-row { background: #fffdf5; }
.diff-cell { background: #fff3cd; }
.missing-cell { color: #667085; background: #f2f4f7; }
.diff-marker { font-weight: 700; color: var(--warn); }
.diff-marker.same { font-weight: 400; color: var(--muted); }
.compare-inputs { margin-bottom: 10px; }
.warnings { margin: 6px 0; padding-left: 20px; color: var(--warn); }
.notice { background: #fff7e6; border-left: 4px solid #f0b429; padding: 8px 10px; border-radius: 4px; }
.path { margin: 8px 0 0; padding: 0; list-style: none; }
.path-step { display: flex; gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--line); }
.path-index { display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px; flex: 0 0 26px; border-radius: 50%; background: #e0ecff; color: #0b4dd8; font-weight: 700; }
.reason { color: var(--muted); margin-top: 3px; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; list-style: none; padding: 0; margin: 8px 0 12px; }
.legend li { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }
.swatch { width: 14px; height: 14px; border-radius: 3px; border: 1px solid var(--line); display: inline-block; }
.swatch-router { background: var(--router); border-color: var(--router-line); }
.swatch-worker { background: var(--worker); border-color: var(--worker-line); }
.swatch-joiner { background: var(--joiner); border-color: var(--joiner-line); }
.swatch-subgraph { background: var(--subgraph); border-color: var(--subgraph-line); }
.swatch-chosen { background: #e0ecff; border-color: var(--accent); }
.swatch-pruned { background: #fee4e2; border-color: var(--error); }
.swatch-exit { background: #dcfce7; border-color: #15803d; }
.svg-wrap { width: 100%; overflow-x: auto; }
#graph-svg { min-width: 420px; width: 100%; height: auto; background: #fbfcfe; border: 1px solid var(--line); border-radius: 9px; }
#graph-svg text { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; fill: var(--text); }
.node rect { stroke-width: 2; }
.node.kind-router rect { fill: var(--router); stroke: var(--router-line); }
.node.kind-worker rect { fill: var(--worker); stroke: var(--worker-line); }
.node.kind-joiner rect { fill: var(--joiner); stroke: var(--joiner-line); }
.node.kind-subgraph rect { fill: var(--subgraph); stroke: var(--subgraph-line); }
.node.kind-unknown rect { fill: #f2f4f7; stroke: #98a2b3; }
.node.chosen rect { stroke: var(--accent); stroke-width: 3.5; }
.node.pruned rect { stroke: var(--error); stroke-width: 2.5; stroke-dasharray: 7 5; fill: #fff5f4; opacity: .92; }
.node-id { font-size: 13px; font-weight: 700; }
.node-kind { font-size: 11px; fill: #475467; }
.node-model { font-size: 10px; fill: #667085; }
.node-badges { font-size: 10px; font-weight: 700; fill: #0b4dd8; }
.node.pruned .node-badges { fill: var(--error); }
.edge { stroke: #98a2b3; stroke-width: 1.5; fill: none; }
.edge.chosen { stroke: var(--accent); stroke-width: 3.2; }
.edge.pruned { stroke: var(--error); stroke-width: 2.6; stroke-dasharray: 8 5; }
.edge-arrow { fill: #98a2b3; }
.edge-arrow.chosen { fill: var(--accent); }
.edge-arrow.pruned { fill: var(--error); }
.edge-label { font-size: 10px; fill: #475467; paint-order: stroke; stroke: #fbfcfe; stroke-width: 4px; }
.filters { display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: end; margin: 12px 0; }
.filters label { display: block; font-size: 12px; color: var(--muted); font-weight: 600; }
.filters input, .filters select { padding: 6px 8px; border: 1px solid #98a2b3; border-radius: 6px; background: #fff; color: var(--text); }
.filters input { min-width: 230px; }
.filters button { padding: 6px 10px; border: 1px solid #98a2b3; border-radius: 6px; background: #fff; color: var(--text); cursor: pointer; }
.filters button:hover { background: #f1f4f9; }
#filter-count { padding-bottom: 6px; }
details.artifact { border: 1px solid var(--line); border-radius: 9px; padding: 8px 10px; margin: 8px 0; }
details.artifact > summary { cursor: pointer; }
details.artifact[open] { background: #fbfcfe; }
.artifact-name { font-weight: 700; }
.artifact-body { margin-top: 6px; }
.errors { margin: 6px 0 0; padding-left: 18px; }
details > summary { color: var(--text); }
a { color: var(--accent); }
:focus-visible { outline: 3px solid #f0b429; outline-offset: 2px; }
@media (max-width: 720px) { body { padding: 12px; } .kv th { width: 150px; } }
"""
