"""Run manifest construction, atomic persistence, and interrupted-run recovery.

A run manifest is the evidence layer for one runtime execution: it pins the graph,
model manifests, renderer/parser/IR versions, profile, budget, seed, environment,
outcome, usage, artifacts, and trace. The file at ``runs/<run-id>/run.json`` is
written provisionally at the start of a run and atomically replaced with the final
manifest at the end. A process that dies leaves the provisional file behind; the
recovery helpers here turn that file into an ``interrupted`` manifest.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelgraph import __version__
from modelgraph.compiler.ir import CompiledGraph
from modelgraph.contracts.envelope import Artifact, PathEntry
from modelgraph.errors import ModelGraphError
from modelgraph.providers.base import Provider
from modelgraph.runtime.budget import Budget
from modelgraph.security import redact_data, redact_text_default
from modelgraph.util import canonical_json, sha256_text, utc_now
from modelgraph.version import PARSER_VERSION, RENDERER_VERSION

MANIFEST_FILENAME = "run.json"
IR_VERSION = "modelgraph/ir/v0"

STATUS_PROVISIONAL = "provisional"
STATUS_FINAL = "final"
STATUS_INTERRUPTED = "interrupted"

OUTCOME_RUNNING = "running"
OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_FAILED = "failed"
OUTCOME_BUDGET_EXHAUSTED = "budget_exhausted"
OUTCOME_INTERRUPTED = "interrupted"

INTERRUPTED_CODE = "MG-2001"
INTERRUPTED_MESSAGE = "run did not finalize; marked interrupted by recovery"


def manifest_path(run_dir: str | Path) -> Path:
    """Return the canonical manifest path for a run directory."""
    return Path(run_dir) / MANIFEST_FILENAME


def usage_snapshot(budget: Budget) -> dict[str, Any]:
    """The usage totals recorded in a manifest (also used by ``RunOutcome``)."""
    return budget.usage()


def deployment_revision(graph: CompiledGraph, *, target: str = "portable") -> str:
    """Hash the inputs that define an immutable deployment revision.

    ADR-0013 defines a revision over the graph hash, model hashes, target, and
    compiler/IR identity. Compiled plan/student hashes do not exist in the portable
    v0 backend, so they are not invented here.
    """
    models = sorted(
        [node.model_ref, node.model_hash]
        for node in graph.nodes.values()
        if node.model_ref is not None
    )
    payload = {
        "graph": graph.graph_hash,
        "models": models,
        "target": target,
        "ir_version": IR_VERSION,
        "compiler_version": __version__,
        "renderer_version": RENDERER_VERSION,
        "parser_version": PARSER_VERSION,
    }
    digest = sha256_text(canonical_json(payload)).split(":", 1)[1]
    return f"rev_{digest[:16]}"


def build_environment() -> dict[str, Any]:
    """Capture cheap, non-secret environment facts for reproducibility."""
    return {
        "modelgraph_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "providers": [],
    }


def build_run_manifest(
    graph: CompiledGraph,
    *,
    run_id: str,
    task: str,
    budget: Budget,
    started_at: str,
    status: str,
    outcome: str,
    seed: int | None = None,
    parent_run_id: str | None = None,
    profile: Mapping[str, Any] | None = None,
    prune_report: Mapping[str, Any] | None = None,
    target: str = "portable",
    usage: Mapping[str, Any] | None = None,
    path: Sequence[PathEntry] = (),
    artifacts: Sequence[Artifact] = (),
    error: Mapping[str, Any] | None = None,
    finished_at: str | None = None,
    model_loads: Sequence[Mapping[str, Any]] | None = None,
    subgraphs: Sequence[Mapping[str, Any]] = (),
    providers: Sequence[Provider] | None = None,
    trace: str = "trace.jsonl",
    environment: Mapping[str, Any] | None = None,
    redact: bool = True,
) -> dict[str, Any]:
    """Build a schema-shaped manifest for either the provisional or final write."""
    models = _model_records(graph)
    exit_names = set(graph.exits)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "deployment_revision": deployment_revision(graph, target=target),
        "target": target,
        "status": status,
        "outcome": outcome,
        "graph": {
            "id": graph.id,
            "version": graph.version,
            "hash": graph.graph_hash,
            "source": graph.source,
        },
        "request": {"task": task},
        "models": models,
        "budget": budget.snapshot().model_dump(mode="json"),
        "seed": seed,
        "environment": _environment_with_providers(environment, providers),
        "renderer_version": RENDERER_VERSION,
        "parser_version": PARSER_VERSION,
        "ir_version": IR_VERSION,
        "profile_hash": _document_hash(profile) if profile is not None else None,
        "prune_hash": _document_hash(prune_report) if prune_report is not None else None,
        "started_at": started_at,
        "finished_at": finished_at,
        "usage": dict(usage) if usage is not None else usage_snapshot(budget),
        "path": [entry.model_dump(mode="json", exclude_none=True) for entry in path],
        "artifacts": [
            {
                "name": artifact.name,
                "type": artifact.type.value,
                "hash": artifact.hash,
                "producer": artifact.producer,
                "bytes": len(artifact.content),
                "exit": artifact.name in exit_names,
            }
            for artifact in artifacts
        ],
        "exit_artifacts": _exit_artifacts(artifacts, graph.exits),
        "trace": trace,
        "error": dict(error) if error is not None else None,
        "parent_run_id": parent_run_id,
        "subgraphs": [dict(item) for item in subgraphs],
    }
    if profile is not None:
        manifest["profile"] = dict(profile)
    if prune_report is not None:
        manifest["prune"] = dict(prune_report)
    if model_loads is not None:
        manifest["model_loads"] = [dict(item) for item in model_loads]
    if redact:
        _redact_manifest_fields(manifest)
    return manifest


def _redact_manifest_fields(manifest: dict[str, Any]) -> None:
    """Redact user/model-derived text that can carry secrets.

    Hashes, ids, and graph source paths are evidence and are left intact. The request text,
    router reasons, and error payload are the manifest fields that can quote unstructured
    content.
    """

    request = manifest.get("request")
    if isinstance(request, dict) and isinstance(request.get("task"), str):
        request["task"] = redact_text_default(request["task"])
    for entry in manifest.get("path") or []:
        if isinstance(entry, dict) and isinstance(entry.get("reason"), str):
            entry["reason"] = redact_text_default(entry["reason"])
    error = manifest.get("error")
    if isinstance(error, Mapping):
        redacted, _report = redact_data(dict(error))
        manifest["error"] = redacted


def write_manifest_atomic(run_dir: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write ``run.json`` with temp-file + rename so readers never see a partial file."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = manifest_path(directory)
    payload = json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{MANIFEST_FILENAME}.", suffix=".tmp", dir=directory
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def read_run_manifest(
    run_dir: str | Path,
    *,
    recover: bool = False,
    stale_after_seconds: float | None = None,
) -> dict[str, Any]:
    """Read a run manifest, optionally recovering a crashed provisional file."""
    directory = Path(run_dir)
    path = manifest_path(directory)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelGraphError(f"run manifest not found: {path}", path=str(path)) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelGraphError(f"run manifest is unreadable: {exc}", path=str(path)) from exc
    if not isinstance(document, dict):
        raise ModelGraphError("run manifest must be a JSON object", path=str(path))
    if recover and document.get("status") == STATUS_PROVISIONAL:
        recovered = recover_interrupted_run(directory, stale_after_seconds=stale_after_seconds)
        if recovered is not None:
            return recovered
    return document


def recover_interrupted_run(
    run_dir: str | Path,
    *,
    stale_after_seconds: float | None = None,
) -> dict[str, Any] | None:
    """Mark one dead provisional run as ``interrupted``; return the new manifest.

    A manifest whose process is still alive (same host and PID) is left untouched.
    If process identity is unavailable, recovery requires an explicit stale age, so a
    moved or hand-written manifest is never marked interrupted by accident.
    """
    directory = Path(run_dir)
    path = manifest_path(directory)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelGraphError(f"run manifest is unreadable: {exc}", path=str(path)) from exc
    if not isinstance(document, dict) or document.get("status") != STATUS_PROVISIONAL:
        return None

    alive = _manifest_process_alive(document)
    if alive is True:
        return None
    if alive is None:
        if stale_after_seconds is None:
            return None
        if not _older_than(document.get("started_at"), stale_after_seconds):
            return None

    recovered = dict(document)
    previous_pid = (document.get("environment") or {}).get("pid")
    recovered["status"] = STATUS_INTERRUPTED
    recovered["outcome"] = OUTCOME_INTERRUPTED
    recovered["finished_at"] = utc_now()
    recovered["error"] = {
        "code": INTERRUPTED_CODE,
        "message": INTERRUPTED_MESSAGE,
        "path": None,
        "details": {
            "previous_status": STATUS_PROVISIONAL,
            "previous_pid": previous_pid,
            "recovery": "process is no longer running",
        },
    }
    write_manifest_atomic(directory, recovered)
    return recovered


def recover_interrupted_runs(
    runs_dir: str | Path,
    *,
    stale_after_seconds: float | None = None,
) -> list[Path]:
    """Recover every dead provisional manifest under a runs directory.

    Both top-level run directories and ``run_dir/subgraphs/<child-run>/`` manifests
    are inspected. Returns the paths that were rewritten.
    """
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    recovered: list[Path] = []
    for directory in _run_directories(root):
        try:
            result = recover_interrupted_run(directory, stale_after_seconds=stale_after_seconds)
        except ModelGraphError:
            # A corrupt old manifest cannot be repaired, but it must not block a new
            # run from recording its own evidence. Direct inspection still raises.
            continue
        if result is not None:
            recovered.append(manifest_path(directory))
    return recovered


def _run_directories(root: Path) -> list[Path]:
    directories: list[Path] = []
    try:
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError:
        return directories
    for candidate in candidates:
        if manifest_path(candidate).is_file():
            directories.append(candidate)
        subgraphs = candidate / "subgraphs"
        if not subgraphs.is_dir():
            continue
        try:
            children = sorted(path for path in subgraphs.iterdir() if path.is_dir())
        except OSError:
            continue
        directories.extend(child for child in children if manifest_path(child).is_file())
    return directories


def _environment_with_providers(
    environment: Mapping[str, Any] | None,
    providers: Sequence[Provider] | None,
) -> dict[str, Any]:
    result = dict(environment) if environment is not None else build_environment()
    if providers is not None:
        result["providers"] = _provider_records(providers)
    return result


def _provider_records(providers: Sequence[Provider]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for provider in providers:
        name = getattr(provider, "name", type(provider).__name__)
        version = getattr(provider, "version", None) or __version__
        records[str(name)] = {"name": str(name), "version": str(version)}
    return [records[name] for name in sorted(records)]


def _model_records(graph: CompiledGraph) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for node in graph.nodes.values():
        if node.model_ref is None:
            continue
        model = node.model
        revision = model.spec.integrity.revision if model is not None else None
        records.append(
            {
                "node": node.id,
                "ref": node.model_ref,
                "version": model.metadata.version if model is not None else None,
                "hash": node.model_hash,
                "provider": model.spec.provider.value if model is not None else None,
                "model": model.spec.model if model is not None else None,
                "resolved_revision": revision,
                "revision_reason": _revision_reason(model is not None, revision),
                "capabilities": (
                    [capability.value for capability in model.spec.capabilities]
                    if model is not None
                    else []
                ),
            }
        )
    return records


def _revision_reason(resolved: bool, revision: str | None) -> str | None:
    if revision:
        return None
    if not resolved:
        return "model was not resolved"
    return "model manifest does not pin spec.integrity.revision"


def _document_hash(document: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(dict(document)))


def _exit_artifacts(
    artifacts: Sequence[Artifact],
    exits: Sequence[str],
) -> list[dict[str, Any]]:
    latest: dict[str, Artifact] = {}
    for artifact in artifacts:
        if artifact.name in exits:
            latest[artifact.name] = artifact
    return [
        {"name": name, "hash": latest[name].hash}
        for name in exits
        if name in latest and latest[name].hash is not None
    ]


def _memory_bytes() -> int | None:
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None
    try:
        pages = int(sysconf("SC_PHYS_PAGES"))
        page_size = int(sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size


def _manifest_process_alive(manifest: Mapping[str, Any]) -> bool | None:
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        return None
    pid = environment.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    hostname = environment.get("hostname")
    if hostname != socket.gethostname():
        return None
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _older_than(timestamp: object, seconds: float) -> bool:
    if not isinstance(timestamp, str) or not timestamp:
        return False
    try:
        started = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return (datetime.now(UTC) - started).total_seconds() > seconds


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
