"""JSON handlers for the local Model Graph GUI API.

The handlers are deliberately free of HTTP details: :func:`handle_api` takes a method, a
request target, and a raw body, and returns an :class:`ApiResponse` that the stdlib server
serializes. This keeps the API testable without sockets and keeps the runtime semantics in
the existing compiler, registry, runtime, and replay modules.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlsplit

from modelgraph import __version__
from modelgraph.compiler import CompiledGraph, compile_graph
from modelgraph.contracts.loader import load_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.errors import ModelGraphError
from modelgraph.providers import build_provider
from modelgraph.providers.base import Provider
from modelgraph.providers.mock import MockProvider
from modelgraph.registry import Registry
from modelgraph.runtime.engine import run_graph
from modelgraph.runtime.replay import replay_run
from modelgraph.runtime.trace import read_events
from modelgraph.util import sha256_text

logger = logging.getLogger(__name__)

# API-specific error codes. The protocol leaves MG-9xxx unused, so these cannot be confused
# with compiler (MG-1xxx), runtime (MG-2xxx), provider (MG-3xxx), or budget (MG-4xxx) codes.
INVALID_REQUEST = "MG-9001"
NOT_FOUND = "MG-9002"
METHOD_NOT_ALLOWED = "MG-9003"
REPLAY_MISMATCH = "MG-9004"
INTERNAL_ERROR = "MG-9005"
REQUEST_TOO_LARGE = "MG-9006"
MODEL_PULL_ERROR = "MG-9007"

DEFAULT_RUN_LIMIT = 20
MAX_RUN_LIMIT = 500
MAX_GRAPH_BYTES = 500_000
MAX_PULL_JOBS = 2
PROVIDERS = frozenset({"mock", "auto"})
PULL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*(?::[A-Za-z0-9._-]+)?$")

_CODE_RE = re.compile(r"MG-\d{4}")
_ALLOWED_HASH_MISMATCH = "graph hash mismatch allowed by flag"

NEW_GRAPH_TEMPLATE = """\
apiVersion: modelgraph/v0
kind: Graph
metadata:
  id: new-graph
  name: New graph
  version: 1.0.0
  description: Describe what this graph does.
spec:
  entry: triage
  inputs:
    - name: task
      type: text
      required: true
      description: The user request.
  exit: [answer]
  defaults:
    max_steps: 6
    max_depth: 2
    max_cost_usd: 0.05
    max_wall_seconds: 120
    max_total_tokens: 24000
  nodes:
    - id: triage
      kind: router
      description: Decides which worker should handle the request.
      model: router-small
      reads: [task]
      routing:
        confidence_threshold: 0.6
        on_low_confidence: fallback
    - id: answer
      kind: worker
      description: Produces the final answer.
      model: coder-7b
      reads: [task]
      instruction: |
        Complete the request and put the result in the artifact named "answer".
      writes:
        - name: answer
          type: text
  edges:
    - from: triage
      to: answer
      when: The request needs a direct answer.
      examples:
        - Answer this request.
      artifact: text
      priority: 0
"""


@dataclass
class PullJob:
    """One local Ollama pull operation; the server never runs a shell."""

    job_id: str
    model: str
    status: str = "running"
    lines: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model": self.model,
            "status": self.status,
            "lines": list(self.lines[-40:]),
            "error": self.error,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class WebContext:
    """The local resources one server instance is allowed to touch."""

    registry: Registry
    runs_dir: Path
    pull_jobs: dict[str, PullJob] = field(default_factory=dict)
    pull_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class ApiResponse:
    status: int
    payload: dict[str, Any]


class ApiError(Exception):
    """A typed error that maps directly to a JSON error envelope."""

    def __init__(self, status: int, code: str, message: str) -> None:
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


def error_payload(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def handle_api(
    context: WebContext,
    method: str,
    raw_target: str,
    body: bytes = b"",
) -> ApiResponse:
    """Route one API request and always return a JSON-serializable response."""
    try:
        split = urlsplit(raw_target)
        segments = [unquote(part) for part in split.path.split("/") if part]
        if not segments or segments[0] != "api":
            raise ApiError(404, NOT_FOUND, "not an API endpoint")
        route = segments[1:]
        query = parse_qs(split.query, keep_blank_values=True)
        if method == "GET":
            return _handle_get(context, route, query)
        if method == "POST":
            return _handle_post(context, route, body)
        raise ApiError(405, METHOD_NOT_ALLOWED, f"method {method} is not allowed")
    except ApiError as exc:
        return ApiResponse(exc.status, error_payload(exc.code, exc.message))
    except ModelGraphError as exc:
        return ApiResponse(500, error_payload(INTERNAL_ERROR, exc.message))
    except Exception:  # pragma: no cover - defensive boundary; never leak a stack trace
        logger.exception("unhandled error while serving %s", raw_target)
        return ApiResponse(500, error_payload(INTERNAL_ERROR, "internal server error"))


def _handle_get(
    context: WebContext,
    route: list[str],
    query: dict[str, list[str]],
) -> ApiResponse:
    if route == ["health"]:
        return ApiResponse(
            200,
            {"ok": True, "version": __version__, "ui": "1"},
        )
    if route == ["graphs"]:
        return _list_graphs(context)
    if len(route) == 2 and route[0] == "graphs":
        return _graph_detail(context, route[1])
    if len(route) == 3 and route[0] == "graphs" and route[2] == "source":
        return _graph_source(context, route[1])
    if route == ["graph-template"]:
        return ApiResponse(200, {"yaml": NEW_GRAPH_TEMPLATE})
    if route == ["models"]:
        return _list_models(context)
    if route == ["models", "installed"]:
        return _installed_models()
    if len(route) == 3 and route[0] == "models" and route[1] == "pull":
        return _pull_status(context, route[2])
    if route == ["runs"]:
        return _list_runs(context, query)
    if len(route) >= 2 and route[0] == "runs":
        run_id = route[1]
        if len(route) == 2:
            return _run_manifest(context, run_id)
        if len(route) == 3 and route[2] == "trace":
            return _run_trace(context, run_id)
        if len(route) == 4 and route[2] == "artifacts":
            return _run_artifact(context, run_id, route[3])
    raise ApiError(404, NOT_FOUND, f"unknown API endpoint {route!r}")


def _handle_post(context: WebContext, route: list[str], body: bytes) -> ApiResponse:
    if route == ["runs"]:
        return _create_run(context, body)
    if len(route) == 3 and route[0] == "runs" and route[2] == "replay":
        return _replay(context, route[1], body)
    if route == ["graphs", "validate"]:
        return _validate_graph_source(context, body)
    if route == ["graphs", "candidates"]:
        return _save_candidate(context, body)
    if route == ["graphs", "run-candidate"]:
        return _run_candidate(context, body)
    if route == ["models", "pull"]:
        return _start_pull(context, body)
    raise ApiError(404, NOT_FOUND, f"unknown API endpoint {route!r}")


def _list_graphs(context: WebContext) -> ApiResponse:
    summaries: list[dict[str, Any]] = []
    refs = sorted(ref for ref in context.registry.graph_sources if "@" not in ref)
    for graph_id in refs:
        source = context.registry.resolve_graph(graph_id)
        if source is None:
            continue
        result = compile_graph(source, registry=context.registry, strict_models=False)
        graph = result.graph
        if graph is None:
            continue
        summaries.append(
            {
                "id": graph.id,
                "version": graph.version,
                "name": _graph_name(source, graph.id),
                "description": graph.description,
                "source": graph.source,
                "hash": graph.graph_hash,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "warnings": list(result.warnings),
            }
        )
    return ApiResponse(200, {"graphs": summaries})


def _graph_detail(context: WebContext, graph_id: str) -> ApiResponse:
    graph, source = _compiled_graph(context, graph_id, strict=False)
    # ``name`` and ``hash`` are additive convenience fields for the UI: the compiled IR is the
    # contract, but it intentionally does not carry the YAML metadata name and ``to_dict``
    # omits the graph hash. Keeping them top-level leaves ``graph`` exactly to_dict().
    return ApiResponse(
        200,
        {
            "graph": graph.to_dict(),
            "name": _graph_name(source, graph.id),
            "hash": graph.graph_hash,
            "warnings": list(graph.warnings),
        },
    )


def _graph_source(context: WebContext, graph_id: str) -> ApiResponse:
    source = context.registry.resolve_graph(graph_id)
    if source is None or not source.is_file():
        raise ApiError(404, NOT_FOUND, f"unknown graph {graph_id!r}")
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ApiError(500, INTERNAL_ERROR, "graph source is unreadable") from exc
    return ApiResponse(
        200,
        {
            "graph_id": graph_id,
            "file": source.name,
            "yaml": text,
        },
    )


def _source_text(payload: dict[str, Any]) -> str:
    text = payload.get("yaml")
    if not isinstance(text, str) or not text.strip():
        raise ApiError(400, INVALID_REQUEST, "yaml is required")
    if len(text.encode("utf-8")) > MAX_GRAPH_BYTES:
        raise ApiError(413, REQUEST_TOO_LARGE, "graph YAML is too large")
    return text


def _compile_yaml(
    context: WebContext,
    text: str,
    *,
    strict: bool,
) -> tuple[CompiledGraph | None, list[str], list[str]]:
    with tempfile.TemporaryDirectory(prefix="mg-gui-graph-") as directory:
        path = Path(directory) / "candidate.graph.yaml"
        path.write_text(text, encoding="utf-8")
        try:
            result = compile_graph(
                path,
                registry=context.registry,
                strict_models=strict,
                graph_paths=[Path(directory)],
            )
        except ModelGraphError as exc:
            return None, [f"[{exc.code}] {exc.message}"], []
    graph = result.graph
    if graph is not None:
        # Keep the temporary compile path out of the browser preview and the run manifest.
        graph = replace(graph, source="gui-candidate")
    errors = [message.replace(str(path), "candidate.graph.yaml") for message in result.errors]
    warnings = [message.replace(str(path), "candidate.graph.yaml") for message in result.warnings]
    return graph, errors, warnings


def _validate_graph_source(context: WebContext, body: bytes) -> ApiResponse:
    payload = _json_object(body)
    text = _source_text(payload)
    graph, errors, warnings = _compile_yaml(context, text, strict=False)
    return ApiResponse(
        200,
        {
            "valid": graph is not None and not errors,
            "errors": errors,
            "warnings": warnings,
            "graph": graph.to_dict() if graph is not None else None,
        },
    )


def _save_candidate(context: WebContext, body: bytes) -> ApiResponse:
    payload = _json_object(body)
    text = _source_text(payload)
    graph, errors, warnings = _compile_yaml(context, text, strict=True)
    if graph is None or errors:
        raise ApiError(
            400,
            _first_code(errors, "MG-1001"),
            "candidate is invalid: " + "; ".join(errors),
        )
    graph_id = graph.id
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", graph_id):
        raise ApiError(400, INVALID_REQUEST, "graph id is not safe for a local candidate file")
    digest = sha256_text(text)
    directory = context.runs_dir / "gui-candidates"
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{graph_id}-{digest.split(':', 1)[-1][:12]}.graph.yaml"
    destination = directory / filename
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    return ApiResponse(
        201,
        {
            "candidate_id": f"cand_{digest.split(':', 1)[-1][:16]}",
            "graph_id": graph_id,
            "path": str(destination.relative_to(context.runs_dir)),
            "hash": digest,
            "warnings": warnings,
        },
    )


def _run_candidate(context: WebContext, body: bytes) -> ApiResponse:
    payload = _json_object(body)
    text = _source_text(payload)
    task = payload.get("task")
    provider_name = payload.get("provider", "mock")
    if not isinstance(task, str) or not task.strip():
        raise ApiError(400, INVALID_REQUEST, "task is required")
    if not isinstance(provider_name, str) or provider_name not in PROVIDERS:
        raise ApiError(400, INVALID_REQUEST, "provider must be 'mock' or 'auto'")
    graph, errors, _ = _compile_yaml(context, text, strict=True)
    if graph is None or errors:
        raise ApiError(
            400,
            _first_code(errors, "MG-1001"),
            "candidate is invalid: " + "; ".join(errors),
        )

    def factory(manifest: ModelManifest) -> Provider:
        if provider_name == "mock":
            return MockProvider(None, max_context=manifest.spec.context_window)
        return build_provider(manifest)

    outcome = run_graph(graph, task, factory, runs_dir=context.runs_dir)
    error = outcome.error or {}
    code = error.get("code")
    if isinstance(code, str) and code.startswith("MG-3"):
        raise ApiError(502, code, str(error.get("message") or "provider failure"))
    return ApiResponse(201, outcome.to_dict())


def _installed_models() -> ApiResponse:
    executable = shutil.which("ollama")
    if executable is None:
        return ApiResponse(
            200,
            {
                "available": False,
                "installed": [],
                "detail": "ollama is not on PATH",
            },
        )
    try:
        completed = subprocess.run(
            [executable, "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ApiResponse(
            200,
            {"available": False, "installed": [], "detail": "ollama list did not complete"},
        )
    installed: list[str] = []
    for line in completed.stdout.splitlines()[1:]:
        name = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        if name:
            installed.append(name)
    return ApiResponse(
        200,
        {
            "available": completed.returncode == 0,
            "installed": sorted(set(installed)),
            "detail": None if completed.returncode == 0 else "ollama list failed",
        },
    )


def _start_pull(context: WebContext, body: bytes) -> ApiResponse:
    payload = _json_object(body)
    name = payload.get("name")
    if not isinstance(name, str) or not PULL_NAME_RE.fullmatch(name.strip()):
        raise ApiError(
            400,
            INVALID_REQUEST,
            "model name must be a plain Ollama tag such as qwen2.5:1.5b-instruct",
        )
    name = name.strip()
    executable = shutil.which("ollama")
    if executable is None:
        raise ApiError(503, MODEL_PULL_ERROR, "ollama is not on PATH")
    with context.pull_lock:
        running = sum(job.status == "running" for job in context.pull_jobs.values())
        if running >= MAX_PULL_JOBS:
            raise ApiError(429, MODEL_PULL_ERROR, "too many model pulls are already running")
        job_id = f"pull_{sha256_text(name + str(time.time()))[7:19]}"
        job = PullJob(job_id=job_id, model=name)
        context.pull_jobs[job_id] = job
    thread = threading.Thread(
        target=_pull_worker,
        args=(executable, name, job, context.pull_lock),
        daemon=True,
    )
    thread.start()
    return ApiResponse(202, job.to_dict())


def _pull_status(context: WebContext, job_id: str) -> ApiResponse:
    with context.pull_lock:
        job = context.pull_jobs.get(job_id)
        if job is None:
            raise ApiError(404, NOT_FOUND, f"unknown pull job {job_id!r}")
        return ApiResponse(200, job.to_dict())


def _pull_worker(executable: str, name: str, job: PullJob, lock: threading.Lock) -> None:
    try:
        process = subprocess.Popen(
            [executable, "pull", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        with lock:
            job.status = "failed"
            job.error = f"could not start ollama pull: {exc}"
        return
    if process.stdout is not None:
        for line in process.stdout:
            with lock:
                job.lines.append(line.rstrip())
                if len(job.lines) > 200:
                    del job.lines[: len(job.lines) - 200]
    returncode = process.wait()
    with lock:
        if returncode == 0:
            job.status = "succeeded"
        else:
            job.status = "failed"
            job.error = f"ollama pull exited with status {returncode}"


def _list_models(context: WebContext) -> ApiResponse:
    rows: list[dict[str, Any]] = []
    for manifest in sorted(context.registry.manifests.values(), key=lambda item: item.metadata.id):
        rows.append(
            {
                "id": manifest.metadata.id,
                "version": manifest.metadata.version,
                "provider": manifest.spec.provider,
                "model": manifest.spec.model,
                "license": manifest.spec.license.name,
                "description": manifest.metadata.description,
            }
        )
    return ApiResponse(200, {"models": rows})


def _list_runs(context: WebContext, query: dict[str, list[str]]) -> ApiResponse:
    limit = _run_limit(query)
    candidates: list[tuple[float, Path]] = []
    try:
        for item in context.runs_dir.iterdir():
            if not item.is_dir() or not (item / "run.json").is_file():
                continue
            try:
                candidates.append((item.stat().st_mtime, item))
            except OSError:
                continue
    except OSError:
        candidates = []

    # The contract is "newest first", which means the recorded ``started_at`` when it exists.
    # Directory mtime is only a fallback, because copying or touching a run directory must not
    # reorder history.
    records: list[tuple[float, float, Path, dict[str, Any]]] = []
    for mtime, run_dir in candidates:
        try:
            manifest = _read_json_file(run_dir / "run.json", "run manifest is unreadable")
        except ApiError:
            continue
        started = _timestamp(manifest.get("started_at"))
        sort_key = started if started is not None else mtime
        records.append((sort_key, mtime, run_dir, manifest))
    records.sort(key=lambda record: (record[0], record[1]), reverse=True)

    runs: list[dict[str, Any]] = []
    for _, _, run_dir, manifest in records[:limit]:
        graph = manifest.get("graph")
        graph = graph if isinstance(graph, dict) else {}
        usage = manifest.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        runs.append(
            {
                "run_id": manifest.get("run_id", run_dir.name),
                "outcome": manifest.get("outcome"),
                "graph_id": graph.get("id"),
                "graph_version": graph.get("version"),
                "started_at": manifest.get("started_at"),
                "tokens": usage.get("total_tokens"),
                "cost_usd": usage.get("cost_usd", 0.0),
                "dir": str(run_dir),
            }
        )
    return ApiResponse(200, {"runs": runs})


def _run_manifest(context: WebContext, run_id: str) -> ApiResponse:
    run_dir = _run_directory(context, run_id)
    return ApiResponse(200, _read_json_file(run_dir / "run.json", "run manifest is unreadable"))


def _run_trace(context: WebContext, run_id: str) -> ApiResponse:
    run_dir = _run_directory(context, run_id)
    manifest = _read_json_file(run_dir / "run.json", "run manifest is unreadable")
    trace_name = manifest.get("trace") or "trace.jsonl"
    trace_path = _safe_child(run_dir, str(trace_name))
    if trace_path is None:
        raise ApiError(500, INTERNAL_ERROR, "run manifest points at an unsafe trace path")
    if not trace_path.is_file():
        raise ApiError(404, NOT_FOUND, f"trace for run {run_id!r} was not found")
    try:
        events = read_events(trace_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(500, INTERNAL_ERROR, "run trace is unreadable") from exc
    return ApiResponse(200, {"events": events})


def _run_artifact(context: WebContext, run_id: str, name: str) -> ApiResponse:
    run_dir = _run_directory(context, run_id)
    manifest = _read_json_file(run_dir / "run.json", "run manifest is unreadable")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        entries = []

    latest: dict[str, Any] | None = None
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name") == name:
            latest = entry
    if latest is None:
        raise ApiError(404, NOT_FOUND, f"artifact {name!r} is not in run {run_id!r}")

    artifact_hash = latest.get("hash")
    content: str | None = None
    if isinstance(artifact_hash, str) and artifact_hash:
        digest = artifact_hash.split(":", 1)[-1]
        content_path = _safe_child(run_dir / "artifacts", digest)
        if content_path is not None and content_path.is_file():
            try:
                content = content_path.read_text(encoding="utf-8")
            except OSError:
                content = None
    if content is None:
        content = _envelope_artifact_content(run_dir, name, artifact_hash)
    if content is None:
        raise ApiError(404, NOT_FOUND, f"artifact {name!r} content is not available")

    artifact: dict[str, Any] = {
        "name": latest.get("name"),
        "type": latest.get("type"),
        "producer": latest.get("producer"),
        "hash": artifact_hash,
        "content": content,
    }
    media_type = latest.get("media_type")
    if isinstance(media_type, str):
        artifact["media_type"] = media_type
    return ApiResponse(200, {"artifact": artifact})


def _create_run(context: WebContext, body: bytes) -> ApiResponse:
    payload = _json_object(body)
    graph_id = payload.get("graph_id")
    task = payload.get("task")
    provider_name = payload.get("provider", "mock")

    if not isinstance(graph_id, str) or not graph_id.strip():
        raise ApiError(400, INVALID_REQUEST, "graph_id is required")
    if not isinstance(task, str) or not task.strip():
        raise ApiError(400, INVALID_REQUEST, "task is required")
    if not isinstance(provider_name, str) or provider_name not in PROVIDERS:
        raise ApiError(400, INVALID_REQUEST, "provider must be 'mock' or 'auto'")

    graph, _ = _compiled_graph(context, graph_id.strip(), strict=True)

    def factory(manifest: ModelManifest) -> Provider:
        if provider_name == "mock":
            return MockProvider(None, max_context=manifest.spec.context_window)
        return build_provider(manifest)

    outcome = run_graph(graph, task, factory, runs_dir=context.runs_dir)
    error = outcome.error or {}
    code = error.get("code")
    if isinstance(code, str) and code.startswith("MG-3"):
        raise ApiError(502, code, str(error.get("message") or "provider failure"))
    return ApiResponse(201, outcome.to_dict())


def _replay(context: WebContext, run_id: str, body: bytes) -> ApiResponse:
    payload = _json_object(body)
    allow_ir_mismatch = payload.get("allow_ir_mismatch", False)
    if not isinstance(allow_ir_mismatch, bool):
        raise ApiError(400, INVALID_REQUEST, "allow_ir_mismatch must be a boolean")

    run_dir = _run_directory(context, run_id)
    try:
        result = replay_run(run_dir, allow_ir_mismatch=allow_ir_mismatch)
    except ModelGraphError as exc:
        raise ApiError(400, _first_code([exc.message], "MG-1001"), exc.message) from exc
    if not result.ok:
        # A caller may explicitly accept an IR hash change. The replay engine records that
        # acceptance in ``mismatches``; it is not a failure unless something else changed.
        blocking = [
            mismatch for mismatch in result.mismatches if mismatch != _ALLOWED_HASH_MISMATCH
        ]
        if not allow_ir_mismatch or blocking:
            raise ApiError(
                409,
                REPLAY_MISMATCH,
                "; ".join(result.mismatches) or "replay did not match the recorded run",
            )
    return ApiResponse(
        200,
        {
            "ok": True,
            "run_id": result.run_id,
            "replay_id": result.replay_id,
            "mismatches": result.mismatches,
        },
    )


def _compiled_graph(
    context: WebContext,
    graph_id: str,
    *,
    strict: bool,
) -> tuple[CompiledGraph, Path]:
    source = context.registry.resolve_graph(graph_id)
    if source is None or not source.is_file():
        raise ApiError(404, NOT_FOUND, f"unknown graph {graph_id!r}")
    result = compile_graph(
        source,
        registry=context.registry,
        strict_models=strict,
        graph_paths=[source.parent],
    )
    if result.graph is None:
        raise ApiError(
            400,
            _first_code(result.errors, "MG-1001"),
            "graph validation failed: " + "; ".join(result.errors),
        )
    return result.graph, source


def _graph_name(source: Path, fallback: str) -> str:
    try:
        document = load_document(source)
    except ModelGraphError:
        return fallback
    metadata = document.get("metadata")
    if isinstance(metadata, dict):
        name = metadata.get("name")
        if isinstance(name, str) and name.strip():
            return name
    return fallback


def _run_limit(query: dict[str, list[str]]) -> int:
    values = query.get("limit")
    if not values or not values[0]:
        return DEFAULT_RUN_LIMIT
    try:
        limit = int(values[0])
    except ValueError as exc:
        raise ApiError(400, INVALID_REQUEST, "limit must be a positive integer") from exc
    if limit < 1 or limit > MAX_RUN_LIMIT:
        raise ApiError(
            400,
            INVALID_REQUEST,
            f"limit must be between 1 and {MAX_RUN_LIMIT}",
        )
    return limit


def _run_directory(context: WebContext, run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id or "\x00" in run_id:
        raise ApiError(404, NOT_FOUND, f"unknown run {run_id!r}")
    root = context.runs_dir.resolve()
    candidate = (root / run_id).resolve()
    if not candidate.is_relative_to(root) or not (candidate / "run.json").is_file():
        raise ApiError(404, NOT_FOUND, f"unknown run {run_id!r}")
    return candidate


def _safe_child(root: Path, name: str) -> Path | None:
    """Resolve ``name`` under ``root`` or return None when it would escape."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        return None
    root_resolved = root.resolve()
    candidate = (root_resolved / name).resolve()
    if not candidate.is_relative_to(root_resolved):
        return None
    return candidate


def _read_json_file(path: Path, message: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(500, INTERNAL_ERROR, message) from exc
    if not isinstance(value, dict):
        raise ApiError(500, INTERNAL_ERROR, message)
    return cast(dict[str, Any], value)


def _timestamp(value: object) -> float | None:
    """Parse a recorded ISO-8601 timestamp, returning None when it is unusable."""
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def _json_object(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(400, INVALID_REQUEST, "request body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ApiError(400, INVALID_REQUEST, "request body must be a JSON object")
    return cast(dict[str, Any], value)


def _envelope_artifact_content(
    run_dir: Path,
    name: str,
    artifact_hash: object,
) -> str | None:
    path = run_dir / "envelope.json"
    if not path.is_file():
        return None
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    artifacts = envelope.get("artifacts") if isinstance(envelope, dict) else None
    if not isinstance(artifacts, list):
        return None
    content: str | None = None
    for entry in artifacts:
        if not isinstance(entry, dict) or entry.get("name") != name:
            continue
        if artifact_hash is not None and entry.get("hash") != artifact_hash:
            continue
        value = entry.get("content")
        if isinstance(value, str):
            content = value
    return content


def _first_code(messages: tuple[str, ...] | list[str], fallback: str) -> str:
    for message in messages:
        match = _CODE_RE.search(message)
        if match:
            return match.group(0)
    return fallback
