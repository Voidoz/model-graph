"""Fast, offline authoring loop: validate, compile, and mock-run a graph on save.

``mg watch`` is intentionally narrow. It:

* compiles the authored graph through the normal compiler/registry path (so validation is
  the same validation ``mg validate`` performs);
* executes the graph with :class:`~modelgraph.providers.mock.MockProvider` only. The model
  manifests may name ``ollama`` or a remote provider, but this module never constructs a
  real provider, reads weights, or performs a network call;
* records the run under ``--runs-dir`` using a deterministic run id derived from the graph
  hash and request, so repeating a cycle on the same inputs overwrites the same run instead
  of accumulating indistinguishable runs;
* compares the current successful cycle with the previous successful cycle in this watch
  session and reports the graph hash, path/decision, artifact, and usage differences.

The module has no filesystem-watch dependency. The CLI polls with :mod:`time` and detects a
save by hashing the graph file's bytes. That also makes the save detector testable without a
real filesystem watcher: :class:`WatchSession.poll_once` returns ``None`` when nothing changed.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from modelgraph.compiler import compile_graph
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.errors import ModelGraphError
from modelgraph.providers.base import Provider
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.registry import Registry
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.engine import run_graph
from modelgraph.runtime.trace import read_events
from modelgraph.util import canonical_json, sha256_text

WATCH_SCHEMA = "modelgraph/watch-cycle/v0"
#: The fixed authoring request. Keeping it fixed makes the output comparable between saves.
WATCH_TASK = "Write a debounce function in TypeScript."
MOCK_PROVIDER = "mock"


def deterministic_run_id(graph_hash: str, task: str = WATCH_TASK) -> str:
    """Return the stable run id for one watched graph/request combination."""
    digest = sha256_text(
        canonical_json({"graph": graph_hash, "provider": MOCK_PROVIDER, "task": task})
    ).split(":", 1)[1]
    return f"watch_{digest[:16]}"


def _file_signature(path: Path) -> tuple[tuple[str, int, int] | None, str | None]:
    """Return (content hash, mtime_ns, size) so an identical save still counts."""
    try:
        stat = path.stat()
        data = path.read_bytes()
        return (hashlib.sha256(data).hexdigest(), stat.st_mtime_ns, stat.st_size), None
    except FileNotFoundError:
        return None, f"graph file not found: {path}"
    except OSError as exc:
        return None, f"cannot read graph file {path}: {exc}"


def _path_summary(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "node": entry.get("node"),
            "kind": entry.get("kind"),
            "decision": entry.get("decision"),
            "to": entry.get("to"),
            "confidence": entry.get("confidence"),
        }
        for entry in entries
    ]


def _artifact_summary(artifacts: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": artifact.name,
            "hash": artifact.hash,
            "bytes": len(artifact.content.encode("utf-8")),
        }
        for artifact in artifacts
    ]


def _totals(outcome: Any) -> dict[str, Any]:
    usage = getattr(outcome, "usage", {}) or {}
    return {
        "steps": usage.get("steps"),
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": usage.get("cost_usd"),
        # Wall-clock elapsed time is not reproducible. Watch reports the latency the mock
        # provider actually recorded, so two cycles over the same graph produce the same JSON.
        "latency_seconds": _provider_latency(outcome),
    }


def _provider_latency(outcome: Any) -> float:
    run_dir = getattr(outcome, "run_dir", None)
    if run_dir is None:
        return 0.0
    trace = Path(run_dir) / "trace.jsonl"
    if not trace.is_file():
        return 0.0
    try:
        events = read_events(trace)
    except (OSError, ValueError):
        return 0.0
    total = 0.0
    for event in events:
        if event.get("event") != "provider.response" or event.get("cached"):
            continue
        value = event.get("latency_seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
    return round(total, 4)


def _previous_snapshot(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "graph": dict(cast(Mapping[str, Any], report.get("graph") or {})),
        "path": list(cast(Sequence[Mapping[str, Any]], report.get("path") or [])),
        "artifacts": list(cast(Sequence[Mapping[str, Any]], report.get("artifacts") or [])),
        "totals": dict(cast(Mapping[str, Any], report.get("totals") or {})),
    }


def _differs(
    current: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> tuple[bool | None, list[str], dict[str, bool]]:
    """Compare one successful cycle with the previous successful cycle."""
    flags = {"graph": False, "path": False, "artifacts": False, "totals": False}
    if previous is None:
        return None, ["first cycle in this watch session; there is no previous success"], flags
    changes: list[str] = []
    old_graph = cast(Mapping[str, Any], previous.get("graph") or {})
    new_graph = cast(Mapping[str, Any], current.get("graph") or {})
    if old_graph.get("hash") != new_graph.get("hash"):
        flags["graph"] = True
        changes.append(
            f"graph hash: {old_graph.get('hash') or 'unknown'} -> "
            f"{new_graph.get('hash') or 'unknown'}"
        )
    old_path = cast(Sequence[Mapping[str, Any]], previous.get("path") or [])
    new_path = cast(Sequence[Mapping[str, Any]], current.get("path") or [])
    if [dict(item) for item in old_path] != [dict(item) for item in new_path]:
        flags["path"] = True
        changes.append(f"path: {_path_text(old_path)} -> {_path_text(new_path)}")
    old_artifacts = cast(Sequence[Mapping[str, Any]], previous.get("artifacts") or [])
    new_artifacts = cast(Sequence[Mapping[str, Any]], current.get("artifacts") or [])
    if [dict(item) for item in old_artifacts] != [dict(item) for item in new_artifacts]:
        flags["artifacts"] = True
        changes.append(
            f"artifacts: {_artifact_text(old_artifacts)} -> {_artifact_text(new_artifacts)}"
        )
    old_totals = cast(Mapping[str, Any], previous.get("totals") or {})
    new_totals = cast(Mapping[str, Any], current.get("totals") or {})
    if dict(old_totals) != dict(new_totals):
        flags["totals"] = True
        changes.append(f"usage: {_totals_text(old_totals)} -> {_totals_text(new_totals)}")
    changed = any(flags.values())
    if not changed:
        changes.append("no change from the previous successful cycle")
    return changed, changes, flags


def watch_cycle(
    graph_path: str | Path,
    *,
    registry: Registry | None = None,
    runs_dir: str | Path | None = None,
    task: str = WATCH_TASK,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate/compile and mock-run one graph cycle.

    ``previous`` is the previous successful cycle report (or ``None``). The returned
    document never includes a generation timestamp, pid, or random id, so the same graph and
    arguments produce the same JSON.
    """
    source = Path(graph_path).expanduser()
    report: dict[str, Any] = {
        "schema": WATCH_SCHEMA,
        "status": "invalid",
        "graph_path": str(source),
        "provider": MOCK_PROVIDER,
        "graph": None,
        "run": None,
        "path": [],
        "artifacts": [],
        "totals": {},
        "errors": [],
        "previous_success": previous is not None,
        "changed_from_previous": None,
        "changes": [],
        "change_flags": {"graph": False, "path": False, "artifacts": False, "totals": False},
    }
    try:
        result = compile_graph(
            source,
            registry=registry,
            strict_models=True,
            graph_paths=[source.parent],
        )
    except ModelGraphError as exc:
        report["errors"] = [f"[{exc.code}] {exc.message}"]
        return report
    except OSError as exc:
        report["errors"] = [f"cannot read graph: {exc}"]
        return report
    if not result.ok or result.graph is None:
        report["errors"] = list(result.errors) or ["graph validation failed"]
        return report

    graph = result.graph
    report["graph"] = {"id": graph.id, "version": graph.version, "hash": graph.graph_hash}
    scenario = MockScenario()

    def factory(manifest: ModelManifest) -> Provider:
        # Deliberately ignores manifest.spec.provider: watch is a mock-only surface.
        return MockProvider(scenario, max_context=manifest.spec.context_window)

    try:
        outcome = run_graph(
            graph,
            task,
            factory,
            runs_dir=runs_dir,
            budget=Budget.from_defaults(graph.defaults),
            run_id=deterministic_run_id(graph.graph_hash, task),
            seed=0,
        )
    except ModelGraphError as exc:
        report["status"] = "failed"
        report["errors"] = [f"[{exc.code}] {exc.message}"]
        return report

    report["status"] = "ok" if outcome.ok else "failed"
    report["run"] = {
        "run_id": outcome.run_id,
        "run_dir": str(outcome.run_dir) if outcome.run_dir else None,
        "outcome": outcome.outcome,
    }
    report["path"] = _path_summary(
        [entry.model_dump(mode="json", exclude_none=True) for entry in outcome.path]
    )
    report["artifacts"] = _artifact_summary(outcome.artifacts)
    report["totals"] = _totals(outcome)
    if outcome.error:
        report["errors"] = [
            f"[{outcome.error.get('code')}] {outcome.error.get('message')}",
        ]
    if report["status"] == "ok":
        changed, changes, flags = _differs(report, previous)
        report["changed_from_previous"] = changed
        report["changes"] = changes
        report["change_flags"] = flags
    elif previous is not None:
        report["changes"] = ["the previous successful cycle is retained for comparison"]
    return report


class WatchSession:
    """One watch session: detect saves by content and remember the last success."""

    def __init__(
        self,
        graph_path: str | Path,
        *,
        registry: Registry | None = None,
        runs_dir: str | Path | None = None,
        task: str = WATCH_TASK,
    ) -> None:
        self.graph_path = Path(graph_path).expanduser()
        self.registry = registry
        self.runs_dir = runs_dir
        self.task = task
        self._last_signature: tuple[str, int, int] | None = None
        self._initialized = False
        self._previous_success: dict[str, Any] | None = None
        self.cycles = 0

    def poll_once(self) -> dict[str, Any] | None:
        """Run a cycle on the first poll or when the file was saved/touched.

        The signature includes the content hash and the filesystem mtime, so an editor save
        that does not change the text still counts as a save. A missing file is reported once;
        it is reported again when it reappears.
        """
        signature, read_error = _file_signature(self.graph_path)
        if self._initialized and read_error is None and signature == self._last_signature:
            return None
        if self._initialized and read_error is not None and self._last_signature is None:
            return None
        self._initialized = True
        self._last_signature = signature
        self.cycles += 1
        report = watch_cycle(
            self.graph_path,
            registry=self.registry,
            runs_dir=self.runs_dir,
            task=self.task,
            previous=self._previous_success,
        )
        if report["status"] == "ok":
            self._previous_success = _previous_snapshot(report)
        return report


def watch_loop(
    session: WatchSession,
    *,
    interval: float,
    emit: Callable[[dict[str, Any]], None],
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll until ``KeyboardInterrupt``; the CLI decides how to render and exit."""
    while True:
        report = session.poll_once()
        if report is not None:
            emit(report)
        sleep(interval)


def _path_text(entries: Sequence[Mapping[str, Any]]) -> str:
    if not entries:
        return "(empty)"
    parts: list[str] = []
    for entry in entries:
        node = str(entry.get("node") or "?")
        decision = entry.get("decision")
        target = entry.get("to")
        if decision or target:
            parts.append(f"{node} [{decision or '?'} -> {target or '?'}]")
        else:
            parts.append(node)
    return " -> ".join(parts)


def _artifact_text(entries: Sequence[Mapping[str, Any]]) -> str:
    if not entries:
        return "(none)"
    return ", ".join(f"{item.get('name')}={item.get('hash')}" for item in entries)


def _totals_text(totals: Mapping[str, Any]) -> str:
    return (
        f"steps={totals.get('steps')} tokens={totals.get('total_tokens')} "
        f"cost=${_as_float(totals.get('cost_usd')):.6f} "
        f"latency={_as_float(totals.get('latency_seconds')):.3f}s"
    )


def format_watch_report(report: Mapping[str, Any]) -> list[str]:
    """Human-readable cycle output. Every value here is untrusted graph/run text."""
    status = str(report.get("status"))
    graph = cast(Mapping[str, Any], report.get("graph") or {})
    lines = [f"watch:    {status.upper()} (mock provider; no model is loaded or called)"]
    if graph:
        lines.append(
            f"graph:    {graph.get('id')}@{graph.get('version')} ({graph.get('hash') or 'no hash'})"
        )
    else:
        lines.append(f"graph:    {report.get('graph_path')}")
    run = cast(Mapping[str, Any] | None, report.get("run"))
    if isinstance(run, Mapping):
        lines.append(
            f"run:      {run.get('run_id')} ({run.get('outcome')}) "
            f"{run.get('run_dir') or ''}".rstrip()
        )
    path = cast(Sequence[Mapping[str, Any]], report.get("path") or [])
    lines.append(f"path:     {_path_text(path)}")
    artifacts = cast(Sequence[Mapping[str, Any]], report.get("artifacts") or [])
    lines.append(f"artifacts: {_artifact_text(artifacts)}")
    totals = cast(Mapping[str, Any], report.get("totals") or {})
    if totals:
        lines.append(f"usage:    {_totals_text(totals)}")
    changed = report.get("changed_from_previous")
    if status != "ok":
        if report.get("previous_success"):
            lines.append(
                "previous: the last successful cycle is retained; this failed cycle did not "
                "replace it"
            )
        else:
            lines.append("previous: no successful cycle yet")
    elif changed is None:
        lines.append("previous: no previous successful cycle in this session")
    elif changed:
        lines.append("previous: differs from the last successful cycle")
    else:
        lines.append("previous: identical to the last successful cycle")
    for item in cast(Sequence[Any], report.get("changes") or []):
        lines.append(f"  - {item}")
    for item in cast(Sequence[Any], report.get("errors") or []):
        lines.append(f"error:    {item}")
    return lines


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0
