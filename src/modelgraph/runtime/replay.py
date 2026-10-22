"""Deterministic replay from a recorded trace."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from modelgraph.compiler import compile_graph
from modelgraph.errors import ModelGraphError
from modelgraph.runtime.engine import RunOutcome, run_graph
from modelgraph.runtime.trace import ReplayProvider, read_events
from modelgraph.util import new_id


@dataclass
class ReplayResult:
    ok: bool
    run_id: str
    replay_id: str
    mismatches: list[str] = field(default_factory=list)
    outcome: RunOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "run_id": self.run_id,
            "replay_id": self.replay_id,
            "mismatches": self.mismatches,
            "outcome": self.outcome.outcome if self.outcome else None,
        }


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "run.json"
    if not path.exists():
        raise ModelGraphError(f"run manifest not found: {path}", path=str(path))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def replay_run(
    run_dir: str | Path,
    *,
    allow_ir_mismatch: bool = False,
    persist_report: bool = True,
) -> ReplayResult:
    directory = Path(run_dir)
    manifest = load_manifest(directory)
    trace_path = directory / manifest.get("trace", "trace.jsonl")
    if not trace_path.exists():
        raise ModelGraphError(f"trace not found: {trace_path}", path=str(trace_path))
    events = read_events(trace_path)

    graph_info = manifest.get("graph", {})
    source = graph_info.get("source")
    if not source:
        raise ModelGraphError("run manifest has no graph source", path=str(directory / "run.json"))
    compiled = compile_graph(source, strict_models=True)
    if not compiled.ok or compiled.graph is None:
        raise ModelGraphError("cannot recompile graph: " + "; ".join(compiled.errors))
    mismatches: list[str] = []
    if compiled.graph.graph_hash != graph_info.get("hash"):
        if not allow_ir_mismatch:
            return ReplayResult(
                ok=False,
                run_id=str(manifest.get("run_id", directory.name)),
                replay_id=new_id("replay"),
                mismatches=[
                    f"graph hash changed: {graph_info.get('hash')} -> {compiled.graph.graph_hash}"
                ],
            )
        mismatches.append("graph hash mismatch allowed by flag")

    replay_provider = ReplayProvider(events)
    outcome = run_graph(
        compiled.graph,
        str(manifest.get("request", {}).get("task", "")),
        lambda _manifest: replay_provider,
        persist=False,
        run_id=new_id("replay"),
    )
    if outcome.outcome != manifest.get("outcome"):
        mismatches.append(f"outcome changed: {manifest.get('outcome')} -> {outcome.outcome}")

    recorded = _exit_hashes(manifest, compiled.graph.exits)
    replayed = {artifact.name: artifact.hash for artifact in outcome.artifacts}
    for name in compiled.graph.exits:
        if recorded.get(name) != replayed.get(name):
            mismatches.append(
                f"artifact {name!r} hash changed: {recorded.get(name)} -> {replayed.get(name)}"
            )
    recorded_path = [
        (entry.get("node"), entry.get("decision"), entry.get("to"))
        for entry in manifest.get("path", [])
    ]
    replayed_path = [(entry.node, entry.decision, entry.to) for entry in outcome.path]
    if recorded_path != replayed_path:
        mismatches.append("routing path changed")

    result = ReplayResult(
        ok=not mismatches,
        run_id=str(manifest.get("run_id", directory.name)),
        replay_id=outcome.run_id,
        mismatches=mismatches,
        outcome=outcome,
    )
    if persist_report:
        (directory / "replay.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    return result


def _exit_hashes(manifest: dict[str, Any], exits: tuple[str, ...]) -> dict[str, str]:
    latest: dict[str, str] = {}
    for artifact in manifest.get("artifacts", []):
        name = str(artifact.get("name"))
        value = artifact.get("hash")
        if name in exits and value:
            latest[name] = str(value)
    return latest
