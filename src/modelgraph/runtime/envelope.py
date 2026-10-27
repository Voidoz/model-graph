"""Request-scoped state, artifacts, and scoped views."""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from modelgraph.compiler.ir import CompiledGraph, CompiledNode
from modelgraph.contracts.envelope import (
    Artifact,
    Envelope,
    GraphRef,
    Note,
    PathEntry,
    Request,
    RunInfo,
)
from modelgraph.contracts.graph import ArtifactType
from modelgraph.errors import RuntimeFailure
from modelgraph.runtime.budget import Budget
from modelgraph.util import new_id, sha256_text, utc_now


@dataclass
class RuntimeState:
    graph: CompiledGraph
    task: str
    budget: Budget
    run_id: str = field(default_factory=lambda: new_id("run"))
    parent_run_id: str | None = None
    depth: int = 0
    started_at: str = field(default_factory=utc_now)
    inputs: list[Artifact] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    path: list[PathEntry] = field(default_factory=list)
    step: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        graph: CompiledGraph,
        task: str,
        *,
        budget: Budget | None = None,
        depth: int = 0,
        parent_run_id: str | None = None,
        run_id: str | None = None,
    ) -> RuntimeState:
        state = cls(
            graph=graph,
            task=task,
            budget=budget or Budget.from_defaults(graph.defaults),
            run_id=run_id or new_id("run"),
            parent_run_id=parent_run_id,
            depth=depth,
        )
        names = [item.name for item in graph.inputs] or ["task"]
        for index, name in enumerate(names):
            artifact_type = (
                graph.inputs[index].type if index < len(graph.inputs) else ArtifactType.text
            )
            media_type = graph.inputs[index].media_type if index < len(graph.inputs) else None
            state.inputs.append(
                _make_artifact(
                    name=name,
                    artifact_type=artifact_type,
                    content=task,
                    producer="$input",
                    media_type=media_type,
                )
            )
        return state

    def scoped(self, node: CompiledNode) -> list[Artifact]:
        with self._lock:
            artifacts = [*self.inputs, *self.artifacts]
        latest = _latest_by_name(artifacts, self.graph)
        selected: list[Artifact] = []
        for name in node.reads:
            found = latest.get(name)
            if found is None:
                raise RuntimeFailure(
                    f"node {node.id!r} reads {name!r}, which is not available",
                    path=f"{self.graph.id}.{node.id}.reads",
                    details={"code": "MG-2201"},
                )
            selected.append(found)
        return selected

    def write(
        self,
        *,
        name: str,
        artifact_type: ArtifactType | str,
        content: str,
        producer: str,
        media_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        artifact = _make_artifact(
            name=name,
            artifact_type=ArtifactType(artifact_type),
            content=content,
            producer=producer,
            media_type=media_type,
            metadata=metadata,
        )
        with self._lock:
            self.artifacts.append(artifact)
        return artifact

    def note(self, producer: str, content: str) -> None:
        if content.strip():
            with self._lock:
                self.notes.append(
                    Note(producer=producer, content=content.strip(), created_at=utc_now())
                )

    def add_path(self, entry: PathEntry) -> None:
        with self._lock:
            self.path.append(entry)

    def latest_artifact(self, name: str) -> Artifact | None:
        with self._lock:
            artifacts = [*self.inputs, *self.artifacts]
        return _latest_by_name(artifacts, self.graph).get(name)

    def results(self) -> list[Artifact]:
        with self._lock:
            artifacts = [*self.inputs, *self.artifacts]
        latest = _latest_by_name(artifacts, self.graph)
        result: list[Artifact] = []
        for name in self.graph.exits:
            found = latest.get(name)
            if found is None:
                raise RuntimeFailure(
                    f"exit artifact {name!r} was not produced",
                    path=f"{self.graph.id}.exit",
                    details={"code": "MG-1006"},
                )
            result.append(found)
        return result

    def ordered_artifacts(self) -> list[Artifact]:
        """Artifacts in deterministic producer order, not thread completion order."""

        ranks = _node_ranks(self.graph)
        with self._lock:
            artifacts = list(self.artifacts)
        # ``sorted`` is stable, so multiple writes by one producer keep their write order.
        return sorted(
            artifacts,
            key=lambda artifact: (
                ranks.get(artifact.producer, len(ranks)),
                artifact.producer,
            ),
        )

    def envelope(self) -> Envelope:
        return Envelope(
            schema="modelgraph/envelope/v0",
            run=RunInfo(
                id=self.run_id,
                graph=GraphRef(
                    id=self.graph.id,
                    version=self.graph.version,
                    hash=self.graph.graph_hash,
                ),
                parent_run_id=self.parent_run_id,
                depth=self.depth,
                step=self.step,
                path=self.path,
                budget=self.budget.snapshot(),
                started_at=self.started_at,
            ),
            request=Request(task=self.task),
            inputs=self.inputs,
            artifacts=self.ordered_artifacts(),
            notes=self.notes,
        )


def _latest_by_name(artifacts: list[Artifact], graph: CompiledGraph) -> dict[str, Artifact]:
    """Pick the latest artifact by deterministic producer rank, not completion order.

    Within one producer the last write wins, preserving repair/overwrite semantics. Across
    concurrent producers the topologically later producer wins, so an exit artifact with
    more than one writer cannot depend on which branch happened to finish first.
    """

    ranks = _node_ranks(graph)
    latest: dict[str, Artifact] = {}
    for item in artifacts:
        current = latest.get(item.name)
        if current is None or ranks.get(item.producer, -1) >= ranks.get(current.producer, -1):
            latest[item.name] = item
    return latest


def _node_ranks(graph: CompiledGraph) -> dict[str, int]:
    """Topological producer order, tie-broken by node id (deterministic under fan-out)."""

    node_ids = sorted(graph.nodes)
    indegree = {node_id: 0 for node_id in node_ids}
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        adjacency[edge.from_].append(edge.to)
        indegree[edge.to] = indegree.get(edge.to, 0) + 1
    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    ordered: list[str] = []
    while queue:
        node_id = queue.popleft()
        ordered.append(node_id)
        for target in sorted(adjacency.get(node_id, [])):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(ordered) != len(node_ids):
        ordered = node_ids
    return {node_id: index for index, node_id in enumerate(ordered)}


def _make_artifact(
    *,
    name: str,
    artifact_type: ArtifactType,
    content: str,
    producer: str,
    media_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Artifact:
    return Artifact(
        name=name,
        type=artifact_type,
        media_type=media_type,
        content=content,
        producer=producer,
        created_at=utc_now(),
        hash=sha256_text(content),
        metadata=metadata or {},
    )
