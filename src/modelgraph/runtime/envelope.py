"""Request-scoped state, artifacts, and scoped views."""

from __future__ import annotations

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
        latest: dict[str, Artifact] = {}
        for item in [*self.inputs, *self.artifacts]:
            latest[item.name] = item
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
        self.artifacts.append(artifact)
        return artifact

    def note(self, producer: str, content: str) -> None:
        if content.strip():
            self.notes.append(
                Note(producer=producer, content=content.strip(), created_at=utc_now())
            )

    def add_path(self, entry: PathEntry) -> None:
        self.path.append(entry)

    def latest_artifact(self, name: str) -> Artifact | None:
        latest: Artifact | None = None
        for artifact in [*self.inputs, *self.artifacts]:
            if artifact.name == name:
                latest = artifact
        return latest

    def results(self) -> list[Artifact]:
        latest: dict[str, Artifact] = {}
        for item in [*self.inputs, *self.artifacts]:
            latest[item.name] = item
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
            artifacts=self.artifacts,
            notes=self.notes,
        )


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
