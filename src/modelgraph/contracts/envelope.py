"""Typed envelope used by the runtime and persisted in run directories."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from modelgraph.contracts.graph import ArtifactType


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class GraphRef(_Model):
    id: str
    version: str
    hash: str


class PathEntry(_Model):
    node: str
    kind: str
    decision: str | None = None
    to: str | None = None
    confidence: float | None = None
    reason: str | None = None
    graph: str | None = None
    graph_hash: str | None = None


class BudgetState(_Model):
    max_steps: int = 8
    used_steps: int = 0
    max_depth: int = 3
    max_total_tokens: int = 32_000
    used_tokens: int = 0
    max_cost_usd: float = 0.25
    used_cost_usd: float = 0.0
    max_wall_seconds: float = 300.0
    elapsed_seconds: float = 0.0
    on_budget_exhausted: str = "fail"


class RunInfo(_Model):
    id: str
    graph: GraphRef
    parent_run_id: str | None = None
    depth: int = 0
    step: int = 0
    path: list[PathEntry] = Field(default_factory=list)
    budget: BudgetState
    started_at: str
    seed: int | None = None


class Request(_Model):
    task: str
    locale: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)


class Artifact(_Model):
    name: str
    type: ArtifactType
    media_type: str | None = None
    content: str
    producer: str
    created_at: str
    hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Note(_Model):
    producer: str
    content: str
    created_at: str | None = None


class Envelope(_Model):
    schema_: str = Field(default="modelgraph/envelope/v0", alias="schema")
    run: RunInfo
    request: Request
    inputs: list[Artifact] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)
