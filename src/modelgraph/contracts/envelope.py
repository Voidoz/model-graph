"""Typed envelope used by the runtime and persisted in run directories."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from modelgraph.contracts.constraints import ArtifactName, ContentHash, NodeId
from modelgraph.contracts.graph import ArtifactType, BudgetExhaustedAction, NodeKind

_PATH_KINDS = tuple(kind.value for kind in NodeKind)
_BUDGET_POLICIES = tuple(action.value for action in BudgetExhaustedAction)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class GraphRef(_Model):
    id: str
    version: str
    hash: ContentHash


class PathEntry(_Model):
    node: NodeId
    kind: str = Field(json_schema_extra={"enum": list(_PATH_KINDS)})
    decision: Literal["route", "abstain"] | None = None
    to: NodeId | None = None
    confidence: float | None = None
    reason: str | None = None
    graph: str | None = None
    graph_hash: ContentHash | None = None

    @field_validator("kind")
    @classmethod
    def _check_kind(cls, value: str) -> str:
        if value not in _PATH_KINDS:
            raise ValueError(f"kind must be one of {list(_PATH_KINDS)}")
        return value


class BudgetState(_Model):
    max_steps: int
    used_steps: int
    max_depth: int
    max_total_tokens: int = 32_000
    used_tokens: int = 0
    max_cost_usd: float = 0.25
    used_cost_usd: float = 0.0
    max_wall_seconds: float = 300.0
    elapsed_seconds: float = 0.0
    on_budget_exhausted: str = Field(
        default=BudgetExhaustedAction.fail,
        json_schema_extra={"enum": list(_BUDGET_POLICIES)},
    )

    @field_validator("on_budget_exhausted")
    @classmethod
    def _check_policy(cls, value: str) -> str:
        if value not in _BUDGET_POLICIES:
            raise ValueError(f"on_budget_exhausted must be one of {list(_BUDGET_POLICIES)}")
        return value


class RunInfo(_Model):
    id: str
    graph: GraphRef
    parent_run_id: str | None = None
    depth: int
    step: int
    path: list[PathEntry]
    budget: BudgetState
    started_at: str
    seed: int | None = None


class Request(_Model):
    task: str
    locale: str | None = None
    constraints: dict[str, Any] = Field(default_factory=dict)


class Artifact(_Model):
    name: ArtifactName
    type: ArtifactType
    media_type: str | None = None
    content: str
    producer: str
    created_at: str
    hash: ContentHash | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Note(_Model):
    producer: str
    content: str
    created_at: str | None = None


class Envelope(_Model):
    schema_: Literal["modelgraph/envelope/v0"] = Field(alias="schema")
    run: RunInfo
    request: Request
    inputs: list[Artifact]
    artifacts: list[Artifact]
    notes: list[Note] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)
