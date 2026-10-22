"""Typed graph documents. The JSON Schema is canonical; these models mirror it for code."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArtifactType(StrEnum):
    text = "text"
    code = "code"
    patch = "patch"
    plan = "plan"
    json = "json"
    report = "report"


class NodeKind(StrEnum):
    router = "router"
    worker = "worker"
    joiner = "joiner"
    subgraph = "subgraph"


class JoinerStrategy(StrEnum):
    merge = "merge"
    pick_best = "pick_best"
    synthesize = "synthesize"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Labels(_Model):
    pass


class GraphMetadata(_Model):
    id: str
    name: str
    version: str
    description: str
    labels: dict[str, str] = Field(default_factory=dict)
    authors: list[str] | None = None
    license: str | None = None
    homepage: str | None = None


class ArtifactInput(_Model):
    name: str
    type: ArtifactType
    required: bool = True
    description: str | None = None
    media_type: str | None = None


class ArtifactWrite(_Model):
    name: str
    type: ArtifactType
    description: str | None = None
    media_type: str | None = None
    strict_json: bool = False


class Limits(_Model):
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    timeout_seconds: float | None = None
    on_overflow: str = "error"


class RoutingConfig(_Model):
    selection: str = "single"
    confidence_threshold: float = 0.7
    on_low_confidence: str = "error"
    max_retries: int = 1
    allow_direct_answer: bool = False


class Node(_Model):
    id: str
    kind: NodeKind
    description: str
    labels: dict[str, str] = Field(default_factory=dict)
    reads: list[str] = Field(default_factory=list)
    writes: list[ArtifactWrite] = Field(default_factory=list)
    model: str | None = None
    instruction: str | None = None
    routing: RoutingConfig | None = None
    limits: Limits | None = None
    routing_card: str | None = None
    graph: str | None = None
    input_map: dict[str, str] | None = None
    output_map: dict[str, str] | None = None
    strategy: JoinerStrategy | None = None

    @model_validator(mode="after")
    def _check_kind(self) -> Node:
        forbidden = {
            NodeKind.router: {
                "writes",
                "instruction",
                "graph",
                "input_map",
                "output_map",
                "strategy",
            },
            NodeKind.worker: {"routing", "graph", "input_map", "output_map", "strategy"},
            NodeKind.joiner: {"routing", "graph", "input_map", "output_map"},
            NodeKind.subgraph: {"model", "instruction", "routing", "limits", "strategy"},
        }[self.kind]
        for field_name in forbidden:
            value = getattr(self, field_name)
            if value not in (None, [], {}):
                raise ValueError(f"{self.kind} node {self.id!r} cannot set {field_name!r}")
        if self.kind in {NodeKind.router, NodeKind.worker, NodeKind.joiner} and not self.model:
            raise ValueError(f"{self.kind} node {self.id!r} requires a model")
        if self.kind == NodeKind.worker and not self.instruction:
            raise ValueError(f"worker node {self.id!r} requires an instruction")
        if self.kind == NodeKind.worker and not self.writes:
            raise ValueError(f"worker node {self.id!r} requires writes")
        if self.kind == NodeKind.joiner and not self.writes:
            raise ValueError(f"joiner node {self.id!r} requires writes")
        if self.kind == NodeKind.subgraph and (
            not self.graph or not self.input_map or not self.output_map or not self.writes
        ):
            raise ValueError(f"subgraph node {self.id!r} requires graph, maps, and writes")
        return self


class Edge(_Model):
    from_: str = Field(alias="from")
    to: str
    when: str | None = None
    examples: list[str] = Field(default_factory=list)
    artifact: ArtifactType | None = None
    priority: int = 0


class Defaults(_Model):
    max_steps: int = 8
    max_depth: int = 3
    max_total_tokens: int = 32_000
    max_cost_usd: float = 0.25
    max_wall_seconds: float = 300.0
    on_budget_exhausted: str = "fail"


class GraphSpec(_Model):
    entry: str
    exit: list[str]
    inputs: list[ArtifactInput] = Field(default_factory=list)
    fallback: str | None = None
    defaults: Defaults = Field(default_factory=Defaults)
    nodes: list[Node]
    edges: list[Edge]


class GraphDocument(_Model):
    apiVersion: str
    kind: str
    metadata: GraphMetadata
    spec: GraphSpec

    def node(self, node_id: str) -> Node:
        for node in self.spec.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def outgoing(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.spec.edges if edge.from_ == node_id]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)
