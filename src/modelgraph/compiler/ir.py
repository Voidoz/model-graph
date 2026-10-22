"""Compiled graph representation consumed by the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modelgraph.contracts.graph import (
    ArtifactInput,
    ArtifactWrite,
    Defaults,
    Edge,
    JoinerStrategy,
    Limits,
    Node,
    NodeKind,
    RoutingConfig,
)
from modelgraph.contracts.manifest import ModelManifest


@dataclass(frozen=True)
class CompiledNode:
    id: str
    kind: NodeKind
    description: str
    model_ref: str | None
    model: ModelManifest | None
    model_hash: str | None
    instruction: str | None
    limits: Limits | None
    reads: tuple[str, ...]
    writes: tuple[ArtifactWrite, ...]
    routing: RoutingConfig
    routing_card: str | None
    graph_ref: str | None
    input_map: dict[str, str]
    output_map: dict[str, str]
    strategy: JoinerStrategy

    @classmethod
    def from_node(
        cls,
        node: Node,
        model: ModelManifest | None,
        model_hash: str | None,
    ) -> CompiledNode:
        return cls(
            id=node.id,
            kind=node.kind,
            description=node.description,
            model_ref=node.model,
            model=model,
            model_hash=model_hash,
            instruction=node.instruction,
            limits=node.limits,
            reads=tuple(node.reads),
            writes=tuple(node.writes),
            routing=node.routing or RoutingConfig(),
            routing_card=node.routing_card,
            graph_ref=node.graph,
            input_map=dict(node.input_map or {}),
            output_map=dict(node.output_map or {}),
            strategy=node.strategy or JoinerStrategy.synthesize,
        )


@dataclass(frozen=True)
class CompiledEdge:
    from_: str
    to: str
    when: str | None
    examples: tuple[str, ...]
    artifact: str | None
    priority: int

    @classmethod
    def from_edge(cls, edge: Edge) -> CompiledEdge:
        return cls(
            from_=edge.from_,
            to=edge.to,
            when=edge.when,
            examples=tuple(edge.examples),
            artifact=edge.artifact.value if edge.artifact else None,
            priority=edge.priority,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_,
            "to": self.to,
            "when": self.when,
            "examples": list(self.examples),
            "artifact": self.artifact,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class CompiledGraph:
    id: str
    version: str
    description: str
    api_version: str
    entry: str
    exits: tuple[str, ...]
    inputs: tuple[ArtifactInput, ...]
    fallback: str | None
    defaults: Defaults
    nodes: dict[str, CompiledNode]
    edges: tuple[CompiledEdge, ...]
    graph_hash: str
    source: str
    warnings: tuple[str, ...] = ()
    subgraphs: dict[str, CompiledGraph] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    def node(self, node_id: str) -> CompiledNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown node {node_id!r} in graph {self.id}") from exc

    def outgoing(self, node_id: str) -> list[CompiledEdge]:
        return [edge for edge in self.edges if edge.from_ == node_id]

    def incoming(self, node_id: str) -> list[CompiledEdge]:
        return [edge for edge in self.edges if edge.to == node_id]

    def input_names(self) -> set[str]:
        names = {item.name for item in self.inputs}
        return names or {"task"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "apiVersion": self.api_version,
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "entry": self.entry,
            "exit": list(self.exits),
            "inputs": [item.model_dump(mode="json", exclude_none=True) for item in self.inputs],
            "fallback": self.fallback,
            "defaults": self.defaults.model_dump(mode="json"),
            "nodes": {
                node_id: {
                    "id": node.id,
                    "kind": node.kind.value,
                    "description": node.description,
                    "model_ref": node.model_ref,
                    "model_hash": node.model_hash,
                    "instruction": node.instruction,
                    "limits": node.limits.model_dump(mode="json", exclude_none=True)
                    if node.limits
                    else None,
                    "reads": list(node.reads),
                    "writes": [
                        write.model_dump(mode="json", exclude_none=True) for write in node.writes
                    ],
                    "routing": node.routing.model_dump(mode="json"),
                    "routing_card": node.routing_card,
                    "graph_ref": node.graph_ref,
                    "input_map": node.input_map,
                    "output_map": node.output_map,
                    "strategy": node.strategy.value,
                }
                for node_id, node in sorted(self.nodes.items())
            },
            "edges": [edge.to_dict() for edge in self.edges],
            "subgraphs": {ref: graph.graph_hash for ref, graph in sorted(self.subgraphs.items())},
        }


@dataclass(frozen=True)
class CompileResult:
    graph: CompiledGraph | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.graph is not None and not self.errors
