"""Semantic graph validation (rules V-01..V-20)."""

from __future__ import annotations

from collections import defaultdict, deque

from modelgraph.contracts.graph import GraphDocument, NodeKind
from modelgraph.contracts.schema import validate_document
from modelgraph.errors import Issue
from modelgraph.registry import Registry


def topological_order(graph: GraphDocument) -> list[str] | None:
    node_ids = [node.id for node in graph.spec.nodes]
    indegree = {node_id: 0 for node_id in node_ids}
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.spec.edges:
        adjacency[edge.from_].append(edge.to)
        indegree[edge.to] = indegree.get(edge.to, 0) + 1
    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    ordered: list[str] = []
    while queue:
        node_id = queue.popleft()
        ordered.append(node_id)
        for target in adjacency[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return ordered if len(ordered) == len(node_ids) else None


def validate_graph(
    graph: GraphDocument,
    *,
    registry: Registry,
    nested: dict[str, GraphDocument | None] | None = None,
    strict_models: bool = False,
    path: str = "",
) -> list[Issue]:
    """Validate a parsed graph, returning errors and warnings.

    Schema errors are returned by the caller; this function assumes a structurally valid
    document parsed into the typed model.
    """
    nested = nested or {}
    issues: list[Issue] = []
    nodes = {node.id: node for node in graph.spec.nodes}
    node_ids = set(nodes)
    edges = graph.spec.edges

    # V-02 unique node ids
    if len(nodes) != len(graph.spec.nodes):
        issues.append(Issue("MG-1001", "node ids must be unique", f"{path}.spec.nodes"))

    # V-03 entry
    if graph.spec.entry not in nodes:
        issues.append(Issue("MG-1011", "entry node does not exist", f"{path}.spec.entry"))
    incoming: dict[str, int] = defaultdict(int)
    for edge in edges:
        incoming[edge.to] += 1
    if graph.spec.entry in nodes and incoming[graph.spec.entry]:
        issues.append(
            Issue("MG-1011", "entry node must have no incoming edges", f"{path}.spec.entry")
        )

    # V-04/V-05 endpoints, self-loops, duplicates
    seen_pairs: set[tuple[str, str]] = set()
    for index, edge in enumerate(edges):
        edge_path = f"{path}.spec.edges[{index}]"
        if edge.from_ not in node_ids:
            issues.append(Issue("MG-1012", f"unknown edge source {edge.from_!r}", edge_path))
        if edge.to not in node_ids:
            issues.append(Issue("MG-1012", f"unknown edge target {edge.to!r}", edge_path))
        if edge.from_ == edge.to:
            issues.append(Issue("MG-1012", "self-loops are not allowed", edge_path))
        pair = (edge.from_, edge.to)
        if pair in seen_pairs:
            issues.append(Issue("MG-1012", "duplicate edge", edge_path))
        seen_pairs.add(pair)

    # V-06 acyclic
    order = topological_order(graph)
    if order is None:
        issues.append(Issue("MG-1002", "graph contains a cycle", f"{path}.spec.edges"))

    if order is not None:
        # V-07 reachability
        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.from_].append(edge.to)
        reachable = {graph.spec.entry} if graph.spec.entry in nodes else set()
        queue = deque(reachable)
        while queue:
            node_id = queue.popleft()
            for target in adjacency[node_id]:
                if target not in reachable:
                    reachable.add(target)
                    queue.append(target)
        for node_id in sorted(node_ids - reachable):
            issues.append(Issue("MG-1010", "node is unreachable from entry", f"{path}.{node_id}"))

        # V-14 reads available from inputs or ancestors
        inputs = {item.name for item in graph.spec.inputs} or {"task"}
        ancestors: dict[str, set[str]] = defaultdict(set)
        writes: dict[str, set[str]] = {
            node_id: {write.name for write in node.writes} for node_id, node in nodes.items()
        }
        for node_id in order:
            for edge in edges:
                if edge.to == node_id:
                    ancestors[node_id] |= ancestors[edge.from_] | {edge.from_}
        for node in graph.spec.nodes:
            available = set(inputs)
            for ancestor in ancestors[node.id]:
                available |= writes.get(ancestor, set())
            for name in node.reads:
                if name not in available:
                    issues.append(
                        Issue(
                            "MG-1013",
                            f"node reads {name!r}, which is not an input or ancestor artifact",
                            f"{path}.{node.id}.reads",
                        )
                    )

    # Node-kind rules and sinks
    adjacency_all: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        adjacency_all[edge.from_].append(edge.to)
    sinks = {node_id for node_id in node_ids if not adjacency_all[node_id]}
    for node in graph.spec.nodes:
        outgoing = [edge for edge in edges if edge.from_ == node.id]
        node_path = f"{path}.{node.id}"
        if node.kind == NodeKind.router:
            if not outgoing:
                issues.append(Issue("MG-1015", "router has no outgoing edges", node_path))
            if node.writes:
                issues.append(Issue("MG-1015", "router must not write artifacts", node_path))
            for edge in outgoing:
                if not edge.when:
                    issues.append(
                        Issue("MG-1015", "router edge requires a 'when' criterion", node_path)
                    )
        else:
            for edge in outgoing:
                target_node = nodes.get(edge.to)
                if target_node is not None and target_node.kind != NodeKind.joiner:
                    issues.append(
                        Issue(
                            "MG-1012",
                            "only routers may route; workers/subgraphs may only connect to a joiner",
                            f"{path}.{node.id}->{edge.to}",
                        )
                    )
            if node.kind in {NodeKind.worker, NodeKind.joiner} and not node.writes:
                issues.append(Issue("MG-1015", f"{node.kind} must write an artifact", node_path))
        if node.kind == NodeKind.joiner and len([e for e in edges if e.to == node.id]) < 2:
            issues.append(
                Issue(
                    "MG-1015",
                    "joiner has fewer than two incoming edges",
                    node_path,
                    "warning",
                )
            )

    # V-08 sinks write, V-09 exits produced by sinks
    producers: dict[str, set[str]] = defaultdict(set)
    for node in graph.spec.nodes:
        for write in node.writes:
            producers[write.name].add(node.id)
    for node_id in sorted(sinks):
        node = nodes[node_id]
        if not node.writes:
            issues.append(Issue("MG-1006", "sink node writes no artifacts", f"{path}.{node_id}"))
    for exit_name in graph.spec.exit:
        if exit_name not in producers:
            issues.append(
                Issue("MG-1006", f"exit artifact {exit_name!r} is never written", f"{path}.exit")
            )
        elif not (producers[exit_name] & sinks):
            issues.append(
                Issue(
                    "MG-1006",
                    f"exit artifact {exit_name!r} is not written by a sink",
                    f"{path}.exit",
                )
            )

    # V-15 conflicting exit types
    for exit_name in graph.spec.exit:
        types = {
            write.type.value
            for node in graph.spec.nodes
            for write in node.writes
            if write.name == exit_name
        }
        if len(types) > 1:
            issues.append(
                Issue(
                    "MG-1016",
                    f"exit artifact {exit_name!r} has conflicting types: {sorted(types)}",
                    f"{path}.exit",
                    "warning",
                )
            )

    # V-16 fallback
    if graph.spec.fallback:
        fallback = graph.spec.fallback
        if fallback not in nodes:
            issues.append(Issue("MG-1014", "fallback node does not exist", f"{path}.spec.fallback"))
        else:
            if order is not None:
                reachable = _reachable(graph)
                if fallback not in reachable:
                    issues.append(
                        Issue("MG-1014", "fallback node is unreachable", f"{path}.spec.fallback")
                    )
            fallback_writes = {write.name for write in nodes[fallback].writes}
            if not set(graph.spec.exit) <= fallback_writes:
                issues.append(
                    Issue(
                        "MG-1014",
                        "fallback does not write every exit artifact",
                        f"{path}.spec.fallback",
                        "warning",
                    )
                )

    # V-17 model resolution
    for node in graph.spec.nodes:
        if node.model:
            manifest = registry.resolve_model(node.model)
            if manifest is None:
                severity = "error" if strict_models else "warning"
                issues.append(
                    Issue(
                        "MG-1003",
                        f"model {node.model!r} is not in the registry",
                        f"{path}.{node.id}.model",
                        severity,
                    )
                )

    # V-18 subgraph resolution and mappings
    for node in graph.spec.nodes:
        if node.kind != NodeKind.subgraph or not node.graph:
            continue
        inner = nested.get(node.graph)
        if inner is None:
            severity = "error" if strict_models else "warning"
            issues.append(
                Issue(
                    "MG-1004",
                    f"subgraph {node.graph!r} is not in the graph registry",
                    f"{path}.{node.id}.graph",
                    severity,
                )
            )
            continue
        input_map = node.input_map or {}
        output_map = node.output_map or {}
        inner_inputs = {item.name for item in inner.spec.inputs} or {"task"}
        if not inner_inputs <= set(input_map.values()):
            missing = sorted(inner_inputs - set(input_map.values()))
            issues.append(
                Issue(
                    "MG-1004",
                    f"subgraph input map is missing {missing}",
                    f"{path}.{node.id}.input_map",
                )
            )
        if not set(inner.spec.exit) <= set(output_map.values()):
            missing = sorted(set(inner.spec.exit) - set(output_map.values()))
            issues.append(
                Issue(
                    "MG-1004",
                    f"subgraph output map is missing {missing}",
                    f"{path}.{node.id}.output_map",
                )
            )
        write_names = {write.name for write in node.writes}
        if set(output_map) != write_names:
            issues.append(
                Issue(
                    "MG-1004",
                    "subgraph output_map keys must match writes names",
                    f"{path}.{node.id}.output_map",
                )
            )

    # V-19 budgets
    defaults = graph.spec.defaults
    for name in ("max_steps", "max_depth", "max_total_tokens"):
        if getattr(defaults, name) < 1:
            issues.append(Issue("MG-1019", f"{name} must be positive", f"{path}.spec.defaults"))
    if defaults.max_wall_seconds <= 0:
        issues.append(
            Issue("MG-1019", "max_wall_seconds must be positive", f"{path}.spec.defaults")
        )

    return issues


def _reachable(graph: GraphDocument) -> set[str]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.spec.edges:
        adjacency[edge.from_].append(edge.to)
    reachable = {graph.spec.entry}
    queue = deque(reachable)
    while queue:
        node_id = queue.popleft()
        for target in adjacency[node_id]:
            if target not in reachable:
                reachable.add(target)
                queue.append(target)
    return reachable


def schema_issues(document: dict[str, object], *, path: str = "") -> list[Issue]:
    return validate_document(document, "graph", path=path)
