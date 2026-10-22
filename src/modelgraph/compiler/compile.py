"""Graph compilation entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from modelgraph.compiler.ir import CompiledEdge, CompiledGraph, CompiledNode, CompileResult
from modelgraph.compiler.validate import validate_graph
from modelgraph.contracts.graph import GraphDocument
from modelgraph.contracts.loader import load_document
from modelgraph.contracts.schema import validate_document
from modelgraph.errors import Issue, ModelGraphError
from modelgraph.registry import Registry, default_registry
from modelgraph.util import canonical_json, sha256_text


class _Nested:
    def __init__(
        self,
        graph: CompiledGraph | None,
        document: GraphDocument | None,
        issues: list[Issue],
    ) -> None:
        self.graph = graph
        self.document = document
        self.issues = issues


def compile_graph(
    path: str | Path,
    *,
    registry: Registry | None = None,
    strict_models: bool = False,
    graph_paths: list[str | Path] | None = None,
) -> CompileResult:
    """Load, validate, and compile one graph.

    ``strict_models=True`` turns unresolved model/subgraph references into errors (used for a
    run). ``False`` keeps them as warnings (used for offline validation).
    """
    source = Path(path).resolve()
    try:
        document = load_document(source)
    except ModelGraphError as exc:
        return CompileResult(None, errors=(exc.message,))

    issues = validate_document(document, "graph", path=str(source))
    if issues:
        return CompileResult(None, errors=tuple(issue.render() for issue in issues))

    try:
        graph = GraphDocument.model_validate(document)
    except ValidationError as exc:
        messages = tuple(
            f"[MG-1001] at {'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        return CompileResult(None, errors=messages)

    reg = registry or default_registry()
    reg.add_graph_path(source)
    for extra in graph_paths or []:
        reg.add_graph_path(extra)

    try:
        compiled = _compile_document(
            graph,
            source=source,
            registry=reg,
            stack=(),
            strict_models=strict_models,
        )
    except ModelGraphError as exc:
        return CompileResult(None, errors=(exc.message,))

    return CompileResult(
        graph=compiled,
        errors=(),
        warnings=compiled.warnings,
    )


def _compile_document(
    graph: GraphDocument,
    *,
    source: Path,
    registry: Registry,
    stack: tuple[str, ...],
    strict_models: bool,
) -> CompiledGraph:
    ref = f"{graph.metadata.id}@{graph.metadata.version}"
    if ref in stack:
        chain = " -> ".join([*stack, ref])
        raise ModelGraphError(f"[MG-1004] subgraph cycle: {chain}", path=str(source))

    nested_graphs: dict[str, CompiledGraph] = {}
    nested_docs: dict[str, GraphDocument | None] = {}
    issues: list[Issue] = []

    for node in graph.spec.nodes:
        if node.kind.value != "subgraph" or not node.graph:
            continue
        if node.graph in nested_docs:
            continue
        sub_path = registry.resolve_graph(node.graph)
        if sub_path is None:
            severity = "error" if strict_models else "warning"
            nested_docs[node.graph] = None
            issues.append(
                Issue(
                    "MG-1004",
                    f"subgraph {node.graph!r} is not in the graph registry",
                    f"{source}.{node.id}.graph",
                    severity,
                )
            )
            continue
        sub_document = load_document(sub_path)
        sub_issues = validate_document(sub_document, "graph", path=str(sub_path))
        if sub_issues:
            issues.extend(sub_issues)
            nested_docs[node.graph] = None
            continue
        try:
            parsed = GraphDocument.model_validate(sub_document)
        except ValidationError as exc:
            issues.append(
                Issue(
                    "MG-1001",
                    f"subgraph {node.graph!r} is invalid: {exc}",
                    f"{source}.{node.id}.graph",
                )
            )
            nested_docs[node.graph] = None
            continue
        nested_docs[node.graph] = parsed
        try:
            nested_graphs[node.graph] = _compile_document(
                parsed,
                source=sub_path,
                registry=registry,
                stack=(*stack, ref),
                strict_models=strict_models,
            )
        except ModelGraphError as exc:
            if strict_models:
                raise
            issues.append(Issue("MG-1004", exc.message, f"{source}.{node.id}.graph", "warning"))

    issues.extend(
        validate_graph(
            graph,
            registry=registry,
            nested=nested_docs,
            strict_models=strict_models,
            path=str(source),
        )
    )

    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise ModelGraphError("; ".join(issue.render() for issue in errors), path=str(source))
    warnings = [issue.render() for issue in issues if issue.severity == "warning"]

    compiled_nodes: dict[str, CompiledNode] = {}
    resolved_models: dict[str, Any] = {}
    for node in graph.spec.nodes:
        model = registry.resolve_model(node.model) if node.model else None
        model_hash = registry.manifest_hash(node.model) if node.model else None
        compiled_nodes[node.id] = CompiledNode.from_node(node, model, model_hash)
        if node.model:
            resolved_models[node.id] = {
                "ref": node.model,
                "hash": model_hash,
            }

    compiled_edges = tuple(CompiledEdge.from_edge(edge) for edge in graph.spec.edges)
    canonical = {
        "apiVersion": graph.apiVersion,
        "graph": graph.to_dict(),
        "resolved_models": resolved_models,
        "subgraphs": {
            ref_name: nested_graphs[ref_name].graph_hash for ref_name in sorted(nested_graphs)
        },
    }
    graph_hash = sha256_text(canonical_json(canonical))

    return CompiledGraph(
        id=graph.metadata.id,
        version=graph.metadata.version,
        description=graph.metadata.description,
        api_version=graph.apiVersion,
        entry=graph.spec.entry,
        exits=tuple(graph.spec.exit),
        inputs=tuple(graph.spec.inputs),
        fallback=graph.spec.fallback,
        defaults=graph.spec.defaults,
        nodes=compiled_nodes,
        edges=compiled_edges,
        graph_hash=graph_hash,
        source=str(source),
        warnings=tuple(warnings),
        subgraphs=nested_graphs,
        labels=dict(graph.metadata.labels),
    )
