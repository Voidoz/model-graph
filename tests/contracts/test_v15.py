"""Validation rule V-15: conflicting types for the same exit artifact."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from modelgraph.compiler import validate_graph
from modelgraph.contracts import load_document, validate_document
from modelgraph.contracts.graph import GraphDocument
from modelgraph.registry import Registry

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "contracts"
V15_GRAPH = FIXTURES / "graph" / "warnings" / "conflicting-exit-types.graph.yaml"


def _registry() -> Registry:
    return Registry.from_paths([FIXTURES / "models"], [FIXTURES / "graphs"])


def test_v15_fixture_is_schema_valid_then_warns() -> None:
    document = load_document(V15_GRAPH)
    assert not validate_document(document, "graph")
    graph = GraphDocument.model_validate(document)

    issues = validate_graph(graph, registry=_registry(), strict_models=True)
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.code == "MG-1016"]
    assert not errors, [issue.render() for issue in errors]
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.severity == "warning"
    assert warning.path == ".exit"
    assert "conflicting types" in warning.message
    assert "answer" in warning.message
    assert "code" in warning.message and "text" in warning.message


def test_v15_fixture_with_one_exit_type_does_not_warn() -> None:
    document = deepcopy(load_document(V15_GRAPH))
    for node in document["spec"]["nodes"]:
        for write in node.get("writes", []):
            if write["name"] == "answer":
                write["type"] = "code"
    graph = GraphDocument.model_validate(document)

    issues = validate_graph(graph, registry=_registry(), strict_models=True)
    assert not [issue for issue in issues if issue.code == "MG-1016"]
