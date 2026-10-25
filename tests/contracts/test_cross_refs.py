"""Cross-reference checks for the fixture registry.

The contract suite only checks that references resolve against the supplied fixture corpus.
Full acyclicity, reachability, mapping, and fallback semantics belong to the compiler suite
(T-0003).
"""

from __future__ import annotations

from pathlib import Path

from modelgraph.contracts import load_document
from modelgraph.contracts.graph import GraphDocument, NodeKind
from modelgraph.registry import Registry

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "contracts"
FIXTURE_MODELS = FIXTURES / "models"
FIXTURE_GRAPHS = FIXTURES / "graphs"
INNER_REF = "fixture-inner@1.0.0"
INNER_PATH = FIXTURE_GRAPHS / "fixture-inner.graph.yaml"


def _load_graph(name: str) -> GraphDocument:
    return GraphDocument.model_validate(load_document(FIXTURE_GRAPHS / name))


def _fixture_registry() -> Registry:
    return Registry.from_paths([FIXTURE_MODELS], [FIXTURE_GRAPHS])


def test_graph_model_references_resolve_in_the_fixture_registry() -> None:
    registry = _fixture_registry()
    graphs = [_load_graph("fixture-outer.graph.yaml"), _load_graph("fixture-inner.graph.yaml")]
    model_refs = {
        node.model for graph in graphs for node in graph.spec.nodes if node.model is not None
    }
    assert model_refs, "the cross-reference fixture graphs should reference models"
    for ref in sorted(model_refs):
        manifest = registry.resolve_model(ref)
        assert manifest is not None, f"model reference {ref!r} did not resolve"
        assert manifest.metadata.id == ref


def test_subgraph_reference_resolves_within_the_fixture_corpus() -> None:
    registry = _fixture_registry()
    outer = _load_graph("fixture-outer.graph.yaml")
    subgraphs = [node for node in outer.spec.nodes if node.kind == NodeKind.subgraph]
    assert subgraphs, "the outer fixture graph should contain a subgraph node"
    for node in subgraphs:
        assert node.graph is not None
        assert registry.resolve_graph(node.graph) == INNER_PATH


def test_unknown_references_do_not_resolve() -> None:
    registry = _fixture_registry()
    assert registry.resolve_model("fixture-missing-model") is None
    assert registry.resolve_model("fixture-worker@9.9.9") is None
    assert registry.resolve_graph("fixture-missing@9.9.9") is None
