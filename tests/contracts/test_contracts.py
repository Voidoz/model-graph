import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from modelgraph.contracts import load_document, load_schema, validate_document, validator
from modelgraph.contracts.envelope import Envelope
from modelgraph.contracts.graph import GraphDocument
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.contracts.ticket import load_ticket
from modelgraph.errors import ModelGraphError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
TICKETS = ROOT / ".scratch" / "tickets"


@pytest.mark.parametrize("name", ["graph", "manifest", "envelope", "ticket"])
def test_schemas_are_valid(name: str) -> None:
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    validator(name)


def test_examples_validate() -> None:
    for graph in sorted(EXAMPLES.glob("*.graph.yaml")):
        document = load_document(graph)
        assert not validate_document(document, "graph", path=str(graph))
        GraphDocument.model_validate(document)
    for model in sorted((EXAMPLES / "models").glob("*.yaml")):
        assert not validate_document(load_document(model), "manifest", path=str(model))
        ModelManifest.model_validate(load_document(model))
    envelope = json.loads((EXAMPLES / "envelope.json").read_text())
    assert not validate_document(envelope, "envelope")
    Envelope.model_validate(envelope)


def test_all_tickets_validate() -> None:
    tickets = sorted(TICKETS.glob("T-*.md"))
    assert len(tickets) >= 30
    for path in tickets:
        load_ticket(path)


def test_duplicate_yaml_keys_fail(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("apiVersion: modelgraph/v0\napiVersion: modelgraph/v0\n", encoding="utf-8")
    with pytest.raises(ModelGraphError, match="duplicate key"):
        load_document(path)


def test_invalid_graph_reports_path(tmp_path: Path) -> None:
    document = {
        "apiVersion": "modelgraph/v0",
        "kind": "Graph",
        "metadata": {
            "id": "bad-graph",
            "name": "Bad",
            "version": "1.0.0",
            "description": "A graph that is missing required structure.",
        },
        "spec": {"entry": "a", "nodes": [], "edges": [], "exit": []},
    }
    path = tmp_path / "bad.graph.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    issues = validate_document(load_document(path), "graph", path=str(path))
    assert issues
    assert any("spec" in issue.path or "nodes" in issue.message for issue in issues)


def test_schema_snapshot_loads() -> None:
    schema = load_schema("graph")
    assert schema["title"] == "Model Graph"
    assert schema["properties"]["apiVersion"]["const"] == "modelgraph/v0"


def test_corrupted_example_fails_without_changing_the_committed_fixture() -> None:
    path = EXAMPLES / "hello-triage.graph.yaml"
    document = load_document(path)
    document["spec"]["nodes"][0]["kind"] = "planner"
    issues = validate_document(document, "graph")
    assert issues
    assert any(issue.path.endswith("nodes[0].kind") for issue in issues)
    # The file on disk is unchanged and still valid.
    assert not validate_document(load_document(path), "graph")
