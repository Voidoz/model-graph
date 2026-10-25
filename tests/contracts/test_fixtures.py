"""Valid and invalid fixture corpus for all four canonical contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from modelgraph.contracts import load_document, validate_document
from modelgraph.contracts.envelope import Envelope
from modelgraph.contracts.graph import GraphDocument
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.contracts.ticket import TicketFrontmatter, parse_frontmatter

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "contracts"

SCHEMA_FIXTURES: dict[str, tuple[str, type[BaseModel]]] = {
    "graph": ("*.graph.yaml", GraphDocument),
    "manifest": ("*.manifest.yaml", ModelManifest),
    "envelope": ("*.envelope.json", Envelope),
    "ticket": ("*.ticket.md", TicketFrontmatter),
}
SUFFIXES = {
    "graph": ".graph.yaml",
    "manifest": ".manifest.yaml",
    "envelope": ".envelope.json",
    "ticket": ".ticket.md",
}


def _load(schema_name: str, path: Path) -> dict[str, Any]:
    if schema_name == "ticket":
        return parse_frontmatter(path.read_text(encoding="utf-8"))
    return load_document(path)


def _valid_paths(schema_name: str) -> list[Path]:
    pattern, _ = SCHEMA_FIXTURES[schema_name]
    return sorted((FIXTURES / schema_name / "valid").glob(pattern))


def _invalid_paths(schema_name: str) -> list[Path]:
    pattern, _ = SCHEMA_FIXTURES[schema_name]
    return sorted((FIXTURES / schema_name / "invalid").glob(pattern))


def _expected_path(path: Path, suffix: str) -> Path:
    assert path.name.endswith(suffix), path
    return path.with_name(path.name[: -len(suffix)] + ".expected.json")


@pytest.mark.parametrize("schema_name", list(SCHEMA_FIXTURES))
def test_each_schema_has_a_valid_fixture_corpus(schema_name: str) -> None:
    paths = _valid_paths(schema_name)
    assert len(paths) >= 3, f"expected at least three valid {schema_name} fixtures"
    _, model = SCHEMA_FIXTURES[schema_name]
    for path in paths:
        document = _load(schema_name, path)
        issues = validate_document(document, schema_name)
        assert not issues, f"{path}: {[issue.render() for issue in issues]}"
        model.model_validate(document)


@pytest.mark.parametrize("schema_name", list(SCHEMA_FIXTURES))
def test_each_schema_has_an_invalid_fixture_corpus(schema_name: str) -> None:
    paths = _invalid_paths(schema_name)
    assert len(paths) >= 3, f"expected at least three invalid {schema_name} fixtures"
    for path in paths:
        expected_path = _expected_path(path, SUFFIXES[schema_name])
        assert expected_path.exists(), f"missing expectation for {path}"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        issues = validate_document(_load(schema_name, path), schema_name)
        assert issues, f"{path}: expected validation to fail"
        matched = [
            issue
            for issue in issues
            if issue.path == expected["path"] and expected["message"] in issue.message
        ]
        assert matched, (
            f"{path}: expected path {expected['path']!r} containing "
            f"{expected['message']!r}; got {[(issue.path, issue.message) for issue in issues]}"
        )


@pytest.mark.parametrize("schema_name", list(SCHEMA_FIXTURES))
def test_invalid_fixtures_have_expectations(schema_name: str) -> None:
    expected = sorted((FIXTURES / schema_name / "invalid").glob("*.expected.json"))
    assert len(expected) >= 3
    for path in expected:
        fixture_name = path.name.removesuffix(".expected.json") + SUFFIXES[schema_name]
        assert (path.parent / fixture_name).exists(), f"stale expectation {path}"
