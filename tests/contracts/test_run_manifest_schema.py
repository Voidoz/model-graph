"""Run-manifest contract: schema validity, fixture corpus, and stable failures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "schemas" / "run-manifest.schema.json"
FIXTURES = ROOT / "tests" / "fixtures" / "contracts" / "run-manifest"
SUFFIX = ".run-manifest.json"

SCHEMA: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)


def _fixtures(kind: str) -> list[Path]:
    return sorted((FIXTURES / kind).glob(f"*{SUFFIX}"))


def _issue_path(error: Any) -> str:
    parts: list[str] = []
    for part in error.absolute_path:
        if isinstance(part, int):
            parts.append(f"[{part}]")
        else:
            parts.append(f".{part}" if parts else str(part))
    return "".join(parts)


def test_schema_is_valid() -> None:
    Draft202012Validator.check_schema(SCHEMA)
    assert SCHEMA["title"] == "Run Manifest"
    assert SCHEMA["$id"] == "urn:modelgraph:schema:run-manifest:v0"


def test_fixture_corpus_is_complete() -> None:
    assert len(_fixtures("valid")) >= 3
    assert len(_fixtures("invalid")) >= 3
    for expected in sorted((FIXTURES / "invalid").glob("*.expected.json")):
        fixture = expected.with_name(expected.name.removesuffix(".expected.json") + SUFFIX)
        assert fixture.exists(), f"stale expectation {expected}"


@pytest.mark.parametrize("path", _fixtures("valid"), ids=lambda path: path.name)
def test_valid_fixtures_validate(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    errors = sorted(VALIDATOR.iter_errors(document), key=lambda error: list(error.path))
    assert not errors, f"{path}: {[(list(error.path), error.message) for error in errors]}"


@pytest.mark.parametrize("path", _fixtures("invalid"), ids=lambda path: path.name)
def test_invalid_fixtures_fail_for_the_expected_reason(path: Path) -> None:
    expected_path = path.with_name(path.name.removesuffix(SUFFIX) + ".expected.json")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    document = json.loads(path.read_text(encoding="utf-8"))
    errors = list(VALIDATOR.iter_errors(document))
    assert errors, f"{path}: expected validation to fail"
    matched = [
        error
        for error in errors
        if _issue_path(error) == expected["path"] and expected["message"] in error.message
    ]
    assert matched, (
        f"{path}: expected path {expected['path']!r} containing {expected['message']!r}; "
        f"got {[(_issue_path(error), error.message) for error in errors]}"
    )


def test_status_and_outcome_cover_the_runtime_lifecycle() -> None:
    statuses = SCHEMA["properties"]["status"]["enum"]
    outcomes = SCHEMA["properties"]["outcome"]["enum"]
    assert statuses == ["provisional", "final", "interrupted"]
    assert set(outcomes) >= {"running", "succeeded", "failed", "budget_exhausted", "interrupted"}
