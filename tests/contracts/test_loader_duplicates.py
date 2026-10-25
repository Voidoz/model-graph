"""Duplicate-key detection for YAML contract documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelgraph.contracts import DuplicateKeyError, load_document

DUPLICATES = Path(__file__).resolve().parent.parent / "fixtures" / "contracts" / "duplicate-keys"

CASES = [
    ("top-level.yaml", "apiVersion", 3),
    ("nested.yaml", "version", 6),
    ("in-sequence.yaml", "kind", 7),
]


@pytest.mark.parametrize(("name", "key", "line"), CASES)
def test_duplicate_yaml_keys_name_the_key_and_line(name: str, key: str, line: int) -> None:
    with pytest.raises(DuplicateKeyError) as caught:
        load_document(DUPLICATES / name)
    error = caught.value
    assert error.code == "MG-1001"
    assert key in error.message
    assert error.path == f"line {line}"


def test_duplicate_key_error_is_a_contract_error() -> None:
    with pytest.raises(DuplicateKeyError, match="duplicate key 'apiVersion'"):
        load_document(DUPLICATES / "top-level.yaml")
