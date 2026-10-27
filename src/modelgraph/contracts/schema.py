"""Canonical JSON Schema loading and validation."""

from __future__ import annotations

import json
from functools import cache
from typing import Any, cast

from jsonschema import Draft202012Validator

from modelgraph.errors import Issue, ModelGraphError
from modelgraph.paths import schema_dir

SCHEMA_FILES = {
    "graph": "graph.schema.json",
    "manifest": "model-manifest.schema.json",
    "envelope": "envelope.schema.json",
    "ticket": "ticket.schema.json",
    "eval-suite": "eval-suite.schema.json",
    "run-manifest": "run-manifest.schema.json",
    "dataset": "dataset.schema.json",
    "change-set": "change-set.schema.json",
    "registry-index": "registry-index.schema.json",
    "registry-eval-report": "registry-eval-report.schema.json",
}


@cache
def load_schema(name: str) -> dict[str, Any]:
    if name not in SCHEMA_FILES:
        raise ModelGraphError(f"unknown schema {name!r}")
    path = schema_dir() / SCHEMA_FILES[name]
    try:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except OSError as exc:
        raise ModelGraphError(f"cannot read schema {path}: {exc}", path=str(path)) from exc


@cache
def validator(name: str) -> Draft202012Validator:
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_document(document: dict[str, Any], name: str, *, path: str = "") -> list[Issue]:
    """Validate a document and return all issues with stable paths."""
    issues: list[Issue] = []
    for error in sorted(validator(name).iter_errors(document), key=lambda e: list(e.path)):
        parts: list[str] = []
        for part in error.absolute_path:
            if isinstance(part, int):
                parts.append(f"[{part}]")
            else:
                parts.append(f".{part}" if parts else str(part))
        issue_path = "".join(parts)
        if path:
            issue_path = f"{path}{('.' + issue_path) if issue_path else ''}"
        issues.append(Issue("MG-1001", error.message, issue_path, "error"))
    return issues
