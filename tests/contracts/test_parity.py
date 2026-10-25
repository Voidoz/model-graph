"""JSON Schema <-> Pydantic parity for required fields, enums, and patterns.

The JSON Schemas in ``schemas/`` are canonical. Each Pydantic model that mirrors a schema
object is compared field by field. The comparison deliberately ignores constraints the
ticket does not name (lengths, numeric bounds, defaults) and focuses on the three that
routinely drift: required fields, enum/const values, and string patterns.

If a model cannot express a constraint through its generated JSON Schema, the constraint
must be asserted separately rather than dropped from the canonical schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from modelgraph.contracts import load_schema
from modelgraph.contracts.envelope import (
    Artifact,
    BudgetState,
    Envelope,
    GraphRef,
    Note,
    PathEntry,
    Request,
    RunInfo,
)
from modelgraph.contracts.graph import (
    ArtifactInput,
    ArtifactWrite,
    Defaults,
    Edge,
    GraphDocument,
    GraphMetadata,
    GraphSpec,
    Limits,
    Node,
    RoutingConfig,
)
from modelgraph.contracts.manifest import (
    Cost,
    EvalInfo,
    Integrity,
    Latency,
    License,
    ManifestMetadata,
    ModelManifest,
    ModelSpec,
    ProviderLimits,
    Resources,
)
from modelgraph.contracts.ticket import TicketFrontmatter


@dataclass(frozen=True)
class ObjectPair:
    schema_name: str
    definition: str | None
    model: type[BaseModel]

    @property
    def label(self) -> str:
        return f"{self.schema_name}:{self.definition or '<root>'}"


PAIRS = [
    # graph.schema.json
    ObjectPair("graph", None, GraphDocument),
    ObjectPair("graph", "$defs.metadata", GraphMetadata),
    ObjectPair("graph", "$defs.artifactInput", ArtifactInput),
    ObjectPair("graph", "$defs.artifactWrite", ArtifactWrite),
    ObjectPair("graph", "$defs.limits", Limits),
    ObjectPair("graph", "$defs.routing", RoutingConfig),
    ObjectPair("graph", "$defs.node", Node),
    ObjectPair("graph", "$defs.edge", Edge),
    ObjectPair("graph", "$defs.defaults", Defaults),
    ObjectPair("graph", "$defs.spec", GraphSpec),
    # model-manifest.schema.json
    ObjectPair("manifest", None, ModelManifest),
    ObjectPair("manifest", "$defs.metadata", ManifestMetadata),
    ObjectPair("manifest", "$defs.spec", ModelSpec),
    ObjectPair("manifest", "$defs.spec.properties.limits", ProviderLimits),
    ObjectPair("manifest", "$defs.spec.properties.cost", Cost),
    ObjectPair("manifest", "$defs.spec.properties.latency_ms", Latency),
    ObjectPair("manifest", "$defs.spec.properties.resources", Resources),
    ObjectPair("manifest", "$defs.spec.properties.license", License),
    ObjectPair("manifest", "$defs.spec.properties.integrity", Integrity),
    ObjectPair("manifest", "$defs.spec.properties.eval", EvalInfo),
    # envelope.schema.json
    ObjectPair("envelope", None, Envelope),
    ObjectPair("envelope", "$defs.graph", GraphRef),
    ObjectPair("envelope", "$defs.pathEntry", PathEntry),
    ObjectPair("envelope", "$defs.budget", BudgetState),
    ObjectPair("envelope", "$defs.run", RunInfo),
    ObjectPair("envelope", "$defs.request", Request),
    ObjectPair("envelope", "$defs.artifact", Artifact),
    ObjectPair("envelope", "$defs.note", Note),
    # ticket.schema.json
    ObjectPair("ticket", None, TicketFrontmatter),
]


@dataclass(frozen=True)
class Constraints:
    """The constraints the parity test compares for one field or array item."""

    enum_values: frozenset[object]
    patterns: frozenset[str]
    items: Constraints | None = None
    additional: Constraints | None = None


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in schema:
        ref = schema["$ref"]
        assert ref.startswith("#/"), ref
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part]
        assert isinstance(target, dict), ref
        schema = target
    return schema


def _non_null_branches(schema: dict[str, Any], root: dict[str, Any]) -> list[dict[str, Any]]:
    schema = _resolve(schema, root)
    branches = schema.get("anyOf") or schema.get("oneOf")
    if not branches:
        return [schema]
    flattened: list[dict[str, Any]] = []
    for branch in branches:
        resolved = _resolve(branch, root)
        if resolved.get("type") == "null":
            continue
        flattened.extend(_non_null_branches(resolved, root))
    return flattened


def _constraints(schema: dict[str, Any], root: dict[str, Any]) -> Constraints:
    enum_values: set[object] = set()
    patterns: set[str] = set()
    items: Constraints | None = None
    additional: Constraints | None = None
    for branch in _non_null_branches(schema, root):
        if "enum" in branch:
            enum_values.update(branch["enum"])
        if "const" in branch:
            enum_values.add(branch["const"])
        if "pattern" in branch:
            patterns.add(branch["pattern"])
        if isinstance(branch.get("items"), dict):
            items = _constraints(branch["items"], root)
        if isinstance(branch.get("additionalProperties"), dict):
            additional = _constraints(branch["additionalProperties"], root)
    return Constraints(
        enum_values=frozenset(enum_values),
        patterns=frozenset(patterns),
        items=items,
        additional=additional,
    )


def _object_schema(
    schema_name: str, definition: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = load_schema(schema_name)
    if definition is None:
        return root, root
    target: Any = root
    for part in definition.split("."):
        target = target[part]
    assert isinstance(target, dict), definition
    return target, root


def parity_problems(pair: ObjectPair) -> list[str]:
    canonical, canonical_root = _object_schema(pair.schema_name, pair.definition)
    generated = pair.model.model_json_schema(by_alias=True)
    problems: list[str] = []

    canonical_required = set(canonical.get("required", []))
    generated_required = set(generated.get("required", []))
    if canonical_required != generated_required:
        problems.append(
            f"{pair.label}: required {sorted(canonical_required)} != {sorted(generated_required)}"
        )

    canonical_properties = canonical.get("properties", {})
    generated_properties = generated.get("properties", {})
    if set(canonical_properties) != set(generated_properties):
        problems.append(
            f"{pair.label}: properties {sorted(canonical_properties)} != "
            f"{sorted(generated_properties)}"
        )

    for name in sorted(set(canonical_properties) & set(generated_properties)):
        expected = _constraints(canonical_properties[name], canonical_root)
        actual = _constraints(generated_properties[name], generated)
        if expected != actual:
            problems.append(f"{pair.label}.{name}: {expected} != {actual}")
    return problems


@pytest.mark.parametrize("pair", PAIRS, ids=lambda pair: pair.label)
def test_pydantic_mirrors_canonical_constraints(pair: ObjectPair) -> None:
    assert parity_problems(pair) == []


def test_required_field_drift_is_detected() -> None:
    class LooseTicket(TicketFrontmatter):
        phase: int = 0

    problems = parity_problems(ObjectPair("ticket", None, LooseTicket))
    assert any("required" in problem for problem in problems)


def test_enum_drift_is_detected() -> None:
    class LooseCost(Cost):
        pricing: str

    problems = parity_problems(ObjectPair("manifest", "$defs.spec.properties.cost", LooseCost))
    assert any("enum_values" in problem for problem in problems)


def test_pattern_drift_is_detected() -> None:
    class LooseMetadata(GraphMetadata):
        id: str

    problems = parity_problems(ObjectPair("graph", "$defs.metadata", LooseMetadata))
    assert any("patterns" in problem for problem in problems)


def test_const_drift_is_detected() -> None:
    class LooseDocument(GraphDocument):
        kind: str

    problems = parity_problems(ObjectPair("graph", None, LooseDocument))
    assert any("enum_values" in problem for problem in problems)
