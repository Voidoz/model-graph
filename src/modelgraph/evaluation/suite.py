"""Evaluation suite loading, validation, selection, and content hashing.

The suite format is versioned separately from the graph contract (``modelgraph/eval/v0``).
The schema is canonical; the Pydantic models mirror it so the runner gets typed cases. Cases
are JSONL so a suite can grow without rewriting a large YAML document, while the JSONL bytes
participate in the suite hash so changing a case invalidates comparisons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from modelgraph.contracts.graph import ArtifactType
from modelgraph.contracts.loader import load_document
from modelgraph.evaluation.errors import SuiteError
from modelgraph.paths import project_root, schema_dir
from modelgraph.util import canonical_json, sha256_text

SUITE_FILENAMES = ("suite.yaml", "suite.yml")
SUITE_API_VERSION = "modelgraph/eval/v0"
SUITE_KIND = "EvalSuite"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SuiteMetadata(_Model):
    id: str
    name: str
    version: str
    description: str
    labels: dict[str, str] = Field(default_factory=dict)
    authors: list[str] | None = None
    license: str | None = None


class NodeTarget(_Model):
    graph: str
    node: str


class RoutingTarget(_Model):
    graph: str
    router: str | None = None


class E2ETarget(_Model):
    graph: str
    profile: str | None = None


class BaselineConfig(_Model):
    """Which baselines an e2e suite requests, and how oracle labels are produced."""

    best_single: bool = True
    cheapest_capable: bool = True
    oracle_routing: bool = True
    random_routing: bool = True
    incumbent: bool = True
    measure_oracle: bool = True
    oracle_ambiguity_delta: float = Field(default=0.05, ge=0.0, le=1.0)
    random_seed: int = 7
    capability_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_envelope_usd: float | None = Field(default=None, ge=0.0)
    notes: str | None = None


class SuiteBudget(_Model):
    max_cost_usd: float | None = None
    max_tokens: int | None = None
    max_case_cost_usd: float | None = None
    max_case_tokens: int | None = None
    max_case_wall_seconds: float | None = None


class SelectionConfig(_Model):
    case_ids: list[str] | None = None
    tags: list[str] | None = None
    tag_mode: Literal["any", "all"] = "any"
    limit: int | None = None
    reason: str | None = None


class JudgeConfig(_Model):
    enabled: bool = False
    model: str | None = None
    rubric: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None


class OracleConfig(_Model):
    labeling_method: str | None = None
    labeling_cost_usd: float = 0.0
    sampled_from: str | None = None
    ambiguity_rule: str | None = None
    notes: str | None = None


class CalibrationConfig(_Model):
    bins: int = 10
    threshold_steps: int = 20
    notes: str | None = None


class Provenance(_Model):
    source: str | None = None
    contamination: Literal["not-checked", "checked", "unknown"] = "not-checked"
    notes: str | None = None
    private: bool = False


class SuiteSpec(_Model):
    level: Literal["node", "routing", "e2e"]
    provider: Literal["mock", "auto", "ollama", "openai-compatible", "llamacpp"] = "mock"
    claim: str | None = None
    target: NodeTarget | RoutingTarget | E2ETarget
    metric: str
    thresholds: dict[str, float] = Field(default_factory=dict)
    budget: SuiteBudget = Field(default_factory=SuiteBudget)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    oracle: OracleConfig | None = None
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    baselines: BaselineConfig = Field(default_factory=BaselineConfig)
    provenance: Provenance = Field(default_factory=Provenance)
    cases: str
    rubric: str | None = None

    @field_validator("target", mode="before")
    @classmethod
    def _coerce_target_for_level(cls, value: Any, info: Any) -> Any:
        """Disambiguate graph-only targets by the declared level.

        ``{graph: ...}`` is a valid routing target and a valid e2e target on the wire, so the
        union cannot choose from the shape alone. The level field is ordered before target in
        this model, so it is available to the validator.
        """
        level = info.data.get("level")
        if not isinstance(value, dict):
            return value
        if level == "e2e":
            if "node" in value or "router" in value:
                return value
            return E2ETarget.model_validate(value)
        if level == "routing":
            if "node" in value:
                return value
            return RoutingTarget.model_validate(value)
        return value


class SuiteDocument(_Model):
    apiVersion: Literal["modelgraph/eval/v0"]
    kind: Literal["EvalSuite"]
    metadata: SuiteMetadata
    spec: SuiteSpec


class CaseArtifact(_Model):
    name: str
    type: ArtifactType = ArtifactType.text
    media_type: str | None = None
    content: str


class CaseInput(_Model):
    task: str
    artifacts: list[CaseArtifact] = Field(default_factory=list)


class OracleLabel(_Model):
    route: str
    quality: float | None = None
    quality_by_route: dict[str, float] = Field(default_factory=dict)
    cost_usd: float | None = None
    ambiguous: bool = False
    misroute_loss: float | None = None
    note: str | None = None
    labeler: str | None = None
    labeled_at: str | None = None
    labeling_cost_usd: float = 0.0


class CheckSpec(_Model):
    type: Literal[
        "artifact",
        "exact_match",
        "contains",
        "regex",
        "json_schema",
        "command",
        "rubric",
    ]
    artifact: str | None = None
    value: str | None = None
    pattern: str | None = None
    flags: Literal["", "i", "m", "s", "im", "is", "ms", "ims"] = ""
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    schema_path: str | None = None
    command: list[str] | None = None
    timeout_seconds: float | None = None
    expect_exit_code: int = 0
    stdout_contains: str | None = None
    min_score: float = 0.6
    normalize: Literal["none", "strip", "casefold", "collapse_whitespace"] = "none"
    non_empty: bool = True
    min_bytes: int = 0

    @model_validator(mode="after")
    def _required_fields(self) -> CheckSpec:
        if self.type in {"exact_match", "contains"} and self.value is None:
            raise ValueError(f"{self.type} check requires 'value'")
        if self.type == "regex" and self.pattern is None:
            raise ValueError("regex check requires 'pattern'")
        if self.type == "json_schema" and self.schema_ is None and self.schema_path is None:
            raise ValueError("json_schema check requires 'schema' or 'schema_path'")
        if self.type == "command" and not self.command:
            raise ValueError("command check requires a non-empty 'command'")
        return self


MockScript = str | dict[str, Any] | list[str | dict[str, Any]]


class EvalCase(_Model):
    id: str
    tags: list[str] = Field(default_factory=list)
    input: CaseInput
    checks: list[CheckSpec] = Field(default_factory=list)
    oracle: OracleLabel | None = None
    mock: MockScript | None = None
    mock_nodes: dict[str, MockScript] = Field(default_factory=dict)
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class EvalSuite:
    """A validated suite plus the files that produced its content hash."""

    document: SuiteDocument
    source_path: Path
    cases_path: Path
    rubric_path: Path | None
    rubric_text: str | None
    cases: tuple[EvalCase, ...]
    content_hash: str
    warnings: tuple[str, ...] = ()
    file_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def directory(self) -> Path:
        return self.source_path.parent

    @property
    def metadata(self) -> SuiteMetadata:
        return self.document.metadata

    @property
    def spec(self) -> SuiteSpec:
        return self.document.spec

    @property
    def level(self) -> str:
        return self.spec.level

    @property
    def provider(self) -> str:
        return self.spec.provider

    @property
    def hash(self) -> str:
        """Alias for the content hash used in reports."""
        return self.content_hash

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.id for case in self.cases)

    def resolve_graph_path(self, *, root: Path | None = None) -> Path:
        """Resolve the target graph relative to the suite directory, then the repo root."""
        raw = Path(self.spec.target.graph)
        if raw.is_absolute():
            return raw
        local = (self.directory / raw).resolve()
        if local.exists():
            return local
        base = (root or project_root()).resolve()
        return (base / raw).resolve()

    def select_cases(
        self,
        *,
        case_ids: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
        tag_mode: Literal["any", "all"] | None = None,
    ) -> tuple[EvalCase, ...]:
        """Select a deterministic subset in JSONL order.

        Explicit arguments override the suite's ``selection`` block. Case ids and tags both
        narrow the set (AND); tags match any by default. ``limit`` truncates after filtering.
        """
        config = self.spec.selection
        selected_ids = config.case_ids if case_ids is None else case_ids
        selected_tags = config.tags if tags is None else tags
        selected_limit = config.limit if limit is None else limit
        mode = tag_mode or config.tag_mode

        cases = list(self.cases)
        if selected_ids:
            wanted = set(selected_ids)
            known = {case.id for case in cases}
            missing = sorted(wanted - known)
            if missing:
                raise SuiteError(
                    f"suite {self.metadata.id!r} selection references unknown case id(s): {missing}"
                )
            cases = [case for case in cases if case.id in wanted]
        if selected_tags:
            if mode == "all":
                cases = [case for case in cases if all(tag in case.tags for tag in selected_tags)]
            else:
                cases = [case for case in cases if any(tag in case.tags for tag in selected_tags)]
        if selected_limit is not None:
            cases = cases[:selected_limit]
        if not cases:
            raise SuiteError(
                f"suite {self.metadata.id!r} selection matched no cases "
                f"(case_ids={selected_ids!r}, tags={selected_tags!r}, limit={selected_limit!r})"
            )
        return tuple(cases)

    def selection_summary(
        self,
        selected: list[EvalCase],
        *,
        case_ids: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        config = self.spec.selection
        return {
            "requested_case_ids": list(case_ids if case_ids is not None else config.case_ids or []),
            "requested_tags": list(tags if tags is not None else config.tags or []),
            "tag_mode": config.tag_mode,
            "limit": limit if limit is not None else config.limit,
            "selected_cases": len(selected),
            "total_cases": len(self.cases),
            "selected_case_ids": [case.id for case in selected],
            "reason": config.reason,
        }

    def target_node_id(self) -> str | None:
        target = self.spec.target
        if isinstance(target, NodeTarget):
            return target.node
        if isinstance(target, RoutingTarget):
            return target.router
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.metadata.id,
            "name": self.metadata.name,
            "version": self.metadata.version,
            "description": self.metadata.description,
            "hash": self.content_hash,
            "path": str(self.source_path),
            "level": self.level,
            "claim": self.spec.claim or self.metadata.description,
            "labels": dict(self.metadata.labels),
        }


@cache
def eval_suite_schema() -> dict[str, Any]:
    path = schema_dir() / "eval-suite.schema.json"
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise SuiteError(f"cannot read eval suite schema {path}: {exc}", path=str(path)) from exc


@cache
def _suite_validator() -> Draft202012Validator:
    schema = eval_suite_schema()
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:  # pragma: no cover - a broken schema is a repo bug
        raise SuiteError(f"invalid eval suite schema: {exc.message}") from exc
    return Draft202012Validator(schema)


@cache
def _case_validator() -> Draft202012Validator:
    schema = eval_suite_schema()
    wrapper = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": "#/$defs/case",
        "$defs": schema["$defs"],
    }
    return Draft202012Validator(wrapper)


def load_suite(path: str | Path) -> EvalSuite:
    """Load one suite directory (or suite.yaml path) and validate every case."""
    source = Path(path)
    if source.is_dir():
        candidates = [source / name for name in SUITE_FILENAMES]
        found = next((candidate for candidate in candidates if candidate.is_file()), None)
        if found is None:
            raise SuiteError(f"no suite.yaml found in {source}", path=str(source))
        source = found
    if not source.is_file():
        raise SuiteError(f"suite file not found: {source}", path=str(source))
    source = source.resolve()

    raw = load_document(source)
    issues = _document_issues(raw, _suite_validator(), source)
    if issues:
        raise SuiteError(
            f"suite {source} failed schema validation: {issues[0]}"
            + (f" (+{len(issues) - 1} more)" if len(issues) > 1 else ""),
            path=str(source),
            details={"issues": issues},
        )
    try:
        document = SuiteDocument.model_validate(raw)
    except ValidationError as exc:
        raise SuiteError(
            f"suite {source} has an invalid document: {exc}", path=str(source)
        ) from exc

    _check_level_target(document, source)
    directory = source.parent
    cases_path = _relative_file(directory, document.spec.cases, "cases", source)
    rubric_name = document.spec.rubric or (
        document.spec.judge.rubric if document.spec.judge.enabled else None
    )
    rubric_path = _relative_file(directory, rubric_name, "rubric", source) if rubric_name else None
    rubric_text = rubric_path.read_text(encoding="utf-8") if rubric_path else None
    cases = _load_cases(cases_path, document.spec.level)

    warnings: list[str] = []
    has_rubric_checks = any(check.type == "rubric" for case in cases for check in case.checks)
    if has_rubric_checks and not document.spec.judge.enabled:
        warnings.append(
            "cases use rubric checks but spec.judge.enabled is false; rubric checks fail "
            "closed at run time"
        )
    if document.spec.judge.enabled and not has_rubric_checks:
        warnings.append("judge is enabled but no case has a rubric check")
    if document.spec.level == "routing" and document.spec.oracle is None:
        raise SuiteError(
            f"routing suite {document.metadata.id!r} must document spec.oracle "
            "(labeling method and cost)",
            path=str(source),
        )

    file_hashes = {
        source.name: sha256_text(source.read_text(encoding="utf-8")),
        _relative_name(cases_path, directory): sha256_text(cases_path.read_text(encoding="utf-8")),
    }
    if rubric_path is not None and rubric_text is not None:
        file_hashes[_relative_name(rubric_path, directory)] = sha256_text(rubric_text)
    hash_payload = {
        "document": document.model_dump(mode="json", by_alias=True, exclude_none=True),
        "files": file_hashes,
    }
    content_hash = sha256_text(canonical_json(hash_payload))
    return EvalSuite(
        document=document,
        source_path=source,
        cases_path=cases_path,
        rubric_path=rubric_path,
        rubric_text=rubric_text,
        cases=tuple(cases),
        content_hash=content_hash,
        warnings=tuple(warnings),
        file_hashes=file_hashes,
    )


def _check_level_target(document: SuiteDocument, source: Path) -> None:
    target = document.spec.target
    if document.spec.level == "node" and not isinstance(target, NodeTarget):
        raise SuiteError(
            f"node suite {source} must declare target.graph and target.node",
            path=str(source),
        )
    if document.spec.level == "routing" and not isinstance(target, RoutingTarget):
        raise SuiteError(
            f"routing suite {source} must declare target.graph (and optional target.router)",
            path=str(source),
        )
    if document.spec.level == "e2e" and not isinstance(target, E2ETarget):
        raise SuiteError(
            f"e2e suite {source} must declare target.graph (and optional target.profile)",
            path=str(source),
        )


def _load_cases(cases_path: Path, level: str) -> list[EvalCase]:
    try:
        text = cases_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SuiteError(
            f"cannot read cases file {cases_path}: {exc}", path=str(cases_path)
        ) from exc
    cases: list[EvalCase] = []
    seen: set[str] = set()
    validator = _case_validator()
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SuiteError(
                f"{cases_path}:{line_number}: invalid JSON: {exc}",
                path=f"{cases_path}:{line_number}",
            ) from exc
        if not isinstance(payload, dict):
            raise SuiteError(
                f"{cases_path}:{line_number}: each case must be a JSON object",
                path=f"{cases_path}:{line_number}",
            )
        issues = _document_issues(payload, validator, cases_path, line=line_number)
        if issues:
            raise SuiteError(
                f"{cases_path}:{line_number}: case failed schema validation: {issues[0]}"
                + (f" (+{len(issues) - 1} more)" if len(issues) > 1 else ""),
                path=f"{cases_path}:{line_number}",
                details={"issues": issues},
            )
        try:
            case = EvalCase.model_validate(payload)
        except ValidationError as exc:
            raise SuiteError(
                f"{cases_path}:{line_number}: invalid case: {exc}",
                path=f"{cases_path}:{line_number}",
            ) from exc
        if case.id in seen:
            raise SuiteError(
                f"{cases_path}:{line_number}: duplicate case id {case.id!r}",
                path=f"{cases_path}:{line_number}",
            )
        seen.add(case.id)
        if level == "node" and not case.checks:
            raise SuiteError(
                f"{cases_path}:{line_number}: node case {case.id!r} requires at least one check",
                path=f"{cases_path}:{line_number}",
            )
        if level == "e2e" and not case.checks:
            raise SuiteError(
                f"{cases_path}:{line_number}: e2e case {case.id!r} requires at least one check",
                path=f"{cases_path}:{line_number}",
            )
        if level == "routing" and case.oracle is None:
            raise SuiteError(
                f"{cases_path}:{line_number}: routing case {case.id!r} requires an oracle route",
                path=f"{cases_path}:{line_number}",
            )
        cases.append(case)
    if not cases:
        raise SuiteError(f"{cases_path} contains no cases", path=str(cases_path))
    return cases


def _document_issues(
    payload: dict[str, Any],
    validator: Draft202012Validator,
    source: Path,
    *,
    line: int | None = None,
) -> list[str]:
    issues: list[str] = []
    for error in sorted(
        validator.iter_errors(payload),
        key=lambda item: [str(part) for part in item.path],
    ):
        parts: list[str] = []
        for part in error.absolute_path:
            parts.append(
                f"[{part}]" if isinstance(part, int) else f".{part}" if parts else str(part)
            )
        location = "".join(parts) or "$"
        prefix = f"{source}" + (f":{line}" if line is not None else "")
        issues.append(f"{prefix} {location}: {error.message}")
    return issues


def _relative_file(directory: Path, name: str, kind: str, source: Path) -> Path:
    raw = Path(name)
    if raw.is_absolute():
        raise SuiteError(
            f"{kind} path must be relative to the suite directory: {name!r}",
            path=str(source),
        )
    resolved = (directory / raw).resolve()
    try:
        resolved.relative_to(directory.resolve())
    except ValueError as exc:
        raise SuiteError(
            f"{kind} path escapes the suite directory: {name!r}",
            path=str(source),
        ) from exc
    if not resolved.is_file():
        raise SuiteError(f"{kind} file not found: {resolved}", path=str(resolved))
    return resolved


def _relative_name(path: Path, directory: Path) -> str:
    try:
        return path.resolve().relative_to(directory.resolve()).as_posix()
    except ValueError:  # pragma: no cover - _relative_file already prevents this
        return path.name


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))
