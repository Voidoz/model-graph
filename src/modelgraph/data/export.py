"""Trace-to-dataset export with provenance, consent, redaction, splits, and hashes.

The exporter treats a recorded run as immutable input. It reads ``run.json``,
``trace.jsonl``, and ``envelope.json``, reconstructs the exact rendered prompts through the
runtime's own replay path, applies configured redaction, labels examples where evidence exists,
deduplicates, assigns group splits, and writes a deterministic JSONL dataset plus a manifest.

It never invents a positive label: a node label is ``unknown`` unless a check or a human label
file supplies evidence, and a routing label is ``unknown`` unless an oracle/human route is
recorded. Failure traces are excluded unless ``include_failures`` is explicit.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from modelgraph import __version__
from modelgraph.compiler import compile_graph
from modelgraph.contracts.envelope import Artifact
from modelgraph.contracts.loader import load_document
from modelgraph.data.dedupe import (
    deduplicate,
    hamming_distance,
    normalize_text,
    simhash,
)
from modelgraph.data.redaction import build_redactor
from modelgraph.errors import ModelGraphError
from modelgraph.evaluation.checks import evaluate_checks
from modelgraph.evaluation.suite import CheckSpec, load_suite
from modelgraph.paths import schema_dir
from modelgraph.providers.base import Capabilities, ChatRequest, ChatResponse, Health
from modelgraph.runtime.engine import run_graph
from modelgraph.runtime.trace import ReplayProvider, read_events
from modelgraph.util import canonical_json, sha256_text, utc_now

DATASET_API_VERSION = "modelgraph/v0"
DATASET_KIND = "Dataset"
RECORD_SCHEMA = "modelgraph/dataset-example/v0"
SPLITS_SCHEMA = "modelgraph/dataset-splits/v0"
REDACTION_SCHEMA = "modelgraph/redaction-report/v0"
DATASET_SCHEMA_FILE = "dataset.schema.json"
EXAMPLE_FILE = "examples.jsonl"
MANIFEST_FILE = "dataset.json"
REDACTION_FILE = "redaction-report.json"
SPLITS_FILE = "splits.json"
SPLIT_DIR = "splits"
SPLIT_NAMES = ("train", "validation", "test")
OUTCOMES = ("success", "failure", "unknown")
GROUP_BY = ("source", "graph", "run", "license")
DEDUPE_MODES = ("none", "exact", "near", "exact+near")
_LABEL_OUTCOMES = frozenset({"success", "failure", "unknown"})


class DatasetExportError(ModelGraphError):
    """The dataset could not be exported safely."""

    code = "MG-5001"


@dataclass(frozen=True)
class DatasetBundle:
    """A loaded dataset, validated against the dataset schema."""

    path: Path
    manifest: dict[str, Any]
    examples: list[dict[str, Any]]

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_FILE

    @property
    def examples_path(self) -> Path:
        return self.path / EXAMPLE_FILE


@dataclass(frozen=True)
class ExportResult:
    """Summary returned by :func:`export_dataset`."""

    dataset_dir: Path
    manifest: dict[str, Any]
    redaction: dict[str, Any]
    examples: int
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": str(self.dataset_dir),
            "manifest": str(self.dataset_dir / MANIFEST_FILE),
            "examples": str(self.dataset_dir / EXAMPLE_FILE),
            "redaction_report": str(self.dataset_dir / REDACTION_FILE),
            "splits": str(self.dataset_dir / SPLITS_FILE),
            "counts": dict(self.counts),
            "example_count": self.examples,
            "by_node": self.manifest["spec"]["examples"]["by_node"],
            "by_route": self.manifest["spec"]["examples"]["by_route"],
            "redaction": self.redaction,
        }


@dataclass
class _RunData:
    path: Path
    manifest: dict[str, Any]
    events: list[dict[str, Any]]
    envelope: dict[str, Any] | None
    manifest_bytes: bytes
    trace_bytes: bytes

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id") or self.path.name)

    @property
    def outcome(self) -> str:
        return str(self.manifest.get("outcome") or "unknown")

    @property
    def task(self) -> str:
        request = self.manifest.get("request")
        if isinstance(request, dict):
            return str(request.get("task") or "")
        return ""

    @property
    def graph(self) -> dict[str, Any]:
        value = self.manifest.get("graph")
        return cast(dict[str, Any], value) if isinstance(value, dict) else {}


@dataclass
class _Occurrence:
    node: str
    node_kind: str
    depth: int
    step: int
    event_index: int
    prompt_hash: str | None = None
    request_events: list[dict[str, Any]] = field(default_factory=list)
    responses: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] | None = None


@dataclass(frozen=True)
class _LabelEntry:
    run_id: str | None
    kind: str | None
    node: str | None
    outcome: str | None
    route: str | None
    ambiguous: bool | None
    reason: str | None
    labeler: str | None
    source: str
    group: str | None

    def matches(self, run_id: str, kind: str, node: str) -> bool:
        if self.run_id not in (None, "*", run_id):
            return False
        if self.kind not in (None, "*", kind):
            return False
        return self.node in (None, "*", node)


@dataclass(frozen=True)
class _CheckEntry:
    run_id: str | None
    node: str
    checks: tuple[CheckSpec, ...]
    suite_dir: Path

    def matches(self, run_id: str, node: str) -> bool:
        return self.node in ("*", node) and self.run_id in (None, "*", run_id)


def export_dataset(
    runs: Sequence[str | Path],
    out_dir: str | Path,
    *,
    dataset_id: str = "trace-export",
    dataset_name: str = "Trace export",
    dataset_version: str = "0.1.0",
    dataset_description: str = "Trace-derived examples exported by mg data export.",
    task: str | None = None,
    license_id: str = "unknown",
    license_uri: str | None = None,
    consent: str | None = None,
    consent_scope: str = "local-training",
    require_consent: bool = True,
    include_failures: bool = False,
    redact: str | Path = "default",
    redact_action: str = "hash",
    labels: Sequence[str | Path] = (),
    checks: Sequence[str | Path] = (),
    allow_command_checks: bool = False,
    eval_suites: Sequence[str | Path] = (),
    dedupe: str = "exact+near",
    near_duplicate_distance: int = 3,
    group_by: str = "source",
    split_ratio: str = "0.8,0.1,0.1",
    split_seed: str = "modelgraph-v0",
    split_plan: str | Path | None = None,
    created_at: str | None = None,
) -> ExportResult:
    """Export one or more recorded runs into a deterministic dataset directory."""
    if not runs:
        raise DatasetExportError("at least one --run is required")
    if group_by not in GROUP_BY:
        raise DatasetExportError(f"group_by must be one of {list(GROUP_BY)}, got {group_by!r}")
    if dedupe not in DEDUPE_MODES:
        raise DatasetExportError(f"dedupe must be one of {list(DEDUPE_MODES)}, got {dedupe!r}")
    if not dataset_id or not re.fullmatch(r"[a-z][a-z0-9-]{2,63}", dataset_id):
        raise DatasetExportError(
            f"dataset id must match ^[a-z][a-z0-9-]{{2,63}}$ (got {dataset_id!r})"
        )
    if not license_id.strip():
        raise DatasetExportError("license must not be empty")
    ratios = _parse_ratios(split_ratio)
    if near_duplicate_distance < 0 or near_duplicate_distance > 64:
        raise DatasetExportError("near-duplicate distance must be between 0 and 64")
    consent_text = (consent or "").strip()
    if require_consent and not consent_text:
        raise DatasetExportError(
            "consent is required for trace export; pass --consent with the recorded "
            "permission, or explicitly opt out with --no-require-consent"
        )

    run_data = [_load_run(Path(run)) for run in runs]
    run_data.sort(key=lambda item: item.run_id)
    eligible: list[_RunData] = []
    skipped_runs: list[dict[str, str]] = []
    for run in run_data:
        if run.outcome == "succeeded" or include_failures:
            eligible.append(run)
        else:
            skipped_runs.append(
                {
                    "run_id": run.run_id,
                    "outcome": run.outcome,
                    "reason": "run outcome is not succeeded; pass --include-failures",
                }
            )
    if not eligible:
        detail = ", ".join(f"{item['run_id']}={item['outcome']}" for item in skipped_runs)
        raise DatasetExportError(
            "no eligible runs: all runs are non-succeeded and --include-failures was not set"
            + (f" ({detail})" if detail else "")
        )

    redactor = build_redactor(redact, redact_action)
    label_entries = _load_labels(labels)
    check_entries = _load_checks(checks)
    group_overrides = _group_overrides(label_entries)

    dataset_task = task or _default_task(eligible)
    source_license = _dataset_license(license_id)
    consent_record = {
        "required": require_consent,
        "recorded": bool(consent_text),
        "statement": consent_text or None,
        "scope": consent_scope or None,
    }
    redacted_consent, _ = redactor.redact_record(consent_record)
    redacted_consent = cast(dict[str, Any], redacted_consent)

    examples: list[dict[str, Any]] = []
    prompt_misses = 0
    prompt_limitations: list[str] = []
    sources: list[dict[str, Any]] = []
    for run in eligible:
        prompts, prompt_note = _capture_prompts(run)
        if prompt_note:
            prompt_limitations.append(prompt_note)
        group = _group_id(run, group_by, group_overrides, license_id)
        artifacts_by_producer = _artifacts_by_producer(run)
        occurrences = _occurrences(run.events)
        for occurrence in occurrences:
            if occurrence.node_kind == "router":
                example = _routing_example(
                    run,
                    occurrence,
                    prompts,
                    label_entries,
                    group,
                    consent_record,
                    license_id,
                    include_failures,
                )
            elif occurrence.node_kind in {"worker", "joiner"}:
                example = _node_example(
                    run,
                    occurrence,
                    prompts,
                    artifacts_by_producer.get(occurrence.node, []),
                    label_entries,
                    check_entries,
                    allow_command_checks,
                    group,
                    consent_record,
                    license_id,
                    include_failures,
                )
            else:
                continue
            if example is None:
                continue
            if not example["input"].get("rendered"):
                prompt_misses += 1
            examples.append(example)

        source = {
            "uri": str(run.path.resolve()),
            "kind": "run",
            "run_id": run.run_id,
            "run_outcome": run.outcome,
            "graph": {
                "id": str(run.graph.get("id") or "unknown"),
                "version": str(run.graph.get("version") or "unknown"),
                "hash": str(run.graph.get("hash") or "unknown"),
            },
            "trace": str(run.manifest.get("trace") or "trace.jsonl"),
            "license": _graph_license(run) or source_license["id"],
            "group": group,
            "manifest_hash": _bytes_hash(run.manifest_bytes),
            "trace_hash": _bytes_hash(run.trace_bytes),
        }
        redacted_source, _ = redactor.redact_record(source)
        sources.append(cast(dict[str, Any], redacted_source))

    if not examples:
        raise DatasetExportError(
            "the selected runs produced no exportable node or routing examples"
        )

    redacted_examples: list[dict[str, Any]] = []
    for example in examples:
        redacted, names = redactor.redact_record(example)
        redacted = cast(dict[str, Any], redacted)
        redacted["redaction"]["patterns"] = names
        redacted["redaction"]["redacted"] = bool(names)
        _refresh_artifact_hashes(redacted)
        redacted_examples.append(redacted)
    examples = redacted_examples

    contamination, examples = _check_contamination(examples, eval_suites, near_duplicate_distance)

    signature_values = [_dedupe_text(example) for example in examples]
    dedupe_result = deduplicate(
        signature_values,
        mode=dedupe,
        near_distance=near_duplicate_distance,
    )
    examples = [examples[index] for index in dedupe_result.kept]

    groups = sorted({str(example["group"]) for example in examples})
    plan_path = Path(split_plan) if split_plan is not None else None
    assignments, plan = _assign_splits(
        groups,
        plan_path=plan_path,
        seed=split_seed,
        ratios=ratios,
    )
    for example in examples:
        example["split"] = assignments[str(example["group"])]

    for example in examples:
        input_hash = sha256_text(canonical_json(example["input"]))
        output_hash = sha256_text(canonical_json(example["output"]))
        example["hashes"] = {
            "input": input_hash,
            "output": output_hash,
            "example": "sha256:",
        }
        body = {key: value for key, value in example.items() if key not in {"hashes", "example_id"}}
        body["hashes"] = {"input": input_hash, "output": output_hash}
        example_hash = sha256_text(canonical_json(body))
        example["hashes"]["example"] = example_hash
        example["example_id"] = "ex_" + example_hash.split(":", 1)[1][:16]

    examples.sort(
        key=lambda item: (
            str(item["source"]["run_id"]),
            str(item["source"]["node"]),
            int(item["source"]["event_index"]),
            str(item["example_id"]),
        )
    )
    _ensure_unique_ids(examples)

    counts = _counts(examples)
    split_texts = {
        split: _jsonl([example for example in examples if example["split"] == split])
        for split in SPLIT_NAMES
    }
    examples_text = _jsonl(examples)
    split_hashes = {split: sha256_text(text) for split, text in split_texts.items()}
    plan_doc = _split_document(plan, assignments, examples, ratios, split_hashes)

    redacted_task, _ = redactor.redact_record({"task": dataset_task})
    redacted_name, _ = redactor.redact_record({"name": dataset_name})
    redacted_description, _ = redactor.redact_record({"description": dataset_description})
    redacted_command, _ = redactor.redact_record(
        {"command": _export_command(runs, include_failures, redact)}
    )
    redaction_report = redactor.report()
    redaction_report["schema"] = REDACTION_SCHEMA

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_dir = out / SPLIT_DIR
    split_dir.mkdir(parents=True, exist_ok=True)
    (out / EXAMPLE_FILE).write_text(examples_text, encoding="utf-8")
    for split, text in split_texts.items():
        (split_dir / f"{split}.jsonl").write_text(text, encoding="utf-8")
    plan_text = json.dumps(plan_doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    (out / SPLITS_FILE).write_text(plan_text, encoding="utf-8")
    redaction_text = (
        json.dumps(redaction_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    (out / REDACTION_FILE).write_text(redaction_text, encoding="utf-8")
    if plan_path is not None and plan_path.resolve() != (out / SPLITS_FILE).resolve():
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan_text, encoding="utf-8")

    limitations = _limitations(
        include_failures=include_failures,
        skipped_runs=skipped_runs,
        prompt_misses=prompt_misses,
        contamination=contamination,
        redaction=redaction_report,
        license_id=license_id,
        labels=label_entries,
        dedupe=dedupe,
    )
    limitations.extend(prompt_limitations)

    file_hashes = {
        EXAMPLE_FILE: sha256_text(examples_text),
        SPLITS_FILE: sha256_text(plan_text),
        REDACTION_FILE: sha256_text(redaction_text),
        **{f"{SPLIT_DIR}/{split}.jsonl": digest for split, digest in split_hashes.items()},
    }
    manifest = _build_manifest(
        dataset_id=dataset_id,
        dataset_name=str(redacted_name["name"]),
        dataset_version=dataset_version,
        dataset_description=str(redacted_description["description"]),
        dataset_task=str(redacted_task["task"]),
        license_obj=source_license,
        license_uri=license_uri,
        consent_record=redacted_consent,
        sources=sources,
        examples=examples,
        counts=counts,
        examples_text=examples_text,
        split_hashes=split_hashes,
        split_ratios=ratios,
        split_seed=split_seed,
        split_plan=plan_doc,
        group_by=group_by,
        group_overrides=len(group_overrides),
        dedupe=dedupe,
        near_duplicate_distance=near_duplicate_distance,
        dedupe_result=dedupe_result,
        contamination=contamination,
        redaction=redaction_report,
        file_hashes=file_hashes,
        skipped_runs=skipped_runs,
        limitations=limitations,
        command=str(redacted_command["command"]),
        created_at=created_at or utc_now(),
    )
    _validate_or_raise(manifest, examples)
    (out / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return ExportResult(
        dataset_dir=out,
        manifest=manifest,
        redaction=redaction_report,
        examples=len(examples),
        counts={split: int(counts.get(split, 0)) for split in SPLIT_NAMES},
    )


def load_dataset(dataset_dir: str | Path, *, validate: bool = True) -> DatasetBundle:
    """Load and optionally validate a dataset written by :func:`export_dataset`."""
    path = Path(dataset_dir)
    manifest_path = path / MANIFEST_FILE
    examples_path = path / EXAMPLE_FILE
    if not manifest_path.is_file():
        raise DatasetExportError(f"dataset manifest not found: {manifest_path}", path=str(path))
    if not examples_path.is_file():
        raise DatasetExportError(f"dataset examples not found: {examples_path}", path=str(path))
    try:
        manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise DatasetExportError(
            f"invalid dataset manifest: {exc}", path=str(manifest_path)
        ) from exc
    examples: list[dict[str, Any]] = []
    try:
        for line in examples_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                examples.append(cast(dict[str, Any], json.loads(line)))
    except json.JSONDecodeError as exc:
        raise DatasetExportError(f"invalid dataset JSONL: {exc}", path=str(examples_path)) from exc
    if validate:
        issues = [f"{MANIFEST_FILE}: {message}" for message in validate_dataset_manifest(manifest)]
        for index, example in enumerate(examples):
            issues.extend(
                f"{EXAMPLE_FILE}:{index + 1}: {message}"
                for message in validate_dataset_example(example)
            )
        if issues:
            raise DatasetExportError("dataset validation failed: " + "; ".join(issues[:5]))
    return DatasetBundle(path=path, manifest=manifest, examples=examples)


# --------------------------------------------------------------------------------------------
# Schema loading and validation
# --------------------------------------------------------------------------------------------


@cache
def dataset_schema() -> dict[str, Any]:
    path = schema_dir() / DATASET_SCHEMA_FILE
    try:
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except OSError as exc:
        raise DatasetExportError(f"cannot read dataset schema {path}: {exc}") from exc


@cache
def _manifest_validator() -> Draft202012Validator:
    schema = dataset_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@cache
def _example_validator() -> Draft202012Validator:
    schema = dataset_schema()
    subschema = {"$defs": schema.get("$defs", {}), "$ref": "#/$defs/example"}
    Draft202012Validator.check_schema(subschema)
    return Draft202012Validator(subschema)


def validate_dataset_manifest(document: dict[str, Any]) -> list[str]:
    return _schema_errors(_manifest_validator(), document)


def validate_dataset_example(document: dict[str, Any]) -> list[str]:
    return _schema_errors(_example_validator(), document)


def _schema_errors(validator: Draft202012Validator, document: Any) -> list[str]:
    messages = []
    for error in sorted(validator.iter_errors(document), key=lambda item: list(item.path)):
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(f"{location}: {error.message}")
    return messages


def _validate_or_raise(manifest: dict[str, Any], examples: list[dict[str, Any]]) -> None:
    issues = validate_dataset_manifest(manifest)
    for index, example in enumerate(examples):
        issues.extend(
            f"example {index + 1}: {message}" for message in validate_dataset_example(example)
        )
    if issues:
        raise DatasetExportError(
            "generated dataset does not validate against schemas/dataset.schema.json: "
            + "; ".join(issues[:5])
        )


# --------------------------------------------------------------------------------------------
# Run and trace loading
# --------------------------------------------------------------------------------------------


def _load_run(path: Path) -> _RunData:
    manifest_path = path / "run.json"
    if not manifest_path.is_file():
        raise DatasetExportError(f"run manifest not found: {manifest_path}", path=str(path))
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = cast(dict[str, Any], json.loads(manifest_bytes))
    except json.JSONDecodeError as exc:
        raise DatasetExportError(f"invalid run manifest: {exc}", path=str(manifest_path)) from exc
    trace_path = path / str(manifest.get("trace") or "trace.jsonl")
    if not trace_path.is_file():
        raise DatasetExportError(f"trace not found: {trace_path}", path=str(path))
    trace_bytes = trace_path.read_bytes()
    try:
        events = read_events(trace_path)
    except json.JSONDecodeError as exc:
        raise DatasetExportError(f"invalid trace JSONL: {exc}", path=str(trace_path)) from exc
    envelope_path = path / "envelope.json"
    envelope: dict[str, Any] | None = None
    if envelope_path.is_file():
        try:
            envelope = cast(dict[str, Any], json.loads(envelope_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise DatasetExportError(f"invalid envelope: {exc}", path=str(envelope_path)) from exc
    return _RunData(
        path=path,
        manifest=manifest,
        events=events,
        envelope=envelope,
        manifest_bytes=manifest_bytes,
        trace_bytes=trace_bytes,
    )


def _artifacts_by_producer(run: _RunData) -> dict[str, list[dict[str, Any]]]:
    if not run.envelope:
        return {}
    values = run.envelope.get("artifacts")
    if not isinstance(values, list):
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        producer = str(value.get("producer") or "")
        content = value.get("content")
        if not producer or not isinstance(content, str):
            continue
        result.setdefault(producer, []).append(
            {
                "name": str(value.get("name") or "artifact"),
                "type": str(value.get("type") or "text"),
                "content": content,
                "hash": str(value.get("hash") or sha256_text(content)),
                "bytes": len(content.encode("utf-8")),
            }
        )
    return result


def _occurrences(events: Sequence[dict[str, Any]]) -> list[_Occurrence]:
    active: list[_Occurrence] = []
    completed: list[_Occurrence] = []
    for index, event in enumerate(events):
        name = event.get("event")
        if name == "node.entered":
            active.append(
                _Occurrence(
                    node=str(event.get("node") or ""),
                    node_kind=str(event.get("kind") or "unknown"),
                    depth=int(event.get("depth") or 0),
                    step=int(event.get("step") or 0),
                    event_index=index,
                )
            )
            continue
        occurrence = _find_occurrence(active, event)
        if occurrence is None:
            continue
        if name == "render.completed":
            occurrence.prompt_hash = str(event.get("prompt_hash") or "") or None
        elif name == "provider.request":
            occurrence.request_events.append(event)
            if not occurrence.prompt_hash and event.get("prompt_hash"):
                occurrence.prompt_hash = str(event["prompt_hash"])
        elif name == "provider.response":
            occurrence.responses.append(event)
        elif name == "provider.error":
            occurrence.errors.append(event)
        elif name == "decision.made":
            occurrence.decision = event
        elif name == "node.exited":
            active.remove(occurrence)
            completed.append(occurrence)
    completed.extend(active)
    return completed


def _find_occurrence(active: Sequence[_Occurrence], event: dict[str, Any]) -> _Occurrence | None:
    node = event.get("node")
    if node is None:
        return None
    depth = event.get("depth")
    for occurrence in reversed(active):
        if occurrence.node != str(node):
            continue
        if depth is not None and occurrence.depth != int(depth):
            continue
        return occurrence
    return None


# --------------------------------------------------------------------------------------------
# Rendered prompt reconstruction through the runtime replay path
# --------------------------------------------------------------------------------------------


class _CapturingReplayProvider:
    name = "replay-capture"

    def __init__(self, inner: ReplayProvider):
        self.inner = inner
        self.requests: list[ChatRequest] = []

    def capabilities(self) -> Capabilities:
        return self.inner.capabilities()

    def health(self) -> Health:
        return self.inner.health()

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return self.inner.chat(request)


def _capture_prompts(
    run: _RunData,
) -> tuple[dict[tuple[str, str], list[list[dict[str, str]]]], str | None]:
    source = run.graph.get("source")
    if not source:
        return {}, f"{run.run_id}: graph source is not recorded; rendered prompts unavailable"
    graph_path = Path(str(source))
    if not graph_path.is_file():
        return (
            {},
            f"{run.run_id}: graph source {graph_path} is missing; rendered prompts unavailable",
        )
    compiled = compile_graph(graph_path, strict_models=True)
    if not compiled.ok or compiled.graph is None:
        return {}, f"{run.run_id}: graph could not be compiled; rendered prompts unavailable"
    if compiled.graph.graph_hash != run.graph.get("hash"):
        return (
            {},
            f"{run.run_id}: graph hash changed since the run; rendered prompts not reconstructed",
        )
    replay = ReplayProvider(run.events)
    capture = _CapturingReplayProvider(replay)
    # The captures up to a failure are still useful, and prompt hashes remain recorded.
    with suppress(Exception):
        run_graph(
            compiled.graph,
            run.task,
            lambda _manifest: capture,
            persist=False,
            run_id="prompt-capture",
        )
    request_events = [
        event
        for event in run.events
        if event.get("event") == "provider.request" and event.get("prompt_hash")
    ]
    prompts: dict[tuple[str, str], list[list[dict[str, str]]]] = {}
    for request, event in zip(capture.requests, request_events, strict=False):
        node = str(event.get("node") or "")
        if request.metadata.get("node_id") != node:
            continue
        prompt_hash = str(event.get("prompt_hash") or "")
        if not prompt_hash:
            continue
        messages = [message.to_dict() for message in request.messages]
        prompts.setdefault((node, prompt_hash), []).append(messages)
    return prompts, None


def _take_prompt(
    prompts: dict[tuple[str, str], list[list[dict[str, str]]]],
    node: str,
    prompt_hash: str | None,
) -> list[dict[str, str]]:
    if not prompt_hash:
        return []
    values = prompts.get((node, prompt_hash))
    if not values:
        return []
    return values.pop(0)


def _menu_from_messages(
    messages: Sequence[dict[str, str]],
) -> dict[str, Any]:
    for message in reversed(messages):
        text = str(message.get("content") or "")
        start = text.find("<routes>")
        end = text.find("</routes>")
        if start == -1 or end == -1:
            continue
        block = text[start : end + len("</routes>")]
        routes = re.findall(r'<route id="([a-z][a-z0-9-]*)"', block)
        return {"available": True, "text": block, "routes": routes}
    return {"available": False, "text": None, "routes": []}


# --------------------------------------------------------------------------------------------
# Example construction
# --------------------------------------------------------------------------------------------


def _node_example(
    run: _RunData,
    occurrence: _Occurrence,
    prompts: dict[tuple[str, str], list[list[dict[str, str]]]],
    artifacts: list[dict[str, Any]],
    labels: Sequence[_LabelEntry],
    checks: Sequence[_CheckEntry],
    allow_command_checks: bool,
    group: str,
    consent: dict[str, Any],
    license_id: str,
    include_failures: bool,
) -> dict[str, Any] | None:
    if not occurrence.responses and not occurrence.errors:
        return None
    if run.outcome != "succeeded" and not include_failures:
        return None
    messages = _take_prompt(prompts, occurrence.node, occurrence.prompt_hash)
    response = occurrence.responses[-1] if occurrence.responses else None
    artifact_dicts = [
        {
            "name": artifact["name"],
            "type": artifact["type"],
            "content": artifact["content"],
            "hash": artifact["hash"],
            "bytes": artifact["bytes"],
        }
        for artifact in artifacts
    ]
    output = {
        "artifacts": artifact_dicts,
        "finish_reason": response.get("finish_reason") if response else None,
        "usage": response.get("usage")
        if response and isinstance(response.get("usage"), dict)
        else {},
        "attempts": max(1, len(occurrence.responses) or len(occurrence.request_events)),
        "error": _occurrence_error(occurrence),
    }
    label = _node_label(
        run,
        occurrence,
        artifacts,
        labels,
        checks,
        allow_command_checks,
        f"{run.run_id}:{occurrence.event_index}",
    )
    return _base_example(
        run=run,
        occurrence=occurrence,
        group=group,
        input_value={
            "task": run.task,
            "prompt_hash": occurrence.prompt_hash,
            "rendered": bool(messages),
            "messages": messages,
        },
        output=output,
        label=label,
        consent=consent,
        license_id=license_id,
    )


def _routing_example(
    run: _RunData,
    occurrence: _Occurrence,
    prompts: dict[tuple[str, str], list[list[dict[str, str]]]],
    labels: Sequence[_LabelEntry],
    group: str,
    consent: dict[str, Any],
    license_id: str,
    include_failures: bool,
) -> dict[str, Any] | None:
    if not occurrence.responses and not occurrence.errors and not occurrence.decision:
        return None
    if run.outcome != "succeeded" and not include_failures:
        return None
    messages = _take_prompt(prompts, occurrence.node, occurrence.prompt_hash)
    response = occurrence.responses[-1] if occurrence.responses else None
    decision = occurrence.decision or {}
    decision_value = decision.get("decision")
    chosen_route = decision.get("to") if decision else None
    output = {
        "chosen_route": chosen_route,
        "chosen_decision": decision_value if decision_value in {"route", "fallback"} else None,
        "confidence": decision.get("confidence"),
        "threshold": decision.get("threshold"),
        "reason": decision.get("reason"),
        "fallback_used": decision_value == "fallback",
        "finish_reason": response.get("finish_reason") if response else None,
        "usage": response.get("usage")
        if response and isinstance(response.get("usage"), dict)
        else {},
        "error": _occurrence_error(occurrence),
    }
    label = _routing_label(
        run,
        occurrence,
        output,
        labels,
        include_failures,
    )
    return _base_example(
        run=run,
        occurrence=occurrence,
        group=group,
        input_value={
            "task": run.task,
            "prompt_hash": occurrence.prompt_hash,
            "rendered": bool(messages),
            "messages": messages,
            "menu": _menu_from_messages(messages),
        },
        output=output,
        label=label,
        consent=consent,
        license_id=license_id,
    )


def _base_example(
    *,
    run: _RunData,
    occurrence: _Occurrence,
    group: str,
    input_value: dict[str, Any],
    output: dict[str, Any],
    label: dict[str, Any],
    consent: dict[str, Any],
    license_id: str,
) -> dict[str, Any]:
    kind = "routing" if occurrence.node_kind == "router" else "node"
    return {
        "schema": RECORD_SCHEMA,
        "kind": kind,
        "split": "train",
        "group": group,
        "source": {
            "run_id": run.run_id,
            "trace": str(run.manifest.get("trace") or "trace.jsonl"),
            "run_outcome": run.outcome,
            "graph": {
                "id": str(run.graph.get("id") or "unknown"),
                "version": str(run.graph.get("version") or "unknown"),
                "hash": str(run.graph.get("hash") or "unknown"),
            },
            "node": occurrence.node,
            "node_kind": occurrence.node_kind,
            "event_index": occurrence.event_index,
        },
        "input": input_value,
        "output": output,
        "label": label,
        "consent": dict(consent),
        "license": _dataset_license(license_id),
        "redaction": {"report": REDACTION_FILE, "redacted": False, "patterns": []},
        "hashes": {},
    }


def _occurrence_error(occurrence: _Occurrence) -> dict[str, str] | None:
    if occurrence.errors:
        error = occurrence.errors[-1]
        return {
            "code": str(error.get("code") or "MG-3000"),
            "message": str(error.get("message") or "provider failed"),
        }
    return None


# --------------------------------------------------------------------------------------------
# Labels and checks
# --------------------------------------------------------------------------------------------


def _node_label(
    run: _RunData,
    occurrence: _Occurrence,
    artifacts: list[dict[str, Any]],
    labels: Sequence[_LabelEntry],
    checks: Sequence[_CheckEntry],
    allow_command_checks: bool,
    temporary_id: str,
) -> dict[str, Any]:
    base = _unknown_label()
    entry = _lookup_label(labels, run.run_id, "node", occurrence.node)
    if run.outcome != "succeeded":
        base.update(
            {
                "outcome": "failure",
                "source": "run_outcome",
                "reason": f"run outcome: {run.outcome}",
            }
        )
        return base
    if entry is not None and entry.outcome is not None:
        base.update(
            {
                "outcome": entry.outcome,
                "source": entry.source,
                "reason": entry.reason,
                "labeler": entry.labeler,
            }
        )
        return base
    check_label, skipped = _apply_checks(
        run, occurrence, artifacts, checks, allow_command_checks, temporary_id
    )
    if check_label is not None:
        return check_label
    if skipped:
        base["reason"] = "no executable checks: " + ", ".join(skipped)
    return base


def _routing_label(
    run: _RunData,
    occurrence: _Occurrence,
    output: dict[str, Any],
    labels: Sequence[_LabelEntry],
    include_failures: bool,
) -> dict[str, Any]:
    base = _unknown_label()
    entry = _lookup_label(labels, run.run_id, "routing", occurrence.node)
    route = entry.route if entry is not None else None
    should_abstain = entry.ambiguous if entry is not None else None
    if route == "abstain":
        should_abstain = True
    elif route is not None and should_abstain is None:
        should_abstain = False
    if entry is not None:
        base["route"] = route
        base["should_abstain"] = should_abstain
        base["labeler"] = entry.labeler
    if run.outcome != "succeeded" and include_failures:
        base.update(
            {
                "outcome": "failure",
                "source": "run_outcome",
                "reason": f"run outcome: {run.outcome}",
            }
        )
        return base
    if entry is not None and entry.outcome is not None:
        base.update(
            {
                "outcome": entry.outcome,
                "source": entry.source,
                "reason": entry.reason,
            }
        )
        return base
    if route:
        chosen = output.get("chosen_route")
        decision = output.get("chosen_decision")
        if route == "abstain":
            matched = decision == "fallback"
        else:
            matched = decision == "route" and chosen == route
        base.update(
            {
                "outcome": "success" if matched else "failure",
                "source": entry.source if entry is not None else "oracle",
                "reason": (
                    None
                    if matched
                    else f"chosen {chosen or 'nothing'} ({decision or 'no decision'}) "
                    f"does not match oracle {route!r}"
                ),
            }
        )
    return base


def _unknown_label() -> dict[str, Any]:
    return {
        "outcome": "unknown",
        "source": "unknown",
        "route": None,
        "should_abstain": None,
        "reason": "no outcome evidence recorded",
        "labeler": None,
        "checks": [],
    }


def _lookup_label(
    labels: Sequence[_LabelEntry], run_id: str, kind: str, node: str
) -> _LabelEntry | None:
    matched = [entry for entry in labels if entry.matches(run_id, kind, node)]
    return matched[-1] if matched else None


def _apply_checks(
    run: _RunData,
    occurrence: _Occurrence,
    artifacts: list[dict[str, Any]],
    checks: Sequence[_CheckEntry],
    allow_command_checks: bool,
    temporary_id: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    matched = [entry for entry in checks if entry.matches(run.run_id, occurrence.node)]
    if not matched:
        return None, []
    artifact_models: list[Artifact] = []
    for item in artifacts:
        try:
            artifact_models.append(
                Artifact.model_validate(
                    {
                        "name": item["name"],
                        "type": item["type"],
                        "content": item["content"],
                        "producer": occurrence.node,
                        "created_at": run.manifest.get("finished_at")
                        or run.manifest.get("started_at")
                        or "1970-01-01T00:00:00Z",
                        "hash": item["hash"],
                    }
                )
            )
        except ValidationError as exc:
            raise DatasetExportError(
                f"cannot apply checks: invalid artifact {item.get('name')!r}: {exc}"
            ) from exc
    results: list[dict[str, Any]] = []
    skipped: list[str] = []
    for entry in matched:
        executable = []
        for check in entry.checks:
            if check.type == "command" and not allow_command_checks:
                skipped.append("command check disabled (pass --allow-command-checks)")
            elif check.type == "rubric":
                skipped.append("rubric check requires a judge; not available in export")
            else:
                executable.append(check)
        if not executable:
            continue
        try:
            outcome = evaluate_checks(
                executable,
                artifact_models,
                artifact_names=[artifact.name for artifact in artifact_models],
                task=run.task,
                case_id=temporary_id,
                rubric_text=None,
                judge=None,
                suite_dir=entry.suite_dir,
                metadata={"run_id": run.run_id, "node": occurrence.node},
                allow_commands=allow_command_checks,
            )
        except (ModelGraphError, OSError, ValueError) as exc:
            raise DatasetExportError(
                f"check evaluation failed for {occurrence.node!r}: {exc}"
            ) from exc
        for result in outcome.results:
            results.append(
                {
                    "type": result.type,
                    "artifact": result.artifact,
                    "passed": result.passed,
                    "detail": result.detail,
                }
            )
    if not results:
        return None, skipped
    passed = all(bool(result["passed"]) for result in results)
    return (
        {
            "outcome": "success" if passed else "failure",
            "source": "eval_check",
            "route": None,
            "should_abstain": None,
            "reason": (
                f"{len(results)} checks passed"
                if passed
                else next(str(result["detail"]) for result in results if not bool(result["passed"]))
            ),
            "labeler": None,
            "checks": results,
        },
        skipped,
    )


def _load_labels(paths: Sequence[str | Path]) -> list[_LabelEntry]:
    entries: list[_LabelEntry] = []
    for path in paths:
        for raw in _load_items(Path(path), "labels"):
            kind = _optional_str(raw.get("kind"))
            if kind is not None and kind not in {"node", "routing"}:
                raise DatasetExportError(
                    f"label entry kind must be node or routing (got {kind!r})", path=str(path)
                )
            outcome = _optional_str(raw.get("outcome"))
            if outcome is not None and outcome not in _LABEL_OUTCOMES:
                raise DatasetExportError(
                    f"label outcome must be one of {sorted(_LABEL_OUTCOMES)} (got {outcome!r})",
                    path=str(path),
                )
            route = _optional_str(raw.get("route")) or _optional_str(raw.get("oracle_route"))
            if route is not None and kind not in (None, "routing"):
                raise DatasetExportError(
                    f"route labels only apply to routing examples (got kind {kind!r})",
                    path=str(path),
                )
            entries.append(
                _LabelEntry(
                    run_id=_optional_str(raw.get("run_id")),
                    kind=kind,
                    node=_optional_str(raw.get("node")),
                    outcome=outcome,
                    route=route,
                    ambiguous=_optional_bool(raw.get("ambiguous")),
                    reason=_optional_str(raw.get("reason")),
                    labeler=_optional_str(raw.get("labeler")),
                    source=_optional_str(raw.get("source")) or "human_label",
                    group=_optional_str(raw.get("group")),
                )
            )
    return entries


def _load_checks(paths: Sequence[str | Path]) -> list[_CheckEntry]:
    entries: list[_CheckEntry] = []
    for path in paths:
        for raw in _load_items(Path(path), "checks"):
            node = _optional_str(raw.get("node"))
            if not node:
                raise DatasetExportError(
                    "check entry requires a node (use '*' for every node)", path=str(path)
                )
            raw_checks = raw.get("checks")
            if not isinstance(raw_checks, list) or not raw_checks:
                raise DatasetExportError(
                    f"check entry for {node!r} requires a non-empty 'checks' list",
                    path=str(path),
                )
            checks: list[CheckSpec] = []
            for item in raw_checks:
                try:
                    checks.append(CheckSpec.model_validate(item))
                except ValidationError as exc:
                    raise DatasetExportError(
                        f"invalid check for {node!r}: {exc}", path=str(path)
                    ) from exc
            entries.append(
                _CheckEntry(
                    run_id=_optional_str(raw.get("run_id")),
                    node=node,
                    checks=tuple(checks),
                    suite_dir=Path(path).resolve().parent,
                )
            )
    return entries


def _load_items(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetExportError(f"file not found: {path}", path=str(path))
    try:
        if path.suffix == ".jsonl":
            raw: Any = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            raw = load_document(path)
    except (json.JSONDecodeError, ModelGraphError) as exc:
        raise DatasetExportError(f"cannot read {path}: {exc}", path=str(path)) from exc
    if isinstance(raw, dict):
        raw = raw.get(key)
    if not isinstance(raw, list):
        raise DatasetExportError(f"{path} must contain a '{key}' list", path=str(path))
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise DatasetExportError(
                f"{path}: {key} entry {index} must be a mapping", path=str(path)
            )
        result.append(cast(dict[str, Any], item))
    return result


def _group_overrides(labels: Sequence[_LabelEntry]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for entry in labels:
        if entry.group and entry.node in (None, "*") and entry.kind in (None, "*") and entry.run_id:
            overrides[entry.run_id] = entry.group
    return overrides


# --------------------------------------------------------------------------------------------
# Contamination, dedupe text, and splits
# --------------------------------------------------------------------------------------------


def _check_contamination(
    examples: list[dict[str, Any]],
    suites: Sequence[str | Path],
    near_distance: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not suites:
        return (
            {
                "status": "not-checked",
                "checked_against": [],
                "excluded": 0,
                "threshold": near_distance,
                "matches": [],
                "notes": "no eval suite was supplied; contamination is not checked",
            },
            examples,
        )
    loaded = []
    for path in suites:
        try:
            suite = load_suite(path)
        except ModelGraphError as exc:
            raise DatasetExportError(
                f"cannot load eval suite {path}: {exc}", path=str(path)
            ) from exc
        loaded.append(suite)
    kept: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    for example in examples:
        task = str(example["input"].get("task") or "")
        normalized = normalize_text(task)
        task_hash = simhash(task)
        match: dict[str, Any] | None = None
        for suite in loaded:
            for case in suite.cases:
                candidate = normalize_text(case.input.task)
                if normalized == candidate:
                    method = "exact"
                elif hamming_distance(task_hash, simhash(case.input.task)) <= near_distance:
                    method = "near-duplicate"
                else:
                    continue
                match = {
                    "example": f"{example['source']['run_id']}:{example['source']['event_index']}",
                    "suite": f"{suite.metadata.id}@{suite.metadata.version}",
                    "suite_hash": suite.content_hash,
                    "case_id": case.id,
                    "method": method,
                }
                break
            if match is not None:
                break
        if match is None:
            kept.append(example)
        else:
            matches.append(match)
    checked = [
        {
            "id": suite.metadata.id,
            "version": suite.metadata.version,
            "hash": suite.content_hash,
        }
        for suite in loaded
    ]
    return (
        {
            "status": "checked",
            "checked_against": checked,
            "excluded": len(matches),
            "threshold": near_distance,
            "matches": matches,
            "notes": (
                "examples matching an eval case task were excluded"
                if matches
                else "no eval-case task matched an exported example"
            ),
        },
        kept,
    )


def _dedupe_text(example: dict[str, Any]) -> str:
    kind = str(example["kind"])
    node = str(example["source"]["node"])
    task = str(example["input"].get("task") or "")
    messages = example["input"].get("messages") or []
    text = "\n".join(str(message.get("content") or "") for message in messages)
    if kind == "routing":
        menu = example["input"].get("menu") or {}
        text = text + "\n" + str(menu.get("text") or "")
    return f"{kind}\0{node}\0{task}\n{text}"


def _parse_ratios(value: str) -> dict[str, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise DatasetExportError("split ratio must be TRAIN,VALIDATION,TEST")
    try:
        ratios = dict(zip(SPLIT_NAMES, (float(part) for part in parts), strict=True))
    except ValueError as exc:
        raise DatasetExportError(f"invalid split ratio {value!r}") from exc
    if any(ratio < 0 for ratio in ratios.values()):
        raise DatasetExportError("split ratios must be non-negative")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise DatasetExportError(f"split ratios must sum to 1.0 (got {total:.6f})")
    return ratios


def _assign_splits(
    groups: Sequence[str],
    *,
    plan_path: Path | None,
    seed: str,
    ratios: dict[str, float],
) -> tuple[dict[str, str], dict[str, Any]]:
    planned: dict[str, str] = {}
    if plan_path is not None and plan_path.is_file():
        try:
            document = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetExportError(
                f"invalid split plan {plan_path}: {exc}", path=str(plan_path)
            ) from exc
        raw_groups = document.get("groups") if isinstance(document, dict) else None
        if not isinstance(raw_groups, dict):
            raise DatasetExportError(
                "split plan must contain a 'groups' mapping", path=str(plan_path)
            )
        for group, value in raw_groups.items():
            split = value.get("split") if isinstance(value, dict) else value
            if split not in SPLIT_NAMES:
                raise DatasetExportError(
                    f"split plan group {group!r} has invalid split {split!r}",
                    path=str(plan_path),
                )
            planned[str(group)] = str(split)
    assignments: dict[str, str] = {}
    for group in sorted(set(groups) | set(planned)):
        if group in planned:
            assignments[group] = planned[group]
            continue
        ratio = int.from_bytes(
            hashlib.sha256(f"{seed}\0{group}".encode()).digest()[:8], "big"
        ) / float(2**64)
        if ratio < ratios["train"]:
            assignments[group] = "train"
        elif ratio < ratios["train"] + ratios["validation"]:
            assignments[group] = "validation"
        else:
            assignments[group] = "test"
    plan = {
        "schema": SPLITS_SCHEMA,
        "seed": seed,
        "ratios": ratios,
        "frozen_test": True,
        "groups": {group: {"split": assignments[group]} for group in sorted(assignments)},
    }
    plan_hash = sha256_text(
        canonical_json(
            {
                "seed": seed,
                "ratios": ratios,
                "groups": {group: assignments[group] for group in sorted(assignments)},
            }
        )
    )
    plan["plan_hash"] = plan_hash
    return assignments, plan


def _split_document(
    plan: dict[str, Any],
    assignments: dict[str, str],
    examples: Sequence[dict[str, Any]],
    ratios: dict[str, float],
    hashes: dict[str, str],
) -> dict[str, Any]:
    counts = {split: 0 for split in SPLIT_NAMES}
    groups: dict[str, dict[str, Any]] = {}
    for example in examples:
        group = str(example["group"])
        split = assignments[group]
        counts[split] += 1
        entry = groups.setdefault(group, {"split": split, "examples": 0})
        entry["examples"] = int(entry["examples"]) + 1
    document = dict(plan)
    document["schema"] = SPLITS_SCHEMA
    document["ratios"] = ratios
    document["counts"] = counts
    document["hashes"] = hashes
    document["groups"] = {
        group: {
            "split": assignments[group],
            "examples": int(groups.get(group, {}).get("examples", 0)),
        }
        for group in sorted(set(assignments) | set(groups))
    }
    document["frozen_test"] = True
    return document


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------


def _build_manifest(
    *,
    dataset_id: str,
    dataset_name: str,
    dataset_version: str,
    dataset_description: str,
    dataset_task: str,
    license_obj: dict[str, Any],
    license_uri: str | None,
    consent_record: dict[str, Any],
    sources: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    counts: dict[str, int],
    examples_text: str,
    split_hashes: dict[str, str],
    split_ratios: dict[str, float],
    split_seed: str,
    split_plan: dict[str, Any],
    group_by: str,
    group_overrides: int,
    dedupe: str,
    near_duplicate_distance: int,
    dedupe_result: Any,
    contamination: dict[str, Any],
    redaction: dict[str, Any],
    file_hashes: dict[str, str],
    skipped_runs: list[dict[str, str]],
    limitations: list[str],
    command: str,
    created_at: str,
) -> dict[str, Any]:
    split_counts = {split: int(counts.get(split, 0)) for split in SPLIT_NAMES}
    by_kind = _count_by(examples, lambda example: str(example["kind"]))
    by_node = _count_by(examples, lambda example: str(example["source"]["node"]))
    by_route = _count_by(
        examples,
        lambda example: (
            str(example["output"].get("chosen_route") or "unknown")
            if example["kind"] == "routing"
            else None
        ),
    )
    by_outcome = _count_by(examples, lambda example: str(example["label"]["outcome"]))
    split_groups = {
        str(group): {
            "split": str(value.get("split")),
            "examples": int(value.get("examples", 0)),
        }
        for group, value in (split_plan.get("groups") or {}).items()
    }
    source_runs = [str(source["run_id"]) for source in sources]
    filters = [
        "run_outcome:any"
        if skipped_runs or any(s["run_outcome"] != "succeeded" for s in sources)
        else "run_outcome:succeeded",
        "consent:required" if consent_record["required"] else "consent:optional",
        f"redaction:{redaction['status']}",
        f"dedupe:{dedupe}",
    ]
    manifest = {
        "apiVersion": DATASET_API_VERSION,
        "kind": DATASET_KIND,
        "metadata": {
            "id": dataset_id,
            "name": dataset_name,
            "version": dataset_version,
            "description": dataset_description,
            "labels": {"format": "jsonl", "producer": "mg data export"},
            "license": license_obj["id"],
            "created_at": created_at,
        },
        "spec": {
            "task": dataset_task,
            "format": "jsonl",
            "record_schema": RECORD_SCHEMA,
            "sources": sources,
            "preprocessing": {
                "dedupe": {
                    "mode": dedupe,
                    "near_duplicate_distance": near_duplicate_distance,
                    "exact_removed": int(dedupe_result.exact_removed),
                    "near_removed": int(dedupe_result.near_removed),
                },
                "filters": filters,
                "split_strategy": "group_hash",
                "group_by": group_by,
                "group_overrides": group_overrides,
            },
            "counts": {
                **split_counts,
                "total": len(examples),
                "by_kind": by_kind,
                "by_node": by_node,
                "by_route": by_route,
                "by_outcome": by_outcome,
            },
            "hash": sha256_text(examples_text),
            "files": file_hashes,
            "eval_contamination": contamination,
            "consent": consent_record,
            "redaction": {
                "enabled": redaction["enabled"],
                "status": redaction["status"],
                "action": redaction["action"],
                "report": REDACTION_FILE,
                "patterns": [pattern["name"] for pattern in redaction["patterns"]],
                "total_redactions": int(redaction["total_redactions"]),
                "clean": bool(redaction["clean"]),
            },
            "splits": {
                "strategy": "group_hash",
                "group_by": group_by,
                "seed": split_seed,
                "ratios": split_ratios,
                "counts": split_counts,
                "frozen_test": True,
                "plan": SPLITS_FILE,
                "plan_hash": str(split_plan.get("plan_hash") or ""),
                "groups": split_groups,
                "hashes": split_hashes,
            },
            "examples": {
                "node": int(by_kind.get("node", 0)),
                "routing": int(by_kind.get("routing", 0)),
                "total": len(examples),
                "by_node": by_node,
                "by_route": by_route,
            },
            "license": {
                "id": license_obj["id"],
                "uri": license_uri,
                "distributable": bool(license_obj["distributable"]),
            },
            "provenance": {
                "tool": "mg data export",
                "version": __version__,
                "command": command,
                "source_runs": source_runs,
            },
            "skipped_runs": skipped_runs,
            "limitations": limitations,
        },
    }
    return manifest


def _counts(examples: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {split: 0 for split in SPLIT_NAMES}
    for example in examples:
        split = str(example.get("split") or "train")
        counts[split] = counts.get(split, 0) + 1
    counts["total"] = len(examples)
    return counts


def _count_by(examples: Sequence[dict[str, Any]], key: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        value = key(example)
        if value is None:
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return {name: counts[name] for name in sorted(counts)}


def _refresh_artifact_hashes(example: dict[str, Any]) -> None:
    """Keep artifact hashes consistent with redacted content, retaining the source hash."""
    output = example.get("output")
    if not isinstance(output, dict):
        return
    artifacts = output.get("artifacts")
    if not isinstance(artifacts, list):
        return
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("content"), str):
            continue
        content = str(artifact["content"])
        new_hash = sha256_text(content)
        if artifact.get("hash") != new_hash:
            artifact["source_hash"] = artifact.get("hash")
            artifact["hash"] = new_hash
            artifact["bytes"] = len(content.encode("utf-8"))


def _ensure_unique_ids(examples: Sequence[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for example in examples:
        value = str(example["example_id"])
        if value in seen:
            raise DatasetExportError(f"duplicate example id after dedupe: {value}")
        seen.add(value)


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def _graph_license(run: _RunData) -> str | None:
    source = run.graph.get("source")
    if not source:
        return None
    path = Path(str(source))
    if not path.is_file():
        return None
    try:
        document = load_document(path)
    except ModelGraphError:
        return None
    if not isinstance(document, dict):
        return None
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("license")
    if not isinstance(value, str) or not value.strip():
        return None
    if value.strip().lower() in {"unknown", "tbd", "none"}:
        return None
    return value.strip()


def _dataset_license(license_id: str) -> dict[str, Any]:
    cleaned = license_id.strip()
    lower = cleaned.lower()
    distributable = lower not in {"", "unknown", "tbd", "none", "proprietary"}
    return {
        "id": cleaned,
        "uri": None,
        "distributable": distributable,
    }


def _default_task(runs: Sequence[_RunData]) -> str:
    graphs = sorted({str(run.graph.get("id") or "unknown") for run in runs})
    return "Trace-derived examples from " + ", ".join(graphs)


def _group_id(
    run: _RunData,
    group_by: str,
    overrides: dict[str, str],
    license_id: str,
) -> str:
    override = overrides.get(run.run_id)
    if override:
        return "group:" + re.sub(r"[^A-Za-z0-9._:-]+", "-", override).strip("-")
    if group_by == "run":
        return "run:" + re.sub(r"[^A-Za-z0-9._:-]+", "-", run.run_id).strip("-")
    if group_by == "graph":
        value = f"{run.graph.get('id') or 'unknown'}@{run.graph.get('version') or 'unknown'}"
        return "graph:" + re.sub(r"[^A-Za-z0-9._:@-]+", "-", value).strip("-")
    if group_by == "license":
        return "license:" + re.sub(r"[^A-Za-z0-9._:-]+", "-", license_id).strip("-")
    source = str(run.graph.get("source") or run.graph.get("id") or run.run_id)
    return "source:" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _limitations(
    *,
    include_failures: bool,
    skipped_runs: list[dict[str, str]],
    prompt_misses: int,
    contamination: dict[str, Any],
    redaction: dict[str, Any],
    license_id: str,
    labels: Sequence[_LabelEntry],
    dedupe: str,
) -> list[str]:
    limitations: list[str] = []
    if include_failures:
        limitations.append(
            "Failure traces were included explicitly; every example from a non-succeeded run "
            "is labeled failure with source=run_outcome."
        )
    if skipped_runs:
        limitations.append(
            f"{len(skipped_runs)} non-succeeded run(s) were skipped because --include-failures "
            "was not set."
        )
    if not labels:
        limitations.append(
            "No label file was supplied: node outcomes default to unknown and routing "
            "outcomes are unknown without an oracle/human route."
        )
    if prompt_misses:
        limitations.append(
            f"{prompt_misses} example(s) have rendered=false; their prompt_hash is recorded but "
            "the exact rendered prompt could not be reconstructed."
        )
    if contamination.get("status") != "checked":
        limitations.append(
            "Contamination against an eval suite was not checked; pass --eval-suite to check."
        )
    if redaction.get("status") == "disabled":
        limitations.append("Redaction was disabled explicitly; inspect the data before sharing it.")
    elif redaction.get("status") == "clean":
        limitations.append(
            "The configured redaction patterns found no matches; this is not proof that the "
            "data contains no secrets or PII."
        )
    if not _dataset_license(license_id)["distributable"]:
        limitations.append(
            f"License {license_id!r} is not known to be distributable; treat the dataset as "
            "local-only until the license is resolved."
        )
    if dedupe == "none":
        limitations.append(
            "Deduplication was disabled; near-duplicates may inflate evaluation scores."
        )
    return limitations


def _export_command(
    runs: Sequence[str | Path],
    include_failures: bool,
    redact: str | Path,
) -> str:
    parts = ["mg", "data", "export"]
    for run in runs:
        parts.extend(["--run", str(run)])
    parts.extend(["--redact", str(redact)])
    if include_failures:
        parts.append("--include-failures")
    return " ".join(parts)


def _jsonl(examples: Sequence[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(example, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for example in examples
    )


def _bytes_hash(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise DatasetExportError("ambiguous must be a boolean")
