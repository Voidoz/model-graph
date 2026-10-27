"""Specialist training kit: dataset ingestion, deterministic rehearsal, manifest, card.

The real 8GB/24GB LoRA/QLoRA recipes are documented in ``training/configs/`` and
``training/sft_lora.py``. They are **unverified in this repository's development
environment**, which has no torch/transformers/PEFT, CUDA driver, or GPU.

What runs here is a deterministic CPU rehearsal. It reads a real T-0023 dataset export,
applies the same filtering/provenance rules, and emits the complete artifact set (model
artifact, model manifest, model card, eval report, reproducibility record) without taking a
real optimizer step. The artifact is marked ``synthetic: true`` and ``quality_claim: false``,
and it is never promotable. That distinction is deliberate: the kit proves the pipeline
shape, not model quality.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modelgraph.contracts import load_document, validate_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.data import load_dataset, validate_dataset_example
from modelgraph.errors import ModelGraphError
from modelgraph.evaluation import EvalReport, run_e2e_suite, run_node_suite
from modelgraph.paths import project_root
from modelgraph.registry import Registry
from modelgraph.registry_store import (
    REPORT_SCHEMA,
    file_hash,
    verify_eval_report,
)
from modelgraph.training.errors import (
    SpecialistArtifactError,
    SpecialistDataError,
    SpecialistPromotionError,
    SpecialistTrainingError,
)
from modelgraph.util import canonical_json, sha256_text, utc_now

CONFIG_SCHEMA = "modelgraph/specialist-training-config/v0"
ARTIFACT_SCHEMA = "modelgraph/specialist-artifact/v0"
ARTIFACT_VERSION = "0.1.0"
SPLIT_NAMES = ("train", "validation", "test")
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "specialist-artifact.schema.json"


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


DEFAULT_CAPABILITIES: list[Literal["chat", "json_mode", "streaming", "seeded", "grammar"]] = [
    "chat"
]


class SpecialistResources(_Config):
    vram_gb: float | None = Field(default=None, ge=0)
    ram_gb: float | None = Field(default=None, ge=0)
    disk_gb: float | None = Field(default=None, ge=0)
    cpu_only: bool = True
    network_access: Literal["none", "loopback", "internet"] = "none"


class SpecialistModel(_Config):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(
        pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?$"
    )
    description: str = Field(min_length=10, max_length=1000)
    base: str = Field(min_length=1)
    base_uri: str | None = None
    base_license: str = Field(min_length=1)
    provider: Literal[
        "mock", "openai-compatible", "ollama", "llamacpp", "custom", "trained-router"
    ] = "mock"
    model: str | None = Field(default=None, max_length=300)
    context_window: int = Field(default=32768, ge=128, le=10_000_000)
    max_output_tokens: int = Field(default=2048, ge=1)
    capabilities: list[Literal["chat", "json_mode", "streaming", "seeded", "grammar"]] = Field(
        default_factory=lambda: list(DEFAULT_CAPABILITIES)
    )
    authors: list[str] = Field(default_factory=lambda: ["Model Graph contributors"])
    labels: dict[str, str] = Field(default_factory=dict)
    resources: SpecialistResources = Field(default_factory=SpecialistResources)


class Hyperparameters(_Config):
    rank: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0, le=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    epochs: float = Field(default=3, gt=0)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    max_seq_length: int = Field(default=2048, ge=128)
    quantization: Literal["none", "4bit", "8bit"] = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    optimizer: str = "paged_adamw_8bit"
    scheduler: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    gradient_checkpointing: bool = True
    notes: str | None = None


class Hardware(_Config):
    tier: Literal["cpu", "8gb", "24gb"]
    gpu: str
    vram_gb: float = Field(ge=0)
    cpu_only: bool = False
    notes: str | None = None
    expected_data: str | None = None
    expected_runtime: str | None = None
    cuda: str | None = None
    driver: str | None = None


class Distillation(_Config):
    teacher: str = Field(min_length=1)
    teacher_license: str = Field(min_length=1)
    teacher_uri: str | None = None
    filter: Literal["recorded", "checks", "judge", "human"] = "recorded"
    min_bytes: int = Field(default=40, ge=0)
    include_failures: bool = False
    notes: str | None = None


class EvaluationHook(_Config):
    node_suite: str = "evals/mock/js-specialist"
    node_graph: str = "examples/coding-specialist.graph.yaml"
    node: str = "js-specialist"
    graph_suite: str = "evals/e2e/coding-specialist"
    graph_graph: str = "examples/coding-specialist.graph.yaml"
    provider: Literal["mock"] = "mock"
    reports_dir: str | None = None


class PromotionGate(_Config):
    require_passing_eval: bool = True


class SpecialistConfig(_Config):
    schema_version: Literal["modelgraph/specialist-training-config/v0"] = Field(
        default="modelgraph/specialist-training-config/v0", alias="schema"
    )
    name: str = Field(min_length=1, max_length=120)
    method: Literal["lora", "qlora", "distill"]
    mode: Literal["synthetic", "real"]
    dataset: str
    node: str | None = None
    output_dir: str
    seed: int = 20261025
    created_at: str = "2026-10-25T00:00:00Z"
    image_digest: str | None = None
    model: SpecialistModel
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    hardware: Hardware
    distillation: Distillation | None = None
    evaluation: EvaluationHook = Field(default_factory=EvaluationHook)
    promotion: PromotionGate = Field(default_factory=PromotionGate)
    notes: str | None = None
    source_path: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_method(self) -> SpecialistConfig:
        if self.method == "distill" and self.distillation is None:
            raise ValueError("method: distill requires a distillation block")
        if self.method != "distill" and self.distillation is not None:
            raise ValueError("distillation is only valid with method: distill")
        if self.method == "qlora" and self.hyperparameters.quantization != "4bit":
            raise ValueError("method: qlora requires hyperparameters.quantization: 4bit")
        if self.method == "lora" and self.hyperparameters.quantization == "4bit":
            raise ValueError("4bit quantization is QLoRA; use method: qlora")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json", by_alias=True, exclude_none=True, exclude={"source_path"}
        )

    def config_hash(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))


def load_specialist_config(path: str | Path) -> SpecialistConfig:
    source = Path(path)
    try:
        document = load_document(source)
    except ModelGraphError as exc:
        raise SpecialistTrainingError(
            f"cannot load specialist config {source}: {exc.message}", path=str(source)
        ) from exc
    try:
        config = SpecialistConfig.model_validate(document)
    except Exception as exc:
        raise SpecialistTrainingError(
            f"invalid specialist config {source}: {exc}", path=str(source)
        ) from exc
    config.source_path = str(source)
    return config


@cache
def specialist_artifact_schema() -> dict[str, Any]:
    try:
        document = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise SpecialistArtifactError("specialist artifact schema must be an object")
        return document
    except OSError as exc:  # pragma: no cover - packaged file is present
        raise SpecialistArtifactError(
            f"cannot read specialist artifact schema {SCHEMA_PATH}: {exc}", path=str(SCHEMA_PATH)
        ) from exc


@cache
def _artifact_validator() -> Draft202012Validator:
    schema = specialist_artifact_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_specialist_artifact(document: dict[str, Any]) -> list[str]:
    errors = [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(_artifact_validator().iter_errors(document), key=lambda e: list(e.path))
    ]
    if errors:
        return errors
    if document.get("synthetic") is True and document.get("quality_claim") is not False:
        errors.append("a synthetic rehearsal must set quality_claim: false")
    if document.get("synthetic") is False and document.get("kind") != "real-specialist-adapter":
        errors.append("a non-synthetic artifact must be a real-specialist-adapter")
    return errors


# ---------------------------------------------------------------------------------------
# Paths and small helpers
# ---------------------------------------------------------------------------------------


def resolve_path(value: str | Path, *, root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = (root or project_root()) / path
    if candidate.exists():
        return candidate.resolve()
    return path.resolve()


def _target_text(record: dict[str, Any]) -> str:
    output = record.get("output") or {}
    artifacts = output.get("artifacts") or []
    for artifact in artifacts:
        if artifact.get("name") == "answer" and isinstance(artifact.get("content"), str):
            return str(artifact["content"])
    for artifact in artifacts:
        content = artifact.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _messages(record: dict[str, Any]) -> tuple[dict[str, str], ...]:
    messages = ((record.get("input") or {}).get("messages")) or []
    result: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(role, str) and isinstance(content, str):
            result.append({"role": role, "content": content})
    return tuple(result)


def _example_license(record: dict[str, Any]) -> tuple[str, str | None, bool]:
    license_info = record.get("license") or {}
    return (
        str(license_info.get("id") or "unknown"),
        license_info.get("uri"),
        bool(license_info.get("distributable", False)),
    )


def _example_consent(record: dict[str, Any]) -> dict[str, Any]:
    return dict(record.get("consent") or {})


def _consent_recorded(consent: dict[str, Any]) -> bool:
    if not consent.get("required", False):
        return True
    return bool(consent.get("recorded", False))


# ---------------------------------------------------------------------------------------
# Dataset ingestion
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecialistExample:
    """One node-SFT example derived from a T-0023 record."""

    record: dict[str, Any]
    target_text: str

    @property
    def example_id(self) -> str:
        return str(self.record.get("example_id") or "")

    @property
    def split(self) -> str:
        return str(self.record.get("split") or "")

    @property
    def group(self) -> str:
        return str(self.record.get("group") or "")

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        return _messages(self.record)

    @property
    def source_node(self) -> str | None:
        return (self.record.get("source") or {}).get("node")

    @property
    def label(self) -> dict[str, Any]:
        return dict(self.record.get("label") or {})

    @property
    def license_id(self) -> str:
        return _example_license(self.record)[0]

    @property
    def consent(self) -> dict[str, Any]:
        return _example_consent(self.record)

    @property
    def example_hash(self) -> str:
        return str((self.record.get("hashes") or {}).get("example") or self.example_id)


@dataclass
class SpecialistDataset:
    """A filtered, provenance-preserving view of a T-0023 dataset export."""

    path: Path
    manifest: dict[str, Any] | None
    examples: list[SpecialistExample]
    node: str | None
    dataset_id: str
    dataset_version: str
    dataset_hash: str
    license_id: str
    license_uri: str | None
    distributable: bool
    licenses: list[str]
    consent: dict[str, Any]
    counts: dict[str, int]
    label_outcomes: dict[str, int]
    label_sources: dict[str, int]
    excluded: dict[str, int]
    sources: list[dict[str, Any]]
    contamination: dict[str, Any]
    redaction: dict[str, Any]
    filter_report: dict[str, Any] = field(default_factory=dict)

    @property
    def digest_material(self) -> bytes:
        hashes = sorted(example.example_hash for example in self.examples)
        return canonical_json(
            {"dataset": self.dataset_hash, "examples": hashes, "node": self.node}
        ).encode("utf-8")

    def split(self, name: str) -> list[SpecialistExample]:
        return [example for example in self.examples if example.split == name]

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.dataset_id,
            "version": self.dataset_version,
            "hash": self.dataset_hash,
            "path": str(self.path),
            "node": self.node,
            "license": self.license_id,
            "license_uri": self.license_uri,
            "distributable": self.distributable,
            "licenses": self.licenses,
            "consent": self.consent,
            "counts": self.counts,
            "label_outcomes": self.label_outcomes,
            "label_sources": self.label_sources,
            "excluded": self.excluded,
            "sources": self.sources,
            "contamination": self.contamination,
            "redaction": self.redaction,
            "filter": self.filter_report,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SpecialistDataError(f"cannot read dataset {path}: {exc}", path=str(path)) from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpecialistDataError(
                f"invalid JSONL at {path}:{index}: {exc}", path=str(path)
            ) from exc
        if not isinstance(record, dict):
            raise SpecialistDataError(f"dataset line {index} is not an object", path=str(path))
        records.append(record)
    return records


def _apply_distill_filter(
    examples: list[SpecialistExample], distillation: Distillation, *, seed: int
) -> tuple[list[SpecialistExample], dict[str, Any]]:
    kept: list[SpecialistExample] = []
    dropped = Counter({"failure_label": 0, "min_bytes": 0, "duplicate_output": 0})
    seen: set[str] = set()
    for example in examples:
        if not distillation.include_failures and example.label.get("outcome") == "failure":
            dropped["failure_label"] += 1
            continue
        if len(example.target_text.encode("utf-8")) < distillation.min_bytes:
            dropped["min_bytes"] += 1
            continue
        digest = sha256_text(example.target_text.strip())
        if digest in seen:
            dropped["duplicate_output"] += 1
            continue
        seen.add(digest)
        kept.append(example)
    report = {
        "teacher": distillation.teacher,
        "teacher_license": distillation.teacher_license,
        "teacher_uri": distillation.teacher_uri,
        "filter": distillation.filter,
        "min_bytes": distillation.min_bytes,
        "include_failures": distillation.include_failures,
        "input": len(examples),
        "kept": len(kept),
        "dropped": dict(sorted(dropped.items())),
        "seed": seed,
        "note": (
            "Offline rehearsal: teacher samples are the recorded node outputs in the T-0023 "
            "dataset. No teacher model was called, and no teacher quality is claimed."
        ),
    }
    return kept, report


def load_specialist_dataset(
    source: str | Path,
    *,
    node: str | None = None,
    distillation: Distillation | None = None,
    seed: int = 20261025,
    validate: bool = True,
) -> SpecialistDataset:
    """Load a T-0023 export (directory with manifest + JSONL, or a bare JSONL file)."""
    path = resolve_path(source)
    manifest: dict[str, Any] | None = None
    raw: list[dict[str, Any]]
    if path.is_dir():
        if (path / "dataset.json").is_file():
            bundle = load_dataset(path, validate=validate)
            manifest = bundle.manifest
            raw = bundle.examples
            spec = manifest.get("spec") or {}
            expected = (spec.get("files") or {}).get("examples.jsonl")
            actual = file_hash(path / "examples.jsonl")
            if expected and expected != actual:
                raise SpecialistDataError(
                    f"dataset examples hash mismatch: {actual} != {expected}", path=str(path)
                )
        elif (path / "examples.jsonl").is_file():
            raw = _read_jsonl(path / "examples.jsonl")
        else:
            raise SpecialistDataError(
                f"no dataset.json or examples.jsonl in {path}", path=str(path)
            )
    elif path.is_file():
        raw = _read_jsonl(path)
    else:
        raise SpecialistDataError(f"dataset not found: {path}", path=str(path))
    for index, record in enumerate(raw, start=1):
        issues = validate_dataset_example(record)
        if issues:
            raise SpecialistDataError(
                f"dataset example {index} is invalid: " + "; ".join(issues[:3]), path=str(path)
            )
    raw = sorted(raw, key=lambda record: str(record.get("example_id") or ""))
    spec = (manifest or {}).get("spec") or {}
    metadata = (manifest or {}).get("metadata") or {}
    license_ids = sorted({_example_license(record)[0] for record in raw})
    dataset_license = str((spec.get("license") or {}).get("id") or "")
    if not dataset_license:
        dataset_license = license_ids[0] if len(license_ids) == 1 else "unknown"
    dataset_license_uri = (spec.get("license") or {}).get("uri")
    distributable = bool((spec.get("license") or {}).get("distributable", False))
    consent = dict(spec.get("consent") or {})
    if not consent:
        consents = [_example_consent(record) for record in raw]
        consent = (
            consents[0]
            if consents
            else {
                "required": False,
                "recorded": False,
                "statement": None,
                "scope": None,
            }
        )
    if consent.get("required") and not consent.get("recorded"):
        raise SpecialistDataError(
            "dataset consent is required but not recorded; refusing to train", path=str(path)
        )
    dataset_hash = str(spec.get("hash") or "")
    if not dataset_hash:
        dataset_hash = file_hash(path) if path.is_file() else file_hash(path / "examples.jsonl")
    dataset_id = str(metadata.get("id") or path.parent.name or "unknown")
    dataset_version = str(metadata.get("version") or "0.0.0")
    excluded = Counter(
        {"wrong_kind": 0, "other_node": 0, "unrendered": 0, "no_output": 0, "duplicate": 0}
    )
    selected: list[SpecialistExample] = []
    seen: set[str] = set()
    available_nodes = Counter(
        str((record.get("source") or {}).get("node") or "unknown")
        for record in raw
        if record.get("kind") == "node"
    )
    for record in raw:
        if record.get("kind") != "node":
            excluded["wrong_kind"] += 1
            continue
        source_node = (record.get("source") or {}).get("node")
        if node is not None and source_node != node:
            excluded["other_node"] += 1
            continue
        messages = _messages(record)
        if not messages:
            excluded["unrendered"] += 1
            continue
        text = _target_text(record)
        if not text.strip():
            excluded["no_output"] += 1
            continue
        identity = str((record.get("hashes") or {}).get("example") or record.get("example_id"))
        if identity in seen:
            excluded["duplicate"] += 1
            continue
        seen.add(identity)
        if not _consent_recorded(_example_consent(record)):
            raise SpecialistDataError(
                f"example {record.get('example_id')!r} consent is not recorded", path=str(path)
            )
        selected.append(SpecialistExample(record=record, target_text=text))
    if not selected:
        hint = ""
        if node is not None:
            hint = f"; available node(s): {dict(sorted(available_nodes.items()))}"
        raise SpecialistDataError(
            f"dataset {path} has no usable node SFT examples for node {node!r}{hint}",
            path=str(path),
        )
    filter_report: dict[str, Any] = {}
    if distillation is not None:
        selected, filter_report = _apply_distill_filter(selected, distillation, seed=seed)
        for key, value in (filter_report.get("dropped") or {}).items():
            excluded[f"distill_{key}"] = int(value)
        if not selected:
            raise SpecialistDataError(
                "all distillation samples were filtered out; refusing to train", path=str(path)
            )
    counts = {
        split: sum(1 for example in selected if example.split == split) for split in SPLIT_NAMES
    }
    if counts["train"] == 0:
        raise SpecialistDataError("dataset has no train-split examples", path=str(path))
    outcomes = Counter(str(example.label.get("outcome") or "unknown") for example in selected)
    label_sources = Counter(str(example.label.get("source") or "unknown") for example in selected)
    sources: dict[str, dict[str, Any]] = {}
    for example in selected:
        source_ref = example.record.get("source") or {}
        run_id = str(source_ref.get("run_id") or "unknown")
        sources.setdefault(
            run_id,
            {
                "run_id": run_id,
                "node": source_ref.get("node"),
                "graph": source_ref.get("graph"),
                "trace": source_ref.get("trace"),
                "license": example.license_id,
                "manifest_hash": source_ref.get("manifest_hash"),
            },
        )
    return SpecialistDataset(
        path=path,
        manifest=manifest,
        examples=selected,
        node=node,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        dataset_hash=dataset_hash,
        license_id=dataset_license,
        license_uri=str(dataset_license_uri) if dataset_license_uri else None,
        distributable=distributable,
        licenses=license_ids,
        consent=consent,
        counts=counts,
        label_outcomes=dict(sorted(outcomes.items())),
        label_sources=dict(sorted(label_sources.items())),
        excluded=dict(sorted(excluded.items())),
        sources=[sources[key] for key in sorted(sources)],
        contamination=dict(spec.get("eval_contamination") or {}),
        redaction=dict(spec.get("redaction") or {}),
        filter_report=filter_report,
    )


# ---------------------------------------------------------------------------------------
# Deterministic synthetic trainer
# ---------------------------------------------------------------------------------------


def _pseudo_values(name: str, dataset: SpecialistDataset, seed: int, count: int) -> list[float]:
    material = f"{seed}:{name}:".encode() + dataset.digest_material
    values: list[float] = []
    counter = 0
    while len(values) < count:
        digest = hashlib.sha256(material + counter.to_bytes(4, "big")).digest()
        for offset in range(0, 32, 4):
            if len(values) >= count:
                break
            raw = int.from_bytes(digest[offset : offset + 4], "big")
            values.append(round((raw / 0xFFFFFFFF) * 2.0 - 1.0, 8))
        counter += 1
    return values


def build_synthetic_artifact(
    config: SpecialistConfig, dataset: SpecialistDataset
) -> dict[str, Any]:
    """Build the rehearsal artifact. No optimizer step, no real weights, no quality claim."""
    modules = sorted(set(config.hyperparameters.target_modules or ["q_proj", "v_proj"]))
    tensors = []
    for module in modules:
        values = _pseudo_values(
            f"{module}.lora_A", dataset, config.seed, config.hyperparameters.rank
        )
        tensors.append(
            {
                "name": f"{module}.lora_A",
                "shape": [1, config.hyperparameters.rank],
                "values": values,
            }
        )
    artifact: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "version": ARTIFACT_VERSION,
        "created_at": config.created_at,
        "kind": "synthetic-specialist-rehearsal",
        "synthetic": True,
        "quality_claim": False,
        "method": config.method,
        "node": dataset.node,
        "base_model": {
            "id": config.model.base,
            "uri": config.model.base_uri,
            "license": config.model.base_license,
        },
        "dataset": {
            "id": dataset.dataset_id,
            "version": dataset.dataset_version,
            "hash": dataset.dataset_hash,
            "license": dataset.license_id,
            "consent": dataset.consent,
            "counts": dataset.counts,
            "sources": dataset.sources,
        },
        "hyperparameters": config.hyperparameters.model_dump(mode="json", exclude_none=True),
        "seed": config.seed,
        "rehearsal": {
            "steps": 0,
            "examples": len(dataset.examples),
            "note": (
                "No optimizer step was taken. The pseudo-values are a deterministic function "
                "of the dataset digest and seed; they prove the artifact pipeline, not training."
            ),
        },
        "weights": {
            "format": "pseudo-sha256",
            "tensors": tensors,
            "note": "Not model weights. Never deploy, publish, or evaluate this as a trained model.",
        },
        "limitations": [
            "Synthetic CPU rehearsal: no real LoRA/QLoRA optimizer step and no usable adapter.",
            "The artifact is not promotable, regardless of any eval report.",
            "The mock eval hook checks the harness shape and the pinned graph, not model quality.",
        ],
    }
    if config.method == "distill" and config.distillation is not None:
        artifact["teacher"] = {
            "id": config.distillation.teacher,
            "license": config.distillation.teacher_license,
            "calls": 0,
            "filter": config.distillation.filter,
        }
    errors = validate_specialist_artifact(artifact)
    if errors:
        raise SpecialistArtifactError(
            "generated synthetic artifact is invalid: " + "; ".join(errors[:5])
        )
    return artifact


# ---------------------------------------------------------------------------------------
# Manifest, card, reproducibility
# ---------------------------------------------------------------------------------------


def _model_ref(config: SpecialistConfig) -> str:
    if config.model.model:
        return config.model.model
    return f"synthetic-specialist://{config.model.id}@{config.model.version}"


def build_model_manifest(
    config: SpecialistConfig,
    dataset: SpecialistDataset,
    *,
    artifact_path: Path,
    artifact_hash: str,
    eval_report_path: str | None,
) -> dict[str, Any]:
    license_notes = [
        f"Base model license: {config.model.base_license}.",
        f"Dataset license: {dataset.license_id}.",
        "License metadata is a declaration, not legal advice or a redistribution guarantee.",
    ]
    if config.method == "distill" and config.distillation is not None:
        license_notes.append(f"Teacher license: {config.distillation.teacher_license}.")
    labels = {
        "method": config.method,
        "mode": config.mode,
        "synthetic": "true" if config.mode == "synthetic" else "false",
        "base": config.model.base,
        **config.model.labels,
    }
    document: dict[str, Any] = {
        "apiVersion": "modelgraph/v0",
        "kind": "Model",
        "metadata": {
            "id": config.model.id,
            "name": config.model.name,
            "version": config.model.version,
            "description": config.model.description,
            "labels": labels,
            "authors": config.model.authors,
        },
        "spec": {
            "provider": config.model.provider,
            "model": _model_ref(config),
            "context_window": config.model.context_window,
            "max_output_tokens": config.model.max_output_tokens,
            "capabilities": config.model.capabilities,
            "limits": {
                "default_temperature": 0.1,
                "default_max_output_tokens": min(config.model.max_output_tokens, 2048),
            },
            "cost": {
                "pricing": "free",
                "notes": "Local artifact; no marginal provider cost in this rehearsal.",
            },
            "resources": config.model.resources.model_dump(mode="json", exclude_none=True),
            "license": {
                "name": config.model.base_license,
                "notes": " ".join(license_notes)[:500],
            },
            "integrity": {
                "sha256": artifact_hash.removeprefix("sha256:"),
                "revision": f"{config.method}-rehearsal-{dataset.dataset_hash.removeprefix('sha256:')[:12]}",
            },
            "eval": {
                "suite": config.evaluation.node_suite,
                **({"report": eval_report_path} if eval_report_path else {}),
                "notes": (
                    "Mock/offline harness gate. A passing report proves the pipeline and the "
                    "pinned graph, not this rehearsal artifact's quality."
                    if config.mode == "synthetic"
                    else "Real-recipe evaluation evidence; see the report for provider details."
                ),
            },
            "notes": (
                "Synthetic specialist rehearsal; do not deploy. "
                f"Artifact: {artifact_path.name}. " + (config.notes or "")
            )[:2000],
        },
    }
    issues = validate_document(document, "manifest")
    if issues:
        raise SpecialistArtifactError(
            "generated model manifest is invalid: "
            + "; ".join(issue.render() for issue in issues[:5])
        )
    ModelManifest.model_validate(document)
    return document


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_model_card(
    config: SpecialistConfig,
    dataset: SpecialistDataset,
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    eval_report: dict[str, Any] | None,
    reproducibility: dict[str, Any],
    *,
    status: str,
) -> str:
    base = config.model
    eval_text = "not run (candidate artifact)"
    if eval_report is not None:
        eval_text = "PASS" if eval_report.get("passed") else "FAIL"
        eval_text += " (mock/offline harness gate; not a model-quality result)"
    source_rows = [
        [
            source.get("run_id"),
            (source.get("graph") or {}).get("id"),
            source.get("license"),
            source.get("trace"),
        ]
        for source in dataset.sources
    ]
    label_rows = [[name, count] for name, count in dataset.label_outcomes.items()]
    hyper_rows = [
        [name, value]
        for name, value in config.hyperparameters.model_dump(mode="json", exclude_none=True).items()
        if not isinstance(value, list)
    ]
    contamination = dataset.contamination or {}
    consent = dataset.consent or {}
    limitations = [
        "This is a deterministic CPU rehearsal, not a trained model. The pseudo-weights are not "
        "usable for inference and the artifact is never promotable.",
        "The eval hook uses the mock provider and scripted responses. A passing gate proves the "
        "instrument, the pinned graph, and the report pipeline, not the model's quality.",
        "A real LoRA/QLoRA specialist trained on narrow data can regress off-distribution; test "
        "the target graph and a held-out domain before claiming a win.",
        "The 8GB and 24GB recipes in training/configs are unverified in this environment: there "
        "is no torch/transformers/PEFT stack, CUDA driver, or GPU here.",
    ]
    if config.method == "distill":
        limitations.append(
            "Distillation inherits teacher mistakes and teacher license terms; the synthetic "
            "rehearsal never calls a teacher and inherits no teacher quality."
        )
    lines = [
        f"# {base.name}",
        "",
        f"- Model: `{base.id}@{base.version}`",
        f"- Status: **{status}** (candidate unless a real, passing eval report is promoted)",
        f"- Method: `{config.method}`",
        f"- Mode: `{config.mode}`",
        (
            "> **Synthetic rehearsal.** This card describes a pipeline rehearsal, not a "
            "trained model or a quality result. The artifact contains deterministic "
            "pseudo-values and must not be deployed."
            if artifact.get("synthetic")
            else "> **Real-recipe artifact.** See the evaluation evidence and limitations."
        ),
        "",
        "## Base model",
        "",
        f"- Base: `{base.base}`",
        f"- URI: `{base.base_uri or 'not recorded'}`",
        f"- Base license: `{base.base_license}`",
        f"- Context window: `{base.context_window}` tokens",
        f"- Provider binding: `{base.provider}` / `{_model_ref(config)}`",
        "",
        "## Training data",
        "",
        f"- Dataset: `{dataset.dataset_id}@{dataset.dataset_version}`",
        f"- Dataset hash: `{dataset.dataset_hash}`",
        f"- Dataset license: `{dataset.license_id}`"
        + (f" (`{dataset.license_uri}`)" if dataset.license_uri else ""),
        f"- License list from examples: `{', '.join(dataset.licenses) or 'unknown'}`",
        f"- Source node filter: `{dataset.node or 'any node'}`",
        "- Splits: " + ", ".join(f"{name}={dataset.counts.get(name, 0)}" for name in SPLIT_NAMES),
        f"- Consent: required=`{consent.get('required')}`, recorded=`{consent.get('recorded')}`, "
        f"scope=`{consent.get('scope')}`",
        f"- Consent statement: {consent.get('statement') or 'not recorded'}",
        f"- Redaction: `{(dataset.redaction or {}).get('status', 'unknown')}`",
        f"- Contamination: `{contamination.get('status', 'not-checked')}`",
        "",
        "### Label sources",
        "",
        _markdown_table(["outcome", "examples"], label_rows or [["unknown", 0]]),
        "",
        "### Source runs",
        "",
        _markdown_table(["run", "graph", "license", "trace"], source_rows[:40])
        if source_rows
        else "No source runs recorded.",
        "",
        "## Method",
        "",
        f"- Method: `{config.method}`"
        + (" (offline distillation rehearsal)" if config.method == "distill" else ""),
        f"- Training mode: `{config.mode}`",
        "- Pipeline: T-0023 dataset -> provenance/consent/license filter -> deterministic "
        "rehearsal artifact -> model manifest -> mock eval hook -> card.",
    ]
    if config.method == "distill" and config.distillation is not None:
        lines.extend(
            [
                f"- Teacher: `{config.distillation.teacher}` "
                f"(`{config.distillation.teacher_license}`)",
                f"- Distillation filter: `{config.distillation.filter}` "
                f"(min_bytes={config.distillation.min_bytes}, "
                f"include_failures={config.distillation.include_failures})",
                "- Offline rehearsal: teacher samples are recorded dataset outputs; zero teacher "
                "calls were made.",
            ]
        )
    lines.extend(
        [
            "",
            "## Hyperparameters",
            "",
            _markdown_table(["parameter", "value"], hyper_rows),
            "",
            "## Evaluation",
            "",
            f"- Result: {eval_text}",
            f"- Node suite: `{config.evaluation.node_suite}`",
            f"- Graph regression suite: `{config.evaluation.graph_suite}`",
            f"- Pinned model: `{base.id}@{base.version}`",
            f"- Evidence: `{manifest['spec'].get('eval', {}).get('report') or 'none'}`",
            "- The model must be evaluated in a graph pinned to this exact id@version; the "
            "registry contribution format records the suite and graph hashes.",
            "",
            "## Intended use",
            "",
            "- Intended: recovering the pipeline and the review evidence for a specialist node.",
            "- Not intended: inference, deployment, quality benchmarks, or a claim that a node "
            "improved. A real adapter must be produced on real hardware and re-evaluated.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(
        [
            "",
            "## License",
            "",
            f"- Model/weights license declaration: `{base.base_license}`",
            f"- Dataset license: `{dataset.license_id}`",
            f"- Base model: `{base.base}` (`{base.base_license}`)",
            (
                f"- Teacher: `{config.distillation.teacher}` "
                f"(`{config.distillation.teacher_license}`)"
                if config.method == "distill" and config.distillation is not None
                else "- Teacher: n/a"
            ),
            "- License metadata is a declaration, not legal advice. A redistribution right is "
            "not established by this card; check the upstream terms yourself.",
            "",
            "## Reproducibility",
            "",
            f"- Command: `{reproducibility.get('command')}`",
            f"- Code commit: `{reproducibility.get('commit')}` "
            f"(dirty={reproducibility.get('dirty')})",
            f"- Config hash: `{reproducibility.get('config', {}).get('hash')}`",
            f"- Dataset hash: `{dataset.dataset_hash}`",
            f"- Artifact hash: `{reproducibility.get('artifact', {}).get('sha256')}`",
            f"- Seed: `{config.seed}`",
            f"- Hardware: `{config.hardware.tier}` / `{config.hardware.gpu}` "
            f"({config.hardware.vram_gb} GB VRAM)",
            f"- Image digest: `{config.image_digest or 'not recorded'}`",
            "",
        ]
    )
    return "\n".join(lines)


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def build_reproducibility(
    config: SpecialistConfig,
    dataset: SpecialistDataset,
    *,
    command: str,
    artifact_path: Path,
    artifact_hash: str,
    manifest_path: Path,
    card_path: Path,
    manifest_hash: str,
    card_hash: str,
) -> dict[str, Any]:
    root = project_root()
    commit = _git_value(root, "rev-parse", "HEAD") or "unknown"
    status = _git_value(root, "status", "--porcelain")
    dirty: bool | None = None if commit == "unknown" else bool(status)
    return {
        "schema": "modelgraph/reproducibility/v0",
        "model": {"id": config.model.id, "version": config.model.version},
        "created_at": config.created_at,
        "command": command,
        "commit": commit,
        "dirty": dirty,
        "config": {
            "path": config.source_path,
            "hash": config.config_hash(),
        },
        "dataset": {
            "id": dataset.dataset_id,
            "version": dataset.dataset_version,
            "hash": dataset.dataset_hash,
            "license": dataset.license_id,
            "consent": dataset.consent,
            "counts": dataset.counts,
            "sources": dataset.sources,
        },
        "artifact": {
            "path": str(artifact_path),
            "sha256": artifact_hash,
        },
        "manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "card": {"path": str(card_path), "sha256": card_hash},
        "seeds": {
            "training": config.seed,
            "evaluation": 1234,
            "dataset_split": (((dataset.manifest or {}).get("spec") or {}).get("splits") or {}).get(
                "seed"
            ),
        },
        "hardware": {
            **config.hardware.model_dump(mode="json", exclude_none=True),
            "observed": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
                "torch": _module_version("torch"),
            },
        },
        "image_digest": config.image_digest,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "limitations": [
            "Synthetic rehearsal: rerunning reproduces the artifact bytes, not a trained adapter.",
            "A real run must record a container image digest and the actual GPU/driver in addition.",
        ],
    }


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


# ---------------------------------------------------------------------------------------
# Eval hook
# ---------------------------------------------------------------------------------------


def write_pinned_graph(
    graph_path: str | Path, node: str, model_ref: str, destination: Path
) -> Path:
    source = resolve_path(graph_path)
    document = load_document(source)
    spec = document.get("spec") or {}
    nodes = spec.get("nodes") or []
    matches = [item for item in nodes if isinstance(item, dict) and item.get("id") == node]
    if not matches:
        raise SpecialistTrainingError(
            f"graph {source} has no node {node!r} to pin", path=str(source)
        )
    matches[0]["model"] = model_ref
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return destination


def _copy_report_evidence(report: EvalReport, destination: Path) -> None:
    if report.report_dir is None:
        raise SpecialistTrainingError("eval report was not persisted; cannot record evidence")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("report.json", "manifest.json", "report.md"):
        source = report.report_dir / name
        if source.is_file():
            shutil.copyfile(source, destination / name)


def _suite_ref(payload: dict[str, Any]) -> dict[str, Any]:
    suite = payload.get("suite") or {}
    return {
        "id": str(suite.get("id") or "unknown"),
        "version": str(suite.get("version") or "0.0.0"),
        "hash": str(suite.get("hash") or "sha256:" + "0" * 64),
    }


def _level_result(payload: dict[str, Any]) -> dict[str, Any]:
    gate = payload.get("gate") or {}
    metrics = payload.get("metrics") or {}
    return {
        "passed": bool(gate.get("passed")),
        "cases": int(metrics.get("cases") or len(payload.get("cases") or [])),
        "task_success": metrics.get("task_success"),
        "validity": metrics.get("validity"),
        "protocol_error_rate": metrics.get("protocol_error_rate"),
        "notes": "; ".join(gate.get("failures") or []),
    }


def run_eval_hook(
    config: SpecialistConfig,
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    command: str,
) -> tuple[dict[str, Any], Path]:
    """Run the node suite and graph regression over a graph pinned to the new model."""
    node_graph = resolve_path(config.evaluation.node_graph)
    graph_graph = resolve_path(config.evaluation.graph_graph)
    if node_graph != graph_graph:
        raise SpecialistTrainingError(
            "the registry eval-report format requires one pinned graph for the node suite and "
            "the graph regression; point evaluation.node_graph and evaluation.graph_graph at "
            "the same graph"
        )
    model_id = str(manifest["metadata"]["id"])
    version = str(manifest["metadata"]["version"])
    model_ref = f"{model_id}@{version}"
    eval_dir = output_dir / "eval"
    pinned = write_pinned_graph(
        node_graph, config.evaluation.node, model_ref, eval_dir / "pinned.graph.yaml"
    )
    registry = Registry.from_paths(
        [
            output_dir,
            project_root() / "registry" / "models",
            project_root() / "training" / "models",
        ],
        [pinned.parent, project_root() / "examples", project_root() / "graphs"],
    )
    reports_root = (
        resolve_path(config.evaluation.reports_dir)
        if config.evaluation.reports_dir
        else eval_dir / "runs"
    )
    node_report = run_node_suite(
        resolve_path(config.evaluation.node_suite),
        graph_path=pinned,
        provider=config.evaluation.provider,
        registry=registry,
        reports_dir=reports_root,
    )
    graph_report = run_e2e_suite(
        resolve_path(config.evaluation.graph_suite),
        graph_path=pinned,
        provider=config.evaluation.provider,
        registry=registry,
        reports_dir=reports_root,
    )
    node_payload = node_report.to_dict()
    graph_payload = graph_report.to_dict()
    _copy_report_evidence(node_report, eval_dir / "node")
    _copy_report_evidence(graph_report, eval_dir / "graph")
    node_report_hash = file_hash(eval_dir / "node" / "report.json")
    node_manifest_hash = file_hash(eval_dir / "node" / "manifest.json")
    graph_report_hash = file_hash(eval_dir / "graph" / "report.json")
    graph_manifest_hash = file_hash(eval_dir / "graph" / "manifest.json")
    passed = bool((node_payload.get("gate") or {}).get("passed")) and bool(
        (graph_payload.get("gate") or {}).get("passed")
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "model": {"id": model_id, "version": version},
        "command": command + " --eval",
        "reproduce": command + " --eval",
        "suite": {
            "node": _suite_ref(node_payload),
            "graph": _suite_ref(graph_payload),
        },
        "graph": {
            "id": str((graph_payload.get("graph") or {}).get("id") or "unknown"),
            "version": str((graph_payload.get("graph") or {}).get("version") or "0.0.0"),
            "hash": str((graph_payload.get("graph") or {}).get("hash") or "sha256:" + "0" * 64),
            "pinned_model": model_ref,
        },
        "evidence": {
            "node": {
                "report": {"path": "eval/node/report.json", "hash": node_report_hash},
                "run_manifest": {"path": "eval/node/manifest.json", "hash": node_manifest_hash},
            },
            "graph": {
                "report": {"path": "eval/graph/report.json", "hash": graph_report_hash},
                "run_manifest": {"path": "eval/graph/manifest.json", "hash": graph_manifest_hash},
            },
        },
        "seeds": {
            "training": config.seed,
            "evaluation": 1234,
        },
        "hardware": {
            "tier": config.hardware.tier,
            "gpu": config.hardware.gpu,
            "vram_gb": config.hardware.vram_gb,
            "cpu_only": config.hardware.cpu_only,
            "synthetic": config.mode == "synthetic",
        },
        "environment": node_payload.get("environment") or {},
        "results": {
            "node": _level_result(node_payload),
            "graph": _level_result(graph_payload),
        },
        "passed": passed,
        "generated_at": utc_now(),
        "limitations": [
            "Mock provider: the suite responses are scripted, so the gate measures the harness "
            "and the pinned graph, not the artifact's quality.",
            "Synthetic artifact: this evidence cannot promote a rehearsal.",
            "The node suite and graph regression must be rerun with the real adapter and a "
            "real provider before a quality claim.",
        ],
    }
    # Dataset split seed is recorded in the artifact dataset block; keep the report seeds simple
    # and valid rather than inventing a value here.
    issues = validate_document(report, "registry-eval-report")
    if issues:
        raise SpecialistTrainingError(
            "generated registry eval report is invalid: "
            + "; ".join(issue.render() for issue in issues[:5])
        )
    destination = output_dir / "eval-report.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_eval_markdown(report, output_dir / "eval-report.md")
    return report, destination


def _write_eval_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Specialist eval hook report",
        "",
        f"- Model: `{report['model']['id']}@{report['model']['version']}`",
        f"- Pinned graph: `{report['graph']['id']}@{report['graph']['version']}` "
        f"(`{report['graph']['hash']}`)",
        f"- Node suite: `{report['suite']['node']['id']}`",
        f"- Graph suite: `{report['suite']['graph']['id']}`",
        f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
        f"- Command: `{report['command']}`",
        "",
        "This hook runs the deterministic mock provider. A pass proves the harness, the pinned "
        "graph, and the evidence format; it is not a model-quality result.",
        "",
    ]
    for level in ("node", "graph"):
        result = report["results"][level]
        lines.append(
            f"- {level}: passed=`{result['passed']}` cases=`{result['cases']}` "
            f"task_success=`{result['task_success']}` validity=`{result['validity']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------


@dataclass
class SpecialistResult:
    config: SpecialistConfig
    dataset: SpecialistDataset
    artifact: dict[str, Any]
    artifact_path: Path
    artifact_hash: str
    manifest: dict[str, Any]
    manifest_path: Path
    card: str
    card_path: Path
    reproducibility: dict[str, Any]
    reproducibility_path: Path
    eval_report: dict[str, Any] | None
    eval_report_path: Path | None
    status: str
    output_dir: Path

    def summary(self) -> dict[str, Any]:
        return {
            "model": f"{self.config.model.id}@{self.config.model.version}",
            "status": self.status,
            "synthetic": bool(self.artifact.get("synthetic")),
            "quality_claim": False,
            "method": self.config.method,
            "mode": self.config.mode,
            "output_dir": str(self.output_dir),
            "artifact": str(self.artifact_path),
            "artifact_sha256": self.artifact_hash,
            "manifest": str(self.manifest_path),
            "card": str(self.card_path),
            "reproducibility": str(self.reproducibility_path),
            "dataset": {
                "id": self.dataset.dataset_id,
                "version": self.dataset.dataset_version,
                "hash": self.dataset.dataset_hash,
                "license": self.dataset.license_id,
                "counts": self.dataset.counts,
                "consent": self.dataset.consent,
                "excluded": self.dataset.excluded,
                "filter": self.dataset.filter_report,
            },
            "eval_report": str(self.eval_report_path) if self.eval_report_path else None,
            "eval_passed": (
                bool(self.eval_report.get("passed")) if self.eval_report is not None else None
            ),
            "limitations": self.artifact.get("limitations", []),
        }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_training_report(result: SpecialistResult) -> Path:
    path = result.output_dir / "training-report.md"
    dataset = result.dataset
    lines = [
        "# Specialist training report",
        "",
        f"- Model: `{result.config.model.id}@{result.config.model.version}`",
        f"- Status: **{result.status}**",
        f"- Method: `{result.config.method}` / mode `{result.config.mode}`",
        f"- Artifact: `{result.artifact_path}` (`{result.artifact_hash}`)",
        f"- Manifest: `{result.manifest_path}`",
        f"- Card: `{result.card_path}`",
        f"- Eval report: `{result.eval_report_path}`",
        f"- Dataset: `{dataset.dataset_id}@{dataset.dataset_version}` (`{dataset.dataset_hash}`)",
        f"- Dataset license: `{dataset.license_id}`",
        f"- Consent recorded: `{dataset.consent.get('recorded')}`",
        f"- Splits: {dataset.counts}",
        f"- Excluded: {dataset.excluded}",
        "",
        "## Honest result",
        "",
        "This run is a deterministic CPU rehearsal. It proves that ingestion, provenance, "
        "artifact, manifest, card, evaluation hook, and reproducibility record fit together. "
        "It does not train a model, does not produce usable weights, and cannot be promoted.",
        "",
        "## Commands",
        "",
        "```bash",
        f"uv run mg train specialist --config {result.config.source_path or '<config>'} --eval",
        "```",
        "",
    ]
    if result.eval_report is not None:
        lines.extend(
            [
                "## Eval hook",
                "",
                f"- Passed: `{result.eval_report.get('passed')}`",
                f"- Node: `{result.eval_report['results']['node']}`",
                f"- Graph regression: `{result.eval_report['results']['graph']}`",
                "- Mock provider: this is a harness check, not a quality claim.",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(result.output_dir / "training-report.json", result.summary())
    return path


def _adopt_eval_report(
    source: str | Path, manifest: dict[str, Any], output_dir: Path
) -> tuple[dict[str, Any], Path]:
    path = resolve_path(source)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecialistTrainingError(
            f"cannot read eval report {path}: {exc}", path=str(path)
        ) from exc
    if not isinstance(document, dict):
        raise SpecialistTrainingError("eval report root must be an object", path=str(path))
    issues, _warnings, _hashes = verify_eval_report(document, manifest, base=path.parent)
    if issues:
        raise SpecialistPromotionError(
            "supplied eval report failed verification: " + "; ".join(issues[:6]),
            path=str(path),
            details={"issues": issues},
        )
    destination = output_dir / "eval-report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(path.read_bytes())
    for level in ("node", "graph"):
        for kind in ("report", "run_manifest"):
            relative = ((document.get("evidence") or {}).get(level) or {}).get(kind, {}).get("path")
            if not isinstance(relative, str) or not _is_relative_evidence(relative):
                continue
            target = (output_dir / relative).resolve()
            try:
                target.relative_to(output_dir.resolve())
            except ValueError:
                continue
            source_file = (path.parent / relative).resolve()
            if source_file.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, target)
    weights = document.get("weights")
    if isinstance(weights, dict) and isinstance(weights.get("path"), str):
        relative = weights["path"]
        if _is_relative_evidence(relative):
            source_file = (path.parent / relative).resolve()
            target = (output_dir / relative).resolve()
            try:
                target.relative_to(output_dir.resolve())
            except ValueError:
                pass
            else:
                if source_file.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_file, target)
    return document, destination


def _is_relative_evidence(path: str) -> bool:
    candidate = PurePosixPath(path)
    return not candidate.is_absolute() and ".." not in candidate.parts


def train_specialist(
    config: SpecialistConfig,
    *,
    run_eval: bool = True,
    eval_report_path: str | Path | None = None,
    promote: bool = False,
    command: str | None = None,
    write: bool = True,
) -> SpecialistResult:
    """Run the deterministic pipeline and (optionally) the offline eval hook."""
    if config.mode != "synthetic":
        raise SpecialistTrainingError(
            "mode: real is not implemented in this CPU pipeline. Use training/sft_lora.py "
            "with the real torch/transformers/PEFT stack; this command refuses to fake a "
            "real LoRA run."
        )
    command = command or f"mg train specialist --config {config.source_path or '<config>'}"
    dataset = load_specialist_dataset(
        config.dataset,
        node=config.node,
        distillation=config.distillation if config.method == "distill" else None,
        seed=config.seed,
    )
    output_dir = resolve_path(config.output_dir)
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
    artifact = build_synthetic_artifact(config, dataset)
    artifact_text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    artifact_path = output_dir / "artifact.json"
    artifact_hash = sha256_text(artifact_text)
    if write:
        artifact_path.write_text(artifact_text, encoding="utf-8")
    report_expected = eval_report_path is not None or run_eval
    manifest = build_model_manifest(
        config,
        dataset,
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
        eval_report_path="eval-report.json" if report_expected else None,
    )
    manifest_path = output_dir / "manifest.yaml"
    manifest_text = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    manifest_hash = sha256_text(manifest_text)
    if write:
        manifest_path.write_text(manifest_text, encoding="utf-8")
    eval_document: dict[str, Any] | None = None
    eval_path: Path | None = None
    if eval_report_path is not None:
        eval_document, eval_path = _adopt_eval_report(eval_report_path, manifest, output_dir)
    elif run_eval:
        eval_document, eval_path = run_eval_hook(config, manifest, output_dir, command=command)
    status = "candidate"
    if promote:
        if not config.promotion.require_passing_eval:
            raise SpecialistPromotionError(
                "promotion always requires a passing eval report; require_passing_eval cannot "
                "be disabled for this artifact class"
            )
        if eval_document is None:
            raise SpecialistPromotionError(
                "refusing to promote without a passing eval report; pass --eval or "
                "--eval-report PATH"
            )
        if eval_document.get("passed") is not True:
            raise SpecialistPromotionError(
                "refusing to promote: the eval report did not pass", path=str(eval_path)
            )
        if artifact.get("synthetic"):
            raise SpecialistPromotionError(
                "refusing to promote a synthetic rehearsal artifact. Rehearsals prove the "
                "pipeline, not quality; train a real adapter on real hardware and submit a real "
                "passing eval report.",
                path=str(artifact_path),
            )
        status = "promoted"
    card_path = output_dir / "card.md"
    reproducibility_path = output_dir / "reproducibility.json"
    reproducibility = build_reproducibility(
        config,
        dataset,
        command=command,
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
        manifest_path=manifest_path,
        card_path=card_path,
        manifest_hash=manifest_hash,
        card_hash="",
    )
    card = build_model_card(
        config, dataset, artifact, manifest, eval_document, reproducibility, status=status
    )
    card_hash = sha256_text(card)
    reproducibility["card"]["sha256"] = card_hash
    if write:
        card_path.write_text(card, encoding="utf-8")
        _write_json(reproducibility_path, reproducibility)
        _write_json(output_dir / "dataset-summary.json", dataset.summary())
        if status == "promoted":
            _write_json(
                output_dir / "promotion.json",
                {
                    "schema": "modelgraph/promotion/v0",
                    "model": {"id": config.model.id, "version": config.model.version},
                    "status": status,
                    "eval_report": str(eval_path) if eval_path else None,
                    "eval_passed": True,
                    "synthetic": False,
                },
            )
    result = SpecialistResult(
        config=config,
        dataset=dataset,
        artifact=artifact,
        artifact_path=artifact_path,
        artifact_hash=artifact_hash,
        manifest=manifest,
        manifest_path=manifest_path,
        card=card,
        card_path=card_path,
        reproducibility=reproducibility,
        reproducibility_path=reproducibility_path,
        eval_report=eval_document,
        eval_report_path=eval_path,
        status=status,
        output_dir=output_dir,
    )
    if write:
        _write_training_report(result)
    return result
