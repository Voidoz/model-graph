"""Build router examples from T-0023 dataset exports.

This is the bridge between the exporter and the trainer. It keeps the rendered prompt, the menu,
the label (oracle/trace/human), the ambiguity flag, the group/split, and the provenance chain
(dataset manifest -> example -> run/node/event). It never invents a label: examples without a
positive routing label are counted as excluded, and contradictory labels are rejected.

The output is deliberately separate from the T-0023 dataset: ``router-examples.jsonl`` plus
``router-dataset.json`` (summary). Both are written by ``mg train dataset``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelgraph.data import DatasetBundle, load_dataset, validate_dataset_example
from modelgraph.training.errors import RouterDataError
from modelgraph.training.features import ABSTAIN_CLASS, MenuRoute, RouterView, view_from_example
from modelgraph.util import canonical_json, sha256_text

ROUTER_EXAMPLE_SCHEMA = "modelgraph/router-example/v0"
ROUTER_DATASET_SCHEMA = "modelgraph/router-dataset/v0"
SPLIT_NAMES = ("train", "validation", "test")
EXCLUDED_REASONS = ("wrong_kind", "unlabeled", "unrendered", "duplicate")


@dataclass(frozen=True)
class RouterExample:
    """One labeled router example derived from a dataset record."""

    example_id: str
    source_example_id: str
    split: str
    group: str
    request: str
    messages: tuple[dict[str, str], ...]
    menu_routes: tuple[MenuRoute, ...]
    target: str
    oracle_route: str | None
    ambiguous: bool
    label_source: str
    label_outcome: str
    labeler: str | None
    prompt_hash: str | None
    output: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def should_abstain(self) -> bool:
        return self.ambiguous or self.target == ABSTAIN_CLASS

    @property
    def class_label(self) -> str:
        return ABSTAIN_CLASS if self.should_abstain else self.target

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ROUTER_EXAMPLE_SCHEMA,
            "example_id": self.example_id,
            "split": self.split,
            "group": self.group,
            "request": self.request,
            "messages": list(self.messages),
            "menu": {
                "routes": [route.id for route in self.menu_routes],
                "route_text": {route.id: route.text for route in self.menu_routes},
            },
            "label": {
                "target": self.class_label,
                "oracle_route": self.oracle_route,
                "ambiguous": self.ambiguous,
                "source": self.label_source,
                "outcome": self.label_outcome,
                "labeler": self.labeler,
            },
            "prompt_hash": self.prompt_hash,
            "output": self.output,
            "provenance": self.provenance,
        }


@dataclass
class RouterDataset:
    path: Path | None
    examples: list[RouterExample]
    summary: dict[str, Any]
    manifest: dict[str, Any] | None = None
    dataset_manifest_hash: str | None = None

    def split(self, name: str) -> list[RouterExample]:
        return [example for example in self.examples if example.split == name]

    @property
    def dataset_id(self) -> str:
        return str(self.summary.get("dataset", {}).get("id") or "unknown")

    @property
    def dataset_version(self) -> str:
        return str(self.summary.get("dataset", {}).get("version") or "0.0.0")

    @property
    def frozen_test(self) -> bool:
        return bool(self.summary.get("dataset", {}).get("frozen_test"))

    @property
    def counts(self) -> dict[str, int]:
        return {split: len(self.split(split)) for split in SPLIT_NAMES}

    def classes(self) -> list[str]:
        return sorted({example.class_label for example in self.examples})

    def write(self, out: str | Path) -> Path:
        destination = Path(out)
        destination.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(example.to_dict(), sort_keys=True, ensure_ascii=False)
            for example in sorted(self.examples, key=lambda item: item.example_id)
        ]
        examples_text = ("\n".join(lines) + "\n") if lines else ""
        (destination / "router-examples.jsonl").write_text(examples_text, encoding="utf-8")
        summary = dict(self.summary)
        summary["schema"] = ROUTER_DATASET_SCHEMA
        summary["files"] = {
            "router-examples.jsonl": sha256_text(examples_text),
        }
        (destination / "router-dataset.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination


def load_router_dataset(
    source: str | Path,
    *,
    strict_labels: bool = False,
    allow_unrendered: bool = False,
    validate: bool = True,
) -> RouterDataset:
    """Load a T-0023 dataset directory, ``examples.jsonl``, or split file."""
    path = Path(source)
    manifest: dict[str, Any] | None = None
    raw_examples: list[dict[str, Any]]
    dataset_id = "unknown"
    dataset_version = "0.0.0"
    manifest_hash: str | None = None
    frozen_test = False
    split_plan_hash: str | None = None
    if path.is_dir():
        if (path / "dataset.json").is_file():
            bundle: DatasetBundle = load_dataset(path, validate=validate)
            manifest = bundle.manifest
            raw_examples = list(bundle.examples)
            raw_metadata = manifest.get("metadata")
            metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
            raw_spec = manifest.get("spec")
            spec: dict[str, Any] = raw_spec if isinstance(raw_spec, dict) else {}
            dataset_id = str(metadata.get("id") or "unknown")
            dataset_version = str(metadata.get("version") or "0.0.0")
            manifest_hash = str(spec.get("hash") or "")
            raw_splits = spec.get("splits")
            splits: dict[str, Any] = raw_splits if isinstance(raw_splits, dict) else {}
            frozen_test = bool(splits.get("frozen_test"))
            split_plan_hash = (
                splits.get("plan_hash") if isinstance(splits.get("plan_hash"), str) else None
            )
        elif (path / "examples.jsonl").is_file():
            raw_examples = _read_jsonl(path / "examples.jsonl")
        else:
            raise RouterDataError(
                f"dataset directory has neither dataset.json nor examples.jsonl: {path}",
                path=str(path),
            )
    elif path.is_file():
        raw_examples = _read_jsonl(path)
        manifest_hash = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        raise RouterDataError(f"dataset path not found: {path}", path=str(path))

    if validate and manifest is None:
        for index, raw in enumerate(raw_examples):
            issues = validate_dataset_example(raw)
            if issues:
                raise RouterDataError(
                    f"dataset example {index + 1} does not validate: " + "; ".join(issues[:3]),
                    path=str(path),
                )
    if not manifest_hash:
        manifest_hash = (
            "sha256:" + hashlib.sha256(canonical_json(raw_examples).encode("utf-8")).hexdigest()
        )

    examples: list[RouterExample] = []
    excluded = {reason: 0 for reason in EXCLUDED_REASONS}
    seen_ids: set[str] = set()
    groups: dict[str, str] = {}
    for index, raw in enumerate(raw_examples):
        if not isinstance(raw, dict):
            raise RouterDataError(f"dataset example {index + 1} is not an object", path=str(path))
        if raw.get("kind") != "routing":
            excluded["wrong_kind"] += 1
            continue
        example, reason = _router_example(
            raw,
            index=index,
            manifest=manifest,
            manifest_hash=manifest_hash,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            allow_unrendered=allow_unrendered,
            strict_labels=strict_labels,
        )
        if reason is not None:
            excluded[reason] += 1
            continue
        assert example is not None
        if example.example_id in seen_ids:
            raise RouterDataError(f"duplicate example_id {example.example_id!r}", path=str(path))
        seen_ids.add(example.example_id)
        previous = groups.get(example.group)
        if previous is not None and previous != example.split:
            raise RouterDataError(
                f"group {example.group!r} appears in both {previous} and {example.split}; "
                "group-safe splits are required",
                path=str(path),
            )
        groups[example.group] = example.split
        examples.append(example)

    if not examples:
        raise RouterDataError("no labeled routing examples were produced", path=str(path))
    summary = _summary(
        examples,
        excluded=excluded,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        manifest_hash=manifest_hash,
        frozen_test=frozen_test,
        split_plan_hash=split_plan_hash,
        manifest=manifest,
        source=str(path),
    )
    return RouterDataset(
        path=path,
        examples=examples,
        summary=summary,
        manifest=manifest,
        dataset_manifest_hash=manifest_hash,
    )


def _router_example(
    raw: dict[str, Any],
    *,
    index: int,
    manifest: dict[str, Any] | None,
    manifest_hash: str,
    dataset_id: str,
    dataset_version: str,
    allow_unrendered: bool,
    strict_labels: bool,
) -> tuple[RouterExample | None, str | None]:
    example_id = str(raw.get("example_id") or "")
    if not example_id:
        raise RouterDataError(f"routing example {index + 1} has no example_id")
    split = str(raw.get("split") or "")
    if split not in SPLIT_NAMES:
        raise RouterDataError(f"routing example {example_id} has invalid split {split!r}")
    group = str(raw.get("group") or "")
    if not group:
        raise RouterDataError(f"routing example {example_id} has no group")
    input_value = raw.get("input")
    if not isinstance(input_value, dict):
        raise RouterDataError(f"routing example {example_id} has no input object")
    if input_value.get("rendered") is not True and not allow_unrendered:
        return None, "unrendered"
    try:
        view = view_from_example(raw)
    except ValueError as exc:
        raise RouterDataError(f"routing example {example_id}: {exc}") from exc
    label = raw.get("label")
    if not isinstance(label, dict):
        raise RouterDataError(f"routing example {example_id} has no label object")
    route = label.get("route")
    if route is not None and not isinstance(route, str):
        raise RouterDataError(f"routing example {example_id} has a non-string route label")
    route = route.strip() if isinstance(route, str) else None
    should_abstain = label.get("should_abstain")
    if should_abstain is not None and not isinstance(should_abstain, bool):
        raise RouterDataError(
            f"routing example {example_id} has a non-boolean should_abstain label"
        )
    if route is None and should_abstain is False:
        raise RouterDataError(
            f"routing example {example_id} says should_abstain=false but has no route label"
        )
    source = str(label.get("source") or "unknown")
    outcome = str(label.get("outcome") or "unknown")
    if source not in {"eval_check", "human_label", "run_outcome", "oracle", "unknown"}:
        raise RouterDataError(f"routing example {example_id} has unknown label source {source!r}")
    if source == "unknown" and route is not None:
        raise RouterDataError(
            f"routing example {example_id} has a route with label source 'unknown'"
        )
    if route == "abstain":
        if should_abstain is False:
            raise RouterDataError(
                f"routing example {example_id} says route=abstain but should_abstain=false"
            )
        should_abstain = True
        oracle_route = None
    elif route is not None:
        if route not in view.route_ids():
            raise RouterDataError(
                f"routing example {example_id} label route {route!r} is not in menu "
                f"{view.route_ids()}"
            )
        oracle_route = route
    else:
        oracle_route = None
    if route is None and not should_abstain:
        if strict_labels:
            raise RouterDataError(
                f"routing example {example_id} has no positive routing label", path=""
            )
        return None, "unlabeled"
    ambiguous = bool(should_abstain)
    target = ABSTAIN_CLASS if ambiguous else str(route)
    if target not in {*view.route_ids(), ABSTAIN_CLASS}:
        raise RouterDataError(f"routing example {example_id} target {target!r} is not scorable")
    source_block = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    raw_output = raw.get("output")
    output_value: dict[str, Any] = raw_output if isinstance(raw_output, dict) else {}
    provenance = {
        "dataset": {"id": dataset_id, "version": dataset_version, "hash": manifest_hash},
        "source_example_id": example_id,
        "source": source_block,
        "label": {
            "source": source,
            "outcome": outcome,
            "labeler": label.get("labeler"),
            "reason": label.get("reason"),
        },
        "license": raw.get("license"),
        "consent": raw.get("consent"),
        "redaction": raw.get("redaction"),
        "hashes": raw.get("hashes"),
        "manifest": (
            {
                "id": manifest.get("metadata", {}).get("id"),
                "version": manifest.get("metadata", {}).get("version"),
                "hash": manifest_hash,
            }
            if manifest is not None
            else None
        ),
    }
    return (
        RouterExample(
            example_id=example_id,
            source_example_id=example_id,
            split=split,
            group=group,
            request=view.request,
            messages=view.messages,
            menu_routes=view.routes,
            target=str(route) if not ambiguous else ABSTAIN_CLASS,
            oracle_route=oracle_route,
            ambiguous=ambiguous,
            label_source=source,
            label_outcome=outcome,
            labeler=str(label.get("labeler")) if label.get("labeler") else None,
            prompt_hash=view.prompt_hash,
            output=output_value,
            provenance=provenance,
        ),
        None,
    )


def _summary(
    examples: Sequence[RouterExample],
    *,
    excluded: dict[str, int],
    dataset_id: str,
    dataset_version: str,
    manifest_hash: str,
    frozen_test: bool,
    split_plan_hash: str | None,
    manifest: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    counts = {split: 0 for split in SPLIT_NAMES}
    class_balance: dict[str, dict[str, int]] = {split: {} for split in SPLIT_NAMES}
    ambiguity: dict[str, int] = {split: 0 for split in SPLIT_NAMES}
    label_sources: dict[str, int] = {}
    groups: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    per_route: dict[str, int] = {}
    for example in examples:
        counts[example.split] += 1
        class_balance[example.split][example.class_label] = (
            class_balance[example.split].get(example.class_label, 0) + 1
        )
        if example.ambiguous:
            ambiguity[example.split] += 1
        label_sources[example.label_source] = label_sources.get(example.label_source, 0) + 1
        groups[example.split].add(example.group)
        per_route[example.class_label] = per_route.get(example.class_label, 0) + 1
    limitations: list[str] = []
    if not frozen_test:
        limitations.append("dataset does not declare a frozen test split")
    if any(not class_balance[split] for split in SPLIT_NAMES):
        missing = [split for split in SPLIT_NAMES if not class_balance[split]]
        limitations.append("empty split(s): " + ", ".join(missing))
    if label_sources.get("unknown"):
        limitations.append("some included examples have unknown label source")
    return {
        "dataset": {
            "id": dataset_id,
            "version": dataset_version,
            "hash": manifest_hash,
            "frozen_test": frozen_test,
            "split_plan_hash": split_plan_hash,
            "source": source,
            "manifest_present": manifest is not None,
        },
        "examples": len(examples),
        "counts": counts,
        "class_balance": {
            split: dict(sorted(values.items())) for split, values in class_balance.items()
        },
        "per_route": dict(sorted(per_route.items())),
        "ambiguity": ambiguity,
        "label_sources": dict(sorted(label_sources.items())),
        "groups": {split: len(names) for split, names in groups.items()},
        "excluded": dict(sorted(excluded.items())),
        "limitations": limitations,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RouterDataError(f"cannot read dataset JSONL {path}: {exc}", path=str(path)) from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RouterDataError(
                f"invalid JSON on line {index} of {path}: {exc}", path=str(path)
            ) from exc
        if not isinstance(value, dict):
            raise RouterDataError(f"line {index} of {path} is not an object", path=str(path))
        examples.append(value)
    return examples


def iter_views(examples: Iterable[RouterExample]) -> Iterable[RouterView]:
    for example in examples:
        yield RouterView(
            request=example.request,
            routes=tuple(example.menu_routes),
            menu_text="",
            messages=example.messages,
            prompt_hash=example.prompt_hash,
        )
