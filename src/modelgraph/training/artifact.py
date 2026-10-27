"""Versioned, self-describing router artifacts.

The artifact is the whole model: method, feature version, classes, weights, calibration,
abstain policy, training parameters, dataset identity, and provenance. Loading validates the
JSON Schema and a few cross-field invariants that JSON Schema cannot express (weight keys match
classes and every weight vector has the declared length).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from modelgraph.training.errors import RouterArtifactError
from modelgraph.training.features import FEATURE_VERSION, feature_dimension

ARTIFACT_SCHEMA = "modelgraph/router-artifact/v0"
ARTIFACT_VERSION = "0.1.0"
_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "router-artifact.schema.json"


@cache
def artifact_schema() -> dict[str, Any]:
    try:
        return cast(dict[str, Any], json.loads(_SCHEMA_PATH.read_text(encoding="utf-8")))
    except OSError as exc:  # pragma: no cover - packaged file is present
        raise RouterArtifactError(
            f"cannot read router artifact schema {_SCHEMA_PATH}: {exc}", path=str(_SCHEMA_PATH)
        ) from exc


@cache
def _validator() -> Draft202012Validator:
    schema = artifact_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_router_artifact(document: dict[str, Any]) -> list[str]:
    """Return stable validation messages; an empty list means the artifact is usable."""
    errors = [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(_validator().iter_errors(document), key=lambda e: list(e.path))
    ]
    if errors:
        return errors
    classes = document.get("classes")
    weights = document.get("weights")
    dimension = document.get("feature_dimension")
    if isinstance(classes, list) and isinstance(weights, dict):
        expected = set(classes)
        actual = set(weights)
        if actual != expected:
            errors.append(
                f"weights classes {sorted(actual)} do not match classes {sorted(expected)}"
            )
        if isinstance(dimension, int):
            for name in sorted(actual & expected):
                vector = weights[name]
                if not isinstance(vector, list) or len(vector) != dimension:
                    errors.append(
                        f"weights[{name!r}] has {len(vector) if isinstance(vector, list) else '?'} "
                        f"values; expected {dimension}"
                    )
    if document.get("feature_version") != FEATURE_VERSION:
        errors.append(
            f"feature_version must be {FEATURE_VERSION!r}, got {document.get('feature_version')!r}"
        )
    threshold = document.get("abstain", {}).get("threshold")
    if isinstance(threshold, (int, float)) and not 0 <= float(threshold) <= 1:
        errors.append("abstain.threshold must be between 0 and 1")
    return errors


def _raise_if_invalid(document: dict[str, Any], *, path: str = "") -> None:
    errors = validate_router_artifact(document)
    if errors:
        raise RouterArtifactError(
            "invalid router artifact: " + "; ".join(errors[:5]),
            path=path,
            details={"errors": errors},
        )


@dataclass(frozen=True)
class RouterArtifact:
    """A validated router artifact and its optional on-disk location."""

    document: dict[str, Any]
    path: Path | None = None

    @classmethod
    def from_dict(cls, document: dict[str, Any], *, path: str = "") -> RouterArtifact:
        _raise_if_invalid(document, path=path)
        return cls(document=document)

    @classmethod
    def load(cls, path: str | Path, *, verify_sha256: str | None = None) -> RouterArtifact:
        source = Path(path)
        try:
            raw = source.read_bytes()
        except OSError as exc:
            raise RouterArtifactError(
                f"cannot read router artifact {source}: {exc}", path=str(source)
            ) from exc
        digest = hashlib.sha256(raw).hexdigest()
        if verify_sha256 is not None:
            expected = verify_sha256.removeprefix("sha256:")
            if digest != expected:
                raise RouterArtifactError(
                    f"router artifact hash mismatch: {digest} != {expected}", path=str(source)
                )
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RouterArtifactError(
                f"router artifact is not valid JSON: {exc}", path=str(source)
            ) from exc
        if not isinstance(document, dict):
            raise RouterArtifactError("router artifact must be a JSON object", path=str(source))
        artifact = cls.from_dict(document, path=str(source))
        return cls(document=artifact.document, path=source)

    @property
    def classes(self) -> list[str]:
        return [str(item) for item in self.document["classes"]]

    @property
    def weights(self) -> dict[str, list[float]]:
        return {
            name: [float(value) for value in values]
            for name, values in self.document["weights"].items()
        }

    @property
    def feature_dimension(self) -> int:
        return int(self.document["feature_dimension"])

    @property
    def temperature(self) -> float:
        return float(self.document["calibration"]["temperature"])

    @property
    def abstain_threshold(self) -> float:
        return float(self.document["abstain"]["threshold"])

    @property
    def feature_version(self) -> str:
        return str(self.document["feature_version"])

    @property
    def bigrams(self) -> bool:
        """Whether request bigram features were enabled when this artifact was trained."""
        features = self.document.get("features")
        if not isinstance(features, dict):
            # Artifacts produced before the feature block existed always used bigrams.
            return True
        return bool(features.get("bigrams", True))

    @property
    def dataset_id(self) -> str:
        return str(self.document["dataset"]["id"])

    @property
    def dataset_version(self) -> str:
        return str(self.document["dataset"]["version"])

    def sha256(self) -> str:
        return hashlib.sha256(canonical_artifact_bytes(self.document)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(json.dumps(self.document)))

    def save(self, path: str | Path) -> Path:
        _raise_if_invalid(self.document, path=str(path))
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_artifact_bytes(self.document)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            with suppress(OSError):
                os.unlink(temporary)
            raise
        return destination


def canonical_artifact_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def build_artifact(
    *,
    classes: list[str],
    weights: dict[str, list[float]],
    dimension: int,
    bigrams: bool,
    seed: int,
    calibration: dict[str, Any],
    abstain: dict[str, Any],
    training: dict[str, Any],
    dataset: dict[str, Any],
    provenance: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_version": ARTIFACT_VERSION,
        "feature_version": FEATURE_VERSION,
        "features": {"bigrams": bool(bigrams)},
        "method": "conditional-logit",
        "seed": int(seed),
        "classes": list(classes),
        "feature_dimension": feature_dimension(dimension),
        "weights": {name: [float(value) for value in values] for name, values in weights.items()},
        "calibration": calibration,
        "abstain": abstain,
        "training": training,
        "dataset": dataset,
        "provenance": provenance,
    }
    if metrics is not None:
        document["metrics"] = metrics
    _raise_if_invalid(document)
    return document


def load_router_artifact(path: str | Path, *, verify_sha256: str | None = None) -> RouterArtifact:
    return RouterArtifact.load(path, verify_sha256=verify_sha256)
