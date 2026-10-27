"""Versioned local model registry: contribution, index, and verification.

The registry is a directory tree:

    registry/
      index.yaml
      models/<id>/<version>/manifest.yaml
      models/<id>/<version>/card.md
      models/<id>/<version>/eval-report.json
      models/<id>/<version>/<evidence files referenced by the report>

The legacy flat starter layout (``registry/models/<id>.yaml``) is read by
:mod:`modelgraph.registry`; this module only manages the versioned tree. Everything is
local and offline: weights are never downloaded, and the registry stores a URI and a
hash, not the weights themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from modelgraph.contracts import load_document, validate_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.errors import ModelGraphError
from modelgraph.paths import project_root
from modelgraph.registry import version_key
from modelgraph.security import manifest_trust_violations
from modelgraph.util import canonical_json, sha256_text

INDEX_API_VERSION = "modelgraph/registry-index/v0"
INDEX_KIND = "RegistryIndex"
INDEX_FILE = "index.yaml"
REPORT_SCHEMA = "modelgraph/registry-eval-report/v0"
MODELS_DIR = "models"
MANIFEST_NAMES = ("manifest.yaml", "manifest.yml", "model.yaml", "model.yml")
CARD_NAME = "card.md"
REPORT_NAME = "eval-report.json"
HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?$"
)

# Placeholder/unknown declarations are rejected for a distributable registry unless a
# selected profile explicitly lists the value or the operator opts in.
UNKNOWN_LICENSES = {
    "",
    "unknown",
    "tbd",
    "tba",
    "none",
    "n/a",
    "na",
    "unlicensed",
    "unset",
}

# A small SPDX seed list used only to warn about declarations the project has not seen
# before. It is not a legal allowlist; a profile's `licenses` list is the policy.
KNOWN_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "gpl-2.0-only",
    "gpl-2.0-or-later",
    "gpl-3.0-only",
    "gpl-3.0-or-later",
    "lgpl-3.0-only",
    "lgpl-3.0-or-later",
    "mpl-2.0",
    "unlicense",
    "openrail",
    "llama3",
    "gemma",
}

CARD_SECTIONS: dict[str, tuple[str, ...]] = {
    "base": ("base model", "base"),
    "data": ("training data", "data summary", "dataset", "data"),
    "method": ("training method", "method", "approach"),
    "evaluation": ("evaluation", "eval", "results"),
    "limitations": ("limitations", "limitation", "known limitations", "failure modes"),
    "license": ("license", "licensing", "licence"),
}
CARD_OPTIONAL_SECTIONS: dict[str, tuple[str, ...]] = {
    "hyperparameters": ("hyperparameters", "hyper-parameters", "training configuration"),
}


class RegistryError(ModelGraphError):
    """A registry contribution, index, or entry is invalid."""

    code = "MG-7100"


class RegistryConflictError(RegistryError):
    """The same id@version already exists with different content."""

    code = "MG-7101"


class RegistryVerificationError(RegistryError):
    """One or more registry entries failed verification."""

    code = "MG-7102"


@dataclass(frozen=True)
class RegistryEntry:
    """One contribution on disk, before or after verification."""

    model_id: str
    version: str
    path: Path
    manifest: dict[str, Any]
    card: str
    report: dict[str, Any]
    hashes: dict[str, str]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"{self.model_id}@{self.version}"

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_index_entry(self) -> dict[str, Any]:
        spec = self.manifest.get("spec") or {}
        license_info = spec.get("license") or {}
        return {
            "id": self.model_id,
            "version": self.version,
            "path": f"{MODELS_DIR}/{self.model_id}/{self.version}",
            "provider": str(spec.get("provider") or "custom"),
            "license": str(license_info.get("name") or "unknown"),
            "hashes": {
                "manifest": self.hashes["manifest"],
                "card": self.hashes["card"],
                "eval_report": self.hashes["eval_report"],
                **({"weights": self.hashes["weights"]} if "weights" in self.hashes else {}),
            },
            "passed": self.passed,
            "warnings": sorted(set(self.warnings)),
        }


@dataclass(frozen=True)
class Contribution:
    """A contribution directory loaded and validated before copying."""

    source: Path
    entry: RegistryEntry
    files: dict[str, bytes]


@dataclass(frozen=True)
class RegistryVerification:
    """The result of verifying a whole registry."""

    root: Path
    entries: list[RegistryEntry]
    index_document: dict[str, Any]
    index_path: Path
    index_current: bool
    errors: list[str]
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return not self.errors and all(entry.passed for entry in self.entries)


# ---------------------------------------------------------------------------------------
# Hashing and path helpers
# ---------------------------------------------------------------------------------------


def file_hash(path: str | Path) -> str:
    """Return ``sha256:<hex>`` for a file's bytes."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def default_registry_root() -> Path:
    return project_root() / "registry"


def entry_directory(root: str | Path, model_id: str, version: str) -> Path:
    return Path(root) / MODELS_DIR / model_id / version


def _read_text(path: Path, *, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"cannot read {label} {path}: {exc}", path=str(path)) from exc


def _read_report(path: Path) -> dict[str, Any]:
    text = _read_text(path, label="eval report")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegistryError(f"eval report {path} is not valid JSON: {exc}", path=str(path)) from exc
    if not isinstance(document, dict):
        raise RegistryError("eval report root must be a JSON object", path=str(path))
    return document


def _schema_messages(document: dict[str, Any], schema: str, *, path: str = "") -> list[str]:
    return [issue.render() for issue in validate_document(document, schema, path=path)]


def _parse_ref(ref: str) -> tuple[str, str | None]:
    if "@" in ref:
        model_id, version = ref.split("@", 1)
        return model_id, version
    return ref, None


def _validate_ref(model_id: str, version: str | None) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,63}", model_id):
        raise RegistryError(f"invalid model id {model_id!r}")
    if version is not None and not SEMVER_PATTERN.fullmatch(version):
        raise RegistryError(f"invalid model version {version!r}")


def _is_relative_inside(path: str) -> bool:
    candidate = Path(path)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _evidence_path(base: Path, relative: str) -> tuple[Path | None, str | None]:
    if not _is_relative_inside(relative):
        return None, f"evidence path {relative!r} must be relative and stay inside the contribution"
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None, f"evidence path {relative!r} escapes the contribution directory"
    if not candidate.is_file():
        return None, f"referenced evidence file is missing: {relative}"
    return candidate, None


# ---------------------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------------------


def _markdown_sections(card: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    in_fence = False
    for line in card.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        if not in_fence:
            match = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
            if match is not None:
                if current_title is not None:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = match.group(2).strip().lower()
                current_lines = []
                continue
        if current_title is not None:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def _has_heading(sections: list[tuple[str, str]], synonyms: tuple[str, ...]) -> str | None:
    for title, body in sections:
        for synonym in synonyms:
            if re.search(rf"\b{re.escape(synonym)}\b", title) and body:
                return title
    return None


def card_issues(card: str) -> list[str]:
    """Check the required model-card sections and that each has content."""
    issues: list[str] = []
    if not card.strip():
        return ["card.md is empty"]
    sections = _markdown_sections(card)
    for name, synonyms in CARD_SECTIONS.items():
        if _has_heading(sections, synonyms) is None:
            issues.append(
                f"card.md is missing a non-empty {name!r} section "
                f"(one of headings: {', '.join(synonyms)})"
            )
    for name, synonyms in CARD_OPTIONAL_SECTIONS.items():
        if _has_heading(sections, synonyms) is None:
            issues.append(f"recommended card.md section {name!r} is missing or empty")
    return issues


# ---------------------------------------------------------------------------------------
# License policy
# ---------------------------------------------------------------------------------------


def license_issues(
    name: str,
    *,
    profile: Any | None = None,
    allow_unknown: bool = False,
    policy: str = "distributable",
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for a declared license and optional profile."""
    errors: list[str] = []
    warnings: list[str] = []
    normalized = name.strip().lower()
    allowed: set[str] | None = None
    if profile is not None and getattr(profile, "licenses", None) is not None:
        allowed = {str(item).strip().lower() for item in profile.licenses}
    if allowed is not None and normalized not in allowed:
        errors.append(
            f"license {name!r} is not allowed by profile {getattr(profile, 'name', '<profile>')!r} "
            f"(allowed: {', '.join(sorted(allowed)) or 'none'})"
        )
    unknown = normalized in UNKNOWN_LICENSES
    explicitly_allowed = bool(allowed is not None and normalized in allowed)
    if unknown:
        if policy == "distributable" and not allow_unknown and not explicitly_allowed:
            errors.append(
                f"license {name!r} is unknown; a distributable registry rejects an undeclared "
                "license unless --allow-unknown-license is explicit or the selected profile "
                "lists it"
            )
        else:
            warnings.append(
                f"license {name!r} is an unknown/placeholder declaration; redistribution is "
                "not established"
            )
    elif normalized not in KNOWN_LICENSES:
        warnings.append(
            f"license {name!r} is not in the built-in SPDX seed list; it is a declaration, "
            "not a legal check"
        )
    if policy == "local" and unknown and not errors:
        warnings.append("local policy permits an unknown license; do not distribute the model")
    return errors, warnings


# ---------------------------------------------------------------------------------------
# Loading and verifying one entry
# ---------------------------------------------------------------------------------------


def _load_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        document = load_document(path)
    except ModelGraphError as exc:
        return None, [exc.message]
    issues = _schema_messages(document, "manifest", path=str(path))
    if issues:
        return document, issues
    violations = manifest_trust_violations(document)
    if violations:
        issues.extend(f"trust policy: {violation}" for violation in violations)
    try:
        ModelManifest.model_validate(document)
    except ValidationError as exc:
        issues.append(f"manifest contract: {exc}")
    return document, issues


def _evidence_issues(
    report: dict[str, Any],
    base: Path,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    """Verify every evidence reference and return hashes plus parsed report payloads."""
    issues: list[str] = []
    hashes: dict[str, str] = {}
    parsed: dict[str, Any] = {}
    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        return ["eval report has no evidence mapping"], hashes, parsed
    for level in ("node", "graph"):
        entry = evidence.get(level)
        if not isinstance(entry, dict):
            issues.append(f"eval report evidence.{level} is missing")
            continue
        for kind in ("report", "run_manifest"):
            ref = entry.get(kind)
            if not isinstance(ref, dict):
                issues.append(f"eval report evidence.{level}.{kind} is missing")
                continue
            relative = ref.get("path")
            expected = ref.get("hash")
            if not isinstance(relative, str) or not isinstance(expected, str):
                issues.append(f"eval report evidence.{level}.{kind} must have path and hash")
                continue
            if not HASH_PATTERN.match(expected):
                issues.append(f"eval report evidence.{level}.{kind}.hash is not a SHA-256 hash")
                continue
            absolute, problem = _evidence_path(base, relative)
            if problem is not None:
                issues.append(problem)
                continue
            assert absolute is not None
            actual = file_hash(absolute)
            if actual != expected:
                issues.append(f"evidence hash mismatch for {relative}: {actual} != {expected}")
                continue
            hashes[f"{level}.{kind}"] = actual
            if kind == "report":
                try:
                    parsed[level] = json.loads(absolute.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    issues.append(f"cannot parse evidence report {relative}: {exc}")
    return issues, hashes, parsed


def _report_issues(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    base: Path | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    issues: list[str] = []
    warnings: list[str] = []
    issues.extend(_schema_messages(report, "registry-eval-report", path=str(base or "<report>")))
    metadata = manifest.get("metadata") or {}
    spec = manifest.get("spec") or {}
    model = report.get("model") or {}
    if model.get("id") != metadata.get("id") or model.get("version") != metadata.get("version"):
        issues.append(
            "eval report model "
            f"{model.get('id')!r}@{model.get('version')!r} does not match the manifest "
            f"{metadata.get('id')!r}@{metadata.get('version')!r}"
        )
    expected_pin = f"{metadata.get('id')}@{metadata.get('version')}"
    graph = report.get("graph") or {}
    if graph.get("pinned_model") != expected_pin:
        issues.append(
            f"eval report graph.pinned_model must be {expected_pin!r} so the regression ran "
            "with the contributed model"
        )
    if report.get("passed") is not True:
        issues.append("eval report is not passing; a registry contribution requires passed: true")
    if not report.get("seeds"):
        issues.append("eval report does not record seeds")
    if not report.get("hardware"):
        issues.append("eval report does not record hardware")
    if not str(report.get("reproduce", "")).strip():
        issues.append("eval report has no reproduce command")
    declared_weights = spec.get("integrity", {}).get("sha256")
    weights = report.get("weights")
    hashes: dict[str, str] = {}
    if declared_weights and not HASH_PATTERN.match(f"sha256:{declared_weights}"):
        issues.append("manifest integrity.sha256 is not a SHA-256 hash")
    if isinstance(weights, dict):
        path = weights.get("path")
        expected = weights.get("hash")
        if not isinstance(path, str) or not isinstance(expected, str):
            issues.append("eval report weights must have a path and hash when present")
        elif not HASH_PATTERN.match(expected):
            issues.append("eval report weights.hash is not a SHA-256 hash")
        elif base is not None:
            absolute, problem = _evidence_path(base, path)
            if problem is not None:
                issues.append(problem)
            else:
                assert absolute is not None
                actual = file_hash(absolute)
                if actual != expected:
                    issues.append(f"weights hash mismatch: {actual} != {expected}")
                else:
                    hashes["weights"] = actual
        if (
            declared_weights
            and isinstance(expected, str)
            and expected != f"sha256:{declared_weights}"
        ):
            issues.append(
                "eval report weights.hash does not match the manifest integrity.sha256 pin"
            )
    elif declared_weights:
        warnings.append(
            "manifest declares integrity.sha256 but the contribution bundles no weights file "
            "to hash; the declaration is unverified"
        )
    if base is not None:
        evidence_issues, evidence_hashes, parsed = _evidence_issues(report, base)
        issues.extend(evidence_issues)
        hashes.update(evidence_hashes)
        for level in ("node", "graph"):
            payload = parsed.get(level)
            level_suite = (report.get("suite") or {}).get(level) or {}
            if isinstance(payload, dict):
                actual_suite = (payload.get("suite") or {}).get("hash")
                if actual_suite and actual_suite != level_suite.get("hash"):
                    issues.append(
                        f"{level} report suite hash {actual_suite!r} does not match the "
                        f"registry report suite hash {level_suite.get('hash')!r}"
                    )
                actual_graph = (payload.get("graph") or {}).get("hash")
                if actual_graph and actual_graph != graph.get("hash"):
                    issues.append(
                        f"{level} report graph hash {actual_graph!r} does not match the "
                        f"registry report graph hash {graph.get('hash')!r}"
                    )
                evidence_gate = payload.get("gate") or {}
                if evidence_gate.get("passed") is not True:
                    issues.append(f"{level} evidence report does not have a passing gate")
    return issues, warnings, hashes


def verify_entry(
    directory: str | Path,
    *,
    profile: Any | None = None,
    allow_unknown: bool = False,
    policy: str = "distributable",
) -> RegistryEntry:
    """Load and verify one ``models/<id>/<version>`` directory."""
    path = Path(directory)
    manifest_path = next((path / name for name in MANIFEST_NAMES if (path / name).is_file()), None)
    if manifest_path is None:
        raise RegistryError(f"no manifest.yaml in {path}", path=str(path))
    manifest, issues = _load_manifest(manifest_path)
    if manifest is None:
        raise RegistryError("invalid registry manifest: " + "; ".join(issues), path=str(path))
    metadata = manifest.get("metadata") or {}
    model_id = str(metadata.get("id") or "")
    version = str(metadata.get("version") or "")
    if not model_id or not version:
        raise RegistryError("manifest metadata.id/version is missing", path=str(path))
    card_path = path / CARD_NAME
    card = ""
    if not card_path.is_file():
        issues.append(f"missing {CARD_NAME} in {path}")
    else:
        card = _read_text(card_path, label="model card")
        issues.extend(card_issues(card))
    report_path = path / REPORT_NAME
    report: dict[str, Any] = {}
    report_warnings: list[str] = []
    evidence_hashes: dict[str, str] = {}
    if not report_path.is_file():
        issues.append(f"missing {REPORT_NAME} in {path}")
    else:
        report = _read_report(report_path)
        report_issues, report_warnings, evidence_hashes = _report_issues(
            report, manifest, base=path
        )
        issues.extend(report_issues)
    license_errors, license_warnings = license_issues(
        str((manifest.get("spec") or {}).get("license", {}).get("name", "")),
        profile=profile,
        allow_unknown=allow_unknown,
        policy=policy,
    )
    issues.extend(license_errors)
    warnings = sorted(set(report_warnings + license_warnings))
    hashes = {
        "manifest": file_hash(manifest_path),
        "card": file_hash(card_path) if card_path.is_file() else sha256_text(""),
        "eval_report": file_hash(report_path) if report_path.is_file() else sha256_text(""),
    }
    if "weights" in evidence_hashes:
        hashes["weights"] = evidence_hashes["weights"]
    return RegistryEntry(
        model_id=model_id,
        version=version,
        path=path,
        manifest=manifest,
        card=card,
        report=report,
        hashes=hashes,
        errors=issues,
        warnings=warnings,
    )


def verify_eval_report(
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    base: str | Path | None = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Public wrapper for report/evidence validation used by the training pipeline."""
    return _report_issues(report, manifest, base=Path(base) if base is not None else None)


# ---------------------------------------------------------------------------------------
# Loading a contribution
# ---------------------------------------------------------------------------------------


def load_contribution(
    source: str | Path,
    *,
    manifest_path: str | Path | None = None,
    card_path: str | Path | None = None,
    report_path: str | Path | None = None,
    profile: Any | None = None,
    allow_unknown: bool = False,
    policy: str = "distributable",
) -> Contribution:
    """Load a contribution directory (or manifest file) and verify it before copying."""
    raw = Path(source)
    manifest_file: Path | None
    if raw.is_file():
        base = raw.parent
        manifest_file = Path(manifest_path) if manifest_path is not None else raw
    else:
        base = raw
        manifest_file = (
            Path(manifest_path)
            if manifest_path is not None
            else next((base / name for name in MANIFEST_NAMES if (base / name).is_file()), None)
        )
        if manifest_file is None:
            raise RegistryError(
                f"no manifest.yaml, manifest.yml, model.yaml, or model.yml in {base}",
                path=str(base),
            )
    card_file = Path(card_path) if card_path is not None else base / CARD_NAME
    report_file = Path(report_path) if report_path is not None else base / REPORT_NAME
    for required, label in (
        (manifest_file, "manifest"),
        (card_file, "card"),
        (report_file, "report"),
    ):
        if not required.is_file():
            raise RegistryError(f"missing {label}: {required}", path=str(required))
    # Build a temporary entry view so we get one code path for hashes and errors.
    manifest, issues = _load_manifest(manifest_file)
    if manifest is None or issues:
        raise RegistryError("invalid contribution manifest: " + "; ".join(issues), path=str(base))
    card = _read_text(card_file, label="model card")
    report = _read_report(report_file)
    report_issues, report_warnings, evidence_hashes = _report_issues(report, manifest, base=base)
    issues = report_issues + card_issues(card)
    license_errors, license_warnings = license_issues(
        str((manifest.get("spec") or {}).get("license", {}).get("name", "")),
        profile=profile,
        allow_unknown=allow_unknown,
        policy=policy,
    )
    issues.extend(license_errors)
    warnings = sorted(set(report_warnings + license_warnings))
    hashes = {
        "manifest": file_hash(manifest_file),
        "card": file_hash(card_file),
        "eval_report": file_hash(report_file),
    }
    if "weights" in evidence_hashes:
        hashes["weights"] = evidence_hashes["weights"]
    entry = RegistryEntry(
        model_id=str(manifest["metadata"]["id"]),
        version=str(manifest["metadata"]["version"]),
        path=base,
        manifest=manifest,
        card=card,
        report=report,
        hashes=hashes,
        errors=issues,
        warnings=warnings,
    )
    if issues:
        raise RegistryVerificationError(
            "contribution failed verification: " + "; ".join(issues[:8]),
            path=str(base),
            details={"errors": issues},
        )
    files: dict[str, bytes] = {
        MANIFEST_NAMES[0]: manifest_file.read_bytes(),
        CARD_NAME: card_file.read_bytes(),
        REPORT_NAME: report_file.read_bytes(),
    }
    for level in ("node", "graph"):
        evidence = report.get("evidence", {}).get(level, {})
        for kind in ("report", "run_manifest"):
            relative = evidence.get(kind, {}).get("path")
            if not isinstance(relative, str):
                continue
            absolute = (base / relative).resolve()
            files[relative] = absolute.read_bytes()
    weights = report.get("weights")
    if isinstance(weights, dict) and isinstance(weights.get("path"), str):
        absolute = (base / weights["path"]).resolve()
        if absolute.is_file():
            files[weights["path"]] = absolute.read_bytes()
    return Contribution(source=base, entry=entry, files=files)


# ---------------------------------------------------------------------------------------
# Scanning, index, and verification
# ---------------------------------------------------------------------------------------


def iter_entry_directories(root: str | Path) -> list[Path]:
    models = Path(root) / MODELS_DIR
    if not models.is_dir():
        return []
    found: list[Path] = []
    for id_dir in sorted(models.iterdir()):
        if not id_dir.is_dir() or id_dir.name.startswith("."):
            continue
        for version_dir in sorted(id_dir.iterdir()):
            if not version_dir.is_dir() or version_dir.name.startswith("."):
                continue
            if any((version_dir / name).is_file() for name in MANIFEST_NAMES):
                found.append(version_dir)
    return sorted(
        found,
        key=lambda path: (path.parent.name, version_key(path.name)),
    )


def _index_payload(entries: Iterable[RegistryEntry]) -> dict[str, Any]:
    models = [entry.to_index_entry() for entry in entries]
    models.sort(key=lambda item: (item["id"], version_key(item["version"])))
    return {"apiVersion": INDEX_API_VERSION, "kind": INDEX_KIND, "models": models}


def build_index(entries: Iterable[RegistryEntry]) -> dict[str, Any]:
    """Build the deterministic index document, including its content hash."""
    payload = _index_payload(entries)
    document = dict(payload)
    document["content_hash"] = sha256_text(canonical_json(payload))
    return document


def index_text(document: dict[str, Any]) -> str:
    return yaml.safe_dump(
        document,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        width=1000,
    )


def load_index(root: str | Path) -> dict[str, Any] | None:
    path = Path(root) / INDEX_FILE
    if not path.is_file():
        return None
    document = load_document(path)
    issues = _schema_messages(document, "registry-index", path=str(path))
    if issues:
        raise RegistryError("invalid registry index: " + "; ".join(issues), path=str(path))
    payload = {key: value for key, value in document.items() if key != "content_hash"}
    expected = sha256_text(canonical_json(payload))
    if document.get("content_hash") != expected:
        raise RegistryError(
            f"registry index content hash mismatch: {document.get('content_hash')} != {expected}",
            path=str(path),
        )
    return document


def verify_registry(
    root: str | Path | None = None,
    *,
    profile: Any | None = None,
    allow_unknown: bool = False,
    policy: str = "distributable",
    write_index: bool = False,
) -> RegistryVerification:
    """Verify every versioned entry and the deterministic index."""
    location = Path(root) if root is not None else default_registry_root()
    entries: list[RegistryEntry] = []
    errors: list[str] = []
    for directory in iter_entry_directories(location):
        entry = verify_entry(
            directory,
            profile=profile,
            allow_unknown=allow_unknown,
            policy=policy,
        )
        entries.append(entry)
        errors.extend(f"{entry.ref}: {error}" for error in entry.errors)
    document = build_index(entries)
    index_path = location / INDEX_FILE
    existing = load_index(location)
    if existing is None:
        index_current = False
        errors.append(
            f"{INDEX_FILE} is missing; run `mg registry verify --write-index` to create it"
        )
    else:
        index_current = canonical_json(existing) == canonical_json(document)
        if not index_current:
            errors.append(
                f"{INDEX_FILE} is stale or was edited; run `mg registry verify --write-index`"
            )
    warnings = [f"{entry.ref}: {warning}" for entry in entries for warning in entry.warnings]
    result = RegistryVerification(
        root=location,
        entries=entries,
        index_document=document,
        index_path=index_path,
        index_current=index_current,
        errors=errors,
        warnings=warnings,
    )
    if write_index:
        if any(entry.errors for entry in entries):
            raise RegistryVerificationError(
                "refusing to write an index while an entry fails verification: "
                + "; ".join(error for entry in entries for error in entry.errors[:3]),
                path=str(location),
                details={"errors": errors},
            )
        index_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(index_path, index_text(document))
        result = replace(
            result,
            index_current=True,
            errors=[error for error in errors if not error.startswith(f"{INDEX_FILE} is")],
        )
    return result


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary)
        raise


# ---------------------------------------------------------------------------------------
# Add, list, inspect
# ---------------------------------------------------------------------------------------


def add_contribution(
    root: str | Path | None,
    source: str | Path,
    *,
    manifest_path: str | Path | None = None,
    card_path: str | Path | None = None,
    report_path: str | Path | None = None,
    profile: Any | None = None,
    allow_unknown: bool = False,
    policy: str = "distributable",
) -> tuple[RegistryEntry, bool]:
    """Copy a verified contribution into the registry.

    Returns ``(entry, changed)``. Adding identical bytes is idempotent; adding the same
    id@version with different bytes raises :class:`RegistryConflictError`.
    """
    location = Path(root) if root is not None else default_registry_root()
    contribution = load_contribution(
        source,
        manifest_path=manifest_path,
        card_path=card_path,
        report_path=report_path,
        profile=profile,
        allow_unknown=allow_unknown,
        policy=policy,
    )
    entry = contribution.entry
    destination = entry_directory(location, entry.model_id, entry.version)
    changed = True
    if destination.exists():
        changed = False
        existing = verify_entry(
            destination,
            profile=profile,
            allow_unknown=allow_unknown,
            policy=policy,
        )
        if existing.errors:
            raise RegistryError(
                f"registry entry {entry.ref} already exists and is invalid: "
                + "; ".join(existing.errors[:5]),
                path=str(destination),
            )
        for relative, content in contribution.files.items():
            current = destination / relative
            if not current.is_file() or current.read_bytes() != content:
                changed = True
                break
        if changed:
            raise RegistryConflictError(
                f"{entry.ref} already exists with different content; a published version is "
                "immutable. Bump the version or remove the local entry deliberately.",
                path=str(destination),
            )
        entry = existing
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{entry.version}.", dir=str(destination.parent)))
        try:
            for relative, content in contribution.files.items():
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        entry = replace(entry, path=destination)
    verification = verify_registry(
        location,
        profile=profile,
        allow_unknown=allow_unknown,
        policy=policy,
    )
    _atomic_write_text(verification.index_path, index_text(verification.index_document))
    return entry, changed


def list_entries(
    root: str | Path | None = None,
    *,
    profile: Any | None = None,
    allow_unknown: bool = False,
    policy: str = "distributable",
) -> list[RegistryEntry]:
    location = Path(root) if root is not None else default_registry_root()
    return [
        verify_entry(
            directory,
            profile=profile,
            allow_unknown=allow_unknown,
            policy=policy,
        )
        for directory in iter_entry_directories(location)
    ]


def inspect_entry(
    root: str | Path | None,
    ref: str,
    *,
    profile: Any | None = None,
    allow_unknown: bool = False,
    policy: str = "distributable",
) -> RegistryEntry:
    location = Path(root) if root is not None else default_registry_root()
    model_id, version = _parse_ref(ref)
    _validate_ref(model_id, version)
    if version is None:
        candidates = [
            candidate
            for candidate in iter_entry_directories(location)
            if candidate.parent.name == model_id
        ]
        if not candidates:
            raise RegistryError(f"model {model_id!r} is not in {location}", path=str(location))
        directory = sorted(candidates, key=lambda path: version_key(path.name))[-1]
    else:
        directory = entry_directory(location, model_id, version)
    if not directory.is_dir():
        raise RegistryError(f"model {ref!r} is not in {location}", path=str(directory))
    return verify_entry(
        directory,
        profile=profile,
        allow_unknown=allow_unknown,
        policy=policy,
    )
