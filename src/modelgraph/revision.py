"""Change sets, impact analysis, and immutable deployment revisions.

This module is the v0 implementation of [ADR-0013](../../docs/adr/0013-immutable-deployment-revisions.md)
and ``docs/iteration.md`` sections 2–5. It deliberately reuses the existing compiler, registry,
run-manifest, and evaluation report contracts:

* a deployment revision id is :func:`modelgraph.runtime.manifest.deployment_revision` over the
  compiled graph hash, resolved model hashes, target, renderer/parser, and compiler version;
* a candidate change is compiled with the normal compiler, so graph/model hashes match the ones
  a run manifest records;
* evidence is a normal evaluation ``report.json`` validated with the eval-report contract and
  its own gate result.

Nothing here executes a model. Impact analysis is static and offline: it loads manifests,
hashes, profiles, and eval reports only.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from modelgraph.compiler import compile_graph
from modelgraph.compiler.ir import CompiledGraph
from modelgraph.contracts.graph import GraphDocument, NodeKind
from modelgraph.contracts.loader import load_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.errors import ModelGraphError
from modelgraph.evaluation.comparison import load_report
from modelgraph.evaluation.report import validate_report
from modelgraph.evaluation.safety import SAFETY_LEVEL
from modelgraph.paths import project_root
from modelgraph.pruning import Profile, load_profile
from modelgraph.registry import Registry, default_registry
from modelgraph.runtime.manifest import deployment_revision
from modelgraph.util import canonical_json, new_id, sha256_text, utc_now

CHANGE_SET_SCHEMA = "modelgraph/change-set/v0"
IMPACT_SCHEMA = "modelgraph/impact-report/v0"
REVISION_SCHEMA = "modelgraph/revision/v0"
ACTIVE_SCHEMA = "modelgraph/active-revision/v0"

CHANGE_STATUSES = (
    "draft",
    "evaluated",
    "compiled",
    "promotable",
    "promoted",
    "rejected",
)
CHANGE_CLASSES = (
    "initial",
    "prompt",
    "adapter_swap",
    "model_swap",
    "base_swap",
    "router_swap",
    "topology",
    "target",
    "student",
)
TARGETS = ("portable", "resident", "fused", "compiled", "distilled", "auto")
COMPILED_TARGETS = frozenset({"fused", "compiled", "distilled"})

STATUS_DRAFT = "draft"
STATUS_EVALUATED = "evaluated"
STATUS_COMPILED = "compiled"
STATUS_PROMOTABLE = "promotable"
STATUS_PROMOTED = "promoted"
STATUS_REJECTED = "rejected"

DEFAULT_GRACE_DAYS = 7.0
REVISION_DIRNAME = "revisions"
CHANGE_DIRNAME = "changes"
ACTIVE_FILENAME = "active.json"
EVENTS_FILENAME = "events.jsonl"

_REVISION_ID_RE = re.compile(r"^rev_[A-Za-z0-9_-]{1,96}$")
_CHANGE_ID_RE = re.compile(r"^chg_[A-Za-z0-9_-]{1,96}$")

# The promotion mapping lives here, in one place, so ``mg change explain`` and
# ``mg promote`` cannot disagree about what evidence a change class needs.
CHANGE_REQUIRED_EVIDENCE: dict[str, tuple[str, ...]] = {
    "initial": (),
    "prompt": ("node",),
    "adapter_swap": ("node",),
    "model_swap": ("node", "e2e"),
    "base_swap": ("node", "routing", "e2e", "safety"),
    "router_swap": ("routing", "e2e"),
    "topology": ("e2e",),
    "target": ("e2e", "safety"),
    "student": ("e2e", "safety"),
}

CHANGE_CLASS_DESCRIPTIONS = {
    "initial": "First deployment revision; no prior active revision to compare against.",
    "prompt": "A node instruction or rendered prompt changed.",
    "adapter_swap": "A model manifest/weights revision changed without changing the base.",
    "model_swap": "A node points at a different model with the same base family/tokenizer.",
    "base_swap": "A node changes base model or tokenizer; this is a migration, not a hot swap.",
    "router_swap": "Router configuration, routing card, or route menu changed.",
    "topology": "Nodes, edges, entry/exit, or artifact contract changed.",
    "target": "The deployment target changed (for example portable to fused).",
    "student": "A distilled student or its source graph changed.",
}


class RevisionError(ModelGraphError):
    """Base error for change sets and revisions."""

    code = "MG-6000"


class RevisionStoreError(RevisionError):
    code = "MG-6001"


class ChangeSetError(RevisionError):
    code = "MG-6002"


class CompatibilityError(RevisionError):
    code = "MG-6003"


class EvidenceError(RevisionError):
    code = "MG-6004"


class RetentionError(RevisionError):
    code = "MG-6005"


# --------------------------------------------------------------------------------------
# Small filesystem and identity helpers
# --------------------------------------------------------------------------------------


def default_store_dir() -> Path:
    """Default local revision store (ignored by git, like runs and caches)."""
    override = os.environ.get("MG_REVISION_STORE")
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd() / ".mg" / "revisions"


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with temp-file + fsync + atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RevisionStoreError(f"file not found: {path}", path=str(path)) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RevisionStoreError(f"cannot read {path}: {exc}", path=str(path)) from exc
    if not isinstance(value, dict):
        raise RevisionStoreError(f"{path} must contain a JSON object", path=str(path))
    return cast(dict[str, Any], value)


def _dump_yaml(payload: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)


def _validate_revision_id(revision_id: str) -> str:
    if not _REVISION_ID_RE.match(revision_id):
        raise RevisionStoreError(
            f"invalid revision id {revision_id!r}; expected rev_ followed by 1-96 "
            "letters, digits, hyphen, or underscore",
            path=revision_id,
        )
    return revision_id


def _validate_change_id(change_id: str) -> str:
    if not _CHANGE_ID_RE.match(change_id):
        raise ChangeSetError(
            f"invalid change id {change_id!r}; expected chg_ followed by 1-96 "
            "letters, digits, hyphen, or underscore",
            path=change_id,
        )
    return change_id


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise RevisionStoreError(f"invalid timestamp {value!r}") from exc
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _age_days(value: str, *, now: datetime | None = None) -> float:
    current = now or datetime.now(UTC)
    return max(0.0, (current - _parse_time(value)).total_seconds() / 86_400.0)


def _hash_document(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


def _iter_nodes(graph: CompiledGraph) -> Iterable[Any]:
    yield from graph.nodes.values()
    for subgraph in graph.subgraphs.values():
        yield from _iter_nodes(subgraph)


def _collect_models(graph: CompiledGraph) -> dict[str, ModelManifest]:
    models: dict[str, ModelManifest] = {}
    for node in _iter_nodes(graph):
        if node.model is not None:
            models.setdefault(node.model.metadata.id, node.model)
    return models


def _model_ref_and_hash(node: Any) -> tuple[str | None, str | None]:
    return node.model_ref, node.model_hash


def derive_model_identity(manifest: ModelManifest) -> dict[str, str]:
    """Derive the best available base/tokenizer identity for one manifest.

    The v0 model manifest has no dedicated ``base_model`` or ``tokenizer`` fields. Labels are
    the explicit escape hatch; without them, identity is derived from ``spec.provider`` and
    ``spec.model``. A derived tokenizer match is conservative evidence, not proof, and the
    impact report says so.
    """
    labels = manifest.metadata.labels or {}
    explicit_base = labels.get("base_model") or labels.get("base")
    explicit_tokenizer = labels.get("tokenizer")
    model_name = manifest.spec.model
    family = model_name.split(":", 1)[0]
    return {
        "model": model_name,
        "family": family,
        "base": explicit_base or model_name,
        "base_source": "metadata.labels.base_model" if explicit_base else "derived-from-spec.model",
        "tokenizer": explicit_tokenizer or family,
        "tokenizer_source": "metadata.labels.tokenizer"
        if explicit_tokenizer
        else "derived-from-spec.model-family",
    }


# --------------------------------------------------------------------------------------
# Revision store
# --------------------------------------------------------------------------------------


class RevisionStore:
    """A local, content-addressed store of immutable deployment revisions."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root is not None else default_store_dir()
        self.revisions_dir.mkdir(parents=True, exist_ok=True)
        self.changes_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def open(cls, root: str | Path | None = None) -> RevisionStore:
        return cls(root)

    @property
    def revisions_dir(self) -> Path:
        return self.root / REVISION_DIRNAME

    @property
    def changes_dir(self) -> Path:
        return self.root / CHANGE_DIRNAME

    @property
    def active_path(self) -> Path:
        return self.root / ACTIVE_FILENAME

    @property
    def events_path(self) -> Path:
        return self.root / EVENTS_FILENAME

    def revision_dir(self, revision_id: str) -> Path:
        return self.revisions_dir / _validate_revision_id(revision_id)

    def change_dir(self, change_id: str) -> Path:
        return self.changes_dir / _validate_change_id(change_id)

    # -- revisions -----------------------------------------------------------------

    def create_revision(
        self,
        graph: CompiledGraph,
        *,
        graph_source: str | Path,
        target: str = "portable",
        bootstrap: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        """Persist an immutable revision; return ``(manifest, created)``.

        The revision directory is a self-contained artifact set: the graph source, any
        subgraph sources, the resolved model manifests, the canonical IR, and the revision
        manifest. It is built in a temporary sibling directory and renamed into place, so a
        reader never sees a partial revision.
        """
        if target not in TARGETS:
            raise ChangeSetError(f"unknown target {target!r}; expected one of {', '.join(TARGETS)}")
        revision_id = deployment_revision(graph, target=target)
        final = self.revision_dir(revision_id)
        if (final / "revision.json").is_file():
            return self.load_revision(revision_id), False

        source = Path(graph_source)
        if not source.is_file():
            raise RevisionStoreError(f"graph source not found: {source}", path=str(source))

        staging = Path(
            tempfile.mkdtemp(prefix=f".{revision_id}.", suffix=".tmp", dir=self.revisions_dir)
        )
        try:
            (staging / "models").mkdir()
            (staging / "graphs").mkdir()
            shutil.copyfile(source, staging / "graph.yaml")
            for ref, subgraph in sorted(graph.subgraphs.items()):
                sub_source = Path(subgraph.source)
                if sub_source.is_file():
                    name = ref.replace("@", "-").replace("/", "-")
                    shutil.copyfile(sub_source, staging / "graphs" / f"{name}.graph.yaml")
            for model_id, model_manifest in sorted(_collect_models(graph).items()):
                (staging / "models" / f"{model_id}.yaml").write_text(
                    _dump_yaml(model_manifest.to_dict()), encoding="utf-8"
                )
            _write_json_atomic(staging / "ir.json", graph.to_dict())
            manifest = {
                "schema": REVISION_SCHEMA,
                "revision_id": revision_id,
                "created_at": utc_now(),
                "target": target,
                "bootstrap": bootstrap,
                "graph": {
                    "id": graph.id,
                    "version": graph.version,
                    "hash": graph.graph_hash,
                    # Reference the stored immutable copy, not the caller's build/staging
                    # path: a swap candidate is compiled from a temporary file that is
                    # removed after this method returns.
                    "source": str(final / "graph.yaml"),
                    "original_source": str(source),
                },
                "models": _revision_model_records(graph),
                "components": {
                    "graph_hash": graph.graph_hash,
                    "model_hashes": _model_hash_map(graph),
                    "target": target,
                    "plan_hash": None,
                    "plan_status": "not_applicable"
                    if target not in COMPILED_TARGETS
                    else "not_built",
                    "student_hash": None,
                    "compiler_version": _compiler_version(),
                    "renderer_version": _renderer_version(),
                    "parser_version": _parser_version(),
                    "ir_version": "modelgraph/ir/v0",
                },
                "artifacts": {
                    "graph": "graph.yaml",
                    "graphs": "graphs/",
                    "models": "models/",
                    "ir": "ir.json",
                },
                "notes": [
                    "A deployment revision is immutable: editing a graph or swapping a model "
                    "creates a new revision.",
                    "Compiled plan and distilled student hashes are null until those targets "
                    "exist; a portable revision does not fake them.",
                ],
            }
            _write_json_atomic(staging / "revision.json", manifest)
            try:
                os.replace(staging, final)
                _fsync_directory(self.revisions_dir)
            except OSError:
                if (final / "revision.json").is_file():
                    return self.load_revision(revision_id), False
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return self.load_revision(revision_id), True

    def load_revision(self, revision_id: str) -> dict[str, Any]:
        path = self.revision_dir(revision_id) / "revision.json"
        if not path.is_file():
            raise RevisionStoreError(
                f"revision {revision_id!r} is not in the store; run `mg status --revisions` "
                "to list retained revisions",
                path=str(path),
            )
        manifest = _read_json(path)
        if manifest.get("schema") != REVISION_SCHEMA:
            raise RevisionStoreError(
                f"revision {revision_id!r} has schema {manifest.get('schema')!r}; "
                f"expected {REVISION_SCHEMA!r}",
                path=str(path),
            )
        return manifest

    def revision_exists(self, revision_id: str) -> bool:
        try:
            return (self.revision_dir(revision_id) / "revision.json").is_file()
        except RevisionStoreError:
            return False

    def compile_revision(self, revision_id: str) -> CompiledGraph:
        """Recompile a stored revision and verify its graph hash."""
        manifest = self.load_revision(revision_id)
        directory = self.revision_dir(revision_id)
        registry = Registry.from_paths([directory / "models"], [directory / "graphs", directory])
        result = compile_graph(
            directory / "graph.yaml",
            registry=registry,
            strict_models=True,
            graph_paths=[directory / "graphs", directory],
        )
        if not result.ok or result.graph is None:
            raise RevisionStoreError(
                f"revision {revision_id!r} cannot be reconstructed: " + "; ".join(result.errors),
                path=str(directory / "graph.yaml"),
            )
        expected = str(manifest.get("graph", {}).get("hash") or "")
        if result.graph.graph_hash != expected:
            raise RevisionStoreError(
                f"revision {revision_id!r} is corrupt: stored graph hash {expected!r} does not "
                f"match recompiled hash {result.graph.graph_hash!r}",
                path=str(directory / "graph.yaml"),
            )
        return result.graph

    def list_revisions(self) -> list[dict[str, Any]]:
        revisions: list[dict[str, Any]] = []
        if not self.revisions_dir.exists():
            return revisions
        for directory in self.revisions_dir.iterdir():
            path = directory / "revision.json"
            if directory.is_dir() and path.is_file():
                try:
                    revisions.append(_read_json(path))
                except RevisionStoreError:
                    continue
        revisions.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return revisions

    def active_pointer(self) -> dict[str, Any] | None:
        if not self.active_path.is_file():
            return None
        return _read_json(self.active_path)

    def active_revision_id(self) -> str | None:
        pointer = self.active_pointer()
        value = pointer.get("revision_id") if pointer else None
        return str(value) if value else None

    def active_revision(self) -> dict[str, Any] | None:
        revision_id = self.active_revision_id()
        if revision_id is None:
            return None
        return self.load_revision(revision_id)

    def previous_revisions(self, limit: int = 10) -> list[str]:
        active = self.active_revision_id()
        found: list[str] = []
        for event in reversed(self.events()):
            previous = event.get("previous_revision")
            if event.get("event") in {"promote", "rollback"} and previous:
                text = str(previous)
                if text != active and text not in found:
                    found.append(text)
            if len(found) >= limit:
                break
        return found

    def events(self) -> list[dict[str, Any]]:
        if not self.events_path.is_file():
            return []
        events: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(cast(dict[str, Any], value))
        return events

    def _append_event(self, event: Mapping[str, Any]) -> None:
        record = {"ts": utc_now(), **dict(event)}
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def promote(
        self,
        revision_id: str,
        *,
        reason: str = "manual promotion",
        evidence: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """Atomically switch the active pointer to ``revision_id``."""
        manifest = self.load_revision(revision_id)
        current = self.active_revision_id()
        if current == revision_id:
            raise ChangeSetError(f"revision {revision_id!r} is already active")
        pointer = {
            "schema": ACTIVE_SCHEMA,
            "revision_id": revision_id,
            "previous_revision": current,
            "updated_at": utc_now(),
            "target": manifest.get("target"),
            "graph": manifest.get("graph", {}).get("id"),
            "reason": reason,
        }
        _write_json_atomic(self.active_path, pointer)
        self._append_event(
            {
                "event": "promote",
                "revision": revision_id,
                "previous_revision": current,
                "reason": reason,
                "evidence": [str(item.get("path", "")) for item in evidence],
            }
        )
        return pointer

    def rollback(
        self,
        revision_id: str,
        *,
        reason: str = "manual rollback",
    ) -> dict[str, Any]:
        """Atomically point active back at a retained revision."""
        manifest = self.load_revision(revision_id)
        current = self.active_revision_id()
        if current == revision_id:
            raise ChangeSetError(f"revision {revision_id!r} is already active")
        pointer = {
            "schema": ACTIVE_SCHEMA,
            "revision_id": revision_id,
            "previous_revision": current,
            "updated_at": utc_now(),
            "target": manifest.get("target"),
            "graph": manifest.get("graph", {}).get("id"),
            "reason": reason,
        }
        _write_json_atomic(self.active_path, pointer)
        self._append_event(
            {
                "event": "rollback",
                "revision": revision_id,
                "previous_revision": current,
                "reason": reason,
            }
        )
        return pointer

    def delete_revision(
        self,
        revision_id: str,
        *,
        now: datetime | None = None,
        grace_days: float = DEFAULT_GRACE_DAYS,
    ) -> dict[str, Any]:
        """Delete one revision if it is outside the grace period and not referenced."""
        if now is not None and now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        manifest = self.load_revision(revision_id)
        current = self.active_revision_id()
        if current == revision_id:
            raise RetentionError(
                f"refusing to delete active revision {revision_id!r}; roll back to another "
                "revision and leave this one outside the grace period first"
            )
        created = str(manifest.get("created_at", ""))
        age = _age_days(created, now=now)
        if age < grace_days:
            raise RetentionError(
                f"refusing to delete revision {revision_id!r}: it is {age:.2f} days old and "
                f"the grace period is {grace_days:g} days. Retrieval for rollback is the point "
                "of retaining revisions; wait or move the clock only in an explicit test."
            )
        open_changes = [
            change
            for change in self.list_changes()
            if revision_id in {change.get("base_revision"), change.get("candidate_revision")}
            and change.get("status") not in {STATUS_PROMOTED, STATUS_REJECTED}
        ]
        if open_changes:
            ids = ", ".join(str(change.get("change_id")) for change in open_changes)
            raise RetentionError(
                f"refusing to delete revision {revision_id!r}: open change set(s) {ids} still "
                "reference it"
            )
        shutil.rmtree(self.revision_dir(revision_id))
        _fsync_directory(self.revisions_dir)
        self._append_event({"event": "delete", "revision": revision_id, "age_days": round(age, 3)})
        return {"deleted": revision_id, "age_days": round(age, 3)}

    def gc(
        self,
        *,
        dry_run: bool = True,
        grace_days: float = DEFAULT_GRACE_DAYS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Report/delete unreferenced revisions outside the grace period."""
        if now is not None and now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        deleted: list[str] = []
        blocked: list[dict[str, Any]] = []
        for manifest in self.list_revisions():
            revision_id = str(manifest.get("revision_id"))
            age = _age_days(str(manifest.get("created_at", "")), now=now)
            if revision_id == self.active_revision_id():
                blocked.append({"revision": revision_id, "reason": "active"})
                continue
            if age < grace_days:
                blocked.append(
                    {"revision": revision_id, "reason": f"inside grace period ({age:.2f} days)"}
                )
                continue
            open_changes = [
                change
                for change in self.list_changes()
                if revision_id in {change.get("base_revision"), change.get("candidate_revision")}
                and change.get("status") not in {STATUS_PROMOTED, STATUS_REJECTED}
            ]
            if open_changes:
                blocked.append(
                    {
                        "revision": revision_id,
                        "reason": "referenced by open change "
                        + ", ".join(str(item.get("change_id")) for item in open_changes),
                    }
                )
                continue
            if dry_run:
                deleted.append(revision_id)
            else:
                self.delete_revision(revision_id, now=now, grace_days=grace_days)
                deleted.append(revision_id)
        return {
            "schema": "modelgraph/revision-gc/v0",
            "store": str(self.root),
            "dry_run": dry_run,
            "grace_days": grace_days,
            "deleted": deleted,
            "blocked": blocked,
        }

    # -- change sets ---------------------------------------------------------------

    def save_change(self, change: dict[str, Any]) -> dict[str, Any]:
        change["updated_at"] = utc_now()
        _write_json_atomic(self.change_dir(str(change["change_id"])) / "change.json", change)
        return change

    def load_change(self, change_id: str) -> dict[str, Any]:
        path = self.change_dir(change_id) / "change.json"
        if not path.is_file():
            raise ChangeSetError(
                f"change set {change_id!r} is not in the store; run `mg change list`",
                path=str(path),
            )
        change = _read_json(path)
        if change.get("schema") != CHANGE_SET_SCHEMA:
            raise ChangeSetError(
                f"change set {change_id!r} has schema {change.get('schema')!r}; "
                f"expected {CHANGE_SET_SCHEMA!r}",
                path=str(path),
            )
        return change

    def list_changes(self) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        if not self.changes_dir.exists():
            return changes
        for directory in self.changes_dir.iterdir():
            path = directory / "change.json"
            if directory.is_dir() and path.is_file():
                try:
                    changes.append(_read_json(path))
                except RevisionStoreError:
                    continue
        changes.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return changes

    def find_change_for_revision(self, revision_id: str) -> dict[str, Any] | None:
        for change in self.list_changes():
            if change.get("candidate_revision") == revision_id:
                return change
        return None

    def load_impact(self, change_id: str) -> dict[str, Any]:
        path = self.change_dir(change_id) / "impact.json"
        if not path.is_file():
            raise ChangeSetError(
                f"change set {change_id!r} has no impact report; recreate the change",
                path=str(path),
            )
        return _read_json(path)

    def create_change(self, change: Mapping[str, Any], impact: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(change)
        record.setdefault("schema", CHANGE_SET_SCHEMA)
        record.setdefault("created_at", utc_now())
        record["updated_at"] = record["created_at"]
        change_id = str(record["change_id"])
        directory = self.change_dir(change_id)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(directory / "impact.json", impact)
        _write_json_atomic(directory / "change.json", record)
        return record


def _compiler_version() -> str:
    from modelgraph import __version__

    return __version__


def _renderer_version() -> str:
    from modelgraph.version import RENDERER_VERSION

    return RENDERER_VERSION


def _parser_version() -> str:
    from modelgraph.version import PARSER_VERSION

    return PARSER_VERSION


def _revision_model_records(graph: CompiledGraph) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for node in _iter_nodes(graph):
        if node.model is None:
            continue
        identity = derive_model_identity(node.model)
        records.append(
            {
                "node": node.id,
                "ref": node.model_ref,
                "version": node.model.metadata.version,
                "hash": node.model_hash,
                "provider": node.model.spec.provider.value,
                "model": node.model.spec.model,
                "base": identity["base"],
                "base_source": identity["base_source"],
                "tokenizer": identity["tokenizer"],
                "tokenizer_source": identity["tokenizer_source"],
                "vram_gb": node.model.spec.resources.vram_gb,
                "license": node.model.spec.license.name,
            }
        )
    records.sort(key=lambda item: (str(item["node"]), str(item["model"])))
    return records


def _model_hash_map(graph: CompiledGraph) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for node in _iter_nodes(graph):
        if node.model_ref:
            result[node.id] = node.model_hash
    return dict(sorted(result.items()))


# --------------------------------------------------------------------------------------
# Graph diff
# --------------------------------------------------------------------------------------


def _node_view(node: Any) -> dict[str, Any]:
    return {
        "id": node.id,
        "kind": node.kind.value,
        "description": node.description,
        "model_ref": node.model_ref,
        "model_hash": node.model_hash,
        "instruction": node.instruction,
        "limits": node.limits.model_dump(mode="json", exclude_none=True) if node.limits else None,
        "reads": list(node.reads),
        "writes": [write.model_dump(mode="json", exclude_none=True) for write in node.writes],
        "routing": node.routing.model_dump(mode="json"),
        "routing_card": node.routing_card,
        "graph_ref": node.graph_ref,
        "input_map": dict(node.input_map),
        "output_map": dict(node.output_map),
        "strategy": node.strategy.value,
    }


def _edge_view(edge: Any) -> dict[str, Any]:
    return cast(dict[str, Any], edge.to_dict())


def _edge_key(edge: Any) -> tuple[str, str, str | None, str | None, int]:
    return (edge.from_, edge.to, edge.when, edge.artifact, edge.priority)


def diff_graphs(before: CompiledGraph, after: CompiledGraph) -> dict[str, Any]:
    """Structural diff of two compiled graphs (topology, prompts, bindings, policy)."""
    before_nodes = set(before.nodes)
    after_nodes = set(after.nodes)
    added = sorted(after_nodes - before_nodes)
    removed = sorted(before_nodes - after_nodes)
    changed: list[dict[str, Any]] = []
    for node_id in sorted(before_nodes & after_nodes):
        old_view = _node_view(before.nodes[node_id])
        new_view = _node_view(after.nodes[node_id])
        fields: dict[str, Any] = {}
        for key in old_view:
            if old_view[key] != new_view[key]:
                fields[key] = {"from": old_view[key], "to": new_view[key]}
        if fields:
            changed.append(
                {
                    "node": node_id,
                    "kind": new_view["kind"],
                    "fields": fields,
                }
            )

    before_edges = {_edge_key(edge): edge for edge in before.edges}
    after_edges = {_edge_key(edge): edge for edge in after.edges}
    edges_added = [
        _edge_view(after_edges[edge_key])
        for edge_key in sorted(set(after_edges) - set(before_edges))
    ]
    edges_removed = [
        _edge_view(before_edges[edge_key])
        for edge_key in sorted(set(before_edges) - set(after_edges))
    ]
    edges_changed: list[dict[str, Any]] = []
    for edge_key in sorted(set(before_edges) & set(after_edges)):
        old_view = _edge_view(before_edges[edge_key])
        new_view = _edge_view(after_edges[edge_key])
        if old_view != new_view:
            edges_changed.append({"from": old_view, "to": new_view})

    policy: dict[str, Any] = {}
    for name in ("entry", "exits", "fallback", "defaults", "inputs", "labels"):
        old_value = _policy_value(before, name)
        new_value = _policy_value(after, name)
        if old_value != new_value:
            policy[name] = {"from": old_value, "to": new_value}

    topology_changed = (
        bool(added or removed or edges_added or edges_removed or edges_changed)
        or before.entry != after.entry
        or before.exits != after.exits
    )

    model_bindings: list[dict[str, Any]] = []
    for node_id in sorted(before_nodes | after_nodes):
        old_node = before.nodes.get(node_id)
        new_node = after.nodes.get(node_id)
        old_ref = old_node.model_ref if old_node else None
        new_ref = new_node.model_ref if new_node else None
        old_hash = old_node.model_hash if old_node else None
        new_hash = new_node.model_hash if new_node else None
        if old_ref != new_ref or old_hash != new_hash:
            model_bindings.append(
                {
                    "node": node_id,
                    "from_ref": old_ref,
                    "to_ref": new_ref,
                    "from_hash": old_hash,
                    "to_hash": new_hash,
                }
            )

    return {
        "schema": "modelgraph/graph-diff/v0",
        "before": {"id": before.id, "version": before.version, "hash": before.graph_hash},
        "after": {"id": after.id, "version": after.version, "hash": after.graph_hash},
        "graph_changed": before.graph_hash != after.graph_hash,
        "topology_changed": topology_changed,
        "nodes_added": [
            {"node": node_id, "view": _node_view(after.nodes[node_id])} for node_id in added
        ],
        "nodes_removed": [
            {"node": node_id, "view": _node_view(before.nodes[node_id])} for node_id in removed
        ],
        "nodes_changed": changed,
        "model_bindings": model_bindings,
        "edges_added": edges_added,
        "edges_removed": edges_removed,
        "edges_changed": edges_changed,
        "policy_changed": policy,
    }


def _policy_value(graph: CompiledGraph, name: str) -> Any:
    if name == "exits":
        return list(graph.exits)
    if name == "inputs":
        return [item.model_dump(mode="json", exclude_none=True) for item in graph.inputs]
    if name == "defaults":
        return graph.defaults.model_dump(mode="json")
    if name == "labels":
        return dict(graph.labels)
    return getattr(graph, name)


# --------------------------------------------------------------------------------------
# Impact analysis
# --------------------------------------------------------------------------------------


def required_evidence(
    change_class: str,
    *,
    routing_affected: bool = False,
    target: str = "portable",
) -> list[str]:
    """Return the required eval levels for one change class.

    This is the single mapping used by ``mg change explain`` and ``mg promote``. A model swap
    only requires routing evidence when the router menu/card/cost/latency actually changed.
    """
    if change_class == "model_swap":
        required = ["node", "e2e"]
        if routing_affected:
            required.insert(1, "routing")
        return required
    if change_class == "base_swap" and target in COMPILED_TARGETS:
        return ["node", "routing", "e2e", "safety"]
    return list(CHANGE_REQUIRED_EVIDENCE.get(change_class, ("e2e",)))


def classify_change(
    before: CompiledGraph | None,
    after: CompiledGraph,
    diff: Mapping[str, Any],
    *,
    base_target: str | None = None,
    target: str = "portable",
) -> str:
    if before is None:
        return "initial"
    if base_target is not None and base_target != target:
        return "target"
    if diff.get("topology_changed"):
        return "topology"
    changed = cast(list[dict[str, Any]], diff.get("nodes_changed") or [])
    bindings = cast(list[dict[str, Any]], diff.get("model_bindings") or [])
    if bindings:
        old_node = before.nodes.get(str(bindings[0].get("node")))
        new_node = after.nodes.get(str(bindings[0].get("node")))
        if old_node is not None and new_node is not None:
            old_model = old_node.model
            new_model = new_node.model
            if old_model is not None and new_model is not None:
                old_id = derive_model_identity(old_model)
                new_id = derive_model_identity(new_model)
                if old_id["base"] != new_id["base"]:
                    return "base_swap"
                if old_id["tokenizer"] != new_id["tokenizer"]:
                    return "base_swap"
                if old_model.spec.model == new_model.spec.model:
                    return "adapter_swap"
        return "model_swap"
    for item in changed:
        fields = cast(dict[str, Any], item.get("fields") or {})
        if any(name in fields for name in ("routing", "routing_card")):
            return "router_swap"
    for item in changed:
        fields = cast(dict[str, Any], item.get("fields") or {})
        if "instruction" in fields:
            return "prompt"
    if changed:
        return "topology"
    return "prompt"


def _binding_changes(before: CompiledGraph, after: CompiledGraph) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for node_id in sorted(set(before.nodes) | set(after.nodes)):
        old_node = before.nodes.get(node_id)
        new_node = after.nodes.get(node_id)
        old_ref, old_hash = _model_ref_and_hash(old_node) if old_node else (None, None)
        new_ref, new_hash = _model_ref_and_hash(new_node) if new_node else (None, None)
        if old_ref is None and new_ref is None:
            continue
        if old_ref == new_ref and old_hash == new_hash:
            continue
        changes.append(
            {
                "node": node_id,
                "from_ref": old_ref,
                "to_ref": new_ref,
                "from_hash": old_hash,
                "to_hash": new_hash,
                "from_model": old_node.model if old_node else None,
                "to_model": new_node.model if new_node else None,
            }
        )
    return changes


def _compatibility(before: CompiledGraph, after: CompiledGraph, *, target: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blocking = False
    migration = False
    for binding in _binding_changes(before, after):
        old_model = cast(ModelManifest | None, binding["from_model"])
        new_model = cast(ModelManifest | None, binding["to_model"])
        node = str(binding["node"])
        check: dict[str, Any] = {
            "node": node,
            "from_ref": binding["from_ref"],
            "to_ref": binding["to_ref"],
            "from_hash": binding["from_hash"],
            "to_hash": binding["to_hash"],
        }
        if old_model is None or new_model is None:
            check["base"] = {
                "from": None,
                "to": new_model.spec.model if new_model else None,
                "status": "unknown",
                "source": "unresolved",
            }
            check["tokenizer"] = {
                "from": None,
                "to": None,
                "status": "unknown",
                "source": "unresolved",
            }
            check["blocking"] = target in COMPILED_TARGETS
            check["message"] = (
                f"node {node!r} adds or removes a model binding; a fused target cannot "
                "validate an unresolved model. Register the manifest and recreate the change."
                if target in COMPILED_TARGETS
                else f"node {node!r} adds or removes a model binding; review the route menu."
            )
            if check["blocking"]:
                blocking = True
            checks.append(check)
            continue
        old_identity = derive_model_identity(old_model)
        new_identity = derive_model_identity(new_model)
        base_status = "same" if old_identity["base"] == new_identity["base"] else "changed"
        tokenizer_status = (
            "same" if old_identity["tokenizer"] == new_identity["tokenizer"] else "changed"
        )
        check["base"] = {
            "from": old_identity["base"],
            "to": new_identity["base"],
            "status": base_status,
            "source": f"{old_identity['base_source']} -> {new_identity['base_source']}",
        }
        check["tokenizer"] = {
            "from": old_identity["tokenizer"],
            "to": new_identity["tokenizer"],
            "status": tokenizer_status,
            "source": (f"{old_identity['tokenizer_source']} -> {new_identity['tokenizer_source']}"),
        }
        check["context_window"] = {
            "from": old_model.spec.context_window,
            "to": new_model.spec.context_window,
            "status": "same"
            if old_model.spec.context_window == new_model.spec.context_window
            else "changed",
        }
        if target in COMPILED_TARGETS and tokenizer_status != "same":
            check["blocking"] = True
            check["message"] = (
                f"tokenizer changed for node {node!r} ({old_identity['tokenizer']} -> "
                f"{new_identity['tokenizer']}); {target!r} shares one tokenizer across the "
                "compiled/fused family. This is a migration, not an adapter swap. Use "
                "--target portable for a cold model swap, or rebuild the model family and "
                "retrain adapters before creating the fused revision."
            )
        elif target in COMPILED_TARGETS and base_status != "same":
            check["blocking"] = True
            check["message"] = (
                f"base model changed for node {node!r} ({old_identity['base']} -> "
                f"{new_identity['base']}); {target!r} shares one base across adapters. Use "
                "--target portable for a cold model swap, or migrate/retrain the adapters and "
                "rebuild the fused family before creating the revision."
            )
        else:
            check["blocking"] = False
            changed = base_status != "same" or tokenizer_status != "same"
            migration = migration or changed
            if changed:
                check["message"] = (
                    f"node {node!r} changes base/tokenizer; treat this as a cold migration, not "
                    "an in-place hot swap."
                )
            else:
                check["message"] = f"node {node!r} is compatible with the same base/tokenizer."
        if check["blocking"]:
            blocking = True
        checks.append(check)

    if not checks:
        status = "compatible"
        message = "No model binding changed."
    elif blocking:
        status = "incompatible"
        message = next(
            (str(check["message"]) for check in checks if check.get("blocking")), "incompatible"
        )
    elif migration:
        status = "migration"
        message = "Base/tokenizer changed; this is a cold migration, not a hot swap."
    else:
        status = "compatible"
        message = "All changed model bindings keep the same base/tokenizer identity."
    return {
        "status": status,
        "blocking": blocking,
        "message": message,
        "checks": checks,
        "source_note": (
            "Model manifests in v0 have no dedicated base_model/tokenizer fields. Explicit "
            "metadata.labels.base_model and metadata.labels.tokenizer are used when present; "
            "otherwise identity is derived from spec.model. Derived matches are conservative "
            "and should be replaced by explicit labels before a shared deployment. No compiled "
            "target exists in this ticket, so incompatible changes are rejected for fused/"
            "compiled/distilled targets rather than built."
        ),
    }


def _affected(
    before: CompiledGraph, after: CompiledGraph, diff: Mapping[str, Any]
) -> dict[str, Any]:
    changed_nodes = {str(item.get("node")) for item in diff.get("nodes_changed", [])}
    changed_nodes.update(str(item.get("node")) for item in diff.get("nodes_added", []))
    changed_nodes.update(str(item.get("node")) for item in diff.get("nodes_removed", []))
    changed_nodes.update(str(item.get("node")) for item in diff.get("model_bindings", []))
    routers: dict[str, set[str]] = {}
    for changed in sorted(changed_nodes):
        node = after.nodes.get(changed)
        if node is not None and node.kind == NodeKind.router:
            routers.setdefault(changed, set()).add("router configuration changed")
        for graph in (after, before):
            candidate = graph.nodes.get(changed)
            if candidate is None:
                continue
            for router in graph.nodes.values():
                if router.kind != NodeKind.router:
                    continue
                for edge in graph.outgoing(router.id):
                    if edge.to != changed:
                        continue
                    if graph is after:
                        routers.setdefault(router.id, set()).add(
                            f"route menu includes changed node {changed!r}"
                        )
                    else:
                        routers.setdefault(router.id, set()).add(
                            f"route menu included removed node {changed!r}"
                        )
    for edge in diff.get("edges_added", []):
        source = str(edge.get("from"))
        node = after.nodes.get(source)
        if node is not None and node.kind == NodeKind.router:
            routers.setdefault(source, set()).add("router edge added")
    for edge in diff.get("edges_removed", []):
        source = str(edge.get("from"))
        node = before.nodes.get(source)
        if node is not None and node.kind == NodeKind.router:
            routers.setdefault(source, set()).add("router edge removed")

    cards: set[str] = set()
    for item in diff.get("nodes_changed", []):
        fields = cast(dict[str, Any], item.get("fields") or {})
        if any(
            name in fields
            for name in ("model_ref", "model_hash", "description", "routing_card", "instruction")
        ):
            cards.add(str(item.get("node")))
    for item in diff.get("nodes_added", []):
        cards.add(str(item.get("node")))
    return {
        "nodes": sorted(changed_nodes),
        "routers": [
            {"router": name, "reasons": sorted(reasons)}
            for name, reasons in sorted(routers.items())
        ],
        "routing_cards": sorted(cards),
        "route_menu_changed": bool(routers),
        "edges_changed": bool(
            diff.get("edges_added") or diff.get("edges_removed") or diff.get("edges_changed")
        ),
    }


def _invalidation(
    name: str,
    status: str,
    reason: str,
    *,
    scope: Sequence[str] = (),
    applicable: bool,
    present: bool = False,
) -> dict[str, Any]:
    return {
        "artifact": name,
        "status": status,
        "applicable": applicable,
        "present": present,
        "reason": reason,
        "scope": list(scope),
    }


def _invalidations(
    before: CompiledGraph,
    after: CompiledGraph,
    *,
    target: str,
    changed_nodes: Sequence[str],
    affected: Mapping[str, Any],
) -> list[dict[str, Any]]:
    scope = sorted(changed_nodes)
    routing_changed = bool(affected.get("route_menu_changed"))
    compiled = target in COMPILED_TARGETS
    results: list[dict[str, Any]] = []
    if not scope and before.graph_hash == after.graph_hash:
        return [
            _invalidation(
                "compiled_plan",
                "unchanged",
                "the compiled graph and model bindings are unchanged",
                applicable=compiled,
            ),
            _invalidation(
                "prefix_cache",
                "unchanged",
                "the compiled graph and model bindings are unchanged",
                applicable=compiled,
            ),
            _invalidation(
                "route_cache",
                "unchanged",
                "routing decisions and menus are unchanged",
                applicable=compiled,
            ),
            _invalidation(
                "distilled_student",
                "unchanged",
                "the source graph is unchanged",
                applicable=False,
            ),
        ]
    if compiled:
        results.append(
            _invalidation(
                "compiled_plan",
                "invalidated",
                "graph/model/prompt hashes changed; the plan is keyed by those hashes. No plan "
                "artifact is built in this ticket, so it must be rebuilt before promotion.",
                scope=scope,
                applicable=True,
            )
        )
        results.append(
            _invalidation(
                "prefix_cache",
                "invalidated",
                "prefix caches are keyed by model, prompt, and plan hashes that changed. A "
                "compiled target must rebuild the affected prefixes before promotion.",
                scope=scope,
                applicable=True,
            )
        )
    else:
        results.append(
            _invalidation(
                "compiled_plan",
                "not_applicable",
                "the portable target has no compiled plan. Reporting it as invalidated would "
                "invent an artifact; the graph/model hashes still changed and a future compiled "
                "target would need a new plan.",
                scope=scope,
                applicable=False,
            )
        )
        results.append(
            _invalidation(
                "prefix_cache",
                "not_applicable",
                "the portable target has no prefix cache. A future compiled target must rebuild "
                "it for the changed nodes.",
                scope=scope,
                applicable=False,
            )
        )
    if compiled and routing_changed:
        results.append(
            _invalidation(
                "route_cache",
                "invalidated",
                "router menu/decision inputs changed; route cache entries for the affected "
                "routers must not be reused.",
                scope=[str(item.get("router")) for item in affected.get("routers", [])],
                applicable=True,
            )
        )
    elif compiled:
        results.append(
            _invalidation(
                "route_cache",
                "unchanged",
                "routing menus and router configuration are unchanged; route cache entries for "
                "untouched routers remain valid.",
                applicable=True,
            )
        )
    else:
        results.append(
            _invalidation(
                "route_cache",
                "not_applicable",
                "the portable target has no route cache. A future compiled/router cache must "
                "not reuse entries across revisions."
                + (" The affected routers changed." if routing_changed else ""),
                scope=[str(item.get("router")) for item in affected.get("routers", [])],
                applicable=False,
            )
        )
    results.append(
        _invalidation(
            "distilled_student",
            "stale",
            "ADR-0013: any source graph change makes a distilled student stale. No student "
            "artifact is tracked in this revision; if one exists, retrain or explicitly keep "
            "serving it with a visible stale badge.",
            applicable=True,
            present=False,
        )
    )
    return results


def _graph_vram(graph: CompiledGraph) -> tuple[float, list[str], bool]:
    models = _collect_models(graph)
    total = 0.0
    unknown: list[str] = []
    for model_id, manifest in sorted(models.items()):
        value = manifest.spec.resources.vram_gb
        if value is None:
            unknown.append(model_id)
        else:
            total += float(value)
    return round(total, 4), unknown, not unknown


def _vram(
    before: CompiledGraph,
    after: CompiledGraph,
    *,
    profile: Profile | None,
) -> dict[str, Any]:
    before_total, before_unknown, before_complete = _graph_vram(before)
    after_total, after_unknown, after_complete = _graph_vram(after)
    admission: dict[str, Any] = {
        "status": "unknown",
        "max_vram_gb": None,
        "required_gb": after_total,
        "headroom_gb": None,
        "reason": "no --profile was supplied; admission cannot be previewed",
    }
    if profile is not None:
        maximum = profile.max_vram_gb
        if maximum is None:
            admission["reason"] = f"profile {profile.name!r} declares no max_vram_gb"
        else:
            headroom = round(maximum - after_total, 4)
            admission = {
                "status": "fits" if headroom >= 0 else "exceeds",
                "profile": profile.name,
                "max_vram_gb": maximum,
                "required_gb": after_total,
                "headroom_gb": headroom,
                "reason": (
                    f"{after_total:g}GB required for the candidate resident set vs "
                    f"{maximum:g}GB in profile {profile.name!r}"
                ),
            }
    return {
        "before_gb": before_total,
        "after_gb": after_total,
        "delta_gb": round(after_total - before_total, 4),
        "complete": before_complete and after_complete,
        "unknown_models": sorted(set(before_unknown) | set(after_unknown)),
        "models": sorted(_collect_models(after)),
        "admission": admission,
    }


def analyse_impact(
    before: CompiledGraph,
    after: CompiledGraph,
    *,
    target: str = "portable",
    base_target: str | None = None,
    base_revision: str | None = None,
    candidate_revision: str | None = None,
    profile: Profile | None = None,
    evidence: Sequence[Mapping[str, Any]] = (),
    store: RevisionStore | None = None,
    grace_days: float = DEFAULT_GRACE_DAYS,
) -> dict[str, Any]:
    """Build a static impact report; this function never loads a model."""
    diff = diff_graphs(before, after)
    affected = _affected(before, after, diff)
    compatibility = _compatibility(before, after, target=target)
    change_class = classify_change(before, after, diff, base_target=base_target, target=target)
    required = required_evidence(
        change_class, routing_affected=bool(affected["route_menu_changed"]), target=target
    )
    coverage = evaluate_coverage(required, evidence)
    invalidations = _invalidations(
        before,
        after,
        target=target,
        changed_nodes=cast(list[str], affected["nodes"]),
        affected=affected,
    )
    vram = _vram(before, after, profile=profile)
    rollback = {
        "target": base_revision,
        "previous": base_revision,
        "grace_days": grace_days,
        "retained": (
            bool(store.revision_exists(base_revision))
            if store is not None and base_revision
            else None
        ),
        "note": (
            "Rollback is a pointer switch to this revision; artifacts stay in the store for "
            "the grace period."
            if base_revision
            else "There is no active revision to roll back to; the first promotion creates one."
        ),
    }
    risks: list[str] = []
    if compatibility["blocking"]:
        risks.append(compatibility["message"])
    if compatibility["status"] == "migration":
        risks.append(
            "Base/tokenizer changed: treat this as a cold migration; do not claim a hot swap."
        )
    if affected["route_menu_changed"]:
        risks.append("Router menus/cards changed; routing evaluation is required before promotion.")
    if vram["admission"]["status"] == "exceeds":
        risks.append(
            f"Candidate needs {vram['after_gb']:g}GB VRAM; profile "
            f"{vram['admission'].get('profile')} allows "
            f"{vram['admission'].get('max_vram_gb')}GB."
        )
    if not vram["complete"]:
        risks.append(
            "Some model manifests have no resources.vram_gb; the VRAM delta is a lower bound."
        )
    impact = {
        "schema": IMPACT_SCHEMA,
        "change_class": change_class,
        "change_class_description": CHANGE_CLASS_DESCRIPTIONS[change_class],
        "target": target,
        "base_target": base_target,
        "base_revision": base_revision,
        "candidate_revision": candidate_revision,
        "graph_diff": diff,
        "compatibility": compatibility,
        "affected": affected,
        "invalidations": invalidations,
        "vram": vram,
        "eval": {
            "required": required,
            "mapping": "CHANGE_REQUIRED_EVIDENCE in modelgraph.revision",
            "coverage": coverage,
        },
        "rollback": rollback,
        "risk": risks,
        "notes": [
            "Impact analysis is conservative: a false stale artifact is cheaper than serving a "
            "wrong one.",
            "No model is loaded and no compiled target is built by this analysis.",
            compatibility["source_note"],
        ],
    }
    return impact


# --------------------------------------------------------------------------------------
# Evidence and coverage
# --------------------------------------------------------------------------------------


def _evidence_level(report: Mapping[str, Any]) -> str:
    suite = report.get("suite") or {}
    suite_id = str(suite.get("id", "")) if isinstance(suite, Mapping) else ""
    labels = suite.get("labels") if isinstance(suite, Mapping) else None
    labelled_safety = isinstance(labels, Mapping) and labels.get("level") == SAFETY_LEVEL
    if report.get("level") == SAFETY_LEVEL or suite_id.startswith("safety/") or labelled_safety:
        return SAFETY_LEVEL
    return str(report.get("level", "unknown"))


def load_evidence(
    path: str | Path,
    *,
    candidate_graph_hash: str,
    candidate_graph_id: str | None = None,
) -> dict[str, Any]:
    """Validate one eval report as promotion evidence for the candidate revision."""
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "report.json"
    record: dict[str, Any] = {
        "path": str(source),
        "valid": False,
        "level": "unknown",
        "report_id": None,
        "suite": None,
        "graph_hash": None,
        "gate_passed": False,
        "cases": 0,
        "sha256": None,
        "errors": [],
    }
    if not source.is_file():
        record["errors"] = [f"report not found: {source}"]
        return record
    try:
        report = load_report(source)
    except ModelGraphError as exc:
        record["errors"] = [exc.message]
        return record
    issues = validate_report(report)
    record["report_id"] = report.get("report_id")
    suite = report.get("suite") or {}
    record["suite"] = suite.get("id") if isinstance(suite, Mapping) else None
    graph = report.get("graph") or {}
    record["graph_hash"] = graph.get("hash") if isinstance(graph, Mapping) else None
    record["level"] = _evidence_level(report)
    gate = report.get("gate") or {}
    record["gate_passed"] = bool(isinstance(gate, Mapping) and gate.get("passed") is True)
    cases = report.get("cases") or []
    record["cases"] = len(cases) if isinstance(cases, list) else 0
    record["sha256"] = _hash_document(report)
    errors: list[str] = list(issues)
    if record["graph_hash"] != candidate_graph_hash:
        errors.append(
            f"report graph hash {record['graph_hash']!r} does not match candidate graph "
            f"{candidate_graph_hash!r}"
        )
    if candidate_graph_id and graph.get("id") != candidate_graph_id:
        errors.append(
            f"report graph id {graph.get('id')!r} does not match candidate graph "
            f"{candidate_graph_id!r}"
        )
    if not record["gate_passed"]:
        errors.append("report gate did not pass; a failing or unmeasured report is not evidence")
    if record["cases"] <= 0:
        errors.append("report has no case results")
    record["errors"] = errors
    record["valid"] = not errors
    return record


def evaluate_coverage(
    required: Sequence[str], evidence: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    valid_by_level: dict[str, list[Mapping[str, Any]]] = {}
    for item in evidence:
        if item.get("valid"):
            valid_by_level.setdefault(str(item.get("level")), []).append(item)
    satisfied = [level for level in required if valid_by_level.get(level)]
    missing = [level for level in required if not valid_by_level.get(level)]
    return {
        "required": list(required),
        "satisfied": satisfied,
        "missing": missing,
        "complete": not missing,
        "provided": [dict(item) for item in evidence],
    }


def attach_evidence(
    store: RevisionStore,
    change: dict[str, Any],
    *,
    candidate_graph_hash: str,
    candidate_graph_id: str | None,
    evidence_paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing = {
        str(item.get("path")): dict(item)
        for item in cast(list[dict[str, Any]], change.get("evidence") or [])
    }
    for path in evidence_paths:
        record = load_evidence(
            path,
            candidate_graph_hash=candidate_graph_hash,
            candidate_graph_id=candidate_graph_id,
        )
        existing[str(record["path"])] = record
    records = list(existing.values())
    impact = store.load_impact(str(change["change_id"]))
    required = cast(list[str], impact.get("eval", {}).get("required") or [])
    coverage = evaluate_coverage(required, records)
    change["evidence"] = records
    change["status"] = status_for_change(change, impact, coverage)
    store.save_change(change)
    return records, coverage


def status_for_change(
    change: Mapping[str, Any], impact: Mapping[str, Any], coverage: Mapping[str, Any]
) -> str:
    if change.get("status") == STATUS_PROMOTED:
        return STATUS_PROMOTED
    compatibility = impact.get("compatibility") or {}
    if isinstance(compatibility, Mapping) and compatibility.get("blocking"):
        return STATUS_REJECTED
    if coverage.get("complete"):
        return STATUS_PROMOTABLE
    if coverage.get("satisfied"):
        return STATUS_EVALUATED
    return STATUS_COMPILED


def refresh_change_status(
    store: RevisionStore, change: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    impact = store.load_impact(str(change["change_id"]))
    required = cast(list[str], impact.get("eval", {}).get("required") or [])
    coverage = evaluate_coverage(required, cast(list[dict[str, Any]], change.get("evidence") or []))
    change["status"] = status_for_change(change, impact, coverage)
    store.save_change(change)
    return impact, coverage


# --------------------------------------------------------------------------------------
# Promotion plans and rollback targets
# --------------------------------------------------------------------------------------


def _require_built_target(revision: Mapping[str, Any], *, action: str) -> None:
    """Refuse to make an unbuilt compiled/fused/distilled target active.

    T-0038 implements the interpreted/resident targets. A revision may describe a compiled
    target for impact analysis, but promoting it without a plan would claim a deployment that
    does not exist; the plan status is recorded honestly and gates the switch.
    """
    target = str(revision.get("target") or "portable")
    if target not in COMPILED_TARGETS:
        return
    components = revision.get("components")
    plan_status = "missing"
    if isinstance(components, Mapping):
        plan_status = str(components.get("plan_status") or "missing")
    if plan_status == "built":
        return
    raise RevisionStoreError(
        f"cannot {action} revision {revision.get('revision_id')!r}: target {target!r} has no "
        f"built plan (plan_status={plan_status!r}). Compiled/fused/distilled targets are not "
        "implemented in v0; use --target portable, or build and record the plan first."
    )


def promotion_plan(
    store: RevisionStore,
    revision_id: str,
    *,
    evidence_paths: Sequence[str | Path] = (),
    persist_evidence: bool = False,
) -> dict[str, Any]:
    """Validate evidence and compatibility for a revision before switching the pointer."""
    revision = store.load_revision(revision_id)
    _require_built_target(revision, action="promote")
    change = store.find_change_for_revision(revision_id)
    active = store.active_revision_id()
    if change is None:
        if not revision.get("bootstrap"):
            raise EvidenceError(
                f"revision {revision_id!r} has no change set; only an explicit initial "
                "`mg promote --graph ...` revision may be promoted without evidence",
                details={"revision": revision_id},
            )
        if active is not None:
            raise EvidenceError(
                f"revision {revision_id!r} is an initial bootstrap revision, but "
                f"{active!r} is already active; create a change set and attach evidence instead "
                "of re-promoting the bootstrap",
                details={"revision": revision_id, "active": active},
            )
        return {
            "allowed": True,
            "revision": revision,
            "change": None,
            "impact": None,
            "coverage": evaluate_coverage([], []),
            "evidence": [],
            "bootstrap": True,
        }
    impact = store.load_impact(str(change["change_id"]))
    compatibility = impact.get("compatibility") or {}
    if isinstance(compatibility, Mapping) and compatibility.get("blocking"):
        raise CompatibilityError(
            str(compatibility.get("message") or "the change is incompatible with the target"),
            details={"change_id": change.get("change_id"), "impact": impact},
        )
    if persist_evidence and evidence_paths:
        records, coverage = attach_evidence(
            store,
            change,
            candidate_graph_hash=str(revision.get("graph", {}).get("hash")),
            candidate_graph_id=str(revision.get("graph", {}).get("id")),
            evidence_paths=evidence_paths,
        )
    else:
        existing = {
            str(item.get("path")): dict(item)
            for item in cast(list[dict[str, Any]], change.get("evidence") or [])
        }
        for path in evidence_paths:
            record = load_evidence(
                path,
                candidate_graph_hash=str(revision.get("graph", {}).get("hash")),
                candidate_graph_id=str(revision.get("graph", {}).get("id")),
            )
            existing[str(record["path"])] = record
        records = list(existing.values())
        required = cast(list[str], impact.get("eval", {}).get("required") or [])
        coverage = evaluate_coverage(required, records)
    if not coverage["complete"]:
        missing = ", ".join(cast(list[str], coverage["missing"]))
        raise EvidenceError(
            f"promotion blocked: missing passing evidence for {missing or 'the change class'}. "
            "Run the required suite and pass the report with --evidence, for example "
            "`mg eval node --suite <suite> --provider mock --json > report.json` then "
            f"`mg promote --revision {revision_id} --evidence report.json`.",
            details={
                "change_id": change.get("change_id"),
                "required": coverage.get("required"),
                "missing": coverage.get("missing"),
                "provided": coverage.get("provided"),
            },
        )
    return {
        "allowed": True,
        "revision": revision,
        "change": change,
        "impact": impact,
        "coverage": coverage,
        "evidence": records,
        "bootstrap": False,
    }


def mark_promoted(
    store: RevisionStore,
    plan: Mapping[str, Any],
    *,
    reason: str = "manual promotion",
) -> dict[str, Any] | None:
    change = plan.get("change")
    if not isinstance(change, dict):
        return None
    impact = cast(dict[str, Any], plan.get("impact") or {})
    coverage = cast(dict[str, Any], plan.get("coverage") or {})
    change["status"] = STATUS_PROMOTED
    change["evidence"] = [
        dict(item) for item in cast(list[dict[str, Any]], plan.get("evidence") or [])
    ]
    change["promotion"] = {
        "promoted_at": utc_now(),
        "revision": plan.get("revision", {}).get("revision_id"),
        "reason": reason,
        "evidence": [str(item.get("path")) for item in plan.get("evidence", [])],
        "coverage": {
            "required": coverage.get("required"),
            "satisfied": coverage.get("satisfied"),
            "complete": coverage.get("complete"),
        },
        "rollback_target": impact.get("rollback", {}).get("target"),
    }
    store.save_change(change)
    return change


# --------------------------------------------------------------------------------------
# Proposing changes (swap and graph edits)
# --------------------------------------------------------------------------------------


def load_model_target(
    ref: str,
    *,
    registry: Registry,
    model_dirs: Sequence[str | Path] = (),
) -> tuple[ModelManifest, Path | None]:
    """Resolve a model manifest by path or by ``id[@version]``.

    The v0 registry keeps one version per model id. Revision work needs to compare versions, so
    an explicit manifest path or an extra directory is accepted; both are loaded with the normal
    ``ModelManifest`` contract.
    """
    candidate = Path(ref).expanduser()
    if candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml", ".json"}:
        try:
            manifest = ModelManifest.model_validate(load_document(candidate))
        except ModelGraphError:
            raise
        except Exception as exc:  # pydantic ValidationError is not a ModelGraphError
            raise RevisionStoreError(
                f"invalid model manifest {candidate}: {exc}", path=str(candidate)
            ) from exc
        return manifest, candidate.resolve()

    resolved = registry.resolve_model(ref)
    if resolved is not None:
        return resolved, None

    directories = [Path(item).expanduser() for item in model_dirs]
    if not directories:
        root = project_root()
        for fallback in (root / "registry" / "models", root / "examples" / "models"):
            if fallback.is_dir():
                directories.append(fallback)
    wanted_id, _, wanted_version = ref.partition("@")
    for directory in directories:
        if not directory.is_dir():
            continue
        paths = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
        paths += sorted(directory.glob("*.json"))
        for path in paths:
            try:
                document = load_document(path)
                found = ModelManifest.model_validate(document)
            except Exception:
                continue
            if found.metadata.id != wanted_id:
                continue
            if wanted_version and found.metadata.version != wanted_version:
                continue
            return found, path.resolve()
    raise RevisionStoreError(
        f"model {ref!r} is not in the registry. The v0 registry allows one version per model id, "
        "so for a version swap pass the manifest path or an extra --registry directory that "
        "contains the requested version.",
        details={"ref": ref},
    )


def _temp_registry_for_swap(
    *,
    base_registry: Registry,
    target_manifest: ModelManifest,
    target_source: Path | None,
    directory: Path,
    graph_paths: Sequence[Path],
) -> Registry:
    models_dir = directory / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for model_id, manifest in sorted(base_registry.manifests.items()):
        if manifest.metadata.id == target_manifest.metadata.id:
            continue
        (models_dir / f"{model_id}.yaml").write_text(
            _dump_yaml(manifest.to_dict()), encoding="utf-8"
        )
    (models_dir / f"{target_manifest.metadata.id}.yaml").write_text(
        _dump_yaml(target_manifest.to_dict()), encoding="utf-8"
    )
    if target_source is not None and target_source.is_file():
        shutil.copyfile(target_source, models_dir / f"{target_manifest.metadata.id}.source.yaml")
    return Registry.from_paths([models_dir], [path for path in graph_paths if path.exists()])


def _compile_with_temp_registry(
    graph_path: Path,
    *,
    base_registry: Registry,
    target_manifest: ModelManifest,
    target_source: Path | None,
    temp_dir: Path,
    graph_paths: Sequence[Path],
) -> CompiledGraph:
    registry = _temp_registry_for_swap(
        base_registry=base_registry,
        target_manifest=target_manifest,
        target_source=target_source,
        directory=temp_dir,
        graph_paths=graph_paths,
    )
    result = compile_graph(
        graph_path,
        registry=registry,
        strict_models=True,
        graph_paths=[*graph_paths, temp_dir],
    )
    if not result.ok or result.graph is None:
        raise RevisionStoreError(
            "candidate graph failed to compile: " + "; ".join(result.errors),
            path=str(graph_path),
        )
    return result.graph


def _base_graph(
    store: RevisionStore,
    *,
    graph_path: str | Path | None,
    base_revision: str | None,
    registry: Registry,
) -> tuple[CompiledGraph, Path, str | None, str | None, Registry]:
    """Resolve the baseline graph and the registry that describes it."""
    if base_revision is not None:
        manifest = store.load_revision(base_revision)
        directory = store.revision_dir(base_revision)
        base_registry = Registry.from_paths(
            [directory / "models"], [directory / "graphs", directory]
        )
        return (
            store.compile_revision(base_revision),
            directory / "graph.yaml",
            base_revision,
            str(manifest.get("target") or "portable"),
            base_registry,
        )
    if graph_path is not None:
        source = Path(graph_path).expanduser().resolve()
        result = compile_graph(
            source, registry=registry, strict_models=True, graph_paths=[source.parent]
        )
        if not result.ok or result.graph is None:
            raise RevisionStoreError(
                "baseline graph failed to compile: " + "; ".join(result.errors),
                path=str(source),
            )
        active = store.active_revision()
        active_id = store.active_revision_id()
        if (
            active is not None
            and active_id is not None
            and active.get("graph", {}).get("hash") == result.graph.graph_hash
        ):
            return result.graph, source, active_id, str(active.get("target")), registry
        return result.graph, source, None, None, registry
    active_id = store.active_revision_id()
    if active_id is None:
        raise ChangeSetError(
            "no active revision and no --graph was supplied. Bootstrap a deployment with "
            "`mg promote --graph <graph.yaml>` or pass --graph to the change command."
        )
    manifest = store.load_revision(active_id)
    directory = store.revision_dir(active_id)
    base_registry = Registry.from_paths([directory / "models"], [directory / "graphs", directory])
    return (
        store.compile_revision(active_id),
        directory / "graph.yaml",
        active_id,
        str(manifest.get("target") or "portable"),
        base_registry,
    )


def _change_payload(
    *,
    change_id: str,
    base_graph: CompiledGraph | None,
    candidate_graph: CompiledGraph,
    base_revision: str | None,
    candidate_revision: str,
    target: str,
    base_target: str | None,
    bindings: Sequence[Mapping[str, Any]],
    summary: str,
) -> dict[str, Any]:
    return {
        "schema": CHANGE_SET_SCHEMA,
        "change_id": change_id,
        "status": STATUS_COMPILED,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "change_class": classify_change(
            base_graph,
            candidate_graph,
            diff_graphs(base_graph, candidate_graph) if base_graph else {},
            base_target=base_target,
            target=target,
        ),
        "target": target,
        "base_target": base_target,
        "base_revision": base_revision,
        "candidate_revision": candidate_revision,
        "summary": summary,
        "graph": _graph_identity(base_graph) if base_graph else None,
        "candidate_graph": _graph_identity(candidate_graph),
        "bindings": [dict(item) for item in bindings],
        "impact_file": "impact.json",
        "evidence": [],
        "promotion": None,
    }


def _graph_identity(graph: CompiledGraph | None) -> dict[str, Any] | None:
    if graph is None:
        return None
    return {
        "id": graph.id,
        "version": graph.version,
        "hash": graph.graph_hash,
        "source": graph.source,
    }


def _save_change_with_impact(
    store: RevisionStore,
    change: dict[str, Any],
    impact: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    change["change_class"] = impact["change_class"]
    change["status"] = status_for_change(change, impact, impact["eval"]["coverage"])
    impact["change_id"] = change["change_id"]
    impact["status"] = change["status"]
    _write_json_atomic(store.change_dir(change["change_id"]) / "impact.json", impact)
    store.save_change(change)
    return change, impact


def propose_swap(
    store: RevisionStore,
    *,
    node: str,
    to_model: str,
    graph_path: str | Path | None = None,
    target: str = "portable",
    base_revision: str | None = None,
    registry: Registry | None = None,
    model_dirs: Sequence[str | Path] = (),
    profile: Profile | str | Path | None = None,
    dry_run: bool = False,
    summary: str | None = None,
) -> dict[str, Any]:
    """Create (or preview) a candidate revision that swaps one node's model."""
    if target not in TARGETS:
        raise ChangeSetError(f"unknown target {target!r}; expected one of {', '.join(TARGETS)}")
    base_registry = registry or default_registry()
    base_graph, base_source, resolved_base_revision, base_target, base_registry = _base_graph(
        store, graph_path=graph_path, base_revision=base_revision, registry=base_registry
    )
    if node not in base_graph.nodes:
        raise ChangeSetError(
            f"node {node!r} is not in graph {base_graph.id}@{base_graph.version}; nodes: "
            + ", ".join(sorted(base_graph.nodes)),
            details={"graph": base_graph.id},
        )
    target_manifest, target_source = load_model_target(
        to_model, registry=base_registry, model_dirs=model_dirs
    )
    old_node = base_graph.nodes[node]
    target_ref = f"{target_manifest.metadata.id}@{target_manifest.metadata.version}"
    if old_node.model_ref == target_ref and old_node.model_hash == _manifest_hash_for(
        target_manifest
    ):
        raise ChangeSetError(
            f"swap is a no-op: node {node!r} already binds {target_ref!r}. A model swap must "
            "produce a different candidate revision.",
            details={"node": node, "model": target_ref},
        )
    document = GraphDocument.model_validate(load_document(base_source))
    document.node(node).model = target_ref
    temp_root = Path(tempfile.mkdtemp(prefix="mg-swap-"))
    try:
        candidate_source = temp_root / "candidate.graph.yaml"
        # Drop fields equal to their defaults so a re-dump does not add empty lists that the
        # wire schema intentionally omits (for example ``writes: []`` on a router).
        candidate_document = document.model_dump(
            mode="json", by_alias=True, exclude_none=True, exclude_defaults=True
        )
        candidate_source.write_text(_dump_yaml(candidate_document), encoding="utf-8")
        graph_paths = [base_source.parent]
        if store.active_revision_id() and resolved_base_revision:
            graph_paths.append(store.revision_dir(resolved_base_revision) / "graphs")
        candidate = _compile_with_temp_registry(
            candidate_source,
            base_registry=base_registry,
            target_manifest=target_manifest,
            target_source=target_source,
            temp_dir=temp_root,
            graph_paths=graph_paths,
        )
        revision_id = deployment_revision(candidate, target=target)
        loaded_profile = load_profile(profile) if isinstance(profile, (str, Path)) else profile
        bindings = [
            {
                "node": node,
                "from_ref": old_node.model_ref,
                "to_ref": target_ref,
                "from_hash": old_node.model_hash,
                "to_hash": candidate.nodes[node].model_hash,
            }
        ]
        impact = analyse_impact(
            base_graph,
            candidate,
            target=target,
            base_target=base_target,
            base_revision=resolved_base_revision,
            candidate_revision=revision_id,
            profile=loaded_profile,
            store=store,
        )
        if impact["compatibility"]["blocking"]:
            raise CompatibilityError(
                str(impact["compatibility"]["message"]),
                details={
                    "change": "model_swap",
                    "node": node,
                    "target": target,
                    "candidate_revision": revision_id,
                },
            )
        change = _change_payload(
            change_id=new_id("chg"),
            base_graph=base_graph,
            candidate_graph=candidate,
            base_revision=resolved_base_revision,
            candidate_revision=revision_id,
            target=target,
            base_target=base_target,
            bindings=bindings,
            summary=summary or f"swap node {node}: {old_node.model_ref} -> {target_ref}",
        )
        if dry_run:
            revision: dict[str, Any] = {
                "revision_id": revision_id,
                "target": target,
                "graph": _graph_identity(candidate),
                "dry_run": True,
            }
        else:
            revision, _created = store.create_revision(
                candidate, graph_source=candidate_source, target=target
            )
            change["candidate_graph"]["source"] = str(
                store.revision_dir(str(revision["revision_id"])) / "graph.yaml"
            )
            change, impact = _save_change_with_impact(store, change, impact)
        return {
            "dry_run": dry_run,
            "change": change,
            "impact": impact,
            "revision": revision,
            "candidate_graph": candidate,
            "active_revision": store.active_revision_id(),
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def propose_graph_change(
    store: RevisionStore,
    *,
    candidate_graph_path: str | Path,
    base_revision: str | None = None,
    graph_path: str | Path | None = None,
    target: str | None = None,
    registry: Registry | None = None,
    profile: Profile | str | Path | None = None,
    dry_run: bool = False,
    summary: str | None = None,
) -> dict[str, Any]:
    """Create (or preview) a change set from an edited graph file."""
    base_registry = registry or default_registry()
    base_graph, base_source, resolved_base_revision, base_target, base_registry = _base_graph(
        store, graph_path=graph_path, base_revision=base_revision, registry=base_registry
    )
    candidate_source = Path(candidate_graph_path).expanduser().resolve()
    result = compile_graph(
        candidate_source,
        registry=base_registry,
        strict_models=True,
        graph_paths=[candidate_source.parent, base_source.parent],
    )
    if not result.ok or result.graph is None:
        raise RevisionStoreError(
            "candidate graph failed to compile: " + "; ".join(result.errors),
            path=str(candidate_source),
        )
    candidate = result.graph
    chosen_target = target or base_target or "portable"
    if chosen_target not in TARGETS:
        raise ChangeSetError(
            f"unknown target {chosen_target!r}; expected one of {', '.join(TARGETS)}"
        )
    revision_id = deployment_revision(candidate, target=chosen_target)
    loaded_profile = load_profile(profile) if isinstance(profile, (str, Path)) else profile
    impact = analyse_impact(
        base_graph,
        candidate,
        target=chosen_target,
        base_target=base_target,
        base_revision=resolved_base_revision,
        candidate_revision=revision_id,
        profile=loaded_profile,
        store=store,
    )
    if impact["compatibility"]["blocking"]:
        raise CompatibilityError(
            str(impact["compatibility"]["message"]),
            details={"candidate_revision": revision_id, "target": chosen_target},
        )
    change = _change_payload(
        change_id=new_id("chg"),
        base_graph=base_graph,
        candidate_graph=candidate,
        base_revision=resolved_base_revision,
        candidate_revision=revision_id,
        target=chosen_target,
        base_target=base_target,
        bindings=[],
        summary=summary or f"update graph {base_graph.id}: {candidate_source.name}",
    )
    if dry_run:
        revision: dict[str, Any] = {
            "revision_id": revision_id,
            "target": chosen_target,
            "graph": _graph_identity(candidate),
            "dry_run": True,
        }
    else:
        revision, _created = store.create_revision(
            candidate, graph_source=candidate_source, target=chosen_target
        )
        change["candidate_graph"]["source"] = str(
            store.revision_dir(str(revision["revision_id"])) / "graph.yaml"
        )
        change, impact = _save_change_with_impact(store, change, impact)
    return {
        "dry_run": dry_run,
        "change": change,
        "impact": impact,
        "revision": revision,
        "candidate_graph": candidate,
        "active_revision": store.active_revision_id(),
    }


def _manifest_hash_for(manifest: ModelManifest) -> str:
    return sha256_text(canonical_json(manifest.to_dict()))


def bootstrap_revision(
    store: RevisionStore,
    *,
    graph_path: str | Path,
    target: str = "portable",
    registry: Registry | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compile and persist an initial deployment revision."""
    if target in COMPILED_TARGETS:
        raise RevisionStoreError(
            f"cannot bootstrap a {target!r} deployment: compiled/fused/distilled targets are "
            "not implemented in v0 and no plan is built. Use --target portable."
        )
    source = Path(graph_path).expanduser().resolve()
    result = compile_graph(
        source,
        registry=registry or default_registry(),
        strict_models=True,
        graph_paths=[source.parent],
    )
    if not result.ok or result.graph is None:
        raise RevisionStoreError(
            "graph failed to compile: " + "; ".join(result.errors), path=str(source)
        )
    if store.active_revision_id() is not None:
        raise ChangeSetError(
            "an active revision already exists; --graph is only for the initial bootstrap. "
            "Create a change set with `mg swap node` or `mg change create` and promote it with "
            "evidence instead."
        )
    if dry_run:
        revision_id = deployment_revision(result.graph, target=target)
        return {
            "dry_run": True,
            "revision_id": revision_id,
            "graph": _graph_identity(result.graph),
            "target": target,
        }
    revision, created = store.create_revision(
        result.graph, graph_source=source, target=target, bootstrap=True
    )
    return {
        "dry_run": False,
        "revision": revision,
        "created": created,
        "graph": _graph_identity(result.graph),
        "target": target,
    }


# --------------------------------------------------------------------------------------
# Human-readable formatting
# --------------------------------------------------------------------------------------


def format_diff(diff: Mapping[str, Any]) -> list[str]:
    lines = [
        f"graph diff: {diff['before']['id']}@{diff['before']['version']} -> "
        f"{diff['after']['id']}@{diff['after']['version']}",
        f"hash:       {diff['before']['hash']} -> {diff['after']['hash']}",
    ]
    if not diff.get("graph_changed"):
        lines.append("result:     identical compiled graph")
        return lines
    if diff["nodes_added"]:
        lines.append("nodes added:   " + ", ".join(item["node"] for item in diff["nodes_added"]))
    if diff["nodes_removed"]:
        lines.append("nodes removed: " + ", ".join(item["node"] for item in diff["nodes_removed"]))
    for item in diff["nodes_changed"]:
        fields = ", ".join(sorted(item["fields"]))
        lines.append(f"node changed:  {item['node']} ({fields})")
        for name, change in item["fields"].items():
            lines.append(f"  {name}: {change['from']!r} -> {change['to']!r}")
    for binding in diff["model_bindings"]:
        lines.append(
            f"binding:       {binding['node']}: {binding['from_ref']} -> {binding['to_ref']}"
        )
    if diff["edges_added"]:
        lines.append("edges added:   " + str(len(diff["edges_added"])))
    if diff["edges_removed"]:
        lines.append("edges removed: " + str(len(diff["edges_removed"])))
    for item in diff["edges_changed"]:
        lines.append(f"edge changed:  {item['from']} -> {item['to']}")
    for name, change in diff["policy_changed"].items():
        lines.append(f"policy:        {name}: {change['from']!r} -> {change['to']!r}")
    return lines


def format_impact(impact: Mapping[str, Any]) -> list[str]:
    compatibility = cast(Mapping[str, Any], impact.get("compatibility") or {})
    affected = cast(Mapping[str, Any], impact.get("affected") or {})
    vram = cast(Mapping[str, Any], impact.get("vram") or {})
    admission = cast(Mapping[str, Any], vram.get("admission") or {})
    rollback = cast(Mapping[str, Any], impact.get("rollback") or {})
    evaluation = cast(Mapping[str, Any], impact.get("eval") or {})
    coverage = cast(Mapping[str, Any], evaluation.get("coverage") or {})
    lines = [
        f"change class:  {impact.get('change_class')} ({impact.get('change_class_description')})",
        f"target:        {impact.get('target')}",
        f"base revision: {impact.get('base_revision') or 'none'}",
        f"candidate:     {impact.get('candidate_revision')}",
        "",
        "Compatibility",
        f"  status:          {compatibility.get('status')}",
        f"  blocking:        {'yes' if compatibility.get('blocking') else 'no'}",
    ]
    for check in compatibility.get("checks", []):
        base = cast(Mapping[str, Any], check.get("base") or {})
        tokenizer = cast(Mapping[str, Any], check.get("tokenizer") or {})
        lines.append(
            f"  node {check.get('node')}: base {base.get('status')} "
            f"({base.get('from')} -> {base.get('to')}); tokenizer {tokenizer.get('status')} "
            f"({tokenizer.get('from')} -> {tokenizer.get('to')})"
        )
    if compatibility.get("message"):
        lines.append(f"  detail:          {compatibility['message']}")
    lines.extend(["", "Affected"])
    lines.append("  nodes:           " + (", ".join(affected.get("nodes", [])) or "none"))
    routers = [str(item.get("router")) for item in affected.get("routers", [])]
    lines.append("  routers:         " + (", ".join(routers) or "none"))
    lines.append("  routing cards:   " + (", ".join(affected.get("routing_cards", [])) or "none"))
    lines.extend(["", "Invalidates"])
    for item in impact.get("invalidations", []):
        scope = ", ".join(item.get("scope") or []) or "-"
        lines.append(f"  {item.get('artifact'):<20} {item.get('status')} (scope: {scope})")
        lines.append(f"    {item.get('reason')}")
    lines.extend(["", "VRAM"])
    lines.append(
        f"  before: {vram.get('before_gb')}GB  after: {vram.get('after_gb')}GB  "
        f"delta: {vram.get('delta_gb'):+}GB"
    )
    lines.append(f"  admission: {admission.get('status')} ({admission.get('reason')})")
    lines.extend(["", "Evaluation required before promotion"])
    lines.append("  required:  " + (", ".join(evaluation.get("required", [])) or "none"))
    lines.append("  satisfied: " + (", ".join(coverage.get("satisfied", [])) or "none"))
    lines.append("  missing:   " + (", ".join(coverage.get("missing", [])) or "none"))
    lines.extend(["", "Rollback"])
    lines.append(
        f"  target: {rollback.get('target') or 'none'} (grace {rollback.get('grace_days')} days)"
    )
    lines.append(f"  {rollback.get('note')}")
    if impact.get("risk"):
        lines.extend(["", "Risk"])
        for item in impact["risk"]:
            lines.append(f"  - {item}")
    if impact.get("notes"):
        lines.extend(["", "Notes"])
        for item in impact["notes"]:
            lines.append(f"  - {item}")
    return lines


def format_change_summary(change: Mapping[str, Any]) -> list[str]:
    return [
        f"change:    {change.get('change_id')}",
        f"status:    {change.get('status')}",
        f"class:     {change.get('change_class')}",
        f"target:    {change.get('target')}",
        f"base:      {change.get('base_revision') or 'none'}",
        f"candidate: {change.get('candidate_revision')}",
        f"summary:   {change.get('summary')}",
    ]


def status_report(store: RevisionStore, *, include_revisions: bool = False) -> dict[str, Any]:
    active_id = store.active_revision_id()
    active = store.active_revision()
    pointer = store.active_pointer()
    report: dict[str, Any] = {
        "schema": "modelgraph/revision-status/v0",
        "store": str(store.root),
        "active_revision": active_id,
        "graph": active.get("graph") if active else None,
        "target": active.get("target") if active else None,
        "previous_revision": pointer.get("previous_revision") if pointer else None,
        "changes": [
            {
                "change_id": change.get("change_id"),
                "status": change.get("status"),
                "change_class": change.get("change_class"),
                "candidate_revision": change.get("candidate_revision"),
                "base_revision": change.get("base_revision"),
                "summary": change.get("summary"),
            }
            for change in store.list_changes()
        ],
        "distilled_student_status": "not_tracked",
        "note": (
            "Portable revisions do not build compiled plans, prefix caches, route caches, or "
            "distilled students; impact analysis reports them as not applicable rather than "
            "pretending a stale artifact exists."
        ),
    }
    if include_revisions:
        revisions = []
        for manifest in store.list_revisions():
            revisions.append(
                {
                    "revision_id": manifest.get("revision_id"),
                    "target": manifest.get("target"),
                    "graph": manifest.get("graph", {}).get("id"),
                    "graph_hash": manifest.get("graph", {}).get("hash"),
                    "created_at": manifest.get("created_at"),
                    "age_days": round(_age_days(str(manifest.get("created_at", ""))), 2),
                    "active": manifest.get("revision_id") == active_id,
                    "bootstrap": bool(manifest.get("bootstrap")),
                    "models": sorted(
                        {
                            str(item.get("model"))
                            for item in cast(list[dict[str, Any]], manifest.get("models") or [])
                        }
                    ),
                }
            )
        report["revisions"] = revisions
        report["previous_revisions"] = store.previous_revisions()
    return report


def annotate_run_trace(run_dir: str | Path, manifest: Mapping[str, Any]) -> bool:
    """Append one ``revision.pinned`` event so a trace names its deployment revision.

    Run manifests already carry ``deployment_revision``. This helper makes the same identity
    visible in ``trace.jsonl`` without changing the runtime's trace schema or re-writing
    recorded provider events. It is idempotent.
    """
    directory = Path(run_dir)
    trace_name = str(manifest.get("trace") or "trace.jsonl")
    path = directory / trace_name
    if not path.is_file():
        return False
    revision = manifest.get("deployment_revision")
    if not revision:
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(event, dict)
            and event.get("event") == "revision.pinned"
            and event.get("deployment_revision") == revision
        ):
            return False
    record = {
        "ts": utc_now(),
        "run_id": manifest.get("run_id"),
        "event": "revision.pinned",
        "deployment_revision": revision,
        "target": manifest.get("target", "portable"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
    return True
