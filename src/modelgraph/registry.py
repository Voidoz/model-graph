"""Local model and graph registry.

The runtime registry resolves graph model references. It understands the legacy flat
``registry/models/<id>.yaml`` layout and the versioned T-0026 layout
``registry/models/<id>/<version>/manifest.yaml``. When an id has several versions,
an unversioned reference resolves to the highest semver and ``id@version`` resolves
the exact entry.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import ValidationError

from modelgraph.contracts.loader import load_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.errors import ModelGraphError
from modelgraph.paths import project_root
from modelgraph.security import manifest_trust_violations
from modelgraph.util import canonical_json, sha256_text

_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z.-]+))?$")
_VERSIONED_NAMES = ("manifest.yaml", "manifest.yml", "model.yaml", "model.yml")


def version_key(version: str) -> tuple[int, int, int, int, str]:
    """Sort key for a semantic version; releases sort after prereleases."""
    match = _SEMVER.match(version)
    if match is None:
        return (-1, -1, -1, -1, version)
    major, minor, patch, prerelease = match.groups()
    return (
        int(major),
        int(minor),
        int(patch),
        1 if prerelease is None else 0,
        prerelease or "",
    )


class Registry:
    """A directory-backed registry of model manifests and graphs."""

    def __init__(self) -> None:
        # ``manifests`` keeps the highest-version manifest per id for existing callers
        # (compiler, GUI, revision tooling). ``by_ref`` keeps every id@version.
        self.manifests: dict[str, ModelManifest] = {}
        self.manifest_hashes: dict[str, str] = {}
        self.by_ref: dict[str, ModelManifest] = {}
        self.by_ref_hashes: dict[str, str] = {}
        self.graph_sources: dict[str, Path] = {}

    @classmethod
    def from_paths(
        cls,
        model_paths: list[str | Path] | None = None,
        graph_paths: list[str | Path] | None = None,
    ) -> Registry:
        registry = cls()
        for path in model_paths or []:
            registry.add_model_path(path)
        for path in graph_paths or []:
            registry.add_graph_path(path)
        return registry

    def add_model_path(self, path: str | Path) -> None:
        source = Path(path)
        if source.is_dir():
            direct = next(
                (source / name for name in _VERSIONED_NAMES if (source / name).is_file()), None
            )
            if direct is not None:
                # A version directory (``models/<id>/<version>``) or a contribution output
                # directory with a single manifest at its root.
                self._add_model(direct)
                return
            for item in sorted(source.iterdir()):
                if item.name in {"index.yaml", "index.yml", "index.json"}:
                    continue
                if item.is_file() and item.suffix in {".yaml", ".yml", ".json"}:
                    self._add_model(item)
                elif item.is_dir():
                    # Versioned layout: models/<id>/<version>/manifest.yaml
                    nested = next(
                        (item / name for name in _VERSIONED_NAMES if (item / name).is_file()),
                        None,
                    )
                    if nested is not None:
                        self._add_model(nested)
                        continue
                    for version_dir in sorted(item.iterdir()):
                        if not version_dir.is_dir():
                            continue
                        candidate = next(
                            (
                                version_dir / name
                                for name in _VERSIONED_NAMES
                                if (version_dir / name).is_file()
                            ),
                            None,
                        )
                        if candidate is not None:
                            self._add_model(candidate)
        elif source.suffix in {".yaml", ".yml", ".json"}:
            self._add_model(source)

    def _add_model(self, path: Path) -> None:
        try:
            document = load_document(path)
        except ModelGraphError as exc:
            raise ModelGraphError(f"invalid model manifest {path}: {exc}", path=str(path)) from exc
        violations = manifest_trust_violations(document)
        if violations:
            raise ModelGraphError(
                f"model manifest {path} violates the trust policy: " + "; ".join(violations),
                path=str(path),
                details={"violations": violations},
            )
        try:
            manifest = ModelManifest.model_validate(document)
        except ValidationError as exc:
            raise ModelGraphError(f"invalid model manifest {path}: {exc}", path=str(path)) from exc
        model_id = manifest.metadata.id
        version = manifest.metadata.version
        ref = f"{model_id}@{version}"
        digest = sha256_text(canonical_json(manifest.to_dict()))
        existing = self.by_ref.get(ref)
        if existing is not None:
            existing_digest = self.by_ref_hashes[ref]
            if existing_digest != digest:
                raise ModelGraphError(
                    f"model {ref} is already registered with different content; "
                    "a version is immutable once loaded",
                    path=str(path),
                )
            return
        self.by_ref[ref] = manifest
        self.by_ref_hashes[ref] = digest
        current = self.manifests.get(model_id)
        if current is None or version_key(version) > version_key(current.metadata.version):
            self.manifests[model_id] = manifest
            self.manifest_hashes[model_id] = digest

    def add_graph_path(self, path: str | Path) -> None:
        source = Path(path)
        if source.is_dir():
            for item in sorted(source.glob("*.graph.yaml")):
                self._add_graph(item)
        elif source.is_file():
            self._add_graph(source)

    def _add_graph(self, path: Path) -> None:
        from modelgraph.contracts.graph import GraphDocument

        try:
            document = GraphDocument.model_validate(load_document(path))
        except (ModelGraphError, ValidationError) as exc:
            raise ModelGraphError(f"invalid graph {path}: {exc}", path=str(path)) from exc
        ref = f"{document.metadata.id}@{document.metadata.version}"
        self.graph_sources[ref] = path
        self.graph_sources.setdefault(document.metadata.id, path)

    def resolve_model(self, ref: str) -> ModelManifest | None:
        if "@" in ref:
            return self.by_ref.get(ref)
        return self.manifests.get(ref)

    def manifest_hash(self, ref: str) -> str | None:
        if "@" in ref:
            return self.by_ref_hashes.get(ref)
        manifest = self.manifests.get(ref)
        if manifest is None:
            return None
        return self.by_ref_hashes.get(
            f"{manifest.metadata.id}@{manifest.metadata.version}",
            self.manifest_hashes.get(ref),
        )

    def versioned(self) -> list[tuple[str, ModelManifest]]:
        """Every loaded manifest as ``(id@version, manifest)``, sorted by id/version."""
        return sorted(
            self.by_ref.items(),
            key=lambda item: (item[1].metadata.id, version_key(item[1].metadata.version)),
        )

    def resolve_graph(self, ref: str) -> Path | None:
        return self.graph_sources.get(ref)


def default_registry() -> Registry:
    """Load examples/models and repo graphs when present."""
    root = project_root()
    model_dir = root / "registry" / "models"
    if not model_dir.exists():
        model_dir = root / "examples" / "models"
    graph_dir = root / "graphs"
    if not graph_dir.exists():
        graph_dir = root / "examples"
    return Registry.from_paths([model_dir], [graph_dir])
