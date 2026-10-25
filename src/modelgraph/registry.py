"""Local model and graph registry."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from modelgraph.contracts.loader import load_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.errors import ModelGraphError
from modelgraph.paths import project_root
from modelgraph.security import manifest_trust_violations
from modelgraph.util import canonical_json, sha256_text


class Registry:
    """A directory-backed registry of model manifests and graphs."""

    def __init__(self) -> None:
        self.manifests: dict[str, ModelManifest] = {}
        self.manifest_hashes: dict[str, str] = {}
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
            for item in sorted(source.glob("*.yaml")):
                self._add_model(item)
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
        existing = self.manifests.get(model_id)
        if existing is not None and existing.metadata.version != manifest.metadata.version:
            raise ModelGraphError(
                f"model id {model_id!r} has multiple versions in the registry; "
                "use distinct ids for the MVP",
                path=str(path),
            )
        self.manifests[model_id] = manifest
        self.manifest_hashes[model_id] = sha256_text(canonical_json(manifest.to_dict()))

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
            model_id, version = ref.split("@", 1)
            manifest = self.manifests.get(model_id)
            if manifest and manifest.metadata.version == version:
                return manifest
            return None
        return self.manifests.get(ref)

    def manifest_hash(self, ref: str) -> str | None:
        manifest = self.resolve_model(ref)
        if manifest is None:
            return None
        return self.manifest_hashes.get(manifest.metadata.id)

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
