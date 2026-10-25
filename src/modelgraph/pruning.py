"""Run profiles, capability filtering, and graph pruning.

A profile describes the machine or policy a graph runs under. Pruning is a runtime
view of the authored graph: it removes nodes whose models cannot run, then removes
branches that can no longer reach an exit artifact. The authored YAML is never edited.

The pruner reads facts from the compiled graph (model manifests resolved at compile
time), so it does not need a registry and cannot silently re-resolve a floating model
reference.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from modelgraph.compiler.ir import CompiledEdge, CompiledGraph, CompiledNode
from modelgraph.contracts.graph import Defaults, NodeKind
from modelgraph.contracts.loader import load_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.errors import ModelGraphError
from modelgraph.paths import project_root
from modelgraph.providers.base import Provider
from modelgraph.runtime.budget import Budget

Network = Literal["none", "loopback", "internet"]

NETWORK_RANK: dict[str, int] = {"none": 0, "loopback": 1, "internet": 2}
PROFILE_SUFFIXES = (".yaml", ".yml", ".json")
BUDGET_OVERRIDE_KEYS = (
    "max_steps",
    "max_depth",
    "max_total_tokens",
    "max_cost_usd",
    "max_wall_seconds",
    "on_budget_exhausted",
)
#: Hard ceilings mirroring the graph schema maxima. A profile (even with
#: --allow-budget-override) cannot raise a run past these; they bound runaway configuration.
HARD_BUDGET_LIMITS: dict[str, int | float] = {
    "max_steps": 64,
    "max_depth": 8,
    "max_total_tokens": 10_000_000,
    "max_wall_seconds": 86_400,
}


class ProfileError(ModelGraphError):
    """A profile could not be loaded or validated."""

    code = "MG-1001"

    def __init__(self, message: str, *, path: str = "") -> None:
        super().__init__(message, path=path, details={"code": self.code})


class Profile(BaseModel):
    """Explicit run policy: what this machine may run, not what the graph contains.

    ``network`` is a ceiling (``none`` < ``loopback`` < ``internet``). ``None`` means
    no ceiling. ``providers``/``licenses`` are allowlists; ``None`` means no filter,
    while an empty list allows nothing. Unknown resource requirements are not treated
    as failures (the manifest omitted a fact, which is a registry problem), but a
    missing ``network_access`` is treated as ``internet`` for safety.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    network: Network | None = None
    gpu: bool = True
    max_vram_gb: float | None = Field(default=None, ge=0)
    max_ram_gb: float | None = Field(default=None, ge=0)
    max_disk_gb: float | None = Field(default=None, ge=0)
    providers: list[str] | None = None
    licenses: list[str] | None = None
    max_cost_usd: float | None = Field(default=None, ge=0)
    budget_overrides: dict[str, int | float | str] = Field(default_factory=dict)
    backend: str = "auto"
    allow_unknown_cost: bool = False

    @model_validator(mode="after")
    def _validate_overrides(self) -> Profile:
        for key, value in self.budget_overrides.items():
            if key not in BUDGET_OVERRIDE_KEYS:
                raise ValueError(f"unknown budget override {key!r}")
            if key == "on_budget_exhausted":
                if value not in {"fail", "best_effort"}:
                    raise ValueError("on_budget_exhausted must be 'fail' or 'best_effort'")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"budget override {key!r} must be a number")
            if key in {"max_steps", "max_depth", "max_total_tokens"}:
                if not isinstance(value, int) or value < 1:
                    raise ValueError(f"budget override {key!r} must be a positive integer")
            elif value < 0:
                raise ValueError(f"budget override {key!r} must be non-negative")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


@dataclass(frozen=True)
class RemovedNode:
    id: str
    reason: str
    graph: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "reason": self.reason}
        if self.graph is not None:
            data["graph"] = self.graph
        return data


@dataclass(frozen=True)
class RemovedEdge:
    from_: str
    to: str
    reason: str
    graph: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"from": self.from_, "to": self.to, "reason": self.reason}
        if self.graph is not None:
            data["graph"] = self.graph
        return data


@dataclass(frozen=True)
class RemovedExit:
    name: str
    reason: str
    graph: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "reason": self.reason}
        if self.graph is not None:
            data["graph"] = self.graph
        return data


@dataclass
class PruneReport:
    """Everything the pruner removed and why."""

    profile: str
    removed_nodes: list[RemovedNode] = field(default_factory=list)
    removed_edges: list[RemovedEdge] = field(default_factory=list)
    removed_exit_artifacts: list[RemovedExit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "removed_nodes": [item.to_dict() for item in self.removed_nodes],
            "removed_edges": [item.to_dict() for item in self.removed_edges],
            "removed_exit_artifacts": [item.to_dict() for item in self.removed_exit_artifacts],
            "warnings": list(self.warnings),
        }

    def merge(self, other: PruneReport) -> None:
        """Fold a surviving nested subgraph's removals into this report."""

        self.removed_nodes.extend(other.removed_nodes)
        self.removed_edges.extend(other.removed_edges)
        self.removed_exit_artifacts.extend(other.removed_exit_artifacts)
        self.warnings.extend(other.warnings)

    @property
    def removed_node_ids(self) -> list[str]:
        return [item.id for item in self.removed_nodes]


class PruneError(ModelGraphError):
    """Pruning removed the entry or made an exit artifact unreachable."""

    code = "MG-1005"

    def __init__(
        self,
        report: PruneReport,
        *,
        entry_lost: bool,
        graph: str,
    ) -> None:
        self.report = report
        missing = [
            item.name for item in report.removed_exit_artifacts if item.graph in (None, graph)
        ]
        parts = []
        if entry_lost:
            parts.append(f"entry {graph!r} was removed")
        if missing:
            parts.append(f"exit artifact(s) {', '.join(missing)} are unreachable")
        if not parts:
            parts.append("pruning left the graph unrunnable")
        super().__init__(
            f"pruning for profile {report.profile!r} failed: " + "; ".join(parts),
            details={
                "code": self.code,
                "report": report.to_dict(),
                "entry_lost": entry_lost,
                "removed_exit_artifacts": missing,
            },
        )


@dataclass(frozen=True)
class PruneResult:
    graph: CompiledGraph
    report: PruneReport


def resolve_profile_path(ref: str | Path) -> Path:
    """Resolve ``--profile PATH|NAME`` to an existing profile file.

    An explicit path wins. A bare name is searched in the conventional profile
    directories (cwd, ``examples/profiles``, repository ``profiles``), in that order.
    """

    candidate = Path(ref).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    looks_like_path = candidate.is_absolute() or candidate.parent != Path(".")
    if looks_like_path:
        raise ProfileError(f"profile file not found: {candidate}", path=str(candidate))

    names = [candidate.name]
    if not candidate.name.endswith(PROFILE_SUFFIXES):
        names = [f"{candidate.name}{suffix}" for suffix in PROFILE_SUFFIXES]
    root = project_root()
    search_dirs = [
        Path.cwd() / "profiles",
        Path.cwd() / "examples" / "profiles",
        root / "profiles",
        root / "examples" / "profiles",
    ]
    seen: list[Path] = []
    for directory in search_dirs:
        for name in names:
            path = directory / name
            seen.append(path)
            if path.is_file():
                return path.resolve()
    looked = ", ".join(str(path) for path in seen)
    raise ProfileError(f"profile {ref!r} not found; looked in: {looked}")


def load_profile(ref: str | Path) -> Profile:
    path = resolve_profile_path(ref)
    try:
        document = load_document(path)
    except ModelGraphError as exc:
        raise ProfileError(f"cannot read profile {path}: {exc.message}", path=str(path)) from exc
    if not isinstance(document, dict):
        raise ProfileError(f"profile {path} must be a mapping", path=str(path))
    try:
        return Profile.model_validate(document)
    except ValidationError as exc:
        messages = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '<profile>'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ProfileError(f"invalid profile {path}: {messages}", path=str(path)) from exc


def format_prune_report(report: PruneReport) -> list[str]:
    """Human-readable report lines, printed by the CLI and useful in logs."""

    lines = [
        (
            f"prune:    profile={report.profile} "
            f"removed_nodes={len(report.removed_nodes)} "
            f"removed_edges={len(report.removed_edges)} "
            f"removed_exits={len(report.removed_exit_artifacts)}"
        )
    ]
    for node in report.removed_nodes:
        where = f" ({node.graph})" if node.graph else ""
        lines.append(f"  removed node {node.id}{where}: {node.reason}")
    for edge in report.removed_edges:
        where = f" ({edge.graph})" if edge.graph else ""
        lines.append(f"  removed edge {edge.from_} -> {edge.to}{where}: {edge.reason}")
    for exit_artifact in report.removed_exit_artifacts:
        where = f" ({exit_artifact.graph})" if exit_artifact.graph else ""
        lines.append(f"  removed exit {exit_artifact.name}{where}: {exit_artifact.reason}")
    for warning in report.warnings:
        lines.append(f"  warning: {warning}")
    return lines


def prune_graph(graph: CompiledGraph, profile: Profile) -> PruneResult:
    """Return a capability-filtered view of ``graph`` and its :class:`PruneReport`.

    The authored graph and its hash are unchanged. Nested subgraphs are pruned with the
    same profile; if a subgraph cannot run, its parent subgraph node is marked
    unsupported and normal branch pruning applies.
    """

    report = PruneReport(profile=profile.name)
    pruned = _prune_graph(graph, profile, report)
    return PruneResult(graph=pruned, report=report)


def apply_budget_overrides(
    budget: Budget,
    overrides: Mapping[str, Any],
    *,
    defaults: Defaults,
    allow_raise: bool = False,
) -> list[str]:
    """Apply profile budget overrides to an existing budget.

    Profile overrides tighten by default. An override that loosens a graph default is
    ignored with a warning unless ``allow_raise`` is true; the caller records the
    effective budget in the run manifest either way.
    """

    warnings: list[str] = []
    for key in sorted(overrides):
        value = overrides[key]
        hard_limit = HARD_BUDGET_LIMITS.get(key)
        if hard_limit is not None and float(value) > float(hard_limit):
            warnings.append(
                f"profile override {key}={value!r} ignored because it exceeds the hard "
                f"limit {hard_limit!r}"
            )
            continue
        if not allow_raise and _loosens_default(key, value, getattr(defaults, key)):
            warnings.append(
                f"profile override {key}={value!r} ignored because it raises the graph default "
                f"{getattr(defaults, key)!r}; pass --allow-budget-override to apply it"
            )
            continue
        setattr(budget, key, value)
    return warnings


def _loosens_default(key: str, value: Any, default: Any) -> bool:
    if key == "on_budget_exhausted":
        return bool(value == "best_effort" and default != "best_effort")
    try:
        return bool(float(value) > float(default))
    except (TypeError, ValueError):
        return False


def _prune_graph(graph: CompiledGraph, profile: Profile, report: PruneReport) -> CompiledGraph:
    unsupported: dict[str, str] = {}
    nested_pruned: dict[str, CompiledGraph] = {}
    nested_reports: dict[str, PruneReport] = {}

    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if node.kind == NodeKind.subgraph:
            ref = node.graph_ref or ""
            inner = graph.subgraphs.get(ref)
            if inner is None:
                unsupported[node.id] = f"subgraph {ref!r} is not compiled"
                continue
            nested_report = PruneReport(profile=profile.name)
            try:
                nested_pruned[ref] = _prune_graph(inner, profile, nested_report)
            except PruneError as exc:
                unsupported[node.id] = (
                    f"subgraph {ref!r} cannot run under profile {profile.name!r}: {exc.message}"
                )
            else:
                nested_reports[ref] = nested_report
            continue
        reason = _unsupported_reason(node, profile)
        if reason is not None:
            unsupported[node.id] = reason

    removed_reasons: dict[str, str] = dict(unsupported)
    candidates = set(graph.nodes) - set(unsupported)
    candidate_edges = [
        edge for edge in graph.edges if edge.from_ in candidates and edge.to in candidates
    ]
    forward = _forward_reachable(graph.entry, candidate_edges, candidates)
    exit_names = set(graph.exits)
    producers = {
        node_id
        for node_id in candidates
        if any(write.name in exit_names for write in graph.nodes[node_id].writes)
    }
    if exit_names & graph.input_names():
        producers.add(graph.entry)
    backward = _can_reach(producers, candidate_edges)
    keep = forward & backward

    for node_id in sorted(set(graph.nodes) - keep):
        if node_id in removed_reasons:
            continue
        if node_id not in forward:
            removed_reasons[node_id] = f"unreachable from entry {graph.entry!r}"
        else:
            removed_reasons[node_id] = (
                f"cannot reach any exit artifact ({', '.join(sorted(exit_names))})"
            )

    kept_edges: list[CompiledEdge] = []
    for edge in graph.edges:
        if edge.from_ in keep and edge.to in keep:
            kept_edges.append(edge)
            continue
        source_removed = edge.from_ not in keep
        target_removed = edge.to not in keep
        if source_removed and target_removed:
            reason = f"source node {edge.from_!r} and target node {edge.to!r} were pruned"
        elif source_removed:
            reason = f"source node {edge.from_!r} was pruned"
        else:
            reason = f"target node {edge.to!r} was pruned"
        report.removed_edges.append(RemovedEdge(edge.from_, edge.to, reason, graph=graph.id))
        if graph.nodes[edge.from_].kind == NodeKind.router:
            report.warnings.append(
                f"branch {edge.from_!r} -> {edge.to!r} is unreachable: "
                f"{removed_reasons.get(edge.to, reason)}"
            )

    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if node.kind == NodeKind.router and node_id in keep:
            routes = sum(1 for edge in kept_edges if edge.from_ == node_id)
            if routes == 1:
                report.warnings.append(
                    f"router {node_id!r} now has one route; consider bypassing it"
                )

    available_exits = set(graph.input_names())
    for node_id in keep:
        available_exits.update(write.name for write in graph.nodes[node_id].writes)

    missing_exits: list[RemovedExit] = []
    for exit_name in graph.exits:
        if exit_name in available_exits:
            continue
        producers_of_exit = [
            node_id
            for node_id in sorted(graph.nodes)
            if any(write.name == exit_name for write in graph.nodes[node_id].writes)
        ]
        if producers_of_exit:
            details = "; ".join(
                f"{node_id}: {removed_reasons.get(node_id, 'pruned')}"
                for node_id in producers_of_exit
            )
            reason = f"every producer of {exit_name!r} was pruned ({details})"
        else:
            reason = f"no node writes exit artifact {exit_name!r}"
        missing_exits.append(RemovedExit(exit_name, reason, graph=graph.id))

    report.removed_nodes.extend(
        RemovedNode(node_id, removed_reasons[node_id], graph=graph.id)
        for node_id in sorted(set(graph.nodes) - keep)
    )
    report.removed_exit_artifacts.extend(missing_exits)

    fallback = graph.fallback
    if fallback is not None and fallback not in keep:
        report.warnings.append(
            f"fallback {fallback!r} was pruned; router abstention will fail with no fallback"
        )
        fallback = None

    subgraphs = dict(graph.subgraphs)
    subgraphs.update(nested_pruned)
    merged_refs: set[str] = set()
    for node_id in sorted(keep):
        node = graph.nodes[node_id]
        if node.kind != NodeKind.subgraph:
            continue
        ref = node.graph_ref or ""
        if ref in merged_refs or ref not in nested_reports:
            continue
        report.merge(nested_reports[ref])
        merged_refs.add(ref)
    pruned = replace(
        graph,
        nodes={node_id: graph.nodes[node_id] for node_id in sorted(keep)},
        edges=tuple(kept_edges),
        fallback=fallback,
        subgraphs=subgraphs,
    )

    entry_lost = graph.entry not in keep
    if entry_lost or missing_exits:
        raise PruneError(report, entry_lost=entry_lost, graph=graph.id)
    return pruned


def _unsupported_reason(node: CompiledNode, profile: Profile) -> str | None:
    if node.kind == NodeKind.subgraph:
        return None
    model = node.model
    if model is None:
        return f"model reference {node.model_ref!r} is not resolved"

    provider = str(getattr(model.spec.provider, "value", model.spec.provider))
    if profile.providers is not None:
        allowed = {item.lower() for item in profile.providers}
        if provider.lower() not in allowed:
            return (
                f"provider {provider!r} is not available in profile {profile.name!r} "
                f"(allowed: {', '.join(sorted(allowed)) or 'none'})"
            )

    if not profile.gpu and model.spec.resources.cpu_only is False:
        return "model requires a GPU; profile disables GPU use"

    if profile.network is not None:
        actual = model.spec.resources.network_access
        ceiling = NETWORK_RANK[profile.network]
        if actual is None:
            if ceiling < NETWORK_RANK["internet"]:
                return (
                    "model does not declare resources.network_access; "
                    f"profile network ceiling is {profile.network!r}"
                )
        elif NETWORK_RANK.get(actual.lower(), NETWORK_RANK["internet"]) > ceiling:
            return (
                f"model requires {actual!r} network access; "
                f"profile network ceiling is {profile.network!r}"
            )

    resource_reason = _resource_reason(model, profile)
    if resource_reason is not None:
        return resource_reason

    if profile.licenses is not None:
        allowed_licenses = {item.lower() for item in profile.licenses}
        license_name = model.spec.license.name.lower()
        if license_name not in allowed_licenses:
            return (
                f"license {model.spec.license.name!r} is not allowed by profile "
                f"{profile.name!r} (allowed: {', '.join(sorted(allowed_licenses)) or 'none'})"
            )

    return _cost_reason(model, profile)


def _resource_reason(model: ModelManifest, profile: Profile) -> str | None:
    resources = model.spec.resources
    checks = (
        ("VRAM", resources.vram_gb, profile.max_vram_gb),
        ("RAM", resources.ram_gb, profile.max_ram_gb),
        ("disk", resources.disk_gb, profile.max_disk_gb),
    )
    for label, actual, ceiling in checks:
        if actual is None or ceiling is None:
            continue
        if actual > ceiling:
            return (
                f"model {model.metadata.id!r} requires {actual:g}GB {label}; "
                f"profile {profile.name!r} allows {ceiling:g}GB"
            )
    return None


def _cost_reason(model: ModelManifest, profile: Profile) -> str | None:
    if profile.max_cost_usd is None:
        return None
    cost = model.spec.cost
    if cost.pricing == "unknown":
        if profile.allow_unknown_cost:
            return None
        return (
            f"pricing for model {model.metadata.id!r} is unknown; "
            f"profile {profile.name!r} caps cost at ${profile.max_cost_usd:g}"
        )
    if cost.pricing == "free":
        return None
    output_tokens = model.spec.limits.default_max_output_tokens or model.spec.max_output_tokens or 1
    minimum = (
        cost.per_request_usd
        + cost.input_per_million_usd / 1_000_000
        + cost.output_per_million_usd / 1_000_000 * output_tokens
    )
    if minimum > profile.max_cost_usd:
        return (
            f"minimum call cost for model {model.metadata.id!r} is "
            f"${minimum:.6f}; profile {profile.name!r} caps cost at "
            f"${profile.max_cost_usd:g}"
        )
    return None


def _forward_reachable(entry: str, edges: Iterable[CompiledEdge], candidates: set[str]) -> set[str]:
    if entry not in candidates:
        return set()
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.from_, []).append(edge.to)
    seen = {entry}
    stack = [entry]
    while stack:
        current = stack.pop()
        for target in adjacency.get(current, []):
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def _can_reach(producers: set[str], edges: Iterable[CompiledEdge]) -> set[str]:
    reverse: dict[str, list[str]] = {}
    for edge in edges:
        reverse.setdefault(edge.to, []).append(edge.from_)
    seen = set(producers)
    stack = list(producers)
    while stack:
        current = stack.pop()
        for source in reverse.get(current, []):
            if source not in seen:
                seen.add(source)
                stack.append(source)
    return seen


class LazyModelLoader:
    """Load model providers on first use, with optional LRU unloading.

    The engine already asks for a provider only when a node is first invoked. This
    wrapper makes that contract explicit and records cold-load time separately from
    provider generation latency. ``cold_load_ms`` measures adapter construction (and,
    when a provider loader actually fetches weights, that fetch); a provider-specific
    implementation is allowed to make it a no-op.

    LRU unloading is opt-in (``max_loaded``). Do not enable it for the portable runtime
    while the engine keeps its own provider cache: an evicted provider would still be
    referenced there and would not be reloaded. Manual ``unload`` is safe once the run
    has finished.
    """

    def __init__(
        self,
        factory: Callable[[ModelManifest], Provider],
        *,
        max_loaded: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_loaded is not None and max_loaded < 1:
            raise ValueError("max_loaded must be at least 1")
        self._factory = factory
        self._max_loaded = max_loaded
        self._clock = clock
        self._loaded: dict[str, Provider] = {}
        self._order: list[str] = []
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        """Mutable load/unload log; pass to ``run_graph(model_loads=...)``."""

        return self._records

    @property
    def loaded_models(self) -> list[str]:
        return list(self._order)

    def provider_for(self, manifest: ModelManifest) -> Provider:
        key = manifest.metadata.id
        if key in self._loaded:
            self._touch(key)
            return self._loaded[key]
        started = self._clock()
        provider = self._factory(manifest)
        elapsed = self._clock() - started
        self._loaded[key] = provider
        self._order.append(key)
        self._records.append(
            {
                "model": key,
                "provider": manifest.spec.provider,
                "status": "loaded",
                "cold_load_ms": round(elapsed * 1000, 3),
            }
        )
        self._evict_excess()
        return provider

    def unload(self, model_id: str) -> bool:
        provider = self._loaded.pop(model_id, None)
        if provider is None:
            return False
        self._order.remove(model_id)
        self._close(provider)
        self._records.append({"model": model_id, "status": "unloaded"})
        return True

    def close(self) -> None:
        for model_id in list(self._order):
            self.unload(model_id)

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
            self._order.append(key)

    def _evict_excess(self) -> None:
        if self._max_loaded is None:
            return
        while len(self._order) > self._max_loaded:
            victim = self._order[0]
            provider = self._loaded.pop(victim)
            self._order.pop(0)
            self._close(provider)
            self._records.append({"model": victim, "status": "evicted", "reason": "lru"})

    @staticmethod
    def _close(provider: Provider) -> None:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
