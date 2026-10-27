"""Budgeted graph execution."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from modelgraph.compiler.ir import CompiledGraph, CompiledNode
from modelgraph.contracts.envelope import Artifact, PathEntry
from modelgraph.contracts.graph import ArtifactInput, NodeKind, RoutingSelection
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.errors import BudgetExceeded, ModelGraphError, RuntimeFailure
from modelgraph.providers.base import Provider
from modelgraph.runtime.budget import Budget, Reservation
from modelgraph.runtime.cache import CacheStats, ResponseCache
from modelgraph.runtime.envelope import RuntimeState
from modelgraph.runtime.executors import (
    Executor,
    RouterExecutor,
    UsageLedger,
    WorkerExecutor,
    max_output_tokens,
)
from modelgraph.runtime.manifest import (
    OUTCOME_RUNNING,
    STATUS_FINAL,
    STATUS_PROVISIONAL,
    build_environment,
    build_run_manifest,
    recover_interrupted_runs,
    usage_snapshot,
    write_manifest_atomic,
)
from modelgraph.runtime.renderer import render_prompt
from modelgraph.runtime.trace import TraceRecorder
from modelgraph.security import RedactionReport, redact_data
from modelgraph.util import sha256_text, utc_now

if TYPE_CHECKING:
    from modelgraph.observability import EventLogger, SpanExporter

ProviderFactory = Callable[[ModelManifest], Provider]

#: Concurrency cap for independent fan-out branches. Four in-flight provider calls is a
#: safe default for a local-first runtime; a caller can raise or lower it per run.
DEFAULT_MAX_CONCURRENCY = 4

#: Typed runtime error for a joiner that waited past its configured deadline.
JOIN_TIMEOUT_CODE = "MG-2105"

#: Typed runtime error for a join that cannot be completed at all (not a timeout).
JOIN_INCOMPLETE_CODE = "MG-2104"


@dataclass
class _ChildRun:
    """A subgraph invocation captured for child manifest persistence."""

    node: CompiledNode
    graph: CompiledGraph
    state: RuntimeState
    completed: bool = False


@dataclass
class _Branch:
    """Accounting for one active path: root, a fan-out branch, or a join continuation."""

    branch_id: str
    kind: str
    parent: str | None
    router: str | None = None
    target: str | None = None
    selection: str | None = None
    start_order: int | None = None
    completion_order: int | None = None
    outcome: str = "pending"
    nodes: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    ledger: UsageLedger = field(default_factory=UsageLedger)
    contributed: bool = False
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        usage = self.ledger.to_dict()
        return {
            "branch_id": self.branch_id,
            "kind": self.kind,
            "parent": self.parent,
            "router": self.router,
            "target": self.target,
            "selection": self.selection,
            "start_order": self.start_order,
            "completion_order": self.completion_order,
            "outcome": self.outcome,
            "nodes": list(self.nodes),
            "steps": len(self.nodes),
            "artifacts": list(self.artifacts),
            "contributed": self.contributed,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "cost_usd": usage["cost_usd"],
            "calls": usage["calls"],
            "cache_hits": usage["cache_hits"],
            "cached_tokens": usage["cached_tokens"],
            "cached_cost_usd": usage["cached_cost_usd"],
            "error": self.error,
        }


class _BranchRegistry:
    """Thread-safe per-run branch table shared by nested schedulers and subgraphs."""

    def __init__(self) -> None:
        self._branches: dict[str, _Branch] = {}
        self._lock = threading.Lock()

    def create(
        self,
        branch_id: str,
        *,
        kind: str,
        parent: str | None,
        router: str | None = None,
        target: str | None = None,
        selection: str | None = None,
    ) -> _Branch:
        with self._lock:
            existing = self._branches.get(branch_id)
            if existing is not None:
                return existing
            branch = _Branch(
                branch_id=branch_id,
                kind=kind,
                parent=parent,
                router=router,
                target=target,
                selection=selection,
            )
            self._branches[branch_id] = branch
            return branch

    def get(self, branch_id: str) -> _Branch | None:
        with self._lock:
            return self._branches.get(branch_id)

    def mark_contributed(self, branch_ids: set[str]) -> None:
        with self._lock:
            for branch_id in branch_ids:
                branch = self._branches.get(branch_id)
                if branch is not None:
                    branch.contributed = True

    def _children(self, branches: list[_Branch]) -> dict[str, list[str]]:
        children: dict[str, list[str]] = defaultdict(list)
        for branch in branches:
            if branch.parent is not None:
                children[branch.parent].append(branch.branch_id)
        return children

    def _subtree_usage(
        self,
        branch_id: str,
        *,
        by_id: dict[str, _Branch],
        children: dict[str, list[str]],
        seen: set[str],
    ) -> tuple[int, float]:
        if branch_id in seen or branch_id not in by_id:
            return 0, 0.0
        seen.add(branch_id)
        usage = by_id[branch_id].ledger.to_dict()
        tokens = int(usage["total_tokens"])
        cost = float(usage["cost_usd"])
        for child in children.get(branch_id, []):
            child_tokens, child_cost = self._subtree_usage(
                child, by_id=by_id, children=children, seen=seen
            )
            tokens += child_tokens
            cost += child_cost
        return tokens, cost

    def subtree_usage(self, branch_id: str) -> dict[str, int | float]:
        """Direct plus nested-subgraph usage for one branch."""

        with self._lock:
            branches = list(self._branches.values())
        by_id = {branch.branch_id: branch for branch in branches}
        children = self._children(branches)
        tokens, cost = self._subtree_usage(branch_id, by_id=by_id, children=children, seen=set())
        return {"total_tokens": tokens, "cost_usd": round(cost, 8)}

    def to_list(self) -> list[dict[str, Any]]:
        with self._lock:
            branches = list(self._branches.values())
        by_id = {branch.branch_id: branch for branch in branches}
        children = self._children(branches)
        records: list[dict[str, Any]] = []
        for branch in sorted(
            branches,
            key=lambda item: (
                item.start_order if item.start_order is not None else 10**9,
                item.branch_id,
            ),
        ):
            data = branch.to_dict()
            tokens, cost = self._subtree_usage(
                branch.branch_id, by_id=by_id, children=children, seen=set()
            )
            data["subtree_total_tokens"] = tokens
            data["subtree_cost_usd"] = round(cost, 8)
            records.append(data)
        return records


@dataclass
class _PendingNode:
    node_id: str
    branch_id: str
    start_order: int
    reservation: Reservation | None = None


@dataclass
class _NodeRun:
    pending: _PendingNode
    step: int
    depth: int
    branch_map: dict[str, str] = field(default_factory=dict)
    completion_order: int | None = None


@dataclass
class _NodeExecution:
    """One node's outcome plus the routes it selected."""

    selected: list[str] = field(default_factory=list)
    selection: str = RoutingSelection.single.value
    fallback: str | None = None
    decision_target: str | None = None
    decision_confidence: float = 0.0
    path_entries: list[PathEntry] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    contributions: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: ModelGraphError | None = None


@dataclass
class RunOutcome:
    run_id: str
    outcome: str
    artifacts: list[Artifact] = field(default_factory=list)
    path: list[PathEntry] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    run_dir: Path | None = None
    manifest: dict[str, Any] | None = None
    branches: list[dict[str, Any]] = field(default_factory=list)
    execution: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome == "succeeded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "outcome": self.outcome,
            "artifacts": [
                {
                    "name": artifact.name,
                    "type": artifact.type.value,
                    "media_type": artifact.media_type,
                    "content": artifact.content,
                    "producer": artifact.producer,
                    "hash": artifact.hash,
                }
                for artifact in self.artifacts
            ],
            "path": [entry.model_dump(mode="json", exclude_none=True) for entry in self.path],
            "usage": self.usage,
            "cache": self.cache,
            "error": self.error,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "branches": self.branches,
            "execution": self.execution,
        }


def run_graph(
    graph: CompiledGraph,
    task: str,
    provider_factory: ProviderFactory,
    *,
    runs_dir: str | Path | None = None,
    persist: bool = True,
    run_id: str | None = None,
    budget: Budget | None = None,
    profile: dict[str, Any] | None = None,
    prune_report: dict[str, Any] | None = None,
    model_loads: list[dict[str, Any]] | None = None,
    seed: int | None = None,
    target: str = "portable",
    redact_persisted: bool = True,
    cache: ResponseCache | None = None,
    logger: EventLogger | None = None,
    span_exporter: SpanExporter | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    join_timeout_seconds: float | None = None,
) -> RunOutcome:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if join_timeout_seconds is not None and join_timeout_seconds <= 0:
        raise ValueError("join_timeout_seconds must be positive")
    state = RuntimeState.create(
        graph,
        task,
        budget=budget or Budget.from_defaults(graph.defaults),
        run_id=run_id,
    )
    trace = TraceRecorder(state.run_id)
    from modelgraph.observability import NullLogger, NullSpanExporter

    log = logger or NullLogger()
    spans = span_exporter or NullSpanExporter()
    cache_stats = CacheStats()
    cache_mode = cache.mode if cache is not None else "off"
    providers: dict[str, Provider] = {}
    provider_lock = threading.Lock()
    branch_registry = _BranchRegistry()
    fanout_graph = _has_fanout(graph)
    directory = Path(runs_dir) if runs_dir else Path.cwd() / "runs"
    run_dir = directory / state.run_id
    environment: dict[str, Any] | None = None
    if persist:
        # A dead provisional manifest from a crashed process is recovered before this
        # run writes its own. A live manifest (including this process) is left alone.
        recover_interrupted_runs(directory)
        run_dir.mkdir(parents=True, exist_ok=True)
        environment = build_environment()
        write_manifest_atomic(
            run_dir,
            build_run_manifest(
                state.graph,
                run_id=state.run_id,
                task=state.task,
                budget=state.budget,
                started_at=state.started_at,
                status=STATUS_PROVISIONAL,
                outcome=OUTCOME_RUNNING,
                seed=seed,
                parent_run_id=state.parent_run_id,
                profile=profile,
                prune_report=prune_report,
                target=target,
                usage=usage_snapshot(state.budget),
                path=state.path,
                artifacts=state.artifacts,
                finished_at=None,
                environment=environment,
                redact=redact_persisted,
            ),
        )

    def provider_for(node: CompiledNode) -> Provider:
        if node.model is None:
            raise RuntimeFailure(f"node {node.id!r} has no model", details={"code": "MG-1003"})
        key = node.model.metadata.id
        with provider_lock:
            if key not in providers:
                providers[key] = provider_factory(node.model)
            return providers[key]

    def executor_maker(
        node: CompiledNode, node_trace: TraceRecorder, ledger: UsageLedger | None
    ) -> Executor:
        return _executor_for(
            node,
            provider_for(node),
            budget=state.budget,
            trace=node_trace,
            seed=seed,
            cache=cache,
            cache_stats=cache_stats,
            ledger=ledger,
        )

    started_fields: dict[str, Any] = {
        "graph": {
            "id": graph.id,
            "version": graph.version,
            "hash": graph.graph_hash,
            "source": graph.source,
        },
        "target": target,
        "task": task,
        "seed": seed,
        "parent_run_id": state.parent_run_id,
        "concurrency_limit": max_concurrency,
        "join_timeout_seconds": join_timeout_seconds,
        "budget": state.budget.snapshot().model_dump(mode="json"),
        "models": {
            node.id: {
                "ref": node.model_ref,
                "hash": node.model_hash,
                "provider": node.model.spec.provider if node.model else None,
            }
            for node in graph.nodes.values()
            if node.model_ref
        },
    }
    if profile is not None:
        started_fields["profile"] = profile
    trace.emit("run.started", **started_fields)
    log.event(
        "run.started",
        run_id=state.run_id,
        graph_id=graph.id,
        graph_hash=graph.graph_hash,
        target=target,
        seed=seed,
        parent_run_id=state.parent_run_id,
    )
    if prune_report is not None:
        trace.emit(
            "prune.applied",
            profile=prune_report.get("profile"),
            report=prune_report,
            warnings=prune_report.get("warnings", []),
        )

    outcome = "succeeded"
    error: ModelGraphError | None = None
    result_artifacts: list[Artifact] = []
    child_runs: list[_ChildRun] = []
    from modelgraph.observability import run_span

    try:
        with run_span(
            spans,
            state.run_id,
            graph_id=graph.id,
            graph_hash=graph.graph_hash,
            parent_run_id=state.parent_run_id,
            target=target,
            seed=seed,
        ):
            _run_dispatch(
                state,
                graph.entry,
                depth=0,
                trace=trace,
                provider_for=provider_for,
                providers=providers,
                executor_maker=executor_maker,
                children=child_runs,
                logger=log,
                span_exporter=spans,
                branch_registry=branch_registry,
                root_branch_id="root",
                max_concurrency=max_concurrency,
                join_timeout_seconds=join_timeout_seconds,
            )
            result_artifacts = state.results()
            trace.emit(
                "run.finished",
                outcome=outcome,
                usage=_usage(state),
                artifacts=[
                    {"name": artifact.name, "hash": artifact.hash} for artifact in result_artifacts
                ],
                branches=branch_registry.to_list(),
            )
    except BudgetExceeded as exc:
        error = exc
        if state.budget.on_budget_exhausted == "best_effort":
            outcome = "budget_exhausted"
            try:
                result_artifacts = state.results()
            except RuntimeFailure:
                result_artifacts = []
        else:
            outcome = "failed"
        trace.emit(
            "run.failed",
            outcome=outcome,
            code=exc.details.get("code", exc.code),
            message=exc.message,
            branches=branch_registry.to_list() if fanout_graph else [],
        )
    except ModelGraphError as exc:
        error = exc
        outcome = "failed"
        trace.emit(
            "run.failed",
            outcome=outcome,
            code=exc.details.get("code", exc.code),
            message=exc.message,
            branches=branch_registry.to_list() if fanout_graph else [],
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        error = RuntimeFailure(f"internal runtime error: {exc}", details={"code": "MG-2000"})
        outcome = "failed"
        trace.emit(
            "run.failed",
            outcome=outcome,
            code="MG-2000",
            message=str(exc),
            branches=branch_registry.to_list() if fanout_graph else [],
        )
    finally:
        for provider in providers.values():
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    error_payload = (
        _redact_error(
            {
                "code": error.details.get("code", error.code),
                "message": error.message,
                "path": error.path,
                "details": error.details,
            }
        )
        if error
        else None
    )
    result = RunOutcome(
        run_id=state.run_id,
        outcome=outcome,
        artifacts=result_artifacts,
        path=state.path,
        usage=_usage(state),
        cache={"mode": cache_mode, **cache_stats.to_dict()},
        error=error_payload,
        run_dir=run_dir if persist else None,
        branches=branch_registry.to_list() if fanout_graph else [],
        execution={
            "fanout": fanout_graph,
            "concurrency_limit": max_concurrency,
            "join_timeout_seconds": join_timeout_seconds,
        },
    )
    if persist:
        result.manifest = _persist_run(
            state,
            trace,
            result,
            run_dir,
            profile=profile,
            prune_report=prune_report,
            model_loads=model_loads,
            providers=list(providers.values()),
            child_runs=child_runs,
            seed=seed,
            target=target,
            environment=environment,
            redact_persisted=redact_persisted,
        )
    log_fields = {
        "run_id": state.run_id,
        "outcome": outcome,
        "usage": result.usage,
        "cache": result.cache,
        "code": result.error.get("code") if result.error else None,
    }
    if outcome == "succeeded":
        log.info("run.finished", **log_fields)
    elif outcome == "budget_exhausted":
        log.warning("run.finished", **log_fields)
    else:
        log.error("run.failed", **log_fields)
    return result


def _executor_for(
    node: CompiledNode,
    provider: Provider,
    *,
    budget: Budget,
    trace: TraceRecorder,
    seed: int | None,
    cache: ResponseCache | None = None,
    cache_stats: CacheStats | None = None,
    ledger: UsageLedger | None = None,
) -> Executor:
    """Create one executor per node and carry the node's configured retry policy."""

    max_retries = node.routing.max_retries if node.kind == NodeKind.router else 0
    return Executor(
        provider,
        budget=budget,
        trace=trace,
        max_retries=max_retries,
        seed=seed,
        cache=cache,
        cache_stats=cache_stats,
        ledger=ledger,
    )


def _redact_error(error: dict[str, Any]) -> dict[str, Any]:
    """Redact a persisted error payload without changing its shape."""

    redacted, _report = redact_data(error)
    return cast(dict[str, Any], redacted)


def _walk(
    state: RuntimeState,
    start: str,
    *,
    depth: int,
    trace: TraceRecorder,
    provider_for: Callable[[CompiledNode], Provider],
    providers: dict[str, Provider],
    executor_factory: Callable[[CompiledNode], Executor],
    children: list[_ChildRun] | None = None,
    logger: EventLogger,
    span_exporter: SpanExporter,
) -> None:
    current: str | None = start
    from modelgraph.observability import node_span

    while current is not None:
        state.budget.begin_step()
        state.step += 1
        node = state.graph.node(current)
        trace.emit(
            "node.entered",
            node=node.id,
            kind=node.kind.value,
            depth=depth,
            step=state.step,
        )
        logger.event(
            "node.entered",
            run_id=state.run_id,
            node=node.id,
            kind=node.kind.value,
            depth=depth,
            step=state.step,
        )
        with node_span(
            span_exporter,
            state.run_id,
            node.id,
            kind=node.kind.value,
            depth=depth,
            step=state.step,
        ):
            current, finished = _walk_node(
                state,
                node,
                depth=depth,
                trace=trace,
                provider_for=provider_for,
                providers=providers,
                executor_factory=executor_factory,
                children=children,
                logger=logger,
                span_exporter=span_exporter,
            )
        trace.emit("node.exited", node=node.id, kind=node.kind.value, depth=depth)
        logger.event(
            "node.exited",
            run_id=state.run_id,
            node=node.id,
            kind=node.kind.value,
            depth=depth,
        )
        if finished:
            return


def _walk_node(
    state: RuntimeState,
    node: CompiledNode,
    *,
    depth: int,
    trace: TraceRecorder,
    provider_for: Callable[[CompiledNode], Provider],
    providers: dict[str, Provider],
    executor_factory: Callable[[CompiledNode], Executor],
    children: list[_ChildRun] | None,
    logger: EventLogger,
    span_exporter: SpanExporter,
) -> tuple[str | None, bool]:
    """Execute one node and return (next node, finished)."""
    if node.kind == NodeKind.router:
        executor = executor_factory(node)
        decision = RouterExecutor(executor).run(node, state.graph, state)
        route = decision.to if decision.action == "route" else state.graph.fallback
        state.add_path(
            PathEntry(
                node=node.id,
                kind=node.kind.value,
                decision="route" if decision.action == "route" else "abstain",
                to=route,
                confidence=decision.confidence,
                reason=decision.reason,
            )
        )
        trace.emit(
            "decision.made",
            node=node.id,
            decision=decision.action,
            to=route,
            confidence=decision.confidence,
            reason=decision.reason,
            threshold=node.routing.confidence_threshold,
        )
        if route is None:
            raise RuntimeFailure(
                "router abstained and the graph has no fallback",
                details={"code": "MG-2103"},
            )
        if route == node.id:
            raise RuntimeFailure(
                "router fallback cannot point to itself", details={"code": "MG-2103"}
            )
        return route, False
    if node.kind == NodeKind.subgraph:
        inner = state.graph.subgraphs.get(node.graph_ref or "")
        if inner is None:
            raise RuntimeFailure(
                f"subgraph {node.graph_ref!r} is not compiled",
                details={"code": "MG-1004"},
            )
        state.budget.check_depth(depth + 1)
        state.add_path(
            PathEntry(
                node=node.id,
                kind=node.kind.value,
                graph=node.graph_ref,
                graph_hash=inner.graph_hash,
            )
        )
        trace.emit(
            "subgraph.entered",
            node=node.id,
            graph=node.graph_ref,
            graph_hash=inner.graph_hash,
            depth=depth + 1,
        )
        child = _child_state(state, node, inner, depth + 1)
        child_record = _ChildRun(node=node, graph=inner, state=child)
        if children is not None:
            children.append(child_record)
        _walk(
            child,
            inner.entry,
            depth=depth + 1,
            trace=trace,
            provider_for=provider_for,
            providers=providers,
            executor_factory=executor_factory,
            children=children,
            logger=logger,
            span_exporter=span_exporter,
        )
        child_record.completed = True
        _map_subgraph_outputs(state, child, node)
        trace.emit("subgraph.exited", node=node.id, child_run_id=child.run_id)
        outgoing = state.graph.outgoing(node.id)
        return (outgoing[0].to, False) if outgoing else (None, True)
    executor = executor_factory(node)
    WorkerExecutor(executor).run(node, state.graph, state)
    state.add_path(PathEntry(node=node.id, kind=node.kind.value))
    outgoing = state.graph.outgoing(node.id)
    return (outgoing[0].to, False) if outgoing else (None, True)


def _has_fanout(graph: CompiledGraph) -> bool:
    """Whether this graph (or a subgraph it contains) declares multi-selection routing."""

    for node in graph.nodes.values():
        if node.kind == NodeKind.router and node.routing.selection != RoutingSelection.single:
            return True
    return any(_has_fanout(inner) for inner in graph.subgraphs.values())


def _run_dispatch(
    state: RuntimeState,
    start: str,
    *,
    depth: int,
    trace: TraceRecorder,
    provider_for: Callable[[CompiledNode], Provider],
    providers: dict[str, Provider],
    executor_maker: Callable[[CompiledNode, TraceRecorder, UsageLedger | None], Executor],
    children: list[_ChildRun] | None,
    logger: EventLogger,
    span_exporter: SpanExporter,
    branch_registry: _BranchRegistry,
    root_branch_id: str,
    max_concurrency: int,
    join_timeout_seconds: float | None,
    root_parent: str | None = None,
) -> None:
    """Run one graph with the legacy walk or the fan-out scheduler.

    A graph with no multi-selection router keeps the exact sequential walk it has always
    used. A graph with fan-out uses the DAG scheduler so independent branches can run at
    the same time; the scheduler still handles its single-selection subpaths.
    """

    if not _has_fanout(state.graph):
        inner_branch: _Branch | None = None
        if root_parent is not None:
            # A single-selection subgraph still belongs to its parent fan-out branch.
            # Record an inner branch so its cost is visible instead of silently vanishing
            # into the subgraph node's zero-cost outer entry.
            inner_branch = branch_registry.create(root_branch_id, kind="root", parent=root_parent)

        def single_executor(node: CompiledNode) -> Executor:
            return executor_maker(
                node, trace, inner_branch.ledger if inner_branch is not None else None
            )

        try:
            _walk(
                state,
                start,
                depth=depth,
                trace=trace,
                provider_for=provider_for,
                providers=providers,
                executor_factory=single_executor,
                children=children,
                logger=logger,
                span_exporter=span_exporter,
            )
        except ModelGraphError:
            if inner_branch is not None:
                inner_branch.outcome = "failed"
                inner_branch.error = {"message": "subgraph failed"}
            raise
        if inner_branch is not None:
            inner_branch.nodes = [entry.node for entry in state.path]
            inner_branch.artifacts = [item.name for item in state.artifacts]
            inner_branch.outcome = "succeeded"
        return
    _run_dag(
        state,
        start,
        depth=depth,
        trace=trace,
        provider_for=provider_for,
        providers=providers,
        executor_maker=executor_maker,
        children=children,
        logger=logger,
        span_exporter=span_exporter,
        branch_registry=branch_registry,
        root_branch_id=root_branch_id,
        root_parent=root_parent,
        max_concurrency=max_concurrency,
        join_timeout_seconds=join_timeout_seconds,
    )


def _run_dag(
    state: RuntimeState,
    start: str,
    *,
    depth: int,
    trace: TraceRecorder,
    provider_for: Callable[[CompiledNode], Provider],
    providers: dict[str, Provider],
    executor_maker: Callable[[CompiledNode, TraceRecorder, UsageLedger | None], Executor],
    children: list[_ChildRun] | None,
    logger: EventLogger,
    span_exporter: SpanExporter,
    branch_registry: _BranchRegistry,
    root_branch_id: str,
    max_concurrency: int,
    join_timeout_seconds: float | None,
    root_parent: str | None = None,
) -> None:
    """Execute a fan-out-capable graph as a dependency DAG on a bounded thread pool.

    A node becomes runnable only after every incoming edge has been resolved: either
    delivered by a branch that ran, or marked inactive because an upstream router did not
    select it. A joiner therefore waits for every declared input, and a missing input is a
    fail-closed ``MG-2201`` rather than a silent partial join.
    """

    from modelgraph.observability import node_span

    graph = state.graph
    registry = branch_registry
    pool = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="mg-branch")
    registry.create(root_branch_id, kind="root", parent=root_parent)
    branch_of: dict[str, str] = {start: root_branch_id}
    incoming: dict[str, int] = {node_id: len(graph.incoming(node_id)) for node_id in graph.nodes}
    resolved: set[tuple[str, str]] = set()
    delivered: dict[str, int] = defaultdict(int)
    activated: set[str] = set()
    dead: set[str] = set()
    ready: list[_PendingNode] = []
    groups: dict[int, list[dict[str, Any]]] = {}
    futures: dict[Future[tuple[list[dict[str, Any]], _NodeExecution]], _NodeRun] = {}
    open_branches: dict[str, int] = defaultdict(int)
    join_clock: dict[str, float] = {}
    owned_branches: list[str] = [root_branch_id]
    next_order = 0
    completion_counter = [0]
    completion_lock = threading.Lock()
    failure: ModelGraphError | None = None
    failure_order = 10**9
    stop = False

    def emit(order: int, event: str, **fields: Any) -> None:
        groups.setdefault(order, []).append(trace.make(event, **fields))

    def activate(node_id: str, branch_id: str) -> None:
        nonlocal next_order
        if node_id in activated or node_id in dead:
            return
        node = graph.node(node_id)
        if node.kind == NodeKind.joiner:
            # Scope the join branch to this scheduler instance. Two subgraph invocations
            # (or a subgraph and the outer graph) can both declare a joiner named
            # ``merge``; a bare ``join:merge`` id would merge their ledgers, node lists,
            # completion order, and costs. ``root_branch_id`` is unique per scheduler
            # instance and is independent of which incoming branch resolves last.
            join_id = f"{root_branch_id}/join:{node_id}"
            registry.create(
                join_id,
                kind="join",
                # A join is a merge, not a subgraph child; keeping it out of the parent
                # chain prevents double-counting its cost in the subtree totals.
                parent=None,
                target=node_id,
            )
            branch_id = join_id
            branch_of[node_id] = join_id
        activated.add(node_id)
        branch_of.setdefault(node_id, branch_id)
        order = next_order
        next_order += 1
        groups.setdefault(order, [])
        branch = registry.get(branch_id)
        if branch is not None:
            branch.start_order = order if branch.start_order is None else branch.start_order
            emit(
                order,
                "branch.started",
                branch_id=branch_id,
                branch_kind=branch.kind,
                node=node_id,
                kind=node.kind.value,
            )
            open_branches[branch_id] += 1
        ready.append(_PendingNode(node_id=node_id, branch_id=branch_id, start_order=order))

    def resolve_edge(from_id: str, to_id: str, is_delivered: bool, branch_id: str) -> None:
        work: list[tuple[str, str, bool, str]] = [(from_id, to_id, is_delivered, branch_id)]
        while work:
            source, target, flag, source_branch = work.pop()
            key = (source, target)
            if key in resolved:
                continue
            resolved.add(key)
            if flag:
                delivered[target] += 1
            incoming[target] = max(0, incoming[target] - 1)
            if target in activated or target in dead:
                continue
            if incoming[target] > 0:
                continue
            if delivered[target] == 0:
                dead.add(target)
                for edge in reversed(graph.outgoing(target)):
                    work.append((target, edge.to, False, source_branch))
                continue
            branch_of.setdefault(target, source_branch)
            activate(target, branch_of[target])
            if graph.node(target).kind == NodeKind.joiner:
                join_clock.setdefault(target, time.monotonic())

    def force_activate(target: str, branch_id: str) -> None:
        if target in activated or target in dead:
            return
        incoming[target] = 0
        delivered[target] += 1
        branch_of.setdefault(target, branch_id)
        activate(target, branch_of[target])

    def downstream_joiners(target: str) -> set[str]:
        return _downstream_joiners(graph, target)

    def next_join_deadline() -> float | None:
        # The wall budget is always an outer bound; an explicit join timeout is tighter.
        deadlines = [
            time.monotonic()
            + max(0.0, state.budget.max_wall_seconds - state.budget.elapsed_seconds)
        ]
        if join_timeout_seconds is not None:
            deadlines.extend(
                started + join_timeout_seconds
                for joiner, started in join_clock.items()
                if joiner not in activated and joiner not in dead
            )
        return min(deadlines)

    def check_join_timeouts() -> None:
        if join_timeout_seconds is None:
            return
        now = time.monotonic()
        for joiner, started in list(join_clock.items()):
            if joiner in activated or joiner in dead:
                join_clock.pop(joiner, None)
                continue
            waited = now - started
            if waited > join_timeout_seconds:
                pending_branches = sorted(
                    branch_id for branch_id, count in open_branches.items() if count > 0
                )
                details = {
                    "code": JOIN_TIMEOUT_CODE,
                    "join": joiner,
                    "timeout_seconds": join_timeout_seconds,
                    "waited_seconds": round(waited, 3),
                    "pending_branches": pending_branches,
                }
                groups.setdefault(10**9, []).append(
                    trace.make("join.timeout", node=joiner, **details)
                )
                raise RuntimeFailure(
                    f"joiner {joiner!r} timed out waiting for its declared inputs "
                    f"after {waited:.3f}s",
                    details=details,
                )

    def ensure_reservations() -> None:
        batch = [pending for pending in ready if pending.reservation is None]
        if not batch:
            return
        state.budget.check_wall()
        estimates = [_estimate_first_call(state, pending.node_id) for pending in batch]
        order = min(pending.start_order for pending in batch)
        event_fields = {
            "nodes": [pending.node_id for pending in batch],
            "branches": [pending.branch_id for pending in batch],
            "estimated_tokens": sum(item[0] for item in estimates),
            "estimated_cost_usd": round(sum(item[1] for item in estimates), 8),
            "concurrency_limit": max_concurrency,
        }
        try:
            reservations = state.budget.reserve_many(estimates)
        except BudgetExceeded as exc:
            emit(
                order,
                "fanout.preflight",
                accepted=False,
                **event_fields,
                code=exc.details.get("code", exc.code),
            )
            raise
        for pending, reservation in zip(batch, reservations, strict=True):
            pending.reservation = reservation
        if len(batch) > 1:
            emit(order, "fanout.preflight", accepted=True, **event_fields)

    def dispatch() -> None:
        nonlocal failure, stop
        while ready and len(futures) < max_concurrency and not stop:
            pending = ready[0]
            try:
                state.budget.begin_step()
            except BudgetExceeded as exc:
                failure = exc
                stop = True
                return
            state.step += 1
            step = state.step
            ready.pop(0)
            node = graph.node(pending.node_id)
            branch = registry.get(pending.branch_id)
            if branch is not None:
                branch.nodes.append(node.id)
            run = _NodeRun(
                pending=pending,
                step=step,
                depth=depth,
                branch_map=dict(branch_of),
            )
            emit(
                pending.start_order,
                "node.entered",
                node=node.id,
                kind=node.kind.value,
                depth=depth,
                step=step,
                branch_id=pending.branch_id,
            )
            logger.event(
                "node.entered",
                run_id=state.run_id,
                node=node.id,
                kind=node.kind.value,
                depth=depth,
                step=step,
                branch_id=pending.branch_id,
                start_order=pending.start_order,
            )

            def task(
                run: _NodeRun,
            ) -> tuple[list[dict[str, Any]], _NodeExecution]:
                local = TraceRecorder(trace.run_id)
                try:
                    with node_span(
                        span_exporter,
                        state.run_id,
                        run.pending.node_id,
                        kind=graph.node(run.pending.node_id).kind.value,
                        depth=depth,
                        step=run.step,
                        start_order=run.pending.start_order,
                        branch_id=run.pending.branch_id,
                    ):
                        result = _dag_node(
                            state,
                            graph.node(run.pending.node_id),
                            reservation=run.pending.reservation,
                            depth=depth,
                            trace=local,
                            provider_for=provider_for,
                            providers=providers,
                            executor_maker=executor_maker,
                            children=children,
                            logger=logger,
                            span_exporter=span_exporter,
                            branch_registry=registry,
                            branch_id=run.pending.branch_id,
                            branch_map=run.branch_map,
                            root_branch_id=(
                                f"{run.pending.branch_id}/subgraph:{run.pending.node_id}"
                            ),
                            max_concurrency=max_concurrency,
                            join_timeout_seconds=join_timeout_seconds,
                        )
                except ModelGraphError as exc:
                    return local.events, _NodeExecution(error=exc)
                except Exception as exc:  # pragma: no cover - defensive task boundary
                    return local.events, _NodeExecution(
                        error=RuntimeFailure(
                            f"internal runtime error: {exc}", details={"code": "MG-2000"}
                        )
                    )
                return local.events, result

            try:
                future = pool.submit(task, run)
            except BaseException:
                state.budget.release(run.pending.reservation)
                raise

            def mark_completed(
                _future: Future[tuple[list[dict[str, Any]], _NodeExecution]],
                run: _NodeRun = run,
            ) -> None:
                with completion_lock:
                    completion_counter[0] += 1
                    run.completion_order = completion_counter[0]

            future.add_done_callback(mark_completed)
            futures[future] = run

    def complete(run: _NodeRun, events: list[dict[str, Any]], result: _NodeExecution) -> None:
        nonlocal failure, failure_order, stop
        completion_order = run.completion_order or 0
        order = run.pending.start_order
        group = groups.setdefault(order, [])
        group.extend(events)
        node = graph.node(run.pending.node_id)
        if result.error is not None:
            group.append(
                trace.make(
                    "node.exited",
                    node=node.id,
                    kind=node.kind.value,
                    depth=depth,
                    outcome="failed",
                    completion_order=completion_order,
                    error_code=result.error.details.get("code", result.error.code),
                )
            )
            if failure is None or run.pending.start_order < failure_order:
                failure = result.error
                failure_order = run.pending.start_order
            stop = True
        else:
            group.append(
                trace.make(
                    "node.exited",
                    node=node.id,
                    kind=node.kind.value,
                    depth=depth,
                    outcome="succeeded",
                    completion_order=completion_order,
                )
            )
            logger.event(
                "node.exited",
                run_id=state.run_id,
                node=node.id,
                kind=node.kind.value,
                depth=depth,
                outcome="succeeded",
                completion_order=completion_order,
            )

        if result.error is None:
            selected = list(result.selected)
            target_branches: dict[str, str] = {}
            if node.kind == NodeKind.router:
                if result.fallback:
                    emit(
                        order,
                        "fanout.fallback",
                        node=node.id,
                        fallback=result.fallback,
                        decision=result.decision_target,
                    )
                elif result.selection != RoutingSelection.single.value and selected:
                    for index, target in enumerate(selected):
                        branch_id = f"{run.pending.branch_id}/{node.id}#{index}:{target}"
                        registry.create(
                            branch_id,
                            kind="fanout",
                            parent=run.pending.branch_id,
                            router=node.id,
                            target=target,
                            selection=result.selection,
                        )
                        owned_branches.append(branch_id)
                        target_branches[target] = branch_id
                    emit(
                        order,
                        "fanout.decision",
                        node=node.id,
                        selection=result.selection,
                        targets=selected,
                        branches=[target_branches[target] for target in selected],
                        confidence=result.decision_confidence,
                    )
                    for target in selected:
                        for joiner in downstream_joiners(target):
                            join_clock.setdefault(joiner, time.monotonic())
                for edge in graph.outgoing(node.id):
                    if edge.to in selected:
                        resolve_edge(
                            node.id,
                            edge.to,
                            True,
                            target_branches.get(edge.to, run.pending.branch_id),
                        )
                    else:
                        resolve_edge(node.id, edge.to, False, run.pending.branch_id)
                if result.fallback:
                    force_activate(result.fallback, run.pending.branch_id)
            else:
                outgoing = graph.outgoing(node.id)
                if outgoing:
                    resolve_edge(node.id, outgoing[0].to, True, run.pending.branch_id)
                    for edge in outgoing[1:]:
                        resolve_edge(node.id, edge.to, False, run.pending.branch_id)
            for artifact_name in result.artifacts:
                branch = registry.get(run.pending.branch_id)
                if branch is not None:
                    branch.artifacts.append(artifact_name)
            if result.contributions:
                registry.mark_contributed({str(item["branch_id"]) for item in result.contributions})
                emit(
                    order,
                    "joiner.joined",
                    node=node.id,
                    contributions=result.contributions,
                )

        branch = registry.get(run.pending.branch_id)
        if branch is not None:
            open_branches[run.pending.branch_id] = max(0, open_branches[run.pending.branch_id] - 1)
            if open_branches[run.pending.branch_id] == 0 and branch.outcome == "pending":
                branch.completion_order = completion_order
                branch.outcome = "failed" if result.error is not None else "succeeded"
                if result.error is not None:
                    branch.error = {
                        "code": result.error.details.get("code", result.error.code),
                        "message": result.error.message,
                    }
                usage = branch.ledger.to_dict()
                emit(
                    order,
                    "branch.completed",
                    branch_id=branch.branch_id,
                    branch_kind=branch.kind,
                    outcome=branch.outcome,
                    start_order=branch.start_order,
                    completion_order=completion_order,
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    total_tokens=usage["total_tokens"],
                    cost_usd=usage["cost_usd"],
                    nodes=list(branch.nodes),
                    artifacts=list(branch.artifacts),
                )
        state.path.extend(result.path_entries)

    try:
        activate(start, root_branch_id)
        while True:
            if not stop:
                try:
                    ensure_reservations()
                except ModelGraphError as exc:
                    if failure is None:
                        failure = exc
                    stop = True
            dispatch()
            if not futures:
                if stop:
                    break
                if not ready:
                    break
                continue
            deadline = next_join_deadline()
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            done, _pending = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                try:
                    check_join_timeouts()
                    state.budget.check_wall()
                except ModelGraphError as exc:
                    if failure is None:
                        failure = exc
                    stop = True
                continue
            for future in sorted(done, key=lambda item: futures[item].pending.start_order):
                run = futures.pop(future)
                try:
                    events, result = future.result()
                except Exception as exc:  # pragma: no cover - future boundary
                    events = []
                    result = _NodeExecution(
                        error=RuntimeFailure(
                            f"internal runtime error: {exc}", details={"code": "MG-2000"}
                        )
                    )
                complete(run, events, result)
            try:
                check_join_timeouts()
            except ModelGraphError as exc:
                if failure is None:
                    failure = exc
                stop = True
        if failure is not None:
            raise failure
        unresolved = sorted(
            node_id
            for node_id in graph.nodes
            if node_id not in activated and node_id not in dead and node_id != start
        )
        if unresolved:
            raise RuntimeFailure(
                f"fan-out join could not resolve node(s): {unresolved}",
                details={"code": JOIN_INCOMPLETE_CODE, "unresolved": unresolved},
            )
    finally:
        for pending in ready:
            state.budget.release(pending.reservation)
        for run in futures.values():
            state.budget.release(run.pending.reservation)
        pool.shutdown(wait=True)
        for branch_id in owned_branches:
            branch = registry.get(branch_id)
            if branch is not None and branch.outcome == "pending":
                branch.outcome = "failed" if failure is not None and branch.nodes else "not_started"
                if branch.start_order is None:
                    branch.start_order = next_order
                    next_order += 1
        for order in sorted(groups):
            for record in groups[order]:
                record.setdefault("start_order", order)
        for order in sorted(groups):
            trace.extend(groups[order])
        _sort_path(state)


def _estimate_first_call(state: RuntimeState, node_id: str) -> tuple[int, float]:
    """Estimate the first provider call for a ready node (tokens, cost).

    The estimate uses the same renderer and output cap the executor will use, so the
    batch reservation is at least the executor's own pre-flight. A node whose prompt
    cannot be rendered yet (for example a joiner before all reads exist) falls back to
    the request length plus its output cap; the executor still fails closed later.
    """

    node = state.graph.node(node_id)
    if node.model is None:
        # A subgraph node has no model call of its own. Its inner scheduler reserves the
        # inner entry's first call; the outer batch cannot see inside it without
        # double-reserving, so a subgraph branch is reported as a zero-direct-cost branch
        # with subtree usage instead.
        return 0, 0.0
    output = max_output_tokens(node)
    try:
        prompt = render_prompt(node, state.graph, state)
        input_tokens = state.budget.estimate_input_tokens(prompt.messages)
        tokens = state.budget.estimate_tokens(prompt.messages, output)
    except RuntimeFailure:
        input_tokens = state.budget.estimate_input_tokens([state.task])
        tokens = input_tokens + min(output, 256)
    cost = state.budget.estimate_cost(node, input_tokens=input_tokens, output_tokens=output)
    return tokens, cost


def _downstream_joiners(graph: CompiledGraph, target: str) -> set[str]:
    """Joiners reachable from ``target``; the first joiner on a path is the boundary."""

    found: set[str] = set()
    seen: set[str] = set()
    queue = deque([target])
    while queue:
        node_id = queue.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        if node_id in graph.nodes and graph.node(node_id).kind == NodeKind.joiner:
            found.add(node_id)
            continue
        queue.extend(edge.to for edge in graph.outgoing(node_id))
    return found


def _dag_node(
    state: RuntimeState,
    node: CompiledNode,
    *,
    reservation: Reservation | None,
    depth: int,
    trace: TraceRecorder,
    provider_for: Callable[[CompiledNode], Provider],
    providers: dict[str, Provider],
    executor_maker: Callable[[CompiledNode, TraceRecorder, UsageLedger | None], Executor],
    children: list[_ChildRun] | None,
    logger: EventLogger,
    span_exporter: SpanExporter,
    branch_registry: _BranchRegistry,
    branch_id: str,
    branch_map: dict[str, str],
    root_branch_id: str,
    max_concurrency: int,
    join_timeout_seconds: float | None,
) -> _NodeExecution:
    """Execute one node inside a branch task and return its control-flow result."""

    branch = branch_registry.get(branch_id)
    ledger = branch.ledger if branch is not None else None
    try:
        if node.kind == NodeKind.router:
            executor = executor_maker(node, trace, ledger)
            decision = RouterExecutor(executor).run(
                node, state.graph, state, reservation=reservation
            )
            selected = list(decision.selected)
            fallback = decision.to if decision.action != "route" else None
            trace.emit(
                "decision.made",
                node=node.id,
                decision=decision.action,
                to=decision.to,
                targets=selected,
                selection=decision.selection,
                confidence=decision.confidence,
                reason=decision.reason,
                threshold=node.routing.confidence_threshold,
            )
            if decision.action == "route" and not selected:
                raise RuntimeFailure(
                    f"router {node.id!r} selected no routes",
                    details={"code": "MG-2101"},
                )
            if decision.action != "route" and not fallback:
                raise RuntimeFailure(
                    "router abstained and the graph has no fallback",
                    details={"code": "MG-2103"},
                )
            if fallback == node.id:
                raise RuntimeFailure(
                    "router fallback cannot point to itself", details={"code": "MG-2103"}
                )
            path_entries: list[PathEntry] = []
            if decision.action == "route":
                for target in selected:
                    path_entries.append(
                        PathEntry(
                            node=node.id,
                            kind=node.kind.value,
                            decision="route",
                            to=target,
                            confidence=decision.confidence,
                            reason=decision.reason,
                        )
                    )
            else:
                path_entries.append(
                    PathEntry(
                        node=node.id,
                        kind=node.kind.value,
                        decision="abstain",
                        to=fallback,
                        confidence=decision.confidence,
                        reason=decision.reason,
                    )
                )
            return _NodeExecution(
                selected=selected,
                selection=decision.selection,
                fallback=fallback,
                decision_target=decision.to,
                decision_confidence=decision.confidence,
                path_entries=path_entries,
            )
        if node.kind == NodeKind.subgraph:
            inner = state.graph.subgraphs.get(node.graph_ref or "")
            if inner is None:
                raise RuntimeFailure(
                    f"subgraph {node.graph_ref!r} is not compiled",
                    details={"code": "MG-1004"},
                )
            state.budget.check_depth(depth + 1)
            state.add_path(
                PathEntry(
                    node=node.id,
                    kind=node.kind.value,
                    graph=node.graph_ref,
                    graph_hash=inner.graph_hash,
                )
            )
            trace.emit(
                "subgraph.entered",
                node=node.id,
                graph=node.graph_ref,
                graph_hash=inner.graph_hash,
                depth=depth + 1,
            )
            child = _child_state(state, node, inner, depth + 1)
            child_record = _ChildRun(node=node, graph=inner, state=child)
            if children is not None:
                children.append(child_record)
            _run_dispatch(
                child,
                inner.entry,
                depth=depth + 1,
                trace=trace,
                provider_for=provider_for,
                providers=providers,
                executor_maker=executor_maker,
                children=children,
                logger=logger,
                span_exporter=span_exporter,
                branch_registry=branch_registry,
                root_branch_id=root_branch_id,
                root_parent=branch_id,
                max_concurrency=max_concurrency,
                join_timeout_seconds=join_timeout_seconds,
            )
            child_record.completed = True
            _map_subgraph_outputs(state, child, node)
            trace.emit("subgraph.exited", node=node.id, child_run_id=child.run_id)
            return _NodeExecution(
                artifacts=[write.name for write in node.writes],
                path_entries=[],
            )
        if node.kind == NodeKind.joiner:
            received: list[dict[str, Any]] = []
            missing: list[str] = []
            contributions: list[dict[str, Any]] = []
            for name in node.reads:
                artifact = state.latest_artifact(name)
                if artifact is None:
                    missing.append(name)
                    continue
                producer_branch = branch_map.get(artifact.producer)
                usage = (
                    branch_registry.subtree_usage(producer_branch)
                    if producer_branch
                    else {"total_tokens": 0, "cost_usd": 0.0}
                )
                item = {
                    "name": name,
                    "producer": artifact.producer,
                    "branch_id": producer_branch,
                    "total_tokens": usage["total_tokens"],
                    "cost_usd": usage["cost_usd"],
                }
                received.append(item)
                if producer_branch:
                    contributions.append(dict(item))
            trace.emit(
                "joiner.inputs",
                node=node.id,
                received=received,
                missing=missing,
                declared=list(node.reads),
            )
            if missing:
                trace.emit(
                    "joiner.incomplete",
                    node=node.id,
                    missing=missing,
                    declared=list(node.reads),
                )
                raise RuntimeFailure(
                    f"joiner {node.id!r} is missing declared input(s): {missing}",
                    path=f"{state.graph.id}.{node.id}.reads",
                    details={
                        "code": "MG-2201",
                        "missing": missing,
                        "node": node.id,
                    },
                )
            executor = executor_maker(node, trace, ledger)
            WorkerExecutor(executor).run(node, state.graph, state, reservation=reservation)
            state.add_path(PathEntry(node=node.id, kind=node.kind.value))
            return _NodeExecution(
                artifacts=[write.name for write in node.writes],
                contributions=contributions,
            )
        executor = executor_maker(node, trace, ledger)
        WorkerExecutor(executor).run(node, state.graph, state, reservation=reservation)
        state.add_path(PathEntry(node=node.id, kind=node.kind.value))
        return _NodeExecution(artifacts=[write.name for write in node.writes])
    finally:
        # Safe after a successful settle: a consumed reservation is ignored.
        state.budget.release(reservation)


def _sort_path(state: RuntimeState) -> None:
    """Sort the recorded path by deterministic node order, not completion order."""

    ranks: dict[str, int] = {}
    for index, node_id in enumerate(_topological_node_order(state.graph)):
        ranks[node_id] = index
    with state._lock:
        state.path.sort(key=lambda entry: (ranks.get(entry.node, len(ranks)), entry.node))


def _topological_node_order(graph: CompiledGraph) -> list[str]:
    node_ids = sorted(graph.nodes)
    indegree = {node_id: 0 for node_id in node_ids}
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        adjacency[edge.from_].append(edge.to)
        indegree[edge.to] = indegree.get(edge.to, 0) + 1
    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    ordered: list[str] = []
    while queue:
        node_id = queue.popleft()
        ordered.append(node_id)
        for target in sorted(adjacency.get(node_id, [])):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return ordered if len(ordered) == len(node_ids) else node_ids


def _child_state(
    parent: RuntimeState,
    node: CompiledNode,
    inner: CompiledGraph,
    depth: int,
) -> RuntimeState:
    child = RuntimeState.create(
        inner,
        parent.task,
        budget=parent.budget,
        depth=depth,
        parent_run_id=parent.run_id,
    )
    mapped: list[Artifact] = []
    for outer_name, inner_name in node.input_map.items():
        artifact = parent.latest_artifact(outer_name)
        if artifact is None:
            raise RuntimeFailure(
                f"subgraph {node.id!r} maps missing artifact {outer_name!r}",
                details={"code": "MG-2301"},
            )
        inner_input = _inner_input(inner, inner_name)
        mapped.append(
            Artifact(
                name=inner_name,
                type=inner_input.type if inner_input else artifact.type,
                media_type=(inner_input.media_type if inner_input else artifact.media_type),
                content=artifact.content,
                producer=artifact.producer,
                created_at=utc_now(),
                hash=sha256_text(artifact.content),
                metadata={"mapped_from": outer_name},
            )
        )
    child.inputs = mapped
    return child


def _map_subgraph_outputs(parent: RuntimeState, child: RuntimeState, node: CompiledNode) -> None:
    writes = {write.name: write for write in node.writes}
    for outer_name, inner_name in node.output_map.items():
        artifact = child.latest_artifact(inner_name)
        if artifact is None:
            raise RuntimeFailure(
                f"subgraph {node.id!r} did not produce inner artifact {inner_name!r}",
                details={"code": "MG-2301"},
            )
        write = writes.get(outer_name)
        if write is None:
            raise RuntimeFailure(
                f"subgraph {node.id!r} maps undeclared output {outer_name!r}",
                details={"code": "MG-2301"},
            )
        parent.write(
            name=outer_name,
            artifact_type=write.type,
            content=artifact.content,
            producer=node.id,
            media_type=write.media_type,
            metadata={"subgraph_run_id": child.run_id, "inner_artifact": inner_name},
        )


def _inner_input(inner: CompiledGraph, name: str) -> ArtifactInput | None:
    for item in inner.inputs:
        if item.name == name:
            return item
    return None


def _usage(state: RuntimeState) -> dict[str, Any]:
    return usage_snapshot(state.budget)


def _persist_run(
    state: RuntimeState,
    trace: TraceRecorder,
    result: RunOutcome,
    run_dir: Path,
    *,
    profile: dict[str, Any] | None = None,
    prune_report: dict[str, Any] | None = None,
    model_loads: list[dict[str, Any]] | None = None,
    providers: list[Provider] | None = None,
    child_runs: list[_ChildRun] | None = None,
    seed: int | None = None,
    target: str = "portable",
    environment: dict[str, Any] | None = None,
    redact_persisted: bool = True,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    ordered = state.ordered_artifacts()
    for artifact in [*state.inputs, *ordered]:
        if artifact.hash:
            digest = artifact.hash.split(":", 1)[-1]
            (artifacts_dir / digest).write_text(artifact.content, encoding="utf-8")
    trace_report = trace.write(run_dir / "trace.jsonl", redact=redact_persisted)
    envelope_value: Any = state.envelope().to_dict()
    envelope_report: RedactionReport | None = None
    if redact_persisted:
        envelope_value, envelope_report = redact_data(envelope_value)
    (run_dir / "envelope.json").write_text(_json_dump(envelope_value), encoding="utf-8")
    if redact_persisted:
        _write_redaction_report(run_dir, trace_report, envelope_report)
    finished_at = utc_now()

    subgraphs: list[dict[str, Any]] = []
    for child in child_runs or []:
        child_outcome = "succeeded" if child.completed else result.outcome
        child_error = None if child.completed else result.error
        child_dir = run_dir / "subgraphs" / child.state.run_id
        child_manifest = build_run_manifest(
            child.graph,
            run_id=child.state.run_id,
            task=child.state.task,
            budget=child.state.budget,
            started_at=child.state.started_at,
            status=STATUS_FINAL,
            outcome=child_outcome,
            seed=seed,
            parent_run_id=child.state.parent_run_id,
            profile=profile,
            prune_report=prune_report,
            target=target,
            usage=usage_snapshot(child.state.budget),
            path=child.state.path,
            artifacts=child.state.ordered_artifacts(),
            error=child_error,
            finished_at=finished_at,
            providers=providers,
            trace="../../trace.jsonl",
            environment=environment,
            redact=redact_persisted,
        )
        write_manifest_atomic(child_dir, child_manifest)
        subgraphs.append(
            {
                "node": child.node.id,
                "graph": child.graph.id,
                "hash": child.graph.graph_hash,
                "run_id": child.state.run_id,
                "parent_run_id": state.run_id,
                "status": "final",
                "outcome": child_outcome,
                "manifest": str(Path("subgraphs") / child.state.run_id / "run.json"),
            }
        )

    manifest = build_run_manifest(
        state.graph,
        run_id=state.run_id,
        task=state.task,
        budget=state.budget,
        started_at=state.started_at,
        status=STATUS_FINAL,
        outcome=result.outcome,
        seed=seed,
        parent_run_id=state.parent_run_id,
        profile=profile,
        prune_report=prune_report,
        target=target,
        usage=result.usage,
        path=state.path,
        artifacts=ordered,
        error=result.error,
        finished_at=finished_at,
        model_loads=model_loads,
        subgraphs=subgraphs,
        providers=providers,
        environment=environment,
        redact=redact_persisted,
    )
    write_manifest_atomic(run_dir, manifest)
    return manifest


def _write_redaction_report(
    run_dir: Path,
    trace_report: RedactionReport | None,
    envelope_report: RedactionReport | None,
) -> None:
    """Persist pattern names/counts/hashes only; never the matched text."""

    combined = RedactionReport()
    files: list[str] = []
    if trace_report is not None:
        combined.merge(trace_report)
        files.append("trace.jsonl")
    if envelope_report is not None:
        combined.merge(envelope_report)
        files.append("envelope.json")
    document = combined.to_dict()
    document["files"] = files
    document["note"] = (
        "Artifact blobs under artifacts/ are run data and are not rewritten; "
        "treat the whole run directory as sensitive."
    )
    (run_dir / "redaction.json").write_text(_json_dump(document), encoding="utf-8")


def _json_dump(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"
