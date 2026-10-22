"""Budgeted graph execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelgraph.compiler.ir import CompiledGraph, CompiledNode
from modelgraph.contracts.envelope import Artifact, PathEntry
from modelgraph.contracts.graph import ArtifactInput, NodeKind
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.errors import BudgetExceeded, ModelGraphError, RuntimeFailure
from modelgraph.providers.base import Provider
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.envelope import RuntimeState
from modelgraph.runtime.executors import Executor, RouterExecutor, WorkerExecutor
from modelgraph.runtime.trace import TraceRecorder
from modelgraph.util import sha256_text, utc_now
from modelgraph.version import PARSER_VERSION, RENDERER_VERSION

ProviderFactory = Callable[[ModelManifest], Provider]


@dataclass
class RunOutcome:
    run_id: str
    outcome: str
    artifacts: list[Artifact] = field(default_factory=list)
    path: list[PathEntry] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    run_dir: Path | None = None
    manifest: dict[str, Any] | None = None

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
            "error": self.error,
            "run_dir": str(self.run_dir) if self.run_dir else None,
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
) -> RunOutcome:
    state = RuntimeState.create(
        graph,
        task,
        budget=budget or Budget.from_defaults(graph.defaults),
        run_id=run_id,
    )
    trace = TraceRecorder(state.run_id)
    providers: dict[str, Provider] = {}
    directory = Path(runs_dir) if runs_dir else Path.cwd() / "runs"
    run_dir = directory / state.run_id
    if persist:
        run_dir.mkdir(parents=True, exist_ok=True)

    def provider_for(node: CompiledNode) -> Provider:
        if node.model is None:
            raise RuntimeFailure(f"node {node.id!r} has no model", details={"code": "MG-1003"})
        key = node.model.metadata.id
        if key not in providers:
            providers[key] = provider_factory(node.model)
        return providers[key]

    trace.emit(
        "run.started",
        graph={
            "id": graph.id,
            "version": graph.version,
            "hash": graph.graph_hash,
            "source": graph.source,
        },
        target="portable",
        task=task,
        budget=state.budget.snapshot().model_dump(mode="json"),
        models={
            node.id: {
                "ref": node.model_ref,
                "hash": node.model_hash,
                "provider": node.model.spec.provider if node.model else None,
            }
            for node in graph.nodes.values()
            if node.model_ref
        },
    )

    outcome = "succeeded"
    error: ModelGraphError | None = None
    result_artifacts: list[Artifact] = []
    try:
        _walk(
            state,
            graph.entry,
            depth=0,
            trace=trace,
            provider_for=provider_for,
            providers=providers,
            executor_factory=lambda node: Executor(
                provider_for(node), budget=state.budget, trace=trace
            ),
        )
        result_artifacts = state.results()
        trace.emit(
            "run.finished",
            outcome=outcome,
            usage=_usage(state),
            artifacts=[
                {"name": artifact.name, "hash": artifact.hash} for artifact in result_artifacts
            ],
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
        )
    except ModelGraphError as exc:
        error = exc
        outcome = "failed"
        trace.emit(
            "run.failed",
            outcome=outcome,
            code=exc.details.get("code", exc.code),
            message=exc.message,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        error = RuntimeFailure(f"internal runtime error: {exc}", details={"code": "MG-2000"})
        outcome = "failed"
        trace.emit("run.failed", outcome=outcome, code="MG-2000", message=str(exc))
    finally:
        for provider in providers.values():
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    result = RunOutcome(
        run_id=state.run_id,
        outcome=outcome,
        artifacts=result_artifacts,
        path=state.path,
        usage=_usage(state),
        error=(
            {
                "code": error.details.get("code", error.code),
                "message": error.message,
                "path": error.path,
                "details": error.details,
            }
            if error
            else None
        ),
        run_dir=run_dir if persist else None,
    )
    if persist:
        result.manifest = _persist_run(state, trace, result, run_dir)
    return result


def _walk(
    state: RuntimeState,
    start: str,
    *,
    depth: int,
    trace: TraceRecorder,
    provider_for: Callable[[CompiledNode], Provider],
    providers: dict[str, Provider],
    executor_factory: Callable[[CompiledNode], Executor],
) -> None:
    current: str | None = start
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
        finished = False
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
            current = route
        elif node.kind == NodeKind.subgraph:
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
            _walk(
                child,
                inner.entry,
                depth=depth + 1,
                trace=trace,
                provider_for=provider_for,
                providers=providers,
                executor_factory=executor_factory,
            )
            _map_subgraph_outputs(state, child, node)
            trace.emit("subgraph.exited", node=node.id, child_run_id=child.run_id)
            outgoing = state.graph.outgoing(node.id)
            if outgoing:
                current = outgoing[0].to
            else:
                finished = True
        else:
            executor = executor_factory(node)
            WorkerExecutor(executor).run(node, state.graph, state)
            state.add_path(PathEntry(node=node.id, kind=node.kind.value))
            outgoing = state.graph.outgoing(node.id)
            if outgoing:
                current = outgoing[0].to
            else:
                finished = True
        trace.emit("node.exited", node=node.id, kind=node.kind.value, depth=depth)
        if finished:
            return


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
    budget = state.budget
    return {
        "steps": budget.used_steps,
        "input_tokens": budget.used_input_tokens,
        "output_tokens": budget.used_output_tokens,
        "total_tokens": budget.used_tokens,
        "cost_usd": round(budget.used_cost_usd, 8),
        "elapsed_seconds": round(budget.elapsed_seconds, 3),
    }


def _persist_run(
    state: RuntimeState,
    trace: TraceRecorder,
    result: RunOutcome,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for artifact in [*state.inputs, *state.artifacts]:
        if artifact.hash:
            digest = artifact.hash.split(":", 1)[-1]
            (artifacts_dir / digest).write_text(artifact.content, encoding="utf-8")
    trace.write(run_dir / "trace.jsonl")
    (run_dir / "envelope.json").write_text(_json_dump(state.envelope().to_dict()), encoding="utf-8")
    finished_at = utc_now()
    models: list[dict[str, Any]] = []
    for node in state.graph.nodes.values():
        if not node.model_ref:
            continue
        models.append(
            {
                "node": node.id,
                "ref": node.model_ref,
                "hash": node.model_hash,
                "provider": node.model.spec.provider if node.model else None,
                "model": node.model.spec.model if node.model else None,
                "version": node.model.metadata.version if node.model else None,
            }
        )
    manifest: dict[str, Any] = {
        "run_id": state.run_id,
        "deployment_revision": f"rev_{state.run_id}",
        "target": "portable",
        "renderer_version": RENDERER_VERSION,
        "parser_version": PARSER_VERSION,
        "outcome": result.outcome,
        "graph": {
            "id": state.graph.id,
            "version": state.graph.version,
            "hash": state.graph.graph_hash,
            "source": state.graph.source,
        },
        "request": {"task": state.task},
        "models": models,
        "budget": state.budget.snapshot().model_dump(mode="json"),
        "usage": result.usage,
        "path": [entry.model_dump(mode="json", exclude_none=True) for entry in state.path],
        "artifacts": [
            {
                "name": artifact.name,
                "type": artifact.type.value,
                "hash": artifact.hash,
                "producer": artifact.producer,
                "bytes": len(artifact.content),
            }
            for artifact in state.artifacts
        ],
        "trace": "trace.jsonl",
        "started_at": state.started_at,
        "finished_at": finished_at,
        "error": result.error,
    }
    (run_dir / "run.json").write_text(_json_dump(manifest), encoding="utf-8")
    return manifest


def _json_dump(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"
