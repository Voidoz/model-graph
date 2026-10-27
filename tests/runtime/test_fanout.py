"""Fan-out selection, concurrent branch execution, joins, budgets, and compatibility.

These tests use only the offline mock provider and the normal compiler/runtime. Concurrency
is real: branch calls block on a barrier or sleep in a worker thread, and the scheduler's
thread pool is the thing that lets them overlap.
"""

from __future__ import annotations

import json
import threading
import time
from io import StringIO
from pathlib import Path
from typing import Any

import yaml

from modelgraph.compiler import compile_graph
from modelgraph.contracts.schema import validate_document
from modelgraph.observability import RecordingSpanExporter, StructuredLogger
from modelgraph.providers.base import (
    Capabilities,
    ChatRequest,
    ChatResponse,
    Health,
    Usage,
)
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.registry import Registry
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.cache import ResponseCache
from modelgraph.runtime.engine import _has_fanout, run_graph
from modelgraph.runtime.envelope import RuntimeState
from modelgraph.runtime.executors import max_output_tokens
from modelgraph.runtime.renderer import render_prompt

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
TASK = "Write a short implementation plan."

ROUTER_ALL = json.dumps(
    {
        "decision": "route",
        "to": "alpha",
        "confidence": 0.9,
        "reason": "run the ensemble",
    }
)


def _decision(target: str) -> str:
    return json.dumps({"decision": "route", "to": target, "confidence": 0.9, "reason": "pick"})


def _ranked(routes: list[tuple[str, float]], *, confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "decision": "route",
            "routes": [{"to": target, "confidence": score} for target, score in routes],
            "confidence": confidence,
            "reason": "ranked",
        }
    )


def _document(
    *,
    branches: tuple[str, ...] = ("alpha", "beta"),
    selection: str = "all",
    k: int | None = None,
    join_reads: list[str] | None = None,
    priorities: dict[str, int] | None = None,
) -> dict[str, Any]:
    models = {"alpha": "coder-7b", "beta": "story-7b", "gamma": "js-specialist-3b"}
    routing: dict[str, Any] = {
        "selection": selection,
        "confidence_threshold": 0.5,
        "on_low_confidence": "error",
    }
    if k is not None:
        routing["k"] = k
    nodes: list[dict[str, Any]] = [
        {
            "id": "triage",
            "kind": "router",
            "description": "Select the specialists for this request.",
            "model": "router-small",
            "reads": ["task"],
            "routing": routing,
        }
    ]
    for name in branches:
        nodes.append(
            {
                "id": name,
                "kind": "worker",
                "description": f"Writes the {name} candidate.",
                "model": models[name],
                "reads": ["task"],
                "instruction": f"Write the {name} candidate.",
                "writes": [{"name": f"{name}_out", "type": "text"}],
            }
        )
    reads = join_reads if join_reads is not None else [f"{name}_out" for name in branches]
    nodes.append(
        {
            "id": "merge",
            "kind": "joiner",
            "description": "Merges the supplied candidates.",
            "model": "coder-7b",
            "reads": reads,
            "strategy": "synthesize",
            "instruction": "Merge the supplied candidates.",
            "writes": [{"name": "answer", "type": "text"}],
        }
    )
    edges: list[dict[str, Any]] = []
    for name in branches:
        edges.append(
            {
                "from": "triage",
                "to": name,
                "when": f"The request matches {name}.",
                "artifact": "text",
                "priority": (priorities or {}).get(name, 0),
            }
        )
    for name in branches:
        edges.append({"from": name, "to": "merge"})
    return {
        "apiVersion": "modelgraph/v0",
        "kind": "Graph",
        "metadata": {
            "id": f"fanout-{'-'.join(branches)}-{selection.replace('_', '-')}",
            "name": "Fan-out Test Graph",
            "version": "1.0.0",
            "description": "A fan-out graph used by the runtime test suite.",
        },
        "spec": {
            "entry": "triage",
            "inputs": [{"name": "task", "type": "text"}],
            "exit": ["answer"],
            "defaults": {
                "max_steps": 12,
                "max_depth": 2,
                "max_total_tokens": 32_000,
                "max_cost_usd": 0.10,
                "max_wall_seconds": 120,
            },
            "nodes": nodes,
            "edges": edges,
        },
    }


def _compile(tmp_path: Path, **kwargs: Any):
    document = _document(**kwargs)
    path = tmp_path / f"{document['metadata']['id']}.graph.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    result = compile_graph(path, strict_models=True)
    assert result.ok, result.errors
    assert result.graph is not None
    assert not validate_document(document, "graph")
    return result.graph


def _factory(scenario: MockScenario):
    providers: list[MockProvider] = []

    def build(model):
        provider = MockProvider(scenario, max_context=model.spec.context_window)
        providers.append(provider)
        return provider

    return build, providers


def _calls(providers: list[MockProvider]) -> list[ChatRequest]:
    return [call for provider in providers for call in provider.calls]


class _NestedFanoutProvider:
    """A stateless provider so two concurrent subgraph invocations cannot race on state."""

    name = "mock"

    def __init__(self, model) -> None:
        self.model = model

    def capabilities(self) -> Capabilities:
        return Capabilities(chat=True, json_mode=True, max_context=32768)

    def health(self) -> Health:
        return Health(ok=True, detail="nested fan-out test provider")

    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.node_id == "inner-triage":
            target = "walpha"
            text = _decision(target)
        elif request.node_id == "outer-triage":
            text = _decision("sub-one")
        elif request.node_id in {"walpha", "wbeta"}:
            text = request.node_id.upper()
        else:
            text = "MERGED"
        return ChatResponse(text=text, usage=Usage(10, 5), finish_reason="stop")


def _trace(outcome) -> list[dict[str, Any]]:
    assert outcome.run_dir is not None
    return [
        json.loads(line)
        for line in (outcome.run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _projection(events: list[dict[str, Any]]) -> list[tuple[str, str | None, int | None]]:
    return [(event["event"], event.get("node"), event.get("start_order")) for event in events]


def _all_responses(branches: tuple[str, ...] = ("alpha", "beta", "gamma")) -> MockScenario:
    responses = {name: [name.upper()] for name in branches}
    responses["triage"] = [ROUTER_ALL]
    responses["merge"] = ["MERGED"]
    return MockScenario(responses=responses)


def test_three_branch_all_join_receives_every_input(tmp_path: Path) -> None:
    graph = _compile(tmp_path, branches=("alpha", "beta", "gamma"), selection="all")
    scenario = _all_responses()
    factory, providers = _factory(scenario)
    outcome = run_graph(graph, TASK, factory, runs_dir=tmp_path / "runs")
    assert outcome.ok, outcome.error
    assert sorted(call.node_id for call in _calls(providers) if call.node_id != "triage") == [
        "alpha",
        "beta",
        "gamma",
        "merge",
    ]
    merge_call = next(call for call in _calls(providers) if call.node_id == "merge")
    prompt = merge_call.prompt_text()
    for name in ("ALPHA", "BETA", "GAMMA"):
        assert name in prompt

    fanout = [branch for branch in outcome.branches if branch["kind"] == "fanout"]
    assert [branch["target"] for branch in fanout] == ["alpha", "beta", "gamma"]
    assert all(branch["contributed"] for branch in fanout)
    assert all(branch["total_tokens"] > 0 for branch in fanout)
    assert (
        sum(branch["total_tokens"] for branch in outcome.branches) == outcome.usage["total_tokens"]
    )
    joined = [event for event in _trace(outcome) if event["event"] == "joiner.inputs"]
    assert joined
    assert [item["name"] for item in joined[0]["received"]] == [
        "alpha_out",
        "beta_out",
        "gamma_out",
    ]
    joined_names = {
        item["name"]
        for event in _trace(outcome)
        if event["event"] == "joiner.joined"
        for item in event["contributions"]
    }
    assert joined_names == {"alpha_out", "beta_out", "gamma_out"}


def test_two_branch_all(tmp_path: Path) -> None:
    graph = _compile(tmp_path, branches=("alpha", "beta"), selection="all")
    factory, providers = _factory(_all_responses(("alpha", "beta")))
    outcome = run_graph(graph, TASK, factory, runs_dir=tmp_path / "runs")
    assert outcome.ok, outcome.error
    assert outcome.artifacts[0].content == "MERGED"
    assert {call.node_id for call in _calls(providers)} == {"triage", "alpha", "beta", "merge"}
    assert [branch["target"] for branch in outcome.branches if branch["kind"] == "fanout"] == [
        "alpha",
        "beta",
    ]


def test_top_k_ties_use_confidence_priority_then_id(tmp_path: Path) -> None:
    graph = _compile(
        tmp_path,
        branches=("alpha", "beta", "gamma"),
        selection="top_k",
        k=2,
        join_reads=["alpha_out", "beta_out"],
        priorities={"alpha": 10, "beta": 10, "gamma": 10},
    )
    scenario = MockScenario(
        responses={
            "triage": [_ranked([("beta", 0.8), ("alpha", 0.8), ("gamma", 0.8)])],
            "alpha": ["ALPHA"],
            "beta": ["BETA"],
            "merge": ["MERGED"],
        }
    )
    factory, providers = _factory(scenario)
    outcome = run_graph(graph, TASK, factory, runs_dir=tmp_path / "runs")
    assert outcome.ok, outcome.error
    targets = [branch["target"] for branch in outcome.branches if branch["kind"] == "fanout"]
    assert targets == ["alpha", "beta"]
    assert "gamma" not in {call.node_id for call in _calls(providers)}
    assert [(entry.node, entry.to) for entry in outcome.path] == [
        ("triage", "alpha"),
        ("triage", "beta"),
        ("alpha", None),
        ("beta", None),
        ("merge", None),
    ]

    # Input order must not change a tie. Reverse the ranking and run again.
    scenario_reversed = MockScenario(
        responses={
            "triage": [_ranked([("gamma", 0.8), ("beta", 0.8), ("alpha", 0.8)])],
            "alpha": ["ALPHA"],
            "beta": ["BETA"],
            "merge": ["MERGED"],
        }
    )
    factory_reversed, _ = _factory(scenario_reversed)
    outcome_reversed = run_graph(graph, TASK, factory_reversed, runs_dir=tmp_path / "runs-reversed")
    assert outcome_reversed.ok, outcome_reversed.error
    assert [
        branch["target"] for branch in outcome_reversed.branches if branch["kind"] == "fanout"
    ] == ["alpha", "beta"]


def test_top_k_one_branch_with_missing_join_input_fails_closed(tmp_path: Path) -> None:
    graph = _compile(
        tmp_path,
        branches=("alpha", "beta"),
        selection="top_k",
        k=1,
        join_reads=["alpha_out", "beta_out"],
    )
    scenario = MockScenario(
        responses={
            "triage": [_ranked([("alpha", 0.9), ("beta", 0.1)])],
            "alpha": ["ALPHA"],
            "merge": ["MERGED"],
        }
    )
    factory, providers = _factory(scenario)
    outcome = run_graph(graph, TASK, factory, runs_dir=tmp_path / "runs")
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-2201"
    assert "beta_out" in outcome.error["details"]["missing"]
    assert "merge" not in {call.node_id for call in _calls(providers)}
    assert any(event["event"] == "joiner.incomplete" for event in _trace(outcome))


def test_one_branch_failure_is_reported_without_hanging_the_join(tmp_path: Path) -> None:
    graph = _compile(tmp_path, branches=("alpha", "beta"), selection="all")
    scenario = MockScenario(
        responses={
            "triage": [ROUTER_ALL],
            "beta": ["BETA"],
            "merge": ["MERGED"],
        },
        failures={"alpha": "MG-3001"},
    )
    factory, providers = _factory(scenario)
    started = time.monotonic()
    outcome = run_graph(
        graph,
        TASK,
        factory,
        runs_dir=tmp_path / "runs",
        max_concurrency=2,
        join_timeout_seconds=5,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 5
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-3001"
    assert "merge" not in {call.node_id for call in _calls(providers)}
    fanout = {branch["target"]: branch for branch in outcome.branches if branch["kind"] == "fanout"}
    assert fanout["alpha"]["outcome"] == "failed"
    assert fanout["alpha"]["error"]["code"] == "MG-3001"
    assert not any(event["event"] == "joiner.joined" for event in _trace(outcome))


def test_join_timeout_is_typed(tmp_path: Path) -> None:
    graph = _compile(tmp_path, branches=("alpha", "beta"), selection="all")

    class SlowProvider(MockProvider):
        def chat(self, request: ChatRequest) -> ChatResponse:
            if request.node_id == "beta":
                time.sleep(0.6)
            return super().chat(request)

    scenario = _all_responses(("alpha", "beta"))
    providers: list[SlowProvider] = []

    def factory(model):
        provider = SlowProvider(scenario, max_context=model.spec.context_window)
        providers.append(provider)
        return provider

    started = time.monotonic()
    outcome = run_graph(
        graph,
        TASK,
        factory,
        runs_dir=tmp_path / "runs",
        max_concurrency=2,
        join_timeout_seconds=0.1,
    )
    elapsed = time.monotonic() - started
    assert elapsed >= 0.1
    assert elapsed < 5
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-2105"
    assert outcome.error["details"]["join"] == "merge"
    assert "merge" not in {call.node_id for call in _calls(providers)}
    assert any(event["event"] == "join.timeout" for event in _trace(outcome))


def test_budget_preflight_vetoes_fanout_before_any_branch_starts(tmp_path: Path) -> None:
    graph = _compile(tmp_path, branches=("alpha", "beta"), selection="all")
    scenario = _all_responses(("alpha", "beta"))
    factory, _ = _factory(scenario)
    warm = run_graph(graph, TASK, factory, runs_dir=tmp_path / "warm")
    assert warm.ok, warm.error
    router_tokens = next(
        branch["total_tokens"] for branch in warm.branches if branch["kind"] == "root"
    )

    estimate_budget = Budget(
        max_steps=12,
        max_depth=2,
        max_total_tokens=32_000,
        max_cost_usd=0.10,
        max_wall_seconds=120,
    )
    state = RuntimeState.create(graph, TASK)
    estimates = []
    for node_id in ("alpha", "beta"):
        node = graph.node(node_id)
        prompt = render_prompt(node, graph, state)
        estimates.append(estimate_budget.estimate_tokens(prompt.messages, max_output_tokens(node)))
    budget = Budget(
        max_steps=12,
        max_depth=2,
        max_total_tokens=router_tokens + sum(estimates) - 1,
        max_cost_usd=0.10,
        max_wall_seconds=120,
    )
    scenario_veto = _all_responses(("alpha", "beta"))
    factory_veto, providers_veto = _factory(scenario_veto)
    outcome = run_graph(
        graph,
        TASK,
        factory_veto,
        runs_dir=tmp_path / "veto",
        budget=budget,
        max_concurrency=2,
    )
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-4002"
    assert {call.node_id for call in _calls(providers_veto)} == {"triage"}
    fanout_branches = [branch for branch in outcome.branches if branch["kind"] == "fanout"]
    assert len(fanout_branches) == 2
    assert all(branch["outcome"] == "not_started" for branch in fanout_branches)
    assert all(branch["total_tokens"] == 0 for branch in fanout_branches)
    preflight = [event for event in _trace(outcome) if event["event"] == "fanout.preflight"]
    assert preflight and preflight[-1]["accepted"] is False


def test_branches_run_concurrently_with_a_barrier(tmp_path: Path) -> None:
    graph = _compile(tmp_path, branches=("alpha", "beta"), selection="all")
    barrier = threading.Barrier(2, timeout=3)

    class BarrierProvider(MockProvider):
        def chat(self, request: ChatRequest) -> ChatResponse:
            if request.node_id in {"alpha", "beta"}:
                barrier.wait()
            return super().chat(request)

    scenario = _all_responses(("alpha", "beta"))
    providers: list[BarrierProvider] = []

    def factory(model):
        provider = BarrierProvider(scenario, max_context=model.spec.context_window)
        providers.append(provider)
        return provider

    outcome = run_graph(
        graph,
        TASK,
        factory,
        runs_dir=tmp_path / "runs",
        max_concurrency=2,
        join_timeout_seconds=5,
    )
    assert outcome.ok, outcome.error


def test_concurrency_limit_bounds_in_flight_branches(tmp_path: Path) -> None:
    graph = _compile(tmp_path, branches=("alpha", "beta", "gamma"), selection="all")
    lock = threading.Lock()
    tracker = {"active": 0, "max_active": 0}

    class TrackingProvider(MockProvider):
        def chat(self, request: ChatRequest) -> ChatResponse:
            with lock:
                tracker["active"] += 1
                tracker["max_active"] = max(tracker["max_active"], tracker["active"])
            try:
                time.sleep(0.12)
                return super().chat(request)
            finally:
                with lock:
                    tracker["active"] -= 1

    scenario = _all_responses()
    providers: list[TrackingProvider] = []

    def factory(model):
        provider = TrackingProvider(scenario, max_context=model.spec.context_window)
        providers.append(provider)
        return provider

    outcome = run_graph(
        graph,
        TASK,
        factory,
        runs_dir=tmp_path / "runs",
        max_concurrency=2,
        join_timeout_seconds=5,
    )
    assert outcome.ok, outcome.error
    assert tracker["max_active"] == 2
    worker_calls = [
        call for call in _calls(providers) if call.node_id in {"alpha", "beta", "gamma"}
    ]
    assert len(worker_calls) == 3


def test_artifacts_and_trace_are_deterministic_for_scripted_responses(tmp_path: Path) -> None:
    graph = _compile(tmp_path, branches=("alpha", "beta", "gamma"), selection="all")
    runs = []
    for index in range(2):
        factory, _ = _factory(_all_responses())
        outcome = run_graph(
            graph,
            TASK,
            factory,
            runs_dir=tmp_path / f"runs-{index}",
            max_concurrency=3,
        )
        assert outcome.ok, outcome.error
        runs.append(outcome)

    first, second = runs
    assert [(artifact.name, artifact.content, artifact.hash) for artifact in first.artifacts] == [
        (artifact.name, artifact.content, artifact.hash) for artifact in second.artifacts
    ]
    first_manifest = json.loads((first.run_dir / "run.json").read_text())
    second_manifest = json.loads((second.run_dir / "run.json").read_text())
    assert first_manifest["artifacts"] == second_manifest["artifacts"]

    def stable(events):
        return [
            (
                event["event"],
                event.get("node"),
                event.get("start_order"),
                event.get("branch_id"),
                event.get("target"),
                event.get("selection"),
            )
            for event in events
        ]

    assert stable(_trace(first)) == stable(_trace(second))

    def order_key(item):
        return item["start_order"] if item["start_order"] is not None else -1

    def normalized(branches):
        return [
            {key: value for key, value in branch.items() if key != "completion_order"}
            for branch in sorted(branches, key=order_key)
        ]

    assert normalized(first.branches) == normalized(second.branches)

    from modelgraph.runtime.replay import replay_run

    replay = replay_run(first.run_dir)
    assert replay.ok, replay.mismatches
    assert replay.mismatches == []


def test_single_selection_path_is_unchanged(tmp_path: Path) -> None:
    result = compile_graph(EXAMPLES / "hello-triage.graph.yaml", strict_models=True)
    assert result.ok, result.errors
    assert result.graph is not None
    assert not _has_fanout(result.graph)
    scenario = MockScenario(
        responses={
            "triage": [_decision("coding")],
            "coding": ["export const answer = 42;\n"],
        }
    )
    factory, providers = _factory(scenario)
    outcome = run_graph(result.graph, "task", factory, runs_dir=tmp_path / "runs")
    assert outcome.ok, outcome.error
    assert [entry.node for entry in outcome.path] == ["triage", "coding"]
    assert outcome.usage["steps"] == 2
    assert outcome.branches == []
    assert outcome.execution["fanout"] is False
    assert len(_calls(providers)) == 2


def test_fanout_is_cache_compatible(tmp_path: Path) -> None:
    graph = _compile(tmp_path, branches=("alpha", "beta"), selection="all")
    cache = ResponseCache(tmp_path / "cache", mode="on")
    first_factory, _ = _factory(_all_responses(("alpha", "beta")))
    first = run_graph(
        graph,
        TASK,
        first_factory,
        runs_dir=tmp_path / "first",
        cache=cache,
    )
    assert first.ok, first.error

    second_scenario = _all_responses(("alpha", "beta"))
    second_factory, second_providers = _factory(second_scenario)
    second = run_graph(
        graph,
        TASK,
        second_factory,
        runs_dir=tmp_path / "second",
        cache=cache,
    )
    assert second.ok, second.error
    assert _calls(second_providers) == []
    assert second.cache["hits"] == 4
    assert [(artifact.name, artifact.hash) for artifact in first.artifacts] == [
        (artifact.name, artifact.hash) for artifact in second.artifacts
    ]


def test_fanout_security_and_observability_compatibility(tmp_path: Path) -> None:
    graph = _compile(tmp_path, branches=("alpha", "beta"), selection="all")
    secret = "sk-proj-abcdefghijklmnop"
    scenario = MockScenario(
        responses={
            "triage": [ROUTER_ALL],
            "alpha": [f"candidate with {secret}"],
            "beta": ["BETA"],
            "merge": ["MERGED"],
        }
    )
    factory, _ = _factory(scenario)
    stream = StringIO()
    logger = StructuredLogger(stream=stream)
    spans = RecordingSpanExporter()
    outcome = run_graph(
        graph,
        f"task with {secret}",
        factory,
        runs_dir=tmp_path / "runs",
        cache=ResponseCache(tmp_path / "cache", mode="on"),
        logger=logger,
        span_exporter=spans,
    )
    assert outcome.ok, outcome.error
    assert (outcome.run_dir / "redaction.json").exists()
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            outcome.run_dir / "run.json",
            outcome.run_dir / "trace.jsonl",
            outcome.run_dir / "envelope.json",
            outcome.run_dir / "redaction.json",
        ]
    )
    assert secret not in persisted
    assert outcome.cache["skipped"] >= 1

    node_names = {
        span.attributes.get("node") for span in spans.spans if span.name == "modelgraph.node"
    }
    assert {"triage", "alpha", "beta", "merge"} <= node_names
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    logged_nodes = {record.get("node") for record in records if record["event"] == "node.entered"}
    assert {"triage", "alpha", "beta", "merge"} <= logged_nodes
    assert all("task" not in record and "messages" not in record for record in records)


def test_nested_subgraph_fanout_keeps_join_branches_separate(tmp_path: Path) -> None:
    """Two invocations of one fan-out subgraph must not share a joiner branch id.

    Both inner graphs declare a joiner named ``merge`` and run concurrently. Branch
    accounting is per invocation: a bare ``join:merge`` id would merge their node lists
    and ledgers into one bogus branch.
    """

    def node(node_id: str, kind: str, **fields: Any) -> dict[str, Any]:
        return {
            "id": node_id,
            "kind": kind,
            "description": f"The {node_id} node in the nested fan-out test.",
            **fields,
        }

    inner = {
        "apiVersion": "modelgraph/v0",
        "kind": "Graph",
        "metadata": {
            "id": "fanout-inner",
            "name": "Inner fan-out",
            "version": "1.0.0",
            "description": "A fan-out graph invoked twice by the outer graph.",
        },
        "spec": {
            "entry": "inner-triage",
            "inputs": [{"name": "task", "type": "text"}],
            "exit": ["inner_result"],
            "defaults": {
                "max_steps": 20,
                "max_depth": 4,
                "max_total_tokens": 100_000,
                "max_cost_usd": 1.0,
                "max_wall_seconds": 60,
            },
            "nodes": [
                node(
                    "inner-triage",
                    "router",
                    model="router-small",
                    reads=["task"],
                    routing={
                        "selection": "all",
                        "confidence_threshold": 0.0,
                        "on_low_confidence": "error",
                    },
                ),
                node(
                    "walpha",
                    "worker",
                    model="coder-7b",
                    reads=["task"],
                    instruction="Write alpha.",
                    writes=[{"name": "alpha_out", "type": "text"}],
                ),
                node(
                    "wbeta",
                    "worker",
                    model="story-7b",
                    reads=["task"],
                    instruction="Write beta.",
                    writes=[{"name": "beta_out", "type": "text"}],
                ),
                node(
                    "merge",
                    "joiner",
                    model="coder-7b",
                    reads=["alpha_out", "beta_out"],
                    instruction="Merge alpha and beta.",
                    writes=[{"name": "inner_result", "type": "text"}],
                ),
            ],
            "edges": [
                {
                    "from": "inner-triage",
                    "to": "walpha",
                    "when": "Alpha fits.",
                    "priority": 1,
                },
                {
                    "from": "inner-triage",
                    "to": "wbeta",
                    "when": "Beta fits.",
                    "priority": 0,
                },
                {"from": "walpha", "to": "merge"},
                {"from": "wbeta", "to": "merge"},
            ],
        },
    }
    outer = {
        "apiVersion": "modelgraph/v0",
        "kind": "Graph",
        "metadata": {
            "id": "fanout-outer",
            "name": "Outer fan-out",
            "version": "1.0.0",
            "description": "Invokes the same fan-out subgraph twice concurrently.",
        },
        "spec": {
            "entry": "outer-triage",
            "inputs": [{"name": "task", "type": "text"}],
            "exit": ["answer"],
            "defaults": {
                "max_steps": 30,
                "max_depth": 4,
                "max_total_tokens": 100_000,
                "max_cost_usd": 1.0,
                "max_wall_seconds": 60,
            },
            "nodes": [
                node(
                    "outer-triage",
                    "router",
                    model="router-small",
                    reads=["task"],
                    routing={
                        "selection": "all",
                        "confidence_threshold": 0.0,
                        "on_low_confidence": "error",
                    },
                ),
                node(
                    "sub-one",
                    "subgraph",
                    graph="fanout-inner@1.0.0",
                    reads=["task"],
                    input_map={"task": "task"},
                    output_map={"s1": "inner_result"},
                    writes=[{"name": "s1", "type": "text"}],
                ),
                node(
                    "sub-two",
                    "subgraph",
                    graph="fanout-inner@1.0.0",
                    reads=["task"],
                    input_map={"task": "task"},
                    output_map={"s2": "inner_result"},
                    writes=[{"name": "s2", "type": "text"}],
                ),
                node(
                    "outer-merge",
                    "joiner",
                    model="coder-7b",
                    reads=["s1", "s2"],
                    instruction="Merge the two subgraph answers.",
                    writes=[{"name": "answer", "type": "text"}],
                ),
            ],
            "edges": [
                {
                    "from": "outer-triage",
                    "to": "sub-one",
                    "when": "The first subtree fits.",
                    "priority": 1,
                },
                {
                    "from": "outer-triage",
                    "to": "sub-two",
                    "when": "The second subtree fits.",
                    "priority": 0,
                },
                {"from": "sub-one", "to": "outer-merge"},
                {"from": "sub-two", "to": "outer-merge"},
            ],
        },
    }
    (tmp_path / "fanout-inner.graph.yaml").write_text(
        yaml.safe_dump(inner, sort_keys=False), encoding="utf-8"
    )
    outer_path = tmp_path / "fanout-outer.graph.yaml"
    outer_path.write_text(yaml.safe_dump(outer, sort_keys=False), encoding="utf-8")
    registry = Registry.from_paths([ROOT / "registry/models"], [tmp_path])
    compiled = compile_graph(
        outer_path, registry=registry, strict_models=True, graph_paths=[tmp_path]
    )
    assert compiled.ok, compiled.errors
    assert compiled.graph is not None

    outcome = run_graph(
        compiled.graph,
        TASK,
        lambda model: _NestedFanoutProvider(model),
        runs_dir=tmp_path / "runs",
        max_concurrency=4,
        join_timeout_seconds=5,
    )
    assert outcome.ok, outcome.error
    assert outcome.artifacts[0].content == "MERGED"

    inner_joins = [
        branch
        for branch in outcome.branches
        if branch["kind"] == "join" and branch["target"] == "merge"
    ]
    assert len(inner_joins) == 2
    assert len({branch["branch_id"] for branch in inner_joins}) == 2
    assert sorted(branch["nodes"] for branch in inner_joins) == [["merge"], ["merge"]]
    assert sorted(branch["total_tokens"] for branch in inner_joins) == [15, 15]
    outer_joins = [
        branch
        for branch in outcome.branches
        if branch["kind"] == "join" and branch["target"] == "outer-merge"
    ]
    assert len(outer_joins) == 1
    # Ten provider calls, each 15 reported tokens; branch ledgers still sum to the run.
    assert sum(branch["total_tokens"] for branch in outcome.branches) == 150
    assert outcome.usage["total_tokens"] == 150


def test_compiler_rejects_invalid_selection_configuration(tmp_path: Path) -> None:
    missing_k = _document(selection="top_k", k=None)
    path = tmp_path / "missing-k.graph.yaml"
    path.write_text(yaml.safe_dump(missing_k, sort_keys=False), encoding="utf-8")
    result = compile_graph(path, strict_models=True)
    assert not result.ok
    assert any("'k' is a required property" in error for error in result.errors)

    k_too_large = _document(selection="top_k", k=5)
    path = tmp_path / "k-too-large.graph.yaml"
    path.write_text(yaml.safe_dump(k_too_large, sort_keys=False), encoding="utf-8")
    result = compile_graph(path, strict_models=True)
    assert not result.ok
    assert any("MG-1020" in error for error in result.errors)

    k_on_single = _document(selection="single", k=2)
    path = tmp_path / "k-on-single.graph.yaml"
    path.write_text(yaml.safe_dump(k_on_single, sort_keys=False), encoding="utf-8")
    result = compile_graph(path, strict_models=True)
    assert not result.ok
    assert any("False schema does not allow" in error for error in result.errors)
