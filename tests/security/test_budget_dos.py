"""Budget and denial-of-service limits: retries, repair, fan-out, subgraphs, profiles."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from security_helpers import (
    ROUTING_DECISION,
    compile_example,
    read_trace,
)

from modelgraph.contracts.graph import Defaults, LowConfidenceAction
from modelgraph.errors import RuntimeFailure
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.pruning import apply_budget_overrides
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.engine import run_graph
from modelgraph.runtime.envelope import RuntimeState
from modelgraph.runtime.executors import (
    MAX_CALLS_PER_NODE,
    MAX_PROVIDER_ATTEMPTS,
    Executor,
)
from modelgraph.runtime.renderer import render_prompt
from modelgraph.runtime.trace import TraceRecorder

LOW_CONFIDENCE = json.dumps(
    {
        "decision": "route",
        "to": "coding",
        "confidence": 0.10,
        "reason": "not sure",
    }
)


def test_retryable_failure_is_capped_at_three_attempts(tmp_path: Path) -> None:
    provider = MockProvider(MockScenario(failures={"triage": "MG-3002"}))
    outcome = run_graph(
        compile_example(),
        "Write code.",
        lambda _model: provider,
        runs_dir=tmp_path / "runs",
    )
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-3002"
    assert len(provider.calls) == MAX_PROVIDER_ATTEMPTS == 3
    events = read_trace(outcome.run_dir)
    requests = [event for event in events if event["event"] == "provider.request"]
    assert len(requests) == 3


def test_budget_vetoes_a_call_before_the_provider_is_invoked(tmp_path: Path) -> None:
    provider = MockProvider(MockScenario(responses={"triage": [ROUTING_DECISION]}))
    budget = Budget(
        max_steps=8,
        max_depth=2,
        max_total_tokens=1,
        max_cost_usd=0.0,
        max_wall_seconds=120,
    )
    outcome = run_graph(
        compile_example(),
        "Write code.",
        lambda _model: provider,
        runs_dir=tmp_path / "runs",
        budget=budget,
    )
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-4002"
    assert provider.calls == []


def test_router_low_confidence_retry_respects_max_retries(tmp_path: Path) -> None:
    graph = compile_example()
    node = graph.node("triage")
    routing = node.routing.model_copy(
        update={
            "confidence_threshold": 0.9,
            "on_low_confidence": LowConfidenceAction.retry,
        }
    )

    no_retry_node = replace(node, routing=routing.model_copy(update={"max_retries": 0}))
    no_retry_graph = replace(graph, nodes={**graph.nodes, "triage": no_retry_node})
    no_retry_provider = MockProvider(
        MockScenario(responses={"triage": [LOW_CONFIDENCE, LOW_CONFIDENCE, LOW_CONFIDENCE]})
    )
    no_retry = run_graph(
        no_retry_graph,
        "Write code.",
        lambda _model: no_retry_provider,
        runs_dir=tmp_path / "no-retry",
    )
    assert no_retry.outcome == "failed"
    assert no_retry.error is not None
    assert no_retry.error["code"] == "MG-2102"
    assert len(no_retry_provider.calls) == 1

    retry_node = replace(node, routing=routing.model_copy(update={"max_retries": 2}))
    retry_graph = replace(graph, nodes={**graph.nodes, "triage": retry_node})
    retry_provider = MockProvider(
        MockScenario(responses={"triage": [LOW_CONFIDENCE, LOW_CONFIDENCE, LOW_CONFIDENCE]})
    )
    retry = run_graph(
        retry_graph,
        "Write code.",
        lambda _model: retry_provider,
        runs_dir=tmp_path / "retry",
    )
    assert retry.outcome == "failed"
    assert retry.error is not None
    assert retry.error["code"] == "MG-2102"
    # One initial call plus at most one low-confidence retry; never above the configured cap.
    assert 1 < len(retry_provider.calls) <= 3
    assert len(retry_provider.calls) <= 2 + 1


def test_repair_loop_is_bounded(tmp_path: Path) -> None:
    bad = "not a router decision"
    provider = MockProvider(MockScenario(responses={"triage": [bad, bad, bad, bad]}))
    outcome = run_graph(
        compile_example(),
        "Write code.",
        lambda _model: provider,
        runs_dir=tmp_path / "runs",
    )
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-2101"
    assert len(provider.calls) == 2  # initial + one repair, not an open-ended loop


def test_subgraph_depth_cannot_exceed_the_configured_limit(tmp_path: Path) -> None:
    provider = MockProvider(
        MockScenario(
            responses={
                "domain-triage": [ROUTING_DECISION],
                "coding-triage": [ROUTING_DECISION],
            }
        )
    )
    budget = Budget(
        max_steps=10,
        max_depth=0,
        max_total_tokens=100_000,
        max_cost_usd=0.0,
        max_wall_seconds=120,
    )
    outcome = run_graph(
        compile_example("composed-assistant.graph.yaml"),
        "Write code.",
        lambda _model: provider,
        runs_dir=tmp_path / "runs",
        budget=budget,
    )
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-4006"
    # Only the outer router ran; the subgraph never invoked an inner model.
    assert len(provider.calls) == 1


def test_step_budget_stops_before_the_next_node(tmp_path: Path) -> None:
    provider = MockProvider(MockScenario(responses={"triage": [ROUTING_DECISION]}))
    budget = Budget(
        max_steps=1,
        max_depth=2,
        max_total_tokens=100_000,
        max_cost_usd=0.0,
        max_wall_seconds=120,
    )
    outcome = run_graph(
        compile_example(),
        "Write code.",
        lambda _model: provider,
        runs_dir=tmp_path / "runs",
        budget=budget,
    )
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-4001"
    assert len(provider.calls) == 1


def test_router_selects_one_branch_and_does_not_fan_out(tmp_path: Path) -> None:
    provider = MockProvider(MockScenario(responses={"triage": [ROUTING_DECISION]}))
    outcome = run_graph(
        compile_example(),
        "Write code.",
        lambda _model: provider,
        runs_dir=tmp_path / "runs",
    )
    assert outcome.ok, outcome.error
    assert [entry.node for entry in outcome.path] == ["triage", "coding"]
    assert len(provider.calls) == 2
    assert all(artifact.name == "answer" for artifact in outcome.artifacts)


def test_profile_override_cannot_raise_a_budget_or_exceed_hard_limits() -> None:
    defaults = Defaults(max_steps=4, max_depth=2, max_total_tokens=1000)
    budget = Budget.from_defaults(defaults)
    warnings = apply_budget_overrides(
        budget,
        {"max_steps": 8, "max_total_tokens": 500, "max_depth": 100},
        defaults=defaults,
    )
    assert budget.max_steps == 4
    assert budget.max_total_tokens == 500
    assert budget.max_depth == 2
    assert any("max_steps" in warning for warning in warnings)
    assert any("hard limit" in warning for warning in warnings)

    # Even an explicit --allow-budget-override cannot pass the schema-aligned hard ceiling.
    relaxed = Budget.from_defaults(defaults)
    warnings = apply_budget_overrides(
        relaxed,
        {"max_steps": 10_000},
        defaults=defaults,
        allow_raise=True,
    )
    assert relaxed.max_steps == 4
    assert warnings and "hard limit" in warnings[0]


def test_executor_has_a_hard_per_node_call_cap() -> None:
    """Defense in depth: a single executor cannot make unbounded provider calls."""

    graph = compile_example()
    node = graph.node("coding")
    state = RuntimeState.create(graph, "Write code.")
    prompt = render_prompt(node, graph, state)
    provider = MockProvider(MockScenario(responses={"coding": ["ok\n"] * (MAX_CALLS_PER_NODE + 1)}))
    budget = Budget(
        max_steps=64,
        max_depth=2,
        max_total_tokens=1_000_000,
        max_cost_usd=1.0,
        max_wall_seconds=120,
    )
    executor = Executor(provider, budget=budget, trace=TraceRecorder("call-cap"))
    for _ in range(MAX_CALLS_PER_NODE):
        executor.call(node, prompt, temperature=0.2, graph=graph)
    with pytest.raises(RuntimeFailure) as excinfo:
        executor.call(node, prompt, temperature=0.2, graph=graph)
    assert excinfo.value.details["code"] == "MG-4005"
    assert len(provider.calls) == MAX_CALLS_PER_NODE
