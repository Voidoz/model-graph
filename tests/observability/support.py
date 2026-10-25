"""Offline mock-run helpers for observability tests."""

from __future__ import annotations

import json
from pathlib import Path

from modelgraph.compiler import compile_graph
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.registry import Registry
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.engine import run_graph

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
TASK = "Write a debounce function in TypeScript."
CODING_DECISION = json.dumps(
    {
        "decision": "route",
        "to": "coding",
        "confidence": 0.95,
        "reason": "the request asks for TypeScript code",
    }
)


def compile_example(name: str = "hello-triage.graph.yaml"):
    # Scope the registry to the example manifests so this suite is independent of the
    # contents of the repository-level registry/ directory.
    registry = Registry.from_paths([EXAMPLES / "models"], [EXAMPLES])
    result = compile_graph(EXAMPLES / name, registry=registry, strict_models=True)
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def factory(scenario: MockScenario):
    def build(model: ModelManifest) -> MockProvider:
        return MockProvider(scenario, max_context=model.spec.context_window)

    return build


def run_mock(runs_dir: Path, *, task: str = TASK) -> Path:
    outcome = run_graph(
        compile_example(),
        task,
        factory(MockScenario(responses={"triage": [CODING_DECISION]})),
        runs_dir=runs_dir,
    )
    assert outcome.ok, outcome.error
    assert outcome.run_dir is not None
    return outcome.run_dir


def run_budget_failure(runs_dir: Path) -> Path:
    budget = Budget(
        max_steps=1,
        max_depth=2,
        max_total_tokens=24_000,
        max_cost_usd=0.05,
        max_wall_seconds=120,
    )
    outcome = run_graph(
        compile_example(),
        TASK,
        factory(MockScenario(responses={"triage": [CODING_DECISION]})),
        runs_dir=runs_dir,
        budget=budget,
    )
    assert outcome.outcome == "failed"
    assert outcome.run_dir is not None
    return outcome.run_dir
