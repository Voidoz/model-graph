"""Rules/keyword baseline output and parser compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from training_support import ROUTES, synthetic_dataset

from modelgraph.compiler import compile_graph
from modelgraph.registry import Registry
from modelgraph.runtime.executors import RouterExecutor
from modelgraph.training.features import RouterView
from modelgraph.training.rules import RuleConfig, RuleRouter, evaluate_rules

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "examples" / "hello-triage.graph.yaml"


def _view(task: str) -> RouterView:
    return RouterView(request=task, routes=ROUTES)


def test_rule_decision_is_canonical_router_json() -> None:
    router = RuleRouter()
    decision = router.decide(_view("Write code that sorts records."))
    payload = decision.to_json()
    assert payload == {
        "decision": "route",
        "to": "coding",
        "confidence": payload["confidence"],
        "reason": payload["reason"],
    }
    assert 0 <= payload["confidence"] <= 1
    assert payload["reason"]
    assert json.loads(decision.text()) == payload


def test_rule_abstains_without_a_signal() -> None:
    router = RuleRouter(RuleConfig(min_score=5.0))
    decision = router.decide(_view("Hello there."))
    assert decision.action == "abstain"
    assert decision.route is None
    payload = decision.to_json()
    assert payload["decision"] == "abstain"
    assert 0 <= payload["confidence"] <= 1


def test_rule_ambiguity_margin_abstains() -> None:
    router = RuleRouter(RuleConfig(min_score=1.0, ambiguity_margin=0.9))
    decision = router.decide(_view("Write code about a poem and a story."))
    assert decision.action == "abstain"


def test_rule_json_is_accepted_by_the_runtime_parser() -> None:
    registry = Registry.from_paths([ROOT / "registry/models"], [ROOT / "examples"])
    compiled = compile_graph(GRAPH, registry=registry, strict_models=True)
    assert compiled.ok, compiled.errors
    assert compiled.graph is not None
    node = compiled.graph.node("triage")
    router = RuleRouter(RuleConfig(keywords={"coding": ["typescript", "debounce"]}))
    text = router.decide(_view("Write a TypeScript debounce function.")).text()
    parsed = RouterExecutor(cast(Any, None))._parse(text, node, compiled.graph)
    assert parsed.action == "route"
    assert parsed.to == "coding"
    assert 0 <= parsed.confidence <= 1


def test_evaluate_rules_reports_cost_and_abstain_metrics() -> None:
    dataset = synthetic_dataset()
    result = evaluate_rules(dataset.split("test"))
    metrics = result["metrics"]
    assert metrics["cases"] == 5
    assert metrics["ambiguous_cases"] == 1
    assert metrics["oracle_agreement"] is not None
    assert metrics["expected_cost_per_case"] is not None
    assert metrics["coverage"] is not None
    assert len(result["predictions"]) == 5
    assert result["config"]["min_score"] == 1.0
