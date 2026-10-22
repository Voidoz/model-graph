import json
import shutil
from pathlib import Path

import pytest
import yaml

from modelgraph.compiler import compile_graph
from modelgraph.contracts.schema import validate_document
from modelgraph.errors import RuntimeFailure
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.engine import run_graph
from modelgraph.runtime.executors import parse_artifacts
from modelgraph.runtime.replay import replay_run

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"

CODING_DECISION = json.dumps(
    {
        "decision": "route",
        "to": "coding",
        "confidence": 0.95,
        "reason": "the request asks for TypeScript code",
    }
)
JS_DECISION = json.dumps(
    {
        "decision": "route",
        "to": "js-specialist",
        "confidence": 0.95,
        "reason": "the request asks for TypeScript code",
    }
)


def _compile(name: str = "hello-triage.graph.yaml"):
    result = compile_graph(EXAMPLES / name, strict_models=True)
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def _factory(scenario: MockScenario):
    def factory(model):
        return MockProvider(scenario, max_context=model.spec.context_window)

    return factory


def test_mock_run_succeeds_and_records_path(tmp_path: Path) -> None:
    scenario = MockScenario(
        responses={
            "triage": [CODING_DECISION],
            "coding": ["export const answer = 42;\n"],
        }
    )
    outcome = run_graph(
        _compile(),
        "Write a debounce function in TypeScript.",
        _factory(scenario),
        runs_dir=tmp_path,
    )
    assert outcome.ok
    assert outcome.error is None
    assert [entry.node for entry in outcome.path] == ["triage", "coding"]
    assert outcome.path[0].decision == "route"
    assert outcome.path[0].to == "coding"
    assert [artifact.name for artifact in outcome.artifacts] == ["answer"]
    assert outcome.artifacts[0].content == "export const answer = 42;\n"
    assert outcome.usage["steps"] == 2
    assert outcome.run_dir is not None


def test_run_writes_trace_manifest_envelope(tmp_path: Path) -> None:
    outcome = run_graph(
        _compile(),
        "Write a debounce function in TypeScript.",
        _factory(MockScenario(responses={"triage": [CODING_DECISION]})),
        runs_dir=tmp_path,
    )
    assert outcome.run_dir
    run_dir = outcome.run_dir
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["outcome"] == "succeeded"
    assert manifest["graph"]["hash"].startswith("sha256:")
    assert manifest["request"]["task"].startswith("Write a debounce")
    assert len(manifest["models"]) == 3
    assert (run_dir / "trace.jsonl").read_text().count("\n") > 5
    envelope = json.loads((run_dir / "envelope.json").read_text())
    assert not validate_document(envelope, "envelope")
    events = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "run.started"
    assert events[-1]["event"] == "run.finished"
    assert any(event["event"] == "decision.made" for event in events)


def test_budget_failure_is_typed(tmp_path: Path) -> None:
    budget = Budget(
        max_steps=1,
        max_depth=2,
        max_total_tokens=24000,
        max_cost_usd=0.05,
        max_wall_seconds=120,
    )
    outcome = run_graph(
        _compile(),
        "task",
        _factory(MockScenario(responses={"triage": [CODING_DECISION]})),
        runs_dir=tmp_path,
        budget=budget,
    )
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-4001"
    assert len(outcome.path) == 1


def test_low_confidence_falls_back(tmp_path: Path) -> None:
    decision = json.dumps(
        {"decision": "route", "to": "coding", "confidence": 0.1, "reason": "unsure"}
    )
    outcome = run_graph(
        _compile(),
        "task",
        _factory(MockScenario(responses={"triage": [decision]})),
        runs_dir=tmp_path,
    )
    assert outcome.ok
    assert [entry.node for entry in outcome.path] == ["triage", "coding"]
    assert outcome.path[0].decision == "abstain"


def test_invalid_router_decision_fails_after_repair(tmp_path: Path) -> None:
    outcome = run_graph(
        _compile(),
        "task",
        _factory(MockScenario(responses={"triage": ["not json at all"]})),
        runs_dir=tmp_path,
    )
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-2101"


def test_subgraph_execution(tmp_path: Path) -> None:
    scenario = MockScenario(
        responses={
            "domain-triage": [CODING_DECISION.replace('"coding"', '"coding"')],
            "coding-triage": [JS_DECISION],
            "js-specialist": ["const js = true;\n"],
        }
    )
    outcome = run_graph(
        _compile("composed-assistant.graph.yaml"),
        "Write a TypeScript function.",
        _factory(scenario),
        runs_dir=tmp_path,
    )
    assert outcome.ok
    assert outcome.artifacts[0].content == "const js = true;\n"
    assert any(entry.kind == "subgraph" for entry in outcome.path)
    assert outcome.artifacts[0].metadata.get("subgraph_run_id")


def test_multi_artifact_protocol(tmp_path: Path) -> None:
    data = yaml.safe_load((EXAMPLES / "hello-triage.graph.yaml").read_text())
    for node in data["spec"]["nodes"]:
        if node["id"] == "coding":
            node["writes"] = [
                {"name": "answer", "type": "code"},
                {"name": "notes", "type": "report"},
            ]
    path = tmp_path / "multi.graph.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    result = compile_graph(path, strict_models=True)
    assert result.ok, result.errors
    scenario = MockScenario(
        responses={
            "triage": [CODING_DECISION],
            "coding": [
                '<<<MG:artifact name="answer" type="code">>>\nconst x = 1;\n<<<MG:end>>>\n'
                '<<<MG:artifact name="notes" type="report">>>\nAssumes a browser.\n<<<MG:end>>>'
            ],
        }
    )
    outcome = run_graph(
        result.graph,
        "Write code.",
        _factory(scenario),
        runs_dir=tmp_path,
    )
    assert outcome.ok
    assert {artifact.name for artifact in outcome.artifacts} >= {"answer"}
    # The graph exit is answer; notes remains in the trace/envelope.
    manifest = json.loads((outcome.run_dir / "run.json").read_text())
    assert {item["name"] for item in manifest["artifacts"]} == {"answer", "notes"}


def test_replay_is_deterministic(tmp_path: Path) -> None:
    outcome = run_graph(
        _compile(),
        "Write a debounce function in TypeScript.",
        _factory(MockScenario(responses={"triage": [CODING_DECISION]})),
        runs_dir=tmp_path,
    )
    assert outcome.run_dir
    result = replay_run(outcome.run_dir)
    assert result.ok
    assert result.mismatches == []
    assert (outcome.run_dir / "replay.json").exists()


def test_replay_rejects_changed_graph(tmp_path: Path) -> None:
    source = tmp_path / "hello.graph.yaml"
    shutil.copy(EXAMPLES / "hello-triage.graph.yaml", source)
    result = compile_graph(source, strict_models=True)
    assert result.ok and result.graph
    outcome = run_graph(
        result.graph,
        "Write a debounce function in TypeScript.",
        _factory(MockScenario(responses={"triage": [CODING_DECISION]})),
        runs_dir=tmp_path / "runs",
    )
    assert outcome.run_dir
    data = yaml.safe_load(source.read_text())
    data["metadata"]["description"] += " Changed."
    source.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    replay = replay_run(outcome.run_dir)
    assert not replay.ok
    assert any("graph hash changed" in mismatch for mismatch in replay.mismatches)


def test_missing_read_fails_before_call(tmp_path: Path) -> None:
    # Compiler catches this first, but the runtime also protects the boundary.
    graph = _compile()
    node = graph.node("coding")
    object.__setattr__(node, "reads", ("missing",))
    outcome = run_graph(graph, "task", _factory(MockScenario()), runs_dir=tmp_path)
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-2201"


def test_parse_single_artifact_strips_code_fence() -> None:
    graph = _compile()
    node = graph.node("coding")
    parsed = parse_artifacts("```ts\nconst x = 1;\n```", node)
    assert parsed[0]["content"] == "const x = 1;"


def test_parse_multi_artifact_errors_on_missing_block() -> None:
    from modelgraph.compiler.ir import CompiledNode

    graph = _compile()
    base = graph.node("coding")
    # A worker with two writes must return both blocks.
    values = dict(base.__dict__)
    values["writes"] = (*base.writes, base.writes[0].model_copy(update={"name": "notes"}))
    node = CompiledNode(**values)
    with pytest.raises(RuntimeFailure):
        parse_artifacts('<<<MG:artifact name="answer" type="code">>>\nx\n<<<MG:end>>>', node)


def test_joiner_execution(tmp_path: Path) -> None:
    data = {
        "apiVersion": "modelgraph/v0",
        "kind": "Graph",
        "metadata": {
            "id": "joiner-demo",
            "name": "Joiner Demo",
            "version": "1.0.0",
            "description": "A router to worker to joiner path.",
        },
        "spec": {
            "entry": "triage",
            "inputs": [{"name": "task", "type": "text"}],
            "exit": ["answer"],
            "nodes": [
                {
                    "id": "triage",
                    "kind": "router",
                    "description": "Choose the one available worker.",
                    "model": "router-small",
                    "reads": ["task"],
                    "routing": {"confidence_threshold": 0.5, "on_low_confidence": "error"},
                },
                {
                    "id": "coding",
                    "kind": "worker",
                    "description": "Write a draft.",
                    "model": "coder-7b",
                    "reads": ["task"],
                    "instruction": "Write a short draft.",
                    "writes": [{"name": "draft", "type": "text"}],
                },
                {
                    "id": "merge",
                    "kind": "joiner",
                    "description": "Polish the draft into the final answer.",
                    "model": "coder-7b",
                    "reads": ["draft"],
                    "strategy": "synthesize",
                    "instruction": "Polish the draft.",
                    "writes": [{"name": "answer", "type": "text"}],
                },
            ],
            "edges": [
                {
                    "from": "triage",
                    "to": "coding",
                    "when": "The request needs a draft.",
                    "artifact": "text",
                },
                {"from": "coding", "to": "merge"},
            ],
        },
    }
    path = tmp_path / "joiner.graph.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    compiled = compile_graph(path, strict_models=True)
    assert compiled.ok, compiled.errors
    scenario = MockScenario(
        responses={
            "triage": [
                json.dumps(
                    {"decision": "route", "to": "coding", "confidence": 0.9, "reason": "only route"}
                )
            ],
            "coding": ["draft text"],
            "merge": ["final text"],
        }
    )
    outcome = run_graph(compiled.graph, "Write a sentence.", _factory(scenario), runs_dir=tmp_path)
    assert outcome.ok, outcome.error
    assert [entry.node for entry in outcome.path] == ["triage", "coding", "merge"]
    assert outcome.artifacts[0].content == "final text"
