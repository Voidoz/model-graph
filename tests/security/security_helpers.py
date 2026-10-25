"""Shared helpers for the offline security suite.

The suite uses the normal compiler and mock provider only. No test in this directory contacts
a network or a real model server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from modelgraph.compiler import compile_graph
from modelgraph.compiler.ir import CompiledGraph
from modelgraph.contracts.graph import ArtifactWrite
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.runtime.engine import RunOutcome, run_graph

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
SECRET_KEY = "sk-proj-abcdefghijklmnop"
SECRET_EMAIL = "alice@example.com"
SECRET_VALUES = (SECRET_KEY, SECRET_EMAIL)


def compile_example(name: str = "hello-triage.graph.yaml") -> CompiledGraph:
    result = compile_graph(EXAMPLES / name, strict_models=True)
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def mock_factory(scenario: MockScenario):
    def factory(model: Any) -> MockProvider:
        return MockProvider(scenario, max_context=model.spec.context_window)

    return factory


def run_hello(
    tmp_path: Path,
    task: str,
    scenario: MockScenario,
    **kwargs: Any,
) -> RunOutcome:
    return run_graph(
        compile_example(),
        task,
        mock_factory(scenario),
        runs_dir=tmp_path / "runs",
        **kwargs,
    )


def read_trace(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def persisted_text(run_dir: Path, *, include_artifacts: bool = False) -> str:
    files = [
        run_dir / "run.json",
        run_dir / "trace.jsonl",
        run_dir / "envelope.json",
        run_dir / "redaction.json",
    ]
    if include_artifacts:
        files.extend(sorted((run_dir / "artifacts").glob("*")))
    return "\n".join(path.read_text(encoding="utf-8") for path in files if path.is_file())


def write_yaml(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def with_writes(
    graph: CompiledGraph,
    node_id: str,
    writes: list[ArtifactWrite],
) -> CompiledGraph:
    from dataclasses import replace

    node = replace(graph.node(node_id), writes=tuple(writes))
    return replace(graph, nodes={**graph.nodes, node_id: node})


ROUTING_DECISION = json.dumps(
    {
        "decision": "route",
        "to": "coding",
        "confidence": 0.95,
        "reason": "the request asks for code",
    }
)


@pytest.fixture
def hello_graph() -> CompiledGraph:
    return compile_example()
