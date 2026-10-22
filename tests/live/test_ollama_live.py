"""Opt-in live tests against a local Ollama server.

Run with: MG_TEST_OLLAMA=1 uv run pytest tests/live -q
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from modelgraph.compiler import compile_graph
from modelgraph.providers import build_provider
from modelgraph.runtime.engine import run_graph

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
pytestmark = pytest.mark.live


def _ollama_available() -> bool:
    try:
        return httpx.get("http://127.0.0.1:11434/api/tags", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(os.environ.get("MG_TEST_OLLAMA") != "1", reason="set MG_TEST_OLLAMA=1")
@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not running")
def test_live_coding_route(tmp_path: Path) -> None:
    compiled = compile_graph(EXAMPLES / "hello-triage.graph.yaml", strict_models=True)
    assert compiled.ok, compiled.errors
    assert compiled.graph
    outcome = run_graph(
        compiled.graph,
        "Write a debounce function in TypeScript.",
        build_provider,
        runs_dir=tmp_path,
    )
    assert outcome.ok, outcome.error
    assert any(entry.node == "coding" for entry in outcome.path)
    assert outcome.artifacts
    assert "debounce" in outcome.artifacts[0].content.lower()
