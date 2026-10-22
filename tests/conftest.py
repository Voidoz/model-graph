from pathlib import Path

import pytest

from modelgraph.compiler import compile_graph
from modelgraph.contracts.loader import load_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.registry import Registry, default_registry

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
MODELS = EXAMPLES / "models"


def compile_example(name: str, *, strict: bool = True):
    result = compile_graph(EXAMPLES / name, strict_models=strict)
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def manifest(name: str) -> ModelManifest:
    return ModelManifest.model_validate(load_document(MODELS / f"{name}.yaml"))


def mock_factory(scenario: MockScenario):
    def factory(model: ModelManifest) -> MockProvider:
        return MockProvider(scenario, max_context=model.spec.context_window)

    return factory


def hello_registry() -> Registry:
    return Registry.from_paths([MODELS], [EXAMPLES])


@pytest.fixture
def registry() -> Registry:
    return hello_registry()


@pytest.fixture
def default_models() -> Registry:
    return default_registry()
