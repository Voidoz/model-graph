import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from modelgraph.contracts.loader import load_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.providers import build_provider
from modelgraph.providers.base import ChatRequest, Message, ProviderFailure
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.providers.ollama import OllamaProvider
from modelgraph.providers.openai_compat import OpenAICompatibleProvider

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "examples" / "models"


def _manifest(name: str, endpoint: str | None = None) -> ModelManifest:
    manifest = ModelManifest.model_validate(load_document(MODELS / f"{name}.yaml"))
    if endpoint:
        manifest.spec.endpoint = endpoint
    return manifest


def _request() -> ChatRequest:
    return ChatRequest(
        model="test",
        messages=[Message("system", "be terse"), Message("user", "hello")],
        max_output_tokens=16,
    )


def test_mock_provider_is_deterministic() -> None:
    scenario = MockScenario(responses={"worker": ["fixed answer"]})
    provider = MockProvider(scenario)
    first = provider.chat(replace(_request(), metadata={"node_id": "worker"}))
    second = provider.chat(replace(_request(), metadata={"node_id": "worker"}))
    assert first.text == second.text == "fixed answer"
    assert first.usage.output_tokens > 0


def test_mock_router_picks_first_route() -> None:
    prompt = '<routes>\n<route id="coding" artifact="code">write code</route>\n</routes>'
    provider = MockProvider()
    response = provider.chat(
        ChatRequest(
            model="test",
            messages=[Message("user", prompt)],
            metadata={"node_id": "triage"},
        )
    )
    decision = json.loads(response.text)
    assert decision["decision"] == "route"
    assert decision["to"] == "coding"


def test_build_provider_for_mock_manifest() -> None:
    provider = build_provider(_manifest("router-small"))
    assert provider.name == "ollama"
    assert provider.capabilities().json_mode


def test_openai_compatible_success() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "req_1",
                "model": "test-model",
                "choices": [
                    {
                        "message": {"content": "hello back"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    manifest = _manifest("router-small", "http://test/v1")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(manifest, client=client)
        response = provider.chat(_request())
    assert captured["path"] == "/v1/chat/completions"
    assert captured["payload"]["model"] == "test"
    assert response.text == "hello back"
    assert response.usage.total_tokens == 5


def test_openai_compatible_errors() -> None:
    manifest = _manifest("router-small", "http://test/v1")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(manifest, client=client)
        with pytest.raises(ProviderFailure) as exc:
            provider.chat(_request())
    assert exc.value.code == "MG-3001"
    assert exc.value.retryable


def test_openai_timeout() -> None:
    manifest = _manifest("router-small", "http://test/v1")

    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(manifest, client=client)
        with pytest.raises(ProviderFailure) as exc:
            provider.chat(_request())
    assert exc.value.code == "MG-3002"


def test_ollama_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={
                "model": "story",
                "message": {"content": "a short story"},
                "prompt_eval_count": 4,
                "eval_count": 3,
                "done_reason": "stop",
            },
        )

    manifest = _manifest("story-7b", "http://test:11434")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaProvider(manifest, client=client)
        response = provider.chat(_request())
    assert response.text == "a short story"
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 3


def test_ollama_model_missing() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    manifest = _manifest("story-7b", "http://test:11434")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaProvider(manifest, client=client)
        with pytest.raises(ProviderFailure) as exc:
            provider.chat(_request())
    assert exc.value.code == "MG-3005"
