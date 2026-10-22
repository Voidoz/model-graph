"""Provider factory."""

from __future__ import annotations

import httpx

from modelgraph.contracts.manifest import ModelManifest
from modelgraph.providers.base import Provider, ProviderFailure
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.providers.ollama import OllamaProvider
from modelgraph.providers.openai_compat import OpenAICompatibleProvider


def build_provider(
    manifest: ModelManifest,
    *,
    scenario: MockScenario | None = None,
    client: httpx.Client | None = None,
) -> Provider:
    provider = manifest.spec.provider
    if provider == "mock":
        return MockProvider(scenario, max_context=manifest.spec.context_window)
    if provider == "ollama":
        return OllamaProvider(manifest, client=client)
    if provider in {"openai-compatible", "llamacpp"}:
        return OpenAICompatibleProvider(manifest, client=client)
    raise ProviderFailure(f"unsupported provider {provider!r}", code="MG-3005", retryable=False)
