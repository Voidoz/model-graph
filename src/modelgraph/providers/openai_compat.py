"""OpenAI-compatible chat completions adapter."""

from __future__ import annotations

import os
from typing import Any

import httpx

from modelgraph.contracts.manifest import ModelManifest
from modelgraph.providers.base import (
    Capabilities,
    ChatRequest,
    ChatResponse,
    Health,
    ProviderFailure,
    Usage,
)
from modelgraph.util import estimate_tokens


def _error_for_status(status: int, body: str) -> ProviderFailure:
    if status in {401, 403}:
        return ProviderFailure(f"authentication failed ({status})", code="MG-3004", retryable=False)
    if status == 404:
        return ProviderFailure("model or endpoint not found", code="MG-3005", retryable=False)
    if status == 429:
        return ProviderFailure("provider rate limit", code="MG-3003", retryable=True)
    if status >= 500:
        return ProviderFailure(
            f"provider server error ({status}): {body[:200]}",
            code="MG-3001",
            retryable=True,
        )
    return ProviderFailure(
        f"provider request failed ({status}): {body[:200]}", code="MG-3006", retryable=False
    )


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(
        self,
        manifest: ModelManifest,
        *,
        client: httpx.Client | None = None,
        api_key: str | None = None,
    ) -> None:
        self.manifest = manifest
        self.model = manifest.spec.model
        endpoint = manifest.spec.endpoint or "http://127.0.0.1:8080/v1"
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key or os.environ.get("MG_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._client = client or httpx.Client(timeout=manifest.spec.limits.timeout_seconds or 120)
        self._owns_client = client is None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            chat=True,
            json_mode="json_mode" in self.manifest.spec.capabilities,
            streaming="streaming" in self.manifest.spec.capabilities,
            seeded="seeded" in self.manifest.spec.capabilities,
            max_context=self.manifest.spec.context_window,
        )

    def health(self) -> Health:
        try:
            response = self._client.get(f"{self.endpoint}/models", timeout=5)
        except httpx.HTTPError as exc:
            return Health(ok=False, detail=str(exc))
        return Health(ok=response.status_code < 500, detail=f"HTTP {response.status_code}")

    def chat(self, request: ChatRequest) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [message.to_dict() for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if request.stop:
            payload["stop"] = request.stop
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = self._client.post(
                f"{self.endpoint}/chat/completions",
                json=payload,
                headers=headers,
                timeout=request.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderFailure(
                "provider request timed out", code="MG-3002", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderFailure(
                f"provider connection failed: {exc}", code="MG-3001", retryable=True
            ) from exc

        if response.status_code >= 400:
            raise _error_for_status(response.status_code, response.text)
        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            usage_data = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderFailure(
                "provider returned a malformed response", code="MG-3006", retryable=False
            ) from exc

        input_tokens = int(usage_data.get("prompt_tokens") or 0)
        output_tokens = int(usage_data.get("completion_tokens") or 0)
        approximated = not (input_tokens or output_tokens)
        if approximated:
            input_tokens = estimate_tokens(request.prompt_text())
            output_tokens = estimate_tokens(text)
        return ChatResponse(
            text=text,
            usage=Usage(input_tokens, output_tokens, approximated),
            finish_reason=str(choice.get("finish_reason") or "unknown"),
            provider_meta={"id": data.get("id"), "model": data.get("model")},
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
