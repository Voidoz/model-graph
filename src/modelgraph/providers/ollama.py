"""Ollama local chat adapter."""

from __future__ import annotations

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


class OllamaProvider:
    name = "ollama"

    def __init__(self, manifest: ModelManifest, *, client: httpx.Client | None = None) -> None:
        self.manifest = manifest
        self.model = manifest.spec.model
        self.endpoint = (manifest.spec.endpoint or "http://127.0.0.1:11434").rstrip("/")
        self._client = client or httpx.Client(timeout=manifest.spec.limits.timeout_seconds or 120)
        self._owns_client = client is None

    def capabilities(self) -> Capabilities:
        return Capabilities(
            chat=True,
            json_mode="json_mode" in self.manifest.spec.capabilities,
            streaming=False,
            seeded="seeded" in self.manifest.spec.capabilities,
            max_context=self.manifest.spec.context_window,
        )

    def health(self) -> Health:
        try:
            response = self._client.get(f"{self.endpoint}/api/tags", timeout=5)
        except httpx.HTTPError as exc:
            return Health(ok=False, detail=str(exc))
        return Health(ok=response.status_code == 200, detail=f"HTTP {response.status_code}")

    def chat(self, request: ChatRequest) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [message.to_dict() for message in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        if request.json_mode:
            payload["format"] = "json"
        if request.stop:
            payload["options"]["stop"] = request.stop
        try:
            response = self._client.post(
                f"{self.endpoint}/api/chat",
                json=payload,
                timeout=request.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderFailure(
                "Ollama request timed out", code="MG-3002", retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderFailure(
                f"Ollama connection failed: {exc}", code="MG-3001", retryable=True
            ) from exc

        if response.status_code == 404:
            raise ProviderFailure(
                f"model {request.model!r} is not available in Ollama",
                code="MG-3005",
                retryable=False,
            )
        if response.status_code >= 500:
            raise ProviderFailure(
                f"Ollama server error ({response.status_code})",
                code="MG-3001",
                retryable=True,
            )
        if response.status_code >= 400:
            raise ProviderFailure(
                f"Ollama request failed ({response.status_code}): {response.text[:200]}",
                code="MG-3006",
                retryable=False,
            )

        try:
            data = response.json()
            text = data["message"]["content"] or ""
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderFailure(
                "Ollama returned a malformed response", code="MG-3006", retryable=False
            ) from exc

        input_tokens = int(data.get("prompt_eval_count") or 0)
        output_tokens = int(data.get("eval_count") or 0)
        approximated = not (input_tokens or output_tokens)
        if approximated:
            input_tokens = estimate_tokens(request.prompt_text())
            output_tokens = estimate_tokens(text)
        return ChatResponse(
            text=text,
            usage=Usage(input_tokens, output_tokens, approximated),
            finish_reason=str(data.get("done_reason") or "stop"),
            provider_meta={
                "model": data.get("model"),
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
