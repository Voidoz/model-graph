"""Deterministic mock provider for tests and offline runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from modelgraph.providers.base import (
    Capabilities,
    ChatRequest,
    ChatResponse,
    Health,
    Message,
    Usage,
)
from modelgraph.util import estimate_tokens

_ROUTE_RE = re.compile(r'<route id="([a-z][a-z0-9-]*)"')


@dataclass
class MockScenario:
    """Scripted responses for one mock provider instance.

    ``responses`` maps a node id to a list of responses served in order. ``failures`` maps a
    node id to a provider failure code for negative tests.
    """

    responses: dict[str, list[str]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


class MockProvider:
    name = "mock"

    def __init__(self, scenario: MockScenario | None = None, *, max_context: int = 32768):
        self.scenario = scenario or MockScenario()
        self._max_context = max_context
        self.calls: list[ChatRequest] = []

    def capabilities(self) -> Capabilities:
        return Capabilities(
            chat=True, json_mode=True, streaming=False, max_context=self._max_context
        )

    def health(self) -> Health:
        return Health(ok=True, detail="mock provider is always available")

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        node_id = request.node_id or ""
        if node_id in self.scenario.failures:
            from modelgraph.providers.base import ProviderFailure

            code = self.scenario.failures[node_id]
            raise ProviderFailure(
                f"mock failure {code}",
                code=code,
                retryable=code in {"MG-3002", "MG-3003"},
            )

        scripted = self.scenario.responses.get(node_id)
        if scripted:
            text = scripted.pop(0) if len(scripted) > 1 else scripted[0]
        else:
            text = self._default_response(request, node_id)
        prompt_tokens = estimate_tokens(request.prompt_text())
        output_tokens = estimate_tokens(text)
        return ChatResponse(
            text=text,
            usage=Usage(prompt_tokens, output_tokens, approximated=True),
            finish_reason="stop",
            provider_meta={"mock": True},
        )

    def _default_response(self, request: ChatRequest, node_id: str) -> str:
        prompt = request.prompt_text()
        routes = _ROUTE_RE.findall(prompt)
        if routes:
            # Deterministic, intentionally simple: pick the first menu entry. Tests script
            # specific decisions when the branch matters.
            return json.dumps(
                {
                    "decision": "route",
                    "to": routes[0],
                    "confidence": 0.9,
                    "reason": "mock default route (first menu entry)",
                }
            )
        lower = prompt.lower()
        if "javascript" in lower or "typescript" in lower or "debounce" in lower:
            return (
                "export function debounce<T extends (...args: any[]) => void>(fn: T, ms = 200) {\n"
                "  let timer: ReturnType<typeof setTimeout> | undefined;\n"
                "  return (...args: Parameters<T>) => {\n"
                "    if (timer) clearTimeout(timer);\n"
                "    timer = setTimeout(() => fn(...args), ms);\n"
                "  };\n"
                "}\n"
            )
        if "story" in lower or "lighthouse" in lower or "poem" in lower:
            return "The lighthouse keeper counted the waves until the light answered."
        return f"Mock response from {node_id or 'worker'}."


def mock_messages(prompt: str) -> list[Message]:
    return [Message(role="user", content=prompt)]


def scenario_from_dict(data: dict[str, Any]) -> MockScenario:
    return MockScenario(
        responses={str(k): [str(item) for item in v] for k, v in data.get("responses", {}).items()},
        failures={str(k): str(v) for k, v in data.get("failures", {}).items()},
    )
