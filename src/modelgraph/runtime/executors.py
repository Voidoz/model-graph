"""Router, worker, and joiner executors."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from modelgraph.compiler.ir import CompiledGraph, CompiledNode
from modelgraph.contracts.graph import ArtifactType, NodeKind
from modelgraph.errors import RuntimeFailure
from modelgraph.providers.base import (
    ChatRequest,
    ChatResponse,
    Provider,
    ProviderFailure,
    Usage,
)
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.envelope import RuntimeState
from modelgraph.runtime.renderer import RenderedPrompt, render_prompt
from modelgraph.runtime.trace import TraceRecorder

_ARTIFACT_BLOCK = re.compile(
    r'<<<MG:artifact name="([a-z][a-z0-9_]*)" type="(text|code|patch|plan|json|report)">>>\n'
    r"(.*?)\n"
    r"<<<MG:end>>>",
    re.S,
)


@dataclass(frozen=True)
class RouteDecision:
    action: str  # route | fallback
    to: str | None
    confidence: float
    reason: str
    note: str | None = None
    raw: str = ""


@dataclass
class Execution:
    artifacts: list[Any] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_seconds: float = 0.0
    text: str = ""


class Executor:
    def __init__(
        self,
        provider: Provider,
        *,
        budget: Budget,
        trace: TraceRecorder,
        max_retries: int = 1,
        seed: int | None = None,
    ) -> None:
        self.provider = provider
        self.budget = budget
        self.trace = trace
        self.max_retries = max_retries
        self.seed = seed

    def _request(
        self, node: CompiledNode, prompt: RenderedPrompt, *, temperature: float
    ) -> ChatRequest:
        model = node.model
        if model is None:
            raise RuntimeFailure(f"node {node.id!r} has no model", details={"code": "MG-1003"})
        max_output = (
            (node.limits.max_output_tokens if node.limits else None)
            or model.spec.limits.default_max_output_tokens
            or model.spec.max_output_tokens
            or 1024
        )
        json_mode = node.kind == NodeKind.router or any(
            write.type == ArtifactType.json for write in node.writes
        )
        return ChatRequest(
            model=model.spec.model,
            messages=prompt.messages,
            temperature=temperature,
            max_output_tokens=max_output,
            json_mode=json_mode and "json_mode" in model.spec.capabilities,
            seed=self.seed,
            timeout_seconds=(
                node.limits.timeout_seconds
                if node.limits and node.limits.timeout_seconds
                else model.spec.limits.timeout_seconds or 120.0
            ),
            metadata={"node_id": node.id},
        )

    def call(
        self, node: CompiledNode, prompt: RenderedPrompt, *, temperature: float
    ) -> ChatResponse:
        request = self._request(node, prompt, temperature=temperature)
        estimated = self.budget.estimate_tokens(prompt.messages, request.max_output_tokens)
        self.budget.check_tokens(estimated)
        self.budget.check_wall()
        attempts = 0
        while True:
            attempts += 1
            self.trace.emit(
                "provider.request",
                node=node.id,
                model=request.model,
                provider=self.provider.name,
                attempt=attempts,
                prompt_hash=prompt.prompt_hash,
            )
            started = time.monotonic()
            try:
                response = self.provider.chat(request)
            except ProviderFailure as exc:
                self.trace.emit(
                    "provider.error",
                    node=node.id,
                    code=exc.code,
                    message=exc.message,
                    retryable=exc.retryable,
                    attempt=attempts,
                )
                if exc.retryable and attempts <= 2:
                    time.sleep(0.05 * attempts)
                    continue
                raise
            latency = time.monotonic() - started
            self.trace.emit(
                "provider.response",
                node=node.id,
                model=request.model,
                text=response.text,
                usage=response.usage.to_dict(),
                finish_reason=response.finish_reason,
                latency_seconds=round(latency, 4),
                attempt=attempts,
            )
            cost = self.budget.charge(response.usage, node)
            self.trace.emit(
                "budget.charged",
                node=node.id,
                tokens=response.usage.total_tokens,
                cost_usd=round(cost, 8),
                used_tokens=self.budget.used_tokens,
                used_cost_usd=round(self.budget.used_cost_usd, 8),
            )
            return response


class RouterExecutor:
    def __init__(self, executor: Executor):
        self.executor = executor

    def run(
        self,
        node: CompiledNode,
        graph: CompiledGraph,
        state: RuntimeState,
    ) -> RouteDecision:
        prompt = render_prompt(node, graph, state)
        self.executor.trace.emit(
            "render.completed",
            node=node.id,
            prompt_hash=prompt.prompt_hash,
            tokens=prompt.estimated_tokens,
        )
        response = self.executor.call(node, prompt, temperature=0.0)
        try:
            decision = self._parse(response.text, node, graph)
        except RuntimeFailure as exc:
            if exc.details.get("code") != "MG-2101":
                raise
            repair = render_prompt(
                node,
                graph,
                state,
                repair_note=(
                    "Invalid router decision. Return JSON with decision, target (to), "
                    "confidence, and reason. Choose only from the menu."
                ),
            )
            response = self.executor.call(node, repair, temperature=0.0)
            decision = self._parse(response.text, node, graph)

        if decision.action == "route":
            threshold = node.routing.confidence_threshold
            if decision.confidence < threshold:
                policy = node.routing.on_low_confidence
                if policy == "fallback":
                    return RouteDecision(
                        "fallback", graph.fallback, decision.confidence, decision.reason
                    )
                if policy == "retry" and self.executor.max_retries > 0:
                    retry = render_prompt(
                        node,
                        graph,
                        state,
                        repair_note=(
                            f"Your confidence was {decision.confidence:.2f}, below the policy "
                            f"threshold {threshold:.2f}. Choose a route more decisively or abstain."
                        ),
                    )
                    response = self.executor.call(node, retry, temperature=0.0)
                    decision = self._parse(response.text, node, graph)
                if decision.confidence < threshold:
                    raise RuntimeFailure(
                        f"router {node.id!r} was below its confidence threshold "
                        f"({decision.confidence:.2f} < {threshold:.2f})",
                        details={"code": "MG-2102"},
                    )
        if decision.note:
            state.note(node.id, decision.note)
        return decision

    def _parse(self, text: str, node: CompiledNode, graph: CompiledGraph) -> RouteDecision:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                cleaned = "\n".join(lines[1:-1]).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeFailure(
                f"router {node.id!r} returned no JSON object",
                details={"code": "MG-2101"},
            )
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeFailure(
                f"router {node.id!r} returned invalid JSON: {exc}",
                details={"code": "MG-2101"},
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeFailure(
                "router decision must be a JSON object", details={"code": "MG-2101"}
            )
        confidence = data.get("confidence")
        reason = str(data.get("reason") or "").strip()
        note = data.get("note")
        menu = {edge.to for edge in graph.outgoing(node.id)}

        # Canonical decision shape, plus the compact route shape emitted by small routers.
        target_value = data.get("route") if "route" in data else data.get("to")
        target = target_value if isinstance(target_value, str) else None
        action: object
        if "route" in data:
            action = "abstain" if target == "abstain" else "route"
        else:
            action = data.get("decision")
        if action == "abstain":
            if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
                confidence = 0.0
            return RouteDecision(
                "fallback", None, float(confidence), reason or "abstained", None, text
            )
        if action != "route":
            raise RuntimeFailure(
                "router decision must be route or abstain", details={"code": "MG-2101"}
            )
        if not isinstance(target, str) or target not in menu:
            raise RuntimeFailure(
                f"router target {target!r} is not in the menu",
                details={"code": "MG-2101"},
            )
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise RuntimeFailure(
                "router confidence must be between 0 and 1", details={"code": "MG-2101"}
            )
        if not reason or reason == "short reason":
            reason = f"router selected {target}"
        return RouteDecision(
            "route",
            target,
            float(confidence),
            reason[:500],
            str(note)[:1000] if note else None,
            text,
        )


class WorkerExecutor:
    def __init__(self, executor: Executor):
        self.executor = executor

    def run(self, node: CompiledNode, graph: CompiledGraph, state: RuntimeState) -> Execution:
        prompt = render_prompt(node, graph, state)
        self.executor.trace.emit(
            "render.completed",
            node=node.id,
            prompt_hash=prompt.prompt_hash,
            tokens=prompt.estimated_tokens,
        )
        temperature = (
            node.limits.temperature
            if node.limits and node.limits.temperature is not None
            else (node.model.spec.limits.default_temperature if node.model else 0.2)
        )
        if temperature is None:
            temperature = 0.2
        response = self.executor.call(node, prompt, temperature=temperature)
        try:
            artifacts = parse_artifacts(response.text, node)
        except RuntimeFailure as exc:
            if exc.details.get("code") != "MG-2202":
                raise
            repair = render_prompt(
                node,
                graph,
                state,
                repair_note=(
                    "Your previous response did not match the artifact contract. "
                    "Return exactly the requested artifact content or the required MG blocks."
                ),
            )
            response = self.executor.call(node, repair, temperature=0.0)
            artifacts = parse_artifacts(response.text, node)
        for artifact in artifacts:
            state.write(
                name=artifact["name"],
                artifact_type=artifact["type"],
                content=artifact["content"],
                producer=node.id,
                media_type=artifact.get("media_type"),
                metadata=artifact.get("metadata", {}),
            )
            self.executor.trace.emit(
                "artifact.written",
                node=node.id,
                name=artifact["name"],
                type=artifact["type"],
                bytes=len(artifact["content"]),
            )
        return Execution(
            artifacts=artifacts,
            usage=response.usage,
            latency_seconds=0.0,
            text=response.text,
        )


def parse_artifacts(text: str, node: CompiledNode) -> list[dict[str, Any]]:
    writes = {write.name: write for write in node.writes}
    if not writes:
        raise RuntimeFailure(f"node {node.id!r} has no writes", details={"code": "MG-2202"})
    if len(writes) == 1:
        write = next(iter(writes.values()))
        content = text
        if write.type == ArtifactType.code:
            content = _strip_single_code_fence(content)
        return [
            {
                "name": write.name,
                "type": write.type,
                "content": content,
                "media_type": write.media_type,
                "metadata": {},
            }
        ]

    found: dict[str, dict[str, Any]] = {}
    for match in _ARTIFACT_BLOCK.finditer(text):
        name, reported_type, content = match.group(1), match.group(2), match.group(3)
        if name not in writes:
            raise RuntimeFailure(
                f"artifact {name!r} was not declared by node {node.id!r}",
                details={"code": "MG-2202"},
            )
        if name in found:
            raise RuntimeFailure(f"duplicate artifact {name!r}", details={"code": "MG-2202"})
        write = writes[name]
        metadata = {} if reported_type == write.type.value else {"reported_type": reported_type}
        found[name] = {
            "name": name,
            "type": write.type,
            "content": content,
            "media_type": write.media_type,
            "metadata": metadata,
        }
    missing = sorted(set(writes) - set(found))
    if missing:
        raise RuntimeFailure(f"missing artifact block(s): {missing}", details={"code": "MG-2202"})
    return [found[name] for name in writes]


def _strip_single_code_fence(content: str) -> str:
    stripped = content.strip()
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])

    # Models often return one fenced code block followed by an explanation. If there is
    # exactly one block, treat it as the artifact and discard the surrounding prose.
    matches: list[str] = re.findall(r"^```[^\n]*\n(.*?)\n```\s*$", content, re.S | re.M)
    if len(matches) == 1:
        return matches[0]
    return content
