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
from modelgraph.runtime.cache import (
    CacheEntry,
    CacheKey,
    CacheStats,
    ResponseCache,
    build_cache_key,
    response_has_default_secret,
)
from modelgraph.runtime.envelope import RuntimeState
from modelgraph.runtime.renderer import RenderedPrompt, render_prompt
from modelgraph.runtime.trace import TraceRecorder
from modelgraph.util import new_id, utc_now

#: Two retries for a retryable transport failure (three attempts total). This is a hard
#: ceiling independent of a node's routing configuration.
MAX_PROVIDER_ATTEMPTS = 3

#: Defense-in-depth cap on total provider calls for one node, including repair and
#: low-confidence retry calls. The graph schema bounds ``routing.max_retries``; this catches a
#: future code path that forgets to respect it.
MAX_CALLS_PER_NODE = 8

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
        cache: ResponseCache | None = None,
        cache_stats: CacheStats | None = None,
    ) -> None:
        self.provider = provider
        self.budget = budget
        self.trace = trace
        self.max_retries = max_retries
        self.seed = seed
        self.cache = cache
        self.cache_stats = cache_stats
        self.calls = 0

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
        self,
        node: CompiledNode,
        prompt: RenderedPrompt,
        *,
        temperature: float,
        graph: CompiledGraph,
    ) -> ChatResponse:
        request = self._request(node, prompt, temperature=temperature)

        # A cache hit bypasses the provider entirely: no token/cost pre-flight, no
        # provider call, and no budget charge. The recorded response is emitted as a
        # normal ``provider.response`` (with ``cached: true``) so trace replay still
        # works, plus a distinct accounting event.
        cache_key: CacheKey | None = None
        if self.cache is not None and self.cache.readable:
            cache_key = build_cache_key(
                node=node, graph=graph, request=request, provider=self.provider.name
            )
            entry = self.cache.get(cache_key)
            if entry is not None:
                self.budget.check_wall()
                return self._cached_response(node, request, entry, cache_key)
            self.trace.emit(
                "cache.miss",
                node=node.id,
                cache_key=cache_key.value,
                mode=self.cache.mode,
                wrote=self.cache.writable,
            )
            if self.cache_stats is not None:
                self.cache_stats.misses += 1

        estimated = self.budget.estimate_tokens(prompt.messages, request.max_output_tokens)
        self.budget.check_tokens(estimated)
        self.budget.check_wall()
        self.calls += 1
        if self.calls > MAX_CALLS_PER_NODE:
            raise RuntimeFailure(
                f"node {node.id!r} exceeded the hard provider-call limit ({MAX_CALLS_PER_NODE})",
                details={"code": "MG-4005", "limit": MAX_CALLS_PER_NODE},
            )
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
                if exc.retryable and attempts < MAX_PROVIDER_ATTEMPTS:
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
            # Only a successful, accepted response is cached: a provider failure or a
            # budget charge that raises leaves no entry behind.
            if cache_key is not None:
                self._store(node, response, cache_key)
            return response

    def _cached_response(
        self,
        node: CompiledNode,
        request: ChatRequest,
        entry: CacheEntry,
        cache_key: CacheKey,
    ) -> ChatResponse:
        hit_id = new_id("cachehit")
        avoided_tokens = entry.usage.total_tokens
        avoided_cost = self.budget.cost_for(entry.usage, node)
        if self.cache_stats is not None:
            self.cache_stats.hits += 1
            self.cache_stats.avoided_tokens += avoided_tokens
            self.cache_stats.avoided_cost_usd += avoided_cost
        self.trace.emit(
            "provider.response",
            node=node.id,
            model=request.model,
            provider=self.provider.name,
            text=entry.text,
            usage=entry.usage.to_dict(),
            finish_reason=entry.finish_reason,
            latency_seconds=0.0,
            attempt=1,
            cached=True,
            cache_hit_id=hit_id,
        )
        self.trace.emit(
            "provider.response.cached",
            node=node.id,
            model=request.model,
            provider=self.provider.name,
            usage=entry.usage.to_dict(),
            finish_reason=entry.finish_reason,
            avoided_tokens=avoided_tokens,
            avoided_cost_usd=round(avoided_cost, 8),
            cached=True,
            cache_hit_id=hit_id,
            cache_key=cache_key.value,
        )
        return ChatResponse(
            text=entry.text,
            usage=entry.usage,
            finish_reason=entry.finish_reason,
            provider_meta=dict(entry.provider_meta),
        )

    def _store(self, node: CompiledNode, response: ChatResponse, cache_key: CacheKey) -> None:
        if self.cache is None or not self.cache.writable:
            return
        patterns = response_has_default_secret(response)
        if patterns:
            # Never cache a response that contains a default secret/PII pattern. The
            # event carries pattern names only, never the matched text or response.
            if self.cache_stats is not None:
                self.cache_stats.skipped += 1
            self.trace.emit(
                "cache.skipped",
                node=node.id,
                reason="default-secret-pattern",
                patterns=patterns,
            )
            return
        entry = CacheEntry(
            text=response.text,
            usage=response.usage,
            finish_reason=response.finish_reason,
            provider_meta=dict(response.provider_meta),
            created_at=utc_now(),
            key=cache_key.value,
            key_metadata=cache_key.metadata,
        )
        if self.cache.put(cache_key, entry):
            if self.cache_stats is not None:
                self.cache_stats.writes += 1
        else:
            if self.cache_stats is not None:
                self.cache_stats.skipped += 1
            self.trace.emit("cache.skipped", node=node.id, reason="write-failed")


def _trace_render(trace: TraceRecorder, node: CompiledNode, prompt: RenderedPrompt) -> None:
    """Record the render plus one visible event per truncation."""

    fields: dict[str, object] = {
        "node": node.id,
        "prompt_hash": prompt.prompt_hash,
        "tokens": prompt.estimated_tokens,
    }
    if prompt.truncated:
        fields["truncated"] = prompt.truncated
    trace.emit("render.completed", **fields)
    for item in prompt.truncated:
        trace.emit("render.truncated", node=node.id, **item)


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
        _trace_render(self.executor.trace, node, prompt)
        response = self.executor.call(node, prompt, temperature=0.0, graph=graph)
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
            response = self.executor.call(node, repair, temperature=0.0, graph=graph)
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
                    response = self.executor.call(node, retry, temperature=0.0, graph=graph)
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
        _trace_render(self.executor.trace, node, prompt)
        temperature = (
            node.limits.temperature
            if node.limits and node.limits.temperature is not None
            else (node.model.spec.limits.default_temperature if node.model else 0.2)
        )
        if temperature is None:
            temperature = 0.2
        response = self.executor.call(node, prompt, temperature=temperature, graph=graph)
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
            response = self.executor.call(node, repair, temperature=0.0, graph=graph)
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
