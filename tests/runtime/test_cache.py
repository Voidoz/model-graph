"""Exact-match response cache: keying, storage, modes, reporting, and replay."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from modelgraph.cli import app
from modelgraph.compiler import compile_graph
from modelgraph.contracts.graph import Limits
from modelgraph.contracts.manifest import ModelManifest, ProviderKind
from modelgraph.observability import project_trace
from modelgraph.providers.base import ChatRequest, Message, Usage
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.registry import Registry
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.cache import (
    CacheEntry,
    ResponseCache,
    build_cache_key,
    response_has_default_secret,
)
from modelgraph.runtime.engine import RunOutcome, run_graph
from modelgraph.runtime.replay import replay_run
from modelgraph.util import utc_now

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
TASK = "Write a debounce function in TypeScript."
CODING_DECISION = json.dumps(
    {
        "decision": "route",
        "to": "coding",
        "confidence": 0.95,
        "reason": "the request asks for TypeScript code",
    }
)
runner = CliRunner()


def _compile(name: str = "hello-triage.graph.yaml", registry: Registry | None = None):
    result = compile_graph(EXAMPLES / name, registry=registry, strict_models=True)
    assert result.ok, result.errors
    assert result.graph is not None
    return result.graph


def _factory(scenario: MockScenario | None = None):
    scenario = scenario or MockScenario(responses={"triage": [CODING_DECISION]})

    def build(model: ModelManifest) -> MockProvider:
        return MockProvider(scenario, max_context=model.spec.context_window)

    return build


def _run(
    cache: ResponseCache | None,
    runs_dir: Path,
    *,
    task: str = TASK,
    scenario: MockScenario | None = None,
    graph=None,
    seed: int | None = None,
    budget: Budget | None = None,
) -> RunOutcome:
    return run_graph(
        graph or _compile(),
        task,
        _factory(scenario),
        runs_dir=runs_dir,
        cache=cache,
        seed=seed,
        budget=budget,
    )


def _events(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _entries(cache: ResponseCache) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(cache.directory.glob("*.json")):
        entries.append(json.loads(path.read_text(encoding="utf-8")))
    return entries


def _entries_by_node(cache: ResponseCache) -> dict[str, dict[str, Any]]:
    return {entry["key_metadata"]["node_id"]: entry for entry in _entries(cache)}


def _update_model(
    model: ModelManifest,
    *,
    revision: str | None = None,
    version: str | None = None,
    provider: ProviderKind | None = None,
) -> ModelManifest:
    spec = model.spec
    if revision is not None:
        spec = spec.model_copy(
            update={"integrity": spec.integrity.model_copy(update={"revision": revision})}
        )
    if provider is not None:
        spec = spec.model_copy(update={"provider": provider})
    metadata = model.metadata
    if version is not None:
        metadata = metadata.model_copy(update={"version": version})
    return model.model_copy(update={"spec": spec, "metadata": metadata})


def _key_fixture() -> tuple[Any, Any, ChatRequest]:
    graph = _compile()
    node = graph.node("coding")
    request = ChatRequest(
        model=node.model.spec.model,
        messages=[
            Message(role="system", content="system text"),
            Message(role="user", content=TASK),
        ],
        temperature=0.2,
        max_output_tokens=128,
        json_mode=False,
        stop=[],
        seed=7,
    )
    return graph, node, request


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------


def test_identical_requests_produce_identical_keys() -> None:
    graph, node, request = _key_fixture()
    first = build_cache_key(node=node, graph=graph, request=request, provider="mock")
    second = build_cache_key(node=node, graph=graph, request=request, provider="mock")
    assert first.value == second.value
    assert first.value.startswith("sha256:")
    assert len(first.digest) == 64


def test_every_key_component_changes_the_key() -> None:
    graph, node, request = _key_fixture()
    base = build_cache_key(node=node, graph=graph, request=request, provider="mock")

    variants: dict[str, dict[str, Any]] = {
        "node_id": {"node": replace(node, id="story")},
        "graph_hash": {"graph": replace(graph, graph_hash="sha256:" + "0" * 64)},
        "renderer_version": {"renderer_version": "999"},
        "model_manifest_hash": {"node": replace(node, model_hash="sha256:" + "1" * 64)},
        "messages": {
            "request": replace(request, messages=[Message(role="user", content="a different task")])
        },
        "temperature": {"request": replace(request, temperature=0.7)},
        "max_output_tokens": {"request": replace(request, max_output_tokens=256)},
        "json_mode": {"request": replace(request, json_mode=True)},
        "stop": {"request": replace(request, stop=["END"])},
        "seed": {"request": replace(request, seed=8)},
        "provider_adapter": {"provider": "ollama"},
        "model_id": {"request": replace(request, model="another-model")},
        "resolved_revision": {
            "node": replace(node, model=_update_model(node.model, revision="rev-2"))
        },
        "model_version": {"node": replace(node, model=_update_model(node.model, version="9.9.9"))},
        "manifest_provider": {
            "node": replace(node, model=_update_model(node.model, provider=ProviderKind.mock))
        },
    }
    seen = {base.value}
    for name, patch in variants.items():
        arguments: dict[str, Any] = {
            "node": node,
            "graph": graph,
            "request": request,
            "provider": "mock",
        }
        arguments.update(patch)
        key = build_cache_key(**arguments)
        assert key.value not in seen, f"changing {name} did not change the cache key"
        seen.add(key.value)


def test_timeout_is_not_part_of_generation_params() -> None:
    graph, node, request = _key_fixture()
    base = build_cache_key(node=node, graph=graph, request=request, provider="mock")
    longer = build_cache_key(
        node=node,
        graph=graph,
        request=replace(request, timeout_seconds=999.0),
        provider="mock",
    )
    assert base.value == longer.value


# ---------------------------------------------------------------------------
# Hits, reporting, and replay
# ---------------------------------------------------------------------------


def test_identical_request_hits_and_reports_avoided_usage(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    first = _run(cache, tmp_path / "runs")
    assert first.ok, first.error
    assert first.cache == {
        "mode": "on",
        "hits": 0,
        "misses": 2,
        "writes": 2,
        "skipped": 0,
        "avoided_tokens": 0,
        "avoided_cost_usd": 0.0,
    }

    second = _run(cache, tmp_path / "runs")
    assert second.ok, second.error
    assert second.cache["hits"] == 2
    assert second.cache["misses"] == 0
    assert second.cache["writes"] == 0
    assert second.usage["total_tokens"] == 0
    assert second.usage["cost_usd"] == 0.0
    assert second.cache["avoided_tokens"] == first.usage["total_tokens"]
    assert second.artifacts[0].content == first.artifacts[0].content
    assert second.artifacts[0].hash == first.artifacts[0].hash

    # The stored response is byte-identical and the cached run makes no provider call
    # and charges no budget.
    entry = _entries_by_node(cache)["coding"]
    assert entry["response"]["text"] == first.artifacts[0].content
    events = _events(second.run_dir)
    assert not [event for event in events if event["event"] == "provider.request"]
    assert not [event for event in events if event["event"] == "budget.charged"]
    hit_events = [event for event in events if event["event"] == "provider.response"]
    assert hit_events and all(event["cached"] is True for event in hit_events)
    avoided = [event for event in events if event["event"] == "provider.response.cached"]
    assert len(avoided) == 2
    assert sum(event["avoided_tokens"] for event in avoided) == first.usage["total_tokens"]

    # The observability projection counts a paired hit once, not twice.
    metrics = project_trace(second.run_dir / "trace.jsonl")
    assert metrics.cache["hits"] == second.cache["hits"] == 2
    assert metrics.cache["misses"] == 0
    assert metrics.provider_requests == 0
    assert metrics.to_dict()["cache"] == {"hits": 2, "misses": 0}

    # `mg stats` is the same projection, not a second event model.
    stats = runner.invoke(app, ["stats", "--runs-dir", str(tmp_path / "runs"), "--json"])
    assert stats.exit_code == 0, stats.output
    assert json.loads(stats.stdout)["cache"] == {"hits": 2, "misses": 2}


def test_changed_request_misses_and_writes_a_new_entry(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    first = _run(cache, tmp_path / "runs", seed=1)
    assert first.cache["writes"] == 2
    same = _run(cache, tmp_path / "runs", seed=1)
    assert same.cache["hits"] == 2
    assert same.cache["writes"] == 0

    different_seed = _run(cache, tmp_path / "runs", seed=2)
    assert different_seed.cache["hits"] == 0
    assert different_seed.cache["misses"] == 2
    assert different_seed.cache["writes"] == 2

    different_prompt = _run(cache, tmp_path / "runs", seed=1, task="Tell a different story.")
    assert different_prompt.cache["hits"] == 0
    assert different_prompt.cache["misses"] == 2
    assert len(_entries(cache)) == 6


def test_end_to_end_generation_params_and_revision_miss(tmp_path: Path) -> None:
    """Changing temperature, max tokens, or the resolved revision misses for real."""

    cache = ResponseCache(tmp_path / "cache")
    base = _run(cache, tmp_path / "runs", seed=1)
    assert base.cache["writes"] == 2
    hit = _run(cache, tmp_path / "runs", seed=1)
    assert hit.cache["hits"] == 2

    graph = _compile()
    coding = graph.node("coding")
    warmer = replace(
        graph,
        nodes={**graph.nodes, "coding": replace(coding, limits=Limits(temperature=0.9))},
    )
    warm = _run(cache, tmp_path / "runs", seed=1, graph=warmer)
    # The router keeps temperature 0.0, so only the coding node's entry changes.
    assert warm.cache["hits"] == 1
    assert warm.cache["misses"] == 1
    assert warm.cache["writes"] == 1

    longer = replace(
        graph,
        nodes={
            **graph.nodes,
            "coding": replace(coding, limits=Limits(max_output_tokens=512)),
        },
    )
    longer_run = _run(cache, tmp_path / "runs", seed=1, graph=longer)
    assert longer_run.cache["hits"] == 1
    assert longer_run.cache["misses"] == 1
    assert longer_run.cache["writes"] == 1

    revised = replace(
        graph,
        nodes={
            **graph.nodes,
            "coding": replace(coding, model=_update_model(coding.model, revision="rev-2")),
        },
    )
    revision_run = _run(cache, tmp_path / "runs", seed=1, graph=revised)
    # Only the coding node's model changed; its router still serves the same request.
    assert revision_run.cache["hits"] == 1
    assert revision_run.cache["misses"] == 1
    assert revision_run.cache["writes"] == 1


def test_cache_hits_still_obey_step_budget_but_not_token_preflight(tmp_path: Path) -> None:
    """A hit saves tokens; it does not bypass the step or wall-clock ceilings."""

    cache = ResponseCache(tmp_path / "cache")
    first = _run(cache, tmp_path / "runs")
    assert first.cache["writes"] == 2

    step_run = _run(
        cache,
        tmp_path / "runs",
        budget=Budget(
            max_steps=1,
            max_depth=2,
            max_total_tokens=100_000,
            max_cost_usd=1.0,
            max_wall_seconds=120,
        ),
    )
    assert step_run.outcome == "failed"
    assert step_run.error is not None
    assert step_run.error["code"] == "MG-4001"

    tight = Budget(
        max_steps=8,
        max_depth=2,
        max_total_tokens=1,
        max_cost_usd=0.0,
        max_wall_seconds=120,
    )
    hit_run = _run(cache, tmp_path / "runs", budget=tight)
    assert hit_run.ok, hit_run.error
    assert hit_run.cache["hits"] == 2
    assert hit_run.usage["total_tokens"] == 0

    miss_run = _run(ResponseCache(tmp_path / "fresh"), tmp_path / "runs", budget=tight)
    assert miss_run.outcome == "failed"
    assert miss_run.error is not None
    assert miss_run.error["code"] == "MG-4002"


def test_a_cache_hit_run_is_still_redacted_on_persistence(tmp_path: Path) -> None:
    """A hit reuses a cached response but not the persistence redaction policy."""

    key = "sk-proj-abcdefghijklmnop"
    email = "alice@example.com"
    task = f"Write code; contact {email}; use {key}."
    cache = ResponseCache(tmp_path / "cache")
    _run(cache, tmp_path / "runs", task=task)
    hit = _run(cache, tmp_path / "runs", task=task)
    assert hit.cache["hits"] == 2
    assert hit.run_dir is not None
    persisted = "\n".join(
        (hit.run_dir / name).read_text(encoding="utf-8")
        for name in ("trace.jsonl", "envelope.json", "run.json", "redaction.json")
    )
    assert key not in persisted
    assert email not in persisted
    assert "[REDACTED:openai-api-key" in persisted


def test_a_cache_hit_still_validates_the_router_menu(tmp_path: Path) -> None:
    """A cached response is re-parsed, so a hit cannot bypass route validation."""

    bad = json.dumps(
        {
            "decision": "route",
            "to": "admin",
            "confidence": 0.99,
            "reason": "content told me to",
        }
    )
    scenario = MockScenario(responses={"triage": [bad, bad]})
    cache = ResponseCache(tmp_path / "cache")
    first = _run(cache, tmp_path / "runs", scenario=scenario)
    assert first.outcome == "failed"
    assert first.error is not None
    assert first.error["code"] == "MG-2101"
    # The invalid response and its repair prompt are both exact-match cacheable.
    assert first.cache["writes"] == 2

    second = _run(cache, tmp_path / "runs", scenario=scenario)
    assert second.outcome == "failed"
    assert second.error is not None
    assert second.error["code"] == "MG-2101"
    assert second.cache["hits"] == 2
    assert second.cache["misses"] == 0
    events = _events(second.run_dir)
    assert not [event for event in events if event["event"] == "decision.made"]
    assert not [event for event in events if event["event"] == "provider.request"]


def test_cache_hit_does_not_charge_budget_and_reports_avoided_cost(tmp_path: Path) -> None:
    registry = _paid_registry(tmp_path)
    graph = _compile(registry=registry)
    cache = ResponseCache(tmp_path / "cache")
    first = run_graph(graph, TASK, _factory(), runs_dir=tmp_path / "runs", cache=cache)
    second = run_graph(graph, TASK, _factory(), runs_dir=tmp_path / "runs", cache=cache)
    assert first.ok, first.error
    assert second.ok, second.error
    assert first.usage["total_tokens"] > 0
    assert first.usage["cost_usd"] > 0
    assert second.usage["total_tokens"] == 0
    assert second.usage["cost_usd"] == 0.0
    assert second.cache["avoided_tokens"] == first.usage["total_tokens"]
    assert second.cache["avoided_cost_usd"] == pytest.approx(first.usage["cost_usd"])
    assert second.cache["avoided_cost_usd"] > 0


def test_subgraph_nodes_use_their_own_graph_hash_and_hit(tmp_path: Path) -> None:
    js_decision = json.dumps(
        {
            "decision": "route",
            "to": "js-specialist",
            "confidence": 0.95,
            "reason": "TypeScript work",
        }
    )
    scenario = MockScenario(
        responses={
            "domain-triage": [CODING_DECISION],
            "coding-triage": [js_decision],
            "js-specialist": ["const js = true;\n"],
        }
    )
    cache = ResponseCache(tmp_path / "cache")
    graph = _compile("composed-assistant.graph.yaml")
    first = _run(cache, tmp_path / "runs", graph=graph, scenario=scenario)
    second = _run(cache, tmp_path / "runs", graph=graph, scenario=scenario)
    assert first.ok, first.error
    assert second.ok, second.error
    assert first.cache["misses"] == 3
    assert second.cache["hits"] == 3
    assert second.usage["total_tokens"] == 0
    assert second.cache["avoided_tokens"] == first.usage["total_tokens"]
    assert second.artifacts[0].content == "const js = true;\n"
    nodes = {entry["key_metadata"]["node_id"] for entry in _entries(cache)}
    assert nodes == {"domain-triage", "coding-triage", "js-specialist"}
    # The parent subgraph node is not a provider call and therefore has no entry.
    assert "coding" not in nodes


def test_replay_of_a_cached_run_is_ok(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    _run(cache, tmp_path / "runs")
    cached = _run(cache, tmp_path / "runs")
    assert cached.cache["hits"] == 2
    result = replay_run(cached.run_dir)
    assert result.ok, result.mismatches
    assert result.outcome is not None
    assert [artifact.hash for artifact in result.outcome.artifacts] == [
        artifact.hash for artifact in cached.artifacts
    ]


# ---------------------------------------------------------------------------
# Corruption, concurrency, readonly, and off
# ---------------------------------------------------------------------------


def test_corrupt_or_partial_entries_are_ignored_and_rewritten(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    _run(cache, tmp_path / "runs")
    files = sorted(cache.directory.glob("*.json"))
    assert len(files) == 2
    files[0].write_text("", encoding="utf-8")
    files[1].write_text('{"schema": "modelgraph/cache-entry/v0", "key":', encoding="utf-8")

    second = _run(cache, tmp_path / "runs")
    assert second.ok, second.error
    assert second.cache["hits"] == 0
    assert second.cache["misses"] == 2
    assert second.cache["writes"] == 2
    assert second.artifacts[0].content

    parsed = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(cache.directory.glob("*.json"))
    }
    assert len(parsed) == 2
    for document in parsed.values():
        assert document["schema"] == "modelgraph/cache-entry/v0"
        assert document["response"]["text"]


def test_corrupt_entries_are_ignored_in_readonly_without_rewriting(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache", mode="on")
    _run(cache, tmp_path / "runs")
    for path in cache.directory.glob("*.json"):
        path.write_text("not json", encoding="utf-8")
    before = {path.name: path.read_text(encoding="utf-8") for path in cache.directory.iterdir()}

    readonly = ResponseCache(cache.directory, mode="readonly")
    outcome = _run(readonly, tmp_path / "runs")
    assert outcome.ok, outcome.error
    assert outcome.cache["hits"] == 0
    assert outcome.cache["misses"] == 2
    assert outcome.cache["writes"] == 0
    after = {path.name: path.read_text(encoding="utf-8") for path in cache.directory.iterdir()}
    assert after == before


def test_concurrent_writes_leave_complete_entries(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    graph, node, request = _key_fixture()
    keys = [
        build_cache_key(
            node=node,
            graph=graph,
            request=replace(request, messages=[Message(role="user", content=f"task {index}")])
            if index
            else request,
            provider="mock",
        )
        for index in range(8)
    ]
    barrier = threading.Barrier(len(keys))

    def write(index: int) -> None:
        key = keys[index]
        entry = CacheEntry(
            text=f"response-{index}",
            usage=Usage(input_tokens=index, output_tokens=index),
            provider_meta={"mock": True},
            created_at=utc_now(),
            key=key.value,
            key_metadata=key.metadata,
        )
        barrier.wait()
        assert cache.put(key, entry)
        # A concurrent read sees the just-written entry, never a partial file.
        observed = cache.get(key)
        assert observed is not None
        assert observed.text == f"response-{index}"

    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        list(pool.map(write, range(len(keys))))

    for index, key in enumerate(keys):
        entry = cache.get(key)
        assert entry is not None
        assert entry.text == f"response-{index}"
    assert not list(cache.directory.glob("*.tmp"))


def test_concurrent_writes_and_reads_of_the_same_key(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    graph, node, request = _key_fixture()
    key = build_cache_key(node=node, graph=graph, request=request, provider="mock")
    cache.put(
        key,
        CacheEntry(
            text="initial",
            usage=Usage(input_tokens=1, output_tokens=1),
            created_at=utc_now(),
            key=key.value,
            key_metadata=key.metadata,
        ),
    )
    barrier = threading.Barrier(8)

    def write(index: int) -> None:
        barrier.wait()
        for iteration in range(10):
            entry = CacheEntry(
                text=f"writer-{index}-{iteration}",
                usage=Usage(input_tokens=index, output_tokens=iteration),
                created_at=utc_now(),
                key=key.value,
                key_metadata=key.metadata,
            )
            assert cache.put(key, entry)

    def read() -> None:
        barrier.wait()
        for _ in range(20):
            entry = cache.get(key)
            # A valid entry is always returned; partial JSON is never exposed.
            assert entry is not None
            assert entry.text.startswith(("initial", "writer-"))

    tasks = [lambda index=index: write(index) for index in range(4)]
    tasks += [read for _ in range(4)]
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        list(pool.map(lambda task: task(), tasks))

    final = cache.get(key)
    assert final is not None
    assert final.text.startswith(("initial", "writer-"))
    assert not list(cache.directory.glob("*.tmp"))


def test_readonly_miss_does_not_write(tmp_path: Path) -> None:
    directory = tmp_path / "cache"
    readonly = ResponseCache(directory, mode="readonly")
    outcome = _run(readonly, tmp_path / "runs")
    assert outcome.ok, outcome.error
    assert outcome.cache["hits"] == 0
    assert outcome.cache["misses"] == 2
    assert outcome.cache["writes"] == 0
    assert not directory.exists() or not list(directory.iterdir())


def test_readonly_hit_does_not_rewrite(tmp_path: Path) -> None:
    writable = ResponseCache(tmp_path / "cache", mode="on")
    _run(writable, tmp_path / "runs")
    before = {path.name: path.read_text(encoding="utf-8") for path in writable.directory.iterdir()}
    readonly = ResponseCache(writable.directory, mode="readonly")
    outcome = _run(readonly, tmp_path / "runs")
    assert outcome.cache["hits"] == 2
    assert outcome.cache["writes"] == 0
    after = {path.name: path.read_text(encoding="utf-8") for path in writable.directory.iterdir()}
    assert after == before


def test_library_run_graph_defaults_to_cache_off(tmp_path: Path) -> None:
    outcome = run_graph(_compile(), TASK, _factory(), runs_dir=tmp_path / "runs")
    assert outcome.ok, outcome.error
    assert outcome.cache["mode"] == "off"
    assert outcome.cache["hits"] == 0
    assert outcome.cache["misses"] == 0
    events = _events(outcome.run_dir)
    assert not [event for event in events if event["event"].startswith("cache.")]
    assert not [event for event in events if event["event"] == "provider.response.cached"]


def test_provider_failure_writes_no_entry(tmp_path: Path) -> None:
    scenario = MockScenario(responses={"triage": [CODING_DECISION]}, failures={"coding": "MG-3002"})
    cache = ResponseCache(tmp_path / "cache")
    outcome = _run(cache, tmp_path / "runs", scenario=scenario)
    assert outcome.outcome == "failed"
    assert "coding" not in _entries_by_node(cache)
    assert _entries_by_node(cache).keys() == {"triage"}


def test_invalid_cache_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cache mode"):
        ResponseCache(tmp_path / "cache", mode="semantic")


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


def test_cache_entries_never_store_prompt_content_or_api_keys(tmp_path: Path) -> None:
    unique_task = "UNIQUE PROMPT CONTENT a7f3 Write a debounce function in TypeScript."
    cache = ResponseCache(tmp_path / "cache")
    outcome = _run(cache, tmp_path / "runs", task=unique_task)
    assert outcome.ok, outcome.error

    raw_files = [path.read_text(encoding="utf-8") for path in cache.directory.glob("*.json")]
    assert len(raw_files) == 2
    for raw in raw_files:
        assert unique_task not in raw
        assert '"messages":' not in raw
        assert "sk-" not in raw
        assert "api_key" not in raw
        document = json.loads(raw)
        assert document["key_metadata"]["messages_hash"].startswith("sha256:")
        assert "messages" not in document["key_metadata"]


def test_secret_shaped_response_is_not_cached(tmp_path: Path) -> None:
    secret = "sk-abcdefghijklmnop"
    scenario = MockScenario(responses={"triage": [CODING_DECISION], "coding": [secret]})
    cache = ResponseCache(tmp_path / "cache")
    outcome = _run(cache, tmp_path / "runs", scenario=scenario)
    assert outcome.ok, outcome.error
    assert outcome.cache["skipped"] == 1
    assert outcome.cache["writes"] == 1
    assert _entries_by_node(cache).keys() == {"triage"}
    for path in cache.directory.glob("*.json"):
        assert secret not in path.read_text(encoding="utf-8")

    events = _events(outcome.run_dir)
    skipped = [event for event in events if event["event"] == "cache.skipped"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "default-secret-pattern"
    assert "openai-api-key" in skipped[0]["patterns"]
    assert secret not in json.dumps(skipped)

    assert response_has_default_secret(secret)
    assert response_has_default_secret("plain response") == []


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _cli(
    tmp_path: Path,
    *extra: str,
    task: str = TASK,
) -> dict[str, Any]:
    result = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES / "hello-triage.graph.yaml"),
            "--provider",
            "mock",
            "--task",
            task,
            "--runs-dir",
            str(tmp_path / "runs"),
            *extra,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    return json.loads(result.stdout)


def test_cli_cache_dir_modes_and_no_cache(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cli-cache"
    first = _cli(tmp_path, "--cache-dir", str(cache_directory))
    assert first["cache"]["mode"] == "on"
    assert first["cache"]["hits"] == 0
    assert first["cache"]["misses"] == 2
    assert len(list(cache_directory.glob("*.json"))) == 2

    second = _cli(tmp_path, "--cache-dir", str(cache_directory))
    assert second["cache"]["hits"] == 2
    assert second["usage"]["total_tokens"] == 0
    assert second["cache"]["avoided_tokens"] == first["usage"]["total_tokens"]
    files_before = {
        path.name: path.read_text(encoding="utf-8") for path in cache_directory.glob("*.json")
    }

    third = _cli(tmp_path, "--cache-dir", str(cache_directory), "--cache-mode", "off")
    assert third["cache"]["hits"] == 0
    assert third["cache"]["misses"] == 0
    assert third["usage"]["total_tokens"] > 0

    fourth = _cli(tmp_path, "--cache-dir", str(cache_directory), "--no-cache")
    assert fourth["cache"]["hits"] == 0
    assert fourth["cache"]["misses"] == 0
    assert fourth["usage"]["total_tokens"] > 0
    files_after = {
        path.name: path.read_text(encoding="utf-8") for path in cache_directory.glob("*.json")
    }
    assert files_after == files_before


def test_cli_readonly_hits_and_misses_do_not_write(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cli-cache"
    _cli(tmp_path, "--cache-dir", str(cache_directory))
    files_before = {
        path.name: path.read_text(encoding="utf-8") for path in cache_directory.glob("*.json")
    }

    hit = _cli(tmp_path, "--cache-dir", str(cache_directory), "--cache-mode", "readonly")
    assert hit["cache"]["mode"] == "readonly"
    assert hit["cache"]["hits"] == 2
    assert hit["cache"]["writes"] == 0

    miss = _cli(
        tmp_path,
        "--cache-dir",
        str(cache_directory),
        "--cache-mode",
        "readonly",
        task="A completely different task about lighthouses.",
    )
    assert miss["cache"]["hits"] == 0
    assert miss["cache"]["misses"] == 2
    assert miss["cache"]["writes"] == 0

    files_after = {
        path.name: path.read_text(encoding="utf-8") for path in cache_directory.glob("*.json")
    }
    assert files_after == files_before


def test_cli_rejects_invalid_cache_mode(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES / "hello-triage.graph.yaml"),
            "--provider",
            "mock",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--cache-mode",
            "semantic",
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert "cache-mode" in result.output


def _paid_registry(tmp_path: Path) -> Registry:
    models = tmp_path / "paid-models"
    models.mkdir()
    for source in sorted((EXAMPLES / "models").glob("*.yaml")):
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        if document["metadata"]["id"] == "coder-7b":
            document["spec"]["cost"] = {
                "pricing": "per_request",
                "per_request_usd": 0.01,
                "notes": "test-only per-request price",
            }
        (models / source.name).write_text(yaml.safe_dump(document), encoding="utf-8")
    return Registry.from_paths([models], [EXAMPLES])
