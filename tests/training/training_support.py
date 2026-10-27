"""Shared synthetic fixtures for the router-training tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modelgraph.training.dataset import RouterDataset, RouterExample
from modelgraph.training.features import MenuRoute

ROUTES: tuple[MenuRoute, ...] = (
    MenuRoute(
        id="coding",
        text=(
            "Writes code, debugs programs, designs data structures, and explains technical "
            "implementation. When: the user asks for code, debugging, an algorithm, a database, "
            "or an API."
        ),
        order=0,
    ),
    MenuRoute(
        id="story",
        text=(
            "Writes fiction, narrative, characters, and poetry in the requested style. "
            "When: the user asks for a story, a poem, a tale, or creative writing."
        ),
        order=1,
    ),
)
HASH = "sha256:" + "a" * 64


def render_messages(
    task: str, routes: tuple[MenuRoute, ...] = ROUTES
) -> tuple[dict[str, str], ...]:
    lines = ["<routes>"]
    for route in routes:
        lines.append(f'<route id="{route.id}" artifact="text">')
        lines.append(route.text)
        lines.append("</route>")
    lines.append("</routes>")
    return (
        {"role": "system", "content": "You are a routing classifier."},
        {
            "role": "user",
            "content": (
                f'<request trust="untrusted">\n{task}\n</request>\n\n'
                + "\n".join(lines)
                + "\n\nRouting policy: confidence below 0.60 is treated as fallback."
            ),
        },
    )


def dataset_example(
    *,
    example_id: str,
    split: str,
    group: str,
    task: str,
    route: str | None,
    ambiguous: bool = False,
    source: str = "oracle",
    outcome: str = "success",
    routes: tuple[MenuRoute, ...] = ROUTES,
    chosen_route: str | None = None,
    confidence: float | None = 0.9,
    rendered: bool = True,
) -> dict[str, Any]:
    messages = list(render_messages(task, routes)) if rendered else []
    menu_text = "\n".join(
        ["<routes>"]
        + [
            f'<route id="{route_item.id}" artifact="text">\n{route_item.text}\n</route>'
            for route_item in routes
        ]
        + ["</routes>"]
    )
    return {
        "schema": "modelgraph/dataset-example/v0",
        "example_id": example_id,
        "kind": "routing",
        "split": split,
        "group": group,
        "source": {
            "run_id": f"run-{group}",
            "trace": "trace.jsonl",
            "run_outcome": "succeeded",
            "graph": {"id": "hello-triage", "version": "1.0.0", "hash": "deadbeef"},
            "node": "triage",
            "node_kind": "router",
            "event_index": 0,
        },
        "input": {
            "task": task,
            "prompt_hash": "prompt-hash",
            "rendered": rendered,
            "messages": messages,
            "menu": {
                "available": bool(messages),
                "text": menu_text if messages else None,
                "routes": [route_item.id for route_item in routes],
            },
        },
        "output": {
            "chosen_route": chosen_route,
            "chosen_decision": "route" if chosen_route else None,
            "confidence": confidence,
            "threshold": 0.6,
            "reason": "recorded decision",
            "fallback_used": False,
            "finish_reason": "stop",
            "usage": {},
            "error": None,
        },
        "label": {
            "outcome": outcome,
            "source": source,
            "route": route,
            "should_abstain": ambiguous if route is not None else None,
            "reason": None,
            "labeler": "test-labeler",
            "checks": [],
        },
        "consent": {
            "required": True,
            "recorded": True,
            "statement": "synthetic test consent",
            "scope": "local-training",
        },
        "license": {"id": "CC0-1.0", "uri": None, "distributable": True},
        "redaction": {"report": "redaction-report.json", "redacted": False, "patterns": []},
        "hashes": {"input": HASH, "output": HASH, "example": HASH},
    }


def router_example(
    *,
    index: int,
    split: str,
    task: str,
    route: str | None,
    ambiguous: bool = False,
    routes: tuple[MenuRoute, ...] = ROUTES,
) -> RouterExample:
    messages = render_messages(task, routes)
    return RouterExample(
        example_id=f"ex_{index:016x}",
        source_example_id=f"ex_{index:016x}",
        split=split,
        group=f"group-{index:04d}",
        request=task,
        messages=messages,
        menu_routes=routes,
        target="__abstain__" if ambiguous else str(route),
        oracle_route=route,
        ambiguous=ambiguous,
        label_source="oracle",
        label_outcome="success",
        labeler="test-labeler",
        prompt_hash="prompt-hash",
        output={
            "chosen_route": route,
            "chosen_decision": "route" if route else None,
            "confidence": 0.8,
        },
        provenance={"dataset": {"id": "synthetic", "version": "1.0.0", "hash": HASH}},
    )


def synthetic_dataset(*, seed: int = 7) -> RouterDataset:
    """A small, clearly separable dataset with train/validation/test groups."""
    examples: list[RouterExample] = []
    index = 0
    coding_train = (
        "Write a Python function that sorts records.",
        "Debug a Rust compiler error in the parser.",
        "Implement a TypeScript API client cache.",
        "Explain why this SQL query returns no rows.",
    )
    story_train = (
        "Tell a story about a lighthouse keeper.",
        "Write a poem about rain and stars.",
        "Describe a dragon who collects teacups.",
        "Write a fairy tale set in an old library.",
    )
    coding_val = (
        "Write a Go function that retries failed requests.",
        "Debug a Java exception in the scheduler.",
    )
    story_val = (
        "Write a short story about a fox in a forest.",
        "Write a poem about the moon.",
    )
    coding_test = (
        "Write a C++ function that parses JSON.",
        "Debug a Python error in the cache.",
    )
    story_test = (
        "Tell a tale about an astronaut and a comet.",
        "Write a narrative about a clockmaker.",
    )
    for task in coding_train:
        examples.append(router_example(index=index, split="train", task=task, route="coding"))
        index += 1
    for task in story_train:
        examples.append(router_example(index=index, split="train", task=task, route="story"))
        index += 1
    for task in coding_val:
        examples.append(router_example(index=index, split="validation", task=task, route="coding"))
        index += 1
    for task in story_val:
        examples.append(router_example(index=index, split="validation", task=task, route="story"))
        index += 1
    examples.append(
        router_example(
            index=index,
            split="validation",
            task="Write a poem that teaches Python generics.",
            route="coding",
            ambiguous=True,
        )
    )
    index += 1
    for task in coding_test:
        examples.append(router_example(index=index, split="test", task=task, route="coding"))
        index += 1
    for task in story_test:
        examples.append(router_example(index=index, split="test", task=task, route="story"))
        index += 1
    examples.append(
        router_example(
            index=index,
            split="test",
            task="Explain caching as a story about a dragon.",
            route="story",
            ambiguous=True,
        )
    )
    summary = {
        "dataset": {
            "id": "synthetic-router",
            "version": "1.0.0",
            "hash": HASH,
            "frozen_test": True,
        },
        "examples": len(examples),
        "counts": {
            split: sum(1 for item in examples if item.split == split)
            for split in ("train", "validation", "test")
        },
        "groups": {"train": 8, "validation": 5, "test": 5},
        "class_balance": {},
        "per_route": {},
        "ambiguity": {},
        "excluded": {},
    }
    return RouterDataset(
        path=None,
        examples=examples,
        summary=summary,
        dataset_manifest_hash=HASH,
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return path
