"""Build a reproducible synthetic router dataset through the T-0023 exporter.

This is a *demonstration* data generator, not a source of truth about real routing quality. It
creates mock runs for the hello-triage graph, attaches synthetic oracle labels, and exports them
with the same group-split/consent/redaction path as a real dataset. It is deterministic given the
seed and explicit run ids, so the recipe in ``docs/training.md`` can be re-run byte-for-byte
apart from timestamps.

Usage:
    uv run python training/scripts/build_demo_dataset.py --seed 20261025 --count 200

Outputs (git-ignored by default because they live under ``runs/``):
    runs/training/demo-runs/                 recorded mock runs
    runs/training/demo-labels.yaml           synthetic oracle labels
    runs/training/router-dataset/            T-0023 dataset export
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from modelgraph.compiler import compile_graph
from modelgraph.data import export_dataset
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.registry import Registry
from modelgraph.runtime.engine import run_graph

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "examples" / "hello-triage.graph.yaml"
SUITE = ROOT / "evals" / "mock" / "triage"
CONSENT = "synthetic operator consent for offline router-training demo"

LANGUAGES = ("Python", "TypeScript", "Rust", "Go", "Java", "C++")
COMPONENTS = ("parser", "scheduler", "cache", "API client", "CLI", "compiler", "queue")
ACTIONS = (
    "validates input",
    "retries failed requests",
    "parses JSON",
    "sorts records",
    "streams a file",
    "compiles a query",
    "deduplicates rows",
)
DOMAINS = ("inventory", "billing", "bookings", "analytics", "shipping")
CHARACTERS = (
    "lighthouse keeper",
    "dragon",
    "robot",
    "detective",
    "clockmaker",
    "fox",
    "astronaut",
)
THINGS = ("rain", "the stars", "teacups", "a hidden door", "the sea", "memories", "a comet")
PLACES = ("a floating city", "an old library", "a forest", "the moon", "a sleeping village")
STORY_ACTIONS = (
    "learns to fly",
    "finds a letter",
    "loses a shadow",
    "waits for sunrise",
    "collects forgotten names",
)

CODING_TEMPLATES = (
    "Implement a {language} function that {action}.",
    "Debug this {language} error in the {component}.",
    "Refactor the {language} {component} to {action}.",
    "Write a unit test for a {language} {component}.",
    "Explain how a {component} works and give {language} example code.",
    "Design a database schema for {domain} and write the SQL query.",
    "Optimize this algorithm so it {action} without extra allocations.",
)
STORY_TEMPLATES = (
    "Tell a story about a {character} who {story_action}.",
    "Write a short poem about {thing}.",
    "Describe a {character} discovering {thing} in {place}.",
    "Write a fairy tale set in {place}.",
    "Write a scene where a {character} meets a {character2}.",
    "Write a mystery story about {thing}.",
    "Write a short narrative about a {character} in {place}.",
)
BOUNDARY_CODING = (
    "Write a {language} function that parses a story file and counts words.",
    "Debug the code that renders a poem to PDF.",
    "Implement a story generator in {language}.",
    "Explain how a narrative engine stores chapters in a database.",
    "Write a {language} function that sorts characters in a fairy tale.",
)
BOUNDARY_STORY = (
    "Write a story about a programmer debugging a {language} function.",
    "Write a poem about a robot learning to code.",
    "Tell a tale about a database that came alive.",
    "Write a scene about a wizard writing {language}.",
    "Write a story about an algorithm that wanted to be a poem.",
)
AMBIGUOUS = (
    "Write a poem that teaches {language} generics.",
    "Explain {concept} as a story about a {character}.",
    "Write a story that also works as a {language} tutorial.",
    "Describe a {component} in the form of a poem.",
    "Write a fairy tale that explains {concept} with code.",
    "Write a scene where a {character} debugs a {component}.",
)
CONCEPTS = ("recursion", "caching", "type inference", "a database index", "concurrency")
STORY_CUES = (
    "story",
    "poem",
    "tale",
    "fiction",
    "narrative",
    "character",
    "dragon",
    "wizard",
    "robot",
    "lighthouse",
    "fairy",
    "poetry",
)


def build_tasks(seed: int, count: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    tasks: list[dict[str, object]] = []
    patterns = (
        ("coding", CODING_TEMPLATES, 0.38),
        ("story", STORY_TEMPLATES, 0.30),
        ("boundary-coding", BOUNDARY_CODING, 0.09),
        ("boundary-story", BOUNDARY_STORY, 0.09),
        ("ambiguous", AMBIGUOUS, 0.14),
    )
    for index in range(count):
        roll = rng.random()
        cumulative = 0.0
        category = patterns[-1][0]
        templates = patterns[-1][1]
        for name, options, weight in patterns:
            cumulative += weight
            if roll <= cumulative:
                category = name
                templates = options
                break
        template = rng.choice(templates)
        text = template.format(
            language=rng.choice(LANGUAGES),
            component=rng.choice(COMPONENTS),
            action=rng.choice(ACTIONS),
            domain=rng.choice(DOMAINS),
            character=rng.choice(CHARACTERS),
            character2=rng.choice(CHARACTERS),
            thing=rng.choice(THINGS),
            place=rng.choice(PLACES),
            story_action=rng.choice(STORY_ACTIONS),
            concept=rng.choice(CONCEPTS),
        )
        # Keep every task distinct so exact dedupe does not silently collapse the fixture.
        text = f"{text} Ref {index + 1:04d}."
        if category == "coding":
            oracle = "coding"
            ambiguous = False
        elif category == "story":
            oracle = "story"
            ambiguous = False
        elif category == "boundary-coding":
            oracle = "coding"
            ambiguous = False
        elif category == "boundary-story":
            oracle = "story"
            ambiguous = False
        else:
            # Ambiguous: both menus plausibly fit. Keep an oracle route for the confusion
            # matrix, but the evaluation expects abstention.
            oracle = "coding" if "teaches" in text or "component" in text else "story"
            ambiguous = True
        tasks.append(
            {
                "id": f"demo-{index + 1:04d}",
                "task": text,
                "oracle": oracle,
                "ambiguous": ambiguous,
                "category": category,
            }
        )
    return tasks


def incumbent_route(task: str) -> str:
    """A deliberately simple scripted incumbent for comparison only.

    It is not a language model and its result is not evidence about a deployed router. It looks
    for a story cue anywhere in the request and otherwise falls back to coding. Distractor
    requests and ambiguous requests expose its weaknesses.
    """
    lower = task.lower()
    if any(cue in lower for cue in STORY_CUES):
        return "story"
    return "coding"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/training/demo-runs")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/training/router-dataset")
    parser.add_argument("--labels", type=Path, default=ROOT / "runs/training/demo-labels.yaml")
    parser.add_argument("--split-plan", type=Path, default=ROOT / "runs/training/demo-splits.json")
    parser.add_argument("--force", action="store_true", help="Replace existing outputs.")
    args = parser.parse_args()
    for path in (args.runs_dir, args.out):
        non_empty = path.exists() and (not path.is_dir() or any(path.iterdir()))
        if non_empty and not args.force:
            parser.error(f"{path} is not empty; pass --force to replace it")
    if args.runs_dir.exists() and args.force:
        import shutil

        shutil.rmtree(args.runs_dir)
    if args.out.exists() and args.force:
        import shutil

        shutil.rmtree(args.out)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    args.split_plan.parent.mkdir(parents=True, exist_ok=True)

    compiled = compile_graph(
        GRAPH,
        registry=Registry.from_paths([ROOT / "registry/models"], [ROOT / "examples"]),
        strict_models=True,
        graph_paths=[GRAPH.parent],
    )
    if not compiled.ok or compiled.graph is None:
        raise SystemExit("hello-triage graph did not compile: " + "; ".join(compiled.errors))
    tasks = build_tasks(args.seed, args.count)
    labels: list[dict[str, object]] = []
    run_paths: list[Path] = []
    for _index, item in enumerate(tasks):
        task = str(item["task"])
        oracle = str(item["oracle"])
        ambiguous = bool(item["ambiguous"])
        incumbent = incumbent_route(task)
        decision = json.dumps(
            {
                "decision": "route",
                "to": incumbent,
                "confidence": 0.8,
                "reason": "scripted incumbent keyword rule (synthetic stand-in)",
            }
        )
        worker = (
            "export function demo() {\n  return 42;\n}\n"
            if incumbent == "coding"
            else "The lighthouse keeper counted the waves until the light answered."
        )
        scenario = MockScenario(responses={"triage": [decision], incumbent: [worker]})
        outcome = run_graph(
            compiled.graph,
            task,
            lambda model, scenario=scenario: MockProvider(
                scenario, max_context=model.spec.context_window
            ),
            runs_dir=args.runs_dir,
            run_id=str(item["id"]),
            seed=args.seed,
        )
        if outcome.run_dir is None or outcome.outcome != "succeeded":
            raise SystemExit(f"run {item['id']} did not succeed: {outcome.error}")
        run_paths.append(outcome.run_dir)
        labels.append(
            {
                "run_id": str(item["id"]),
                "kind": "routing",
                "node": "triage",
                "route": oracle,
                "ambiguous": ambiguous,
                "outcome": "success",
                "source": "oracle",
                "labeler": "synthetic-oracle",
                "reason": f"synthetic {item['category']} oracle",
            }
        )
    args.labels.write_text(yaml.safe_dump({"labels": labels}, sort_keys=False), encoding="utf-8")
    result = export_dataset(
        run_paths,
        args.out,
        dataset_id="router-triage-demo",
        dataset_name="Synthetic triage router demo",
        dataset_version="1.0.0",
        dataset_description=(
            "Mock hello-triage runs with synthetic oracle labels; a reproducible fixture for "
            "the router training recipe, not a quality claim."
        ),
        task="Route a request to coding or storytelling.",
        license_id="CC0-1.0",
        consent=CONSENT,
        labels=[args.labels],
        eval_suites=[SUITE],
        dedupe="exact",
        group_by="run",
        split_ratio="0.6,0.2,0.2",
        split_seed=f"router-demo-{args.seed}",
        split_plan=args.split_plan,
        created_at="2026-10-25T00:00:00Z",
    )
    print(f"runs:       {args.runs_dir} ({len(run_paths)})")
    print(f"dataset:    {result.dataset_dir}")
    print(f"examples:   {result.examples}")
    print(f"counts:     {result.counts}")
    print(f"by_route:   {result.manifest['spec']['examples']['by_route']}")
    print(f"routing:    {result.manifest['spec']['examples']['routing']}")
    print(f"labels:     {args.labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
