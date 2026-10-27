"""Build a reproducible synthetic specialist dataset through the T-0023 exporter.

This is a *demonstration* data generator, not a source of truth about specialist quality. It
creates mock coding-specialist runs, then exports them with the same consent, redaction,
dedupe, contamination, and group-split path as a real dataset. It is deterministic given the
seed and explicit run ids, so the T-0025 recipe can be re-run byte-for-byte apart from the
timestamps the exporter records.

Usage:
    uv run python training/scripts/build_specialist_demo_dataset.py --seed 20261025 --count 40

Outputs (git-ignored because they live under ``runs/``):
    runs/training/specialist-demo-runs/   recorded mock runs
    runs/training/specialist-dataset/     T-0023 dataset export
    runs/training/specialist-splits.json  frozen group-split plan
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from modelgraph.compiler import compile_graph
from modelgraph.data import export_dataset
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.registry import Registry
from modelgraph.runtime.engine import run_graph

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "examples" / "coding-specialist.graph.yaml"
SUITE = ROOT / "evals/mock/js-specialist"
CONSENT = "synthetic operator consent for offline specialist-training rehearsal"

ACTIONS = (
    "formats a duration as hours, minutes, and seconds",
    "groups records by a caller-supplied key without mutating the input",
    "retries an async request with exponential backoff and jitter",
    "normalizes Unicode text before comparing two labels",
    "turns a flat list of parent ids into a nested tree",
    "reads a JSON file and validates every required field",
    "calculates a rolling average over a typed numeric array",
    "escapes user text before inserting it into an HTML template",
    "merges two sorted arrays while removing duplicate values",
    "parses a comma-separated query string into a typed object",
    "batches an array into fixed-size chunks",
    "memoizes an expensive pure function with a bounded cache",
    "sorts records by a priority field and then by name",
    "converts a nested object into URL-safe form fields",
)
STYLES = ("TypeScript", "modern JavaScript", "TypeScript with strict mode")


def build_tasks(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    tasks = [f"Write a {rng.choice(STYLES)} helper that {rng.choice(ACTIONS)}." for _ in range(count)]
    # Keep every request distinct so exact dedupe does not silently collapse the fixture.
    return [f"{task} Fixture ref {index + 1:04d}." for index, task in enumerate(tasks)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/training/specialist-demo-runs")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/training/specialist-dataset")
    parser.add_argument(
        "--split-plan", type=Path, default=ROOT / "runs/training/specialist-splits.json"
    )
    parser.add_argument("--force", action="store_true", help="Replace existing outputs.")
    args = parser.parse_args()
    for path in (args.runs_dir, args.out):
        non_empty = path.exists() and (not path.is_dir() or any(path.iterdir()))
        if non_empty and not args.force:
            parser.error(f"{path} is not empty; pass --force to replace it")
    if args.force:
        for path in (args.runs_dir, args.out):
            if path.exists():
                shutil.rmtree(path)
        if args.split_plan.exists():
            args.split_plan.unlink()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.split_plan.parent.mkdir(parents=True, exist_ok=True)

    compiled = compile_graph(
        GRAPH,
        registry=Registry.from_paths([ROOT / "registry/models"], [ROOT / "examples"]),
        strict_models=True,
        graph_paths=[GRAPH.parent],
    )
    if not compiled.ok or compiled.graph is None:
        raise SystemExit("coding-specialist graph did not compile: " + "; ".join(compiled.errors))
    run_paths: list[Path] = []
    for index, task in enumerate(build_tasks(args.seed, args.count), start=1):
        decision = json.dumps(
            {
                "decision": "route",
                "to": "js-specialist",
                "confidence": 0.95,
                "reason": "synthetic TypeScript helper fixture",
            }
        )
        answer = (
            "export function helper<T>(input: readonly T[]): T[] {\n"
            "  // Deterministic fixture output; replace with the teacher/worker in a real run.\n"
            f"  return input.filter((_item, index) => index >= 0);\n"
            f"}}\n// fixture {index:04d}\n"
        )
        scenario = MockScenario(responses={"coding-triage": [decision], "js-specialist": [answer]})
        outcome = run_graph(
            compiled.graph,
            task,
            lambda model, scenario=scenario: MockProvider(
                scenario, max_context=model.spec.context_window
            ),
            runs_dir=args.runs_dir,
            run_id=f"specialist-demo-{index:04d}",
            seed=args.seed,
        )
        if outcome.run_dir is None or outcome.outcome != "succeeded":
            raise SystemExit(f"run {index} did not succeed: {outcome.error}")
        run_paths.append(outcome.run_dir)
    result = export_dataset(
        run_paths,
        args.out,
        dataset_id="js-specialist-demo",
        dataset_name="Synthetic JavaScript specialist demo",
        dataset_version="1.0.0",
        dataset_description=(
            "Mock coding-specialist runs with synthetic TypeScript outputs; a reproducible "
            "fixture for the T-0025 specialist pipeline, not a quality claim."
        ),
        task="Implement JavaScript/TypeScript helpers from a short natural-language request.",
        license_id="CC0-1.0",
        consent=CONSENT,
        eval_suites=[SUITE],
        dedupe="exact",
        group_by="run",
        split_ratio="0.7,0.15,0.15",
        split_seed=f"specialist-demo-{args.seed}",
        split_plan=args.split_plan,
        created_at="2026-10-25T00:00:00Z",
    )
    print(f"runs:       {args.runs_dir} ({len(run_paths)})")
    print(f"dataset:    {result.dataset_dir}")
    print(f"examples:   {result.examples}")
    print(f"counts:     {result.counts}")
    print(f"by_node:    {result.manifest['spec']['examples']['by_node']}")
    print(f"contamination: {result.manifest['spec']['eval_contamination']['status']}")
    print(f"splits:     {args.split_plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
