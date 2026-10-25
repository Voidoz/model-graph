"""Regenerate the committed CI baselines.

Run from the repository root:

    make baselines
    # or: uv run python evals/baselines/update.py

The reports are written to `evals/baselines/<level>.json` and are reviewed as a separate
commit. This script deliberately does not commit anything: baseline changes need review, and a
normal feature PR must not silently move the reference point. The e2e baseline excludes the
deliberately invalid mock case so the release hard gate ("validity must be 100%") is meaningful;
CI uses the same case selection. The safety baseline is the zero-violation fixture report.
"""

from __future__ import annotations

import json
from pathlib import Path

from modelgraph.evaluation import run_e2e_suite, run_node_suite, run_routing_suite
from modelgraph.evaluation.safety import run_safety_suite
from modelgraph.paths import runs_dir

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "evals" / "baselines"
E2E_CASES = [
    "js-debounce",
    "js-types",
    "general-rust",
    "story-lighthouse",
    "story-misrouted",
]


def main() -> int:
    reports = {
        "node": run_node_suite(
            ROOT / "evals" / "mock" / "js-specialist",
            provider="mock",
            reports_dir=ROOT / "runs" / "baselines",
            persist=False,
        ),
        "routing": run_routing_suite(
            ROOT / "evals" / "mock" / "triage",
            provider="mock",
            reports_dir=ROOT / "runs" / "baselines",
            persist=False,
        ),
        "e2e": run_e2e_suite(
            ROOT / "evals" / "e2e" / "composed-assistant",
            graph_path=ROOT / "examples" / "composed-assistant.graph.yaml",
            profile="local-8gb",
            provider="mock",
            reports_dir=ROOT / "runs" / "baselines",
            case_ids=E2E_CASES,
            persist=False,
        ),
        "safety": run_safety_suite(
            ROOT / "evals" / "safety" / "model-safety",
            provider="mock",
            reports_dir=ROOT / "runs" / "baselines",
            persist=False,
        ),
    }
    BASELINES.mkdir(parents=True, exist_ok=True)
    destinations: dict[str, Path] = {}
    for level, report in reports.items():
        issues = report.validate()
        if issues:
            raise SystemExit(f"refusing to write invalid {level} baseline: {issues[0]}")
        destination = BASELINES / f"{level}.json"
        destination.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        destinations[level] = destination
        print(f"wrote {destination.relative_to(ROOT)} ({report.report_id})")
    # The ticket's smoke command uses `mock.json`; keep it as an explicit e2e alias rather than
    # an undocumented convention. The level-specific files remain the CI source of truth.
    alias = BASELINES / "mock.json"
    alias.write_text(destinations["e2e"].read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {alias.relative_to(ROOT)} (alias of e2e.json)")
    print(
        "Baseline files are not committed automatically. Review the semantic diff "
        "(suite/graph hashes, metrics, cases) and commit them separately."
    )
    print(f"Generated run directories are under {runs_dir() / 'baselines'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
