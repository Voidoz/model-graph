"""Run the offline safety fixture suite and emit a gate-ready report.

    uv run python evals/safety/run.py
    uv run python evals/safety/run.py --json > runs/evals/ci/safety.json
    uv run mg eval compare --baseline evals/baselines/safety.json \
        --candidate runs/evals/ci/safety.json

Exit codes match the rest of the evaluation tooling: 0 when every declared safety rule passes,
1 when a safety violation is measured, and 3 on an explicit input/contract error. No network or
model server is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from modelgraph.errors import ModelGraphError
from modelgraph.evaluation.safety import (
    SAFETY_CASE_PASS_RATE_METRIC,
    SAFETY_METRIC,
    SAFETY_VIOLATION_RATE_METRIC,
    run_safety_suite,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = ROOT / "evals" / "safety" / "model-safety"
DEFAULT_REPORTS = ROOT / "runs" / "evals" / "safety"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--json", action="store_true", help="print the full report JSON")
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="do not write report.json/report.md/cases under --reports-dir",
    )
    args = parser.parse_args(argv)
    try:
        report = run_safety_suite(
            args.suite,
            reports_dir=args.reports_dir,
            persist=not args.no_persist,
        )
    except ModelGraphError as exc:
        print(f"safety suite error: {exc}", file=sys.stderr)
        return 3
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        metrics = payload["metrics"]
        print(f"suite:    {payload['suite']['id']} ({payload['suite']['hash']})")
        print(f"gate:     {'PASS' if payload['gate'] and payload['gate']['passed'] else 'FAIL'}")
        print(
            f"safety:   {metrics.get(SAFETY_METRIC)} violations, "
            f"violation rate {metrics.get(SAFETY_VIOLATION_RATE_METRIC):.4f}, "
            f"case pass rate {metrics.get(SAFETY_CASE_PASS_RATE_METRIC):.4f}"
        )
        if report.report_dir is not None:
            print(f"report:   {report.report_dir}")
    return 0 if report.gate is not None and report.gate.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
