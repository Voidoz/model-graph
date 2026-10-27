"""Paired comparison of two evaluation reports.

Comparisons require the same suite content hash so the case set and checks are identical.
Per-case deltas are paired by case id; the aggregate delta and a seeded bootstrap interval
are reported alongside win/loss/tie counts. A comparison is evidence about a candidate
against an incumbent, not an absolute quality score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modelgraph.evaluation.errors import EvalError
from modelgraph.evaluation.metrics import (
    bootstrap_ci,
    mean_or_none,
    percentile,
    safe_div,
)
from modelgraph.evaluation.report import (
    EvalReport,
    determinism_fingerprint,
    environment_info,
    new_report_id,
    validate_report,
)
from modelgraph.util import canonical_json, sha256_text


def compare_reports(
    candidate: EvalReport | dict[str, Any] | str | Path,
    baseline: EvalReport | dict[str, Any] | str | Path,
    *,
    allow_suite_mismatch: bool = False,
    reports_dir: str | Path | None = None,
    command: str | None = None,
    persist: bool = True,
    bootstrap_resamples: int = 2000,
    seed: int = 1234,
) -> EvalReport:
    candidate_data, candidate_path = _coerce(candidate)
    baseline_data, baseline_path = _coerce(baseline)
    for label, payload in (("candidate", candidate_data), ("baseline", baseline_data)):
        issues = validate_report(payload)
        if issues:
            raise EvalError(f"{label} report is invalid: {issues[0]}")
    candidate_suite = candidate_data["suite"]["hash"]
    baseline_suite = baseline_data["suite"]["hash"]
    if candidate_suite != baseline_suite and not allow_suite_mismatch:
        raise EvalError(
            "cannot compare reports from different suite versions: "
            f"candidate={candidate_suite}, baseline={baseline_suite}. "
            "Re-run both suites or pass allow_suite_mismatch=True explicitly.",
            details={"candidate_suite": candidate_suite, "baseline_suite": baseline_suite},
        )
    candidate_cases = {case.get("case_id"): case for case in candidate_data["cases"]}
    baseline_cases = {case.get("case_id"): case for case in baseline_data["cases"]}
    common = sorted(set(candidate_cases) & set(baseline_cases))
    if not common:
        raise EvalError("candidate and baseline reports share no case ids")
    missing = sorted(set(baseline_cases) ^ set(candidate_cases))
    direction = candidate_data["primary_metric"].get("direction", "max")
    paired: list[dict[str, Any]] = []
    deltas: list[float] = []
    improvements: list[float] = []
    cost_deltas: list[float] = []
    latency_deltas: list[float] = []
    for case_id in common:
        candidate_case = candidate_cases[case_id]
        baseline_case = baseline_cases[case_id]
        candidate_value = _case_value(candidate_data, candidate_case)
        baseline_value = _case_value(baseline_data, baseline_case)
        delta = None
        improvement = None
        if candidate_value is not None and baseline_value is not None:
            delta = candidate_value - baseline_value
            improvement = delta if direction == "max" else -delta
            deltas.append(delta)
            improvements.append(improvement)
        cost_delta = float(candidate_case.get("cost_usd", 0.0)) - float(
            baseline_case.get("cost_usd", 0.0)
        )
        latency_delta = float(candidate_case.get("latency_seconds", 0.0)) - float(
            baseline_case.get("latency_seconds", 0.0)
        )
        cost_deltas.append(cost_delta)
        latency_deltas.append(latency_delta)
        paired.append(
            {
                "case_id": case_id,
                "baseline": {
                    "value": baseline_value,
                    "success": baseline_case.get("success"),
                    "cost_usd": baseline_case.get("cost_usd"),
                    "latency_seconds": baseline_case.get("latency_seconds"),
                },
                "candidate": {
                    "value": candidate_value,
                    "success": candidate_case.get("success"),
                    "cost_usd": candidate_case.get("cost_usd"),
                    "latency_seconds": candidate_case.get("latency_seconds"),
                },
                "delta": delta,
                "improvement": improvement,
                "cost_delta_usd": cost_delta,
                "latency_delta_seconds": latency_delta,
            }
        )
    wins = sum(
        1
        for item in paired
        if isinstance(item["improvement"], float) and item["improvement"] > 1e-9
    )
    losses = sum(
        1
        for item in paired
        if isinstance(item["improvement"], float) and item["improvement"] < -1e-9
    )
    ties = sum(
        1
        for item in paired
        if isinstance(item["improvement"], float) and abs(item["improvement"]) <= 1e-9
    )
    paired_count = len(deltas)
    primary_name = candidate_data["primary_metric"].get("name", "metric")
    baseline_value = baseline_data["primary_metric"].get("value")
    candidate_value = candidate_data["primary_metric"].get("value")
    aggregate_delta = (
        candidate_value - baseline_value
        if isinstance(candidate_value, (int, float)) and isinstance(baseline_value, (int, float))
        else None
    )
    aggregate_improvement = (
        aggregate_delta
        if aggregate_delta is not None and direction == "max"
        else (-aggregate_delta if aggregate_delta is not None else None)
    )
    primary_ci = bootstrap_ci(deltas, resamples=bootstrap_resamples, seed=seed)
    cost_ci = bootstrap_ci(cost_deltas, resamples=bootstrap_resamples, seed=seed)
    # ``delta_p50_*`` is a median, so its interval must resample the median too. Bootstrapping
    # the mean and labelling it p50 made the point estimate and interval describe two different
    # statistics (and could hide a bimodal latency regression behind a small mean).
    latency_p50 = percentile(latency_deltas, 0.5)
    latency_ci = bootstrap_ci(
        latency_deltas,
        statistic=lambda sample: percentile(sample, 0.5),
        resamples=bootstrap_resamples,
        seed=seed,
    )
    slices = _comparison_slices(
        candidate_data,
        baseline_data,
        common,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    candidate_costs = [float(case.get("cost_usd", 0.0)) for case in candidate_data["cases"]]
    baseline_costs = [float(case.get("cost_usd", 0.0)) for case in baseline_data["cases"]]
    candidate_latencies = [
        float(case.get("latency_seconds", 0.0)) for case in candidate_data["cases"]
    ]
    baseline_latencies = [
        float(case.get("latency_seconds", 0.0)) for case in baseline_data["cases"]
    ]
    candidate_hash = sha256_text(canonical_json(candidate_data))
    baseline_hash = sha256_text(canonical_json(baseline_data))
    worse = [
        item
        for item in paired
        if isinstance(item["improvement"], float) and item["improvement"] < -1e-9
    ]
    failures = [
        {
            "case_id": item["case_id"],
            "reason": (
                f"candidate {item['candidate']['value']!r} vs "
                f"baseline {item['baseline']['value']!r} "
                f"(improvement {item['improvement']!r})"
            ),
        }
        for item in worse
    ]
    metrics = {
        "paired_cases": paired_count,
        "missing_cases": len(missing),
        "baseline_primary": baseline_value,
        "candidate_primary": candidate_value,
        "primary_delta": aggregate_delta,
        "primary_delta_ci95": primary_ci,
        "primary_improvement": aggregate_improvement,
        "primary_improvement_direction": direction,
        "win_rate": safe_div(wins, paired_count),
        "loss_rate": safe_div(losses, paired_count),
        "tie_rate": safe_div(ties, paired_count),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "cost_delta_per_task_usd": safe_div(sum(cost_deltas), len(cost_deltas)),
        "cost_delta_per_task_usd_ci95": cost_ci,
        "baseline_cost_per_task_usd": mean_or_none(baseline_costs),
        "candidate_cost_per_task_usd": mean_or_none(candidate_costs),
        "latency_delta_p50_seconds": percentile(latency_deltas, 0.5),
        "latency_delta_mean_seconds": mean_or_none(latency_deltas),
        "baseline_latency_p95_seconds": percentile(baseline_latencies, 0.95),
        "candidate_latency_p95_seconds": percentile(candidate_latencies, 0.95),
    }
    limitations = [
        "Paired comparison uses the same suite cases; a suite change invalidates it.",
        "The delta is a measurement, not a decision rule; pre-register thresholds before looking.",
        "Wall latency deltas include harness overhead and vary between runs.",
    ]
    if candidate_data["suite"]["hash"] != baseline_data["suite"]["hash"]:
        limitations.append(
            "Suite hashes differ; the comparison was explicitly allowed but is not a clean "
            "matched-suite comparison."
        )
    if paired_count < 20:
        limitations.append(
            f"Only {paired_count} paired cases; the confidence interval is wide and the win "
            "rate is unstable."
        )
    if missing:
        limitations.append(f"{len(missing)} case id(s) appear in only one report.")
    if command is None:
        command = (
            "mg eval compare --candidate "
            f"{candidate_data['report_id']} --baseline {baseline_data['report_id']}"
        )
    reproduce = _reproduce_command(candidate_path, baseline_path)
    report = EvalReport(
        report_id=new_report_id(),
        level="comparison",
        claim=(
            f"Candidate {candidate_data['report_id']} vs baseline "
            f"{baseline_data['report_id']} on {primary_name}"
        ),
        suite=candidate_data["suite"],
        graph=candidate_data.get("graph"),
        target=candidate_data.get("target", {}),
        provider=candidate_data.get("provider", {}),
        command=command,
        commands=[command, reproduce],
        reproduce=reproduce,
        environment=environment_info(),
        selection={
            "paired_cases": paired_count,
            "selected_cases": paired_count,
            "total_cases": len(candidate_data.get("cases", [])),
            "missing_case_ids": missing,
            "baseline_selection": baseline_data.get("selection", {}),
            "candidate_selection": candidate_data.get("selection", {}),
        },
        budget={
            "baseline": baseline_data.get("budget", {}),
            "candidate": candidate_data.get("budget", {}),
        },
        metrics=metrics,
        primary_metric={
            "name": f"{primary_name}_delta",
            "value": aggregate_delta,
            "ci95": primary_ci,
            "threshold": None,
            "direction": direction,
            "passed": None,
        },
        thresholds=[],
        gate=None,
        cases=paired,
        failures=failures,
        slices=slices,
        protocol_errors={},
        costs={
            "baseline_per_task_usd": mean_or_none(baseline_costs),
            "candidate_per_task_usd": mean_or_none(candidate_costs),
            "delta_per_task_usd": mean_or_none(cost_deltas),
            "delta_ci95": cost_ci,
        },
        latency={
            "baseline_p95_seconds": percentile(baseline_latencies, 0.95),
            "candidate_p95_seconds": percentile(candidate_latencies, 0.95),
            "delta_p50_seconds": latency_p50,
            "delta_p50_ci95": latency_ci,
            "paired_deltas": len(latency_deltas),
        },
        usage=candidate_data.get("usage", {}),
        determinism={
            "scripted_provider": bool(
                candidate_data.get("determinism", {}).get("scripted_provider")
                and baseline_data.get("determinism", {}).get("scripted_provider")
            ),
            "fingerprint": determinism_fingerprint(
                {
                    "candidate": candidate_hash,
                    "baseline": baseline_hash,
                    "deltas": deltas,
                    "direction": direction,
                }
            ),
            "excludes": ["generated_at", "report_id", "latency_seconds", "report_dir"],
        },
        limitations=limitations,
        comparisons=[
            {
                "name": "baseline",
                "baseline_report_id": baseline_data["report_id"],
                "candidate_report_id": candidate_data["report_id"],
                "delta": aggregate_delta,
                "improvement": aggregate_improvement,
                "ci95": primary_ci,
                "direction": direction,
            }
        ],
        baseline={
            "report_id": baseline_data["report_id"],
            "report_hash": baseline_hash,
            "suite_hash": baseline_data["suite"]["hash"],
            "graph_hash": (baseline_data.get("graph") or {}).get("hash"),
            "primary": baseline_value,
            "path": str(baseline_path) if baseline_path else None,
        },
        candidate={
            "report_id": candidate_data["report_id"],
            "report_hash": candidate_hash,
            "suite_hash": candidate_data["suite"]["hash"],
            "graph_hash": (candidate_data.get("graph") or {}).get("hash"),
            "primary": candidate_value,
            "path": str(candidate_path) if candidate_path else None,
        },
    )
    if persist:
        report.write(reports_dir)
    return report


def load_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read report {source}: {exc}", path=str(source)) from exc
    if not isinstance(payload, dict):
        raise EvalError(f"report {source} must be a JSON object", path=str(source))
    return payload


def _coerce(
    value: EvalReport | dict[str, Any] | str | Path,
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, EvalReport):
        path = value.report_dir / "report.json" if value.report_dir else None
        return value.to_dict(), path
    if isinstance(value, dict):
        return value, None
    path = Path(value)
    return load_report(path), path


def _case_value(report: dict[str, Any], case: dict[str, Any]) -> float | None:
    name = report.get("primary_metric", {}).get("name")
    if name in {"task_success", "oracle_agreement", "correctness"}:
        if "correct" in case:
            return 1.0 if case.get("correct") else 0.0
        return 1.0 if case.get("success") else 0.0
    if name in {"validity"}:
        return 1.0 if case.get("valid") else 0.0
    if name in {"run_success_rate"}:
        return 1.0 if case.get("run_success") else 0.0
    if name in {"quality_score"}:
        value = case.get("quality")
        return float(value) if isinstance(value, (int, float)) else None
    if name in {"check_pass_rate"}:
        total = int(case.get("checks_total", 0))
        passed = int(case.get("checks_passed_count", 0))
        return passed / total if total else None
    if name in {"cost_per_task_usd"}:
        return float(case.get("cost_usd", 0.0))
    if name in {"latency_p95_seconds", "latency_mean_seconds", "latency_p50_seconds"}:
        return float(case.get("latency_seconds", 0.0))
    if name in {"protocol_error_rate", "protocol_errors_per_100_calls"}:
        return 1.0 if case.get("protocol_errors") else 0.0
    if name in {"abstain_precision", "abstain_recall"}:
        if case.get("chosen") == "abstain":
            return 1.0 if case.get("oracle", {}).get("ambiguous") else 0.0
        return None
    return None


def _slice_keys(case: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    tags = case.get("tags")
    if isinstance(tags, list):
        keys.update(str(tag) for tag in tags if tag)
    metadata = case.get("metadata")
    if isinstance(metadata, dict):
        for dimension in ("domain", "difficulty"):
            value = metadata.get(dimension)
            if value is not None:
                keys.add(f"{dimension}:{value}")
    return keys


def _comparison_slices(
    candidate_data: dict[str, Any],
    baseline_data: dict[str, Any],
    common: list[str],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Per-slice paired deltas with seeded bootstrap intervals.

    The incumbent slice tables are model-level aggregates. A comparison needs paired deltas,
    so this recomputes the same tags/dimensions over the shared case ids and reports uncertainty
    per slice. A slice can be omitted if neither report labels any case with it.
    """
    candidate_cases = {str(case.get("case_id")): case for case in candidate_data["cases"]}
    baseline_cases = {str(case.get("case_id")): case for case in baseline_data["cases"]}
    keys: set[str] = set()
    for case_id in common:
        keys.update(_slice_keys(candidate_cases[case_id]))
        keys.update(_slice_keys(baseline_cases[case_id]))
    direction = str(candidate_data.get("primary_metric", {}).get("direction", "max"))
    slices: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        case_ids = [
            case_id
            for case_id in common
            if key in _slice_keys(candidate_cases[case_id])
            or key in _slice_keys(baseline_cases[case_id])
        ]
        if not case_ids:
            continue
        candidate_values: list[float] = []
        baseline_values: list[float] = []
        value_deltas: list[float] = []
        cost_deltas: list[float] = []
        latency_deltas: list[float] = []
        wins = losses = ties = 0
        candidate_costs: list[float] = []
        baseline_costs: list[float] = []
        candidate_latencies: list[float] = []
        baseline_latencies: list[float] = []
        for case_id in case_ids:
            candidate_case = candidate_cases[case_id]
            baseline_case = baseline_cases[case_id]
            candidate_value = _case_value(candidate_data, candidate_case)
            baseline_value = _case_value(baseline_data, baseline_case)
            if candidate_value is not None:
                candidate_values.append(candidate_value)
            if baseline_value is not None:
                baseline_values.append(baseline_value)
            if candidate_value is not None and baseline_value is not None:
                delta = candidate_value - baseline_value
                value_deltas.append(delta)
                improvement = delta if direction == "max" else -delta
                if improvement > 1e-9:
                    wins += 1
                elif improvement < -1e-9:
                    losses += 1
                else:
                    ties += 1
            candidate_cost = float(candidate_case.get("cost_usd", 0.0))
            baseline_cost = float(baseline_case.get("cost_usd", 0.0))
            candidate_costs.append(candidate_cost)
            baseline_costs.append(baseline_cost)
            cost_deltas.append(candidate_cost - baseline_cost)
            candidate_latency = float(candidate_case.get("latency_seconds", 0.0))
            baseline_latency = float(baseline_case.get("latency_seconds", 0.0))
            candidate_latencies.append(candidate_latency)
            baseline_latencies.append(baseline_latency)
            latency_deltas.append(candidate_latency - baseline_latency)
        slices[key] = {
            "primary_metric": candidate_data.get("primary_metric", {}).get("name"),
            "cases": len(case_ids),
            "baseline_primary": mean_or_none(baseline_values),
            "candidate_primary": mean_or_none(candidate_values),
            "primary_delta": mean_or_none(value_deltas),
            "primary_delta_ci95": (
                bootstrap_ci(value_deltas, resamples=bootstrap_resamples, seed=seed)
                if value_deltas
                else None
            ),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "baseline_cost_per_task_usd": mean_or_none(baseline_costs),
            "candidate_cost_per_task_usd": mean_or_none(candidate_costs),
            "cost_delta_per_task_usd": mean_or_none(cost_deltas),
            "cost_delta_ci95": bootstrap_ci(cost_deltas, resamples=bootstrap_resamples, seed=seed),
            "baseline_latency_p50_seconds": percentile(baseline_latencies, 0.5),
            "candidate_latency_p50_seconds": percentile(candidate_latencies, 0.5),
            "latency_delta_p50_seconds": percentile(latency_deltas, 0.5),
            "latency_delta_ci95": bootstrap_ci(
                latency_deltas,
                statistic=lambda sample: percentile(sample, 0.5),
                resamples=bootstrap_resamples,
                seed=seed,
            ),
        }
    return slices


def _reproduce_command(candidate_path: Path | None, baseline_path: Path | None) -> str:
    if candidate_path is not None and baseline_path is not None:
        return (
            'python -c "from modelgraph.evaluation import compare_reports; '
            f'compare_reports({str(candidate_path)!r}, {str(baseline_path)!r})"'
        )
    return (
        'python -c "from modelgraph.evaluation import compare_reports; '
        "compare_reports('candidate/report.json', 'baseline/report.json')\""
    )
