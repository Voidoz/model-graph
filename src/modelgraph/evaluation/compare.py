"""Contribution gates over two evaluation reports.

This module is the decision layer, not a second measurement layer. It reuses
:func:`modelgraph.evaluation.comparison.compare_reports` for paired case matching and report
construction, then adds the checks described in ``docs/evaluation.md`` section 6:

* per-metric baseline/candidate deltas with seeded paired-bootstrap confidence intervals;
* per-level quality/reliability gates for node, routing, and e2e reports;
* hard gates for validity, schema, safety, budget, and secret scanning;
* an explicit, reviewed waiver file whose entries are rendered into the report.

A hard-gate failure is never waived. A missing baseline is an error, not an empty comparison.
The default policy lives in ``evals/baselines/policy.json`` so the thresholds have one reviewed
source; set ``MG_GATE_POLICY`` to use a different policy during research.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

from modelgraph.evaluation.comparison import _case_value, compare_reports, load_report
from modelgraph.evaluation.errors import EvalError
from modelgraph.evaluation.metrics import bootstrap_ci
from modelgraph.evaluation.report import (
    EvalReport,
    validate_report,
)
from modelgraph.util import canonical_json, sha256_text

DEFAULT_POLICY_PATH = "evals/baselines/policy.json"
GATE_POLICY_SCHEMA = "modelgraph/gate-policy/v0"
GATE_WAIVER_SCHEMA = "modelgraph/gate-waiver/v0"
HARD_GATE_NAMES = ("schema", "validity", "safety", "budget", "secret_scan")

# Patterns are deliberately narrow and generic: enough to catch credentials accidentally copied
# into prompts/responses, without flagging ordinary report fields such as max_tokens.
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    (
        "bearer-token",
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
    ),
    (
        "assigned-secret",
        re.compile(
            r"(?i)(?:api[_-]?key|client[_-]?secret|password|access[_-]?token)"
            r"\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{20,}"
        ),
    ),
)


class GateComparisonReport(EvalReport):
    """An ``EvalReport`` whose Markdown renderer includes the gate decision details."""

    def to_markdown(self) -> str:
        lines = [super().to_markdown().rstrip(), ""]
        gate = self.gate or {}
        passed = bool(gate.get("passed"))
        lines.append("## CI gates")
        lines.append("")
        lines.append(f"**Result:** {'PASS' if passed else 'FAIL'}")
        lines.append("")
        policy = gate.get("policy") or {}
        if policy:
            lines.append(
                f"- Policy: `{policy.get('path')}` (`{policy.get('hash')}`), "
                f"source {policy.get('source') or 'unspecified'}"
            )
        if gate.get("waived"):
            lines.append(f"- Waived checks: {', '.join(gate['waived'])}")
        if gate.get("hard_failures"):
            lines.append(
                "- Hard-gate failures: "
                + ", ".join(f"hard.{name}" for name in gate["hard_failures"])
            )
        lines.append("")
        checks = gate.get("checks") or []
        if checks:
            lines.append("| check | baseline | candidate | delta | 95% CI | tolerance | result |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
            for check in checks:
                lines.append(
                    f"| `{check.get('id')}` | {_fmt(check.get('baseline'))} | "
                    f"{_fmt(check.get('candidate'))} | {_fmt_signed(check.get('delta'))} | "
                    f"{_fmt_ci(check.get('ci95'))} | {_fmt_tolerance(check)} | "
                    f"{check.get('status', 'unknown')} |"
                )
            lines.append("")
        hard = gate.get("hard_gates") or {}
        if hard:
            lines.append("### Hard gates")
            lines.append("")
            lines.append("| hard gate | result | detail |")
            lines.append("| --- | --- | --- |")
            for name in HARD_GATE_NAMES:
                entry = hard.get(name)
                if not isinstance(entry, dict):
                    continue
                lines.append(
                    f"| `{name}` | {entry.get('status', 'unknown')} | "
                    f"{_escape(str(entry.get('detail') or ''))} |"
                )
            lines.append("")
        waivers = gate.get("waivers") or []
        if waivers:
            lines.append("### Waivers")
            lines.append("")
            lines.append("| file | approved by | reason | expires | applied | rejected |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for waiver in waivers:
                lines.append(
                    f"| `{_escape(str(waiver.get('path')))}` | "
                    f"{_escape(str(waiver.get('approved_by') or ''))} | "
                    f"{_escape(str(waiver.get('reason') or ''))} | "
                    f"{_escape(str(waiver.get('expires') or ''))} | "
                    f"{_escape(', '.join(waiver.get('applied') or []))} | "
                    f"{_escape(', '.join(waiver.get('rejected') or []))} |"
                )
            lines.append("")
        not_measured = [
            name
            for name in HARD_GATE_NAMES
            if isinstance(hard.get(name), dict) and hard[name].get("status") == "not_measured"
        ]
        if not_measured:
            lines.append(
                "Hard gates not measured by this suite: " + ", ".join(not_measured) + ". "
                "They are reported explicitly rather than treated as passing evidence."
            )
            lines.append("")
        return "\n".join(lines)


def load_policy(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    """Load and minimally validate the gate policy.

    ``None`` resolves ``MG_GATE_POLICY``, then the repository default. A missing policy is an
    explicit error so a comparison cannot silently fall back to permissive thresholds.
    """
    source = Path(path).expanduser() if path is not None else _default_policy_path()
    if not source.is_file():
        raise EvalError(
            f"gate policy not found: {source}. Expected {DEFAULT_POLICY_PATH} in the repository "
            "or pass --policy.",
            path=str(source),
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read gate policy {source}: {exc}", path=str(source)) from exc
    if not isinstance(payload, dict):
        raise EvalError(f"gate policy {source} must be a JSON object", path=str(source))
    if payload.get("schema") != GATE_POLICY_SCHEMA:
        raise EvalError(
            f"gate policy {source} has schema {payload.get('schema')!r}; "
            f"expected {GATE_POLICY_SCHEMA!r}",
            path=str(source),
        )
    levels = payload.get("levels")
    if not isinstance(levels, dict) or not levels:
        raise EvalError(f"gate policy {source} has no levels object", path=str(source))
    return payload, source


def load_baseline(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a baseline report from ``report.json`` or a report directory."""
    source = _report_path(path, "baseline")
    data = _clean_payload(load_report(source))
    issues = validate_report(data)
    if issues:
        raise EvalError(
            f"baseline report {source} is invalid: {issues[0]}",
            path=str(source),
            details={"issues": issues},
        )
    return data, source


def load_waiver(path: str | Path) -> dict[str, Any]:
    """Load and validate one waiver file.

    Required fields: ``approved_by``, ``reason``, ``checks``, and ``expires`` (an ISO date or
    the literal ``pr``). ``suite_hash``/``graph_hash`` optionally narrow the waiver.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise EvalError(f"waiver file not found: {source}", path=str(source))
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read waiver {source}: {exc}", path=str(source)) from exc
    if not isinstance(payload, dict):
        raise EvalError(f"waiver {source} must be a JSON object", path=str(source))
    schema = payload.get("schema", GATE_WAIVER_SCHEMA)
    if schema != GATE_WAIVER_SCHEMA:
        raise EvalError(
            f"waiver {source} has schema {schema!r}; expected {GATE_WAIVER_SCHEMA!r}",
            path=str(source),
        )
    for field in ("approved_by", "reason", "expires"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise EvalError(
                f"waiver {source} requires a non-empty {field!r} field", path=str(source)
            )
    checks = payload.get("checks")
    if (
        not isinstance(checks, list)
        or not checks
        or not all(isinstance(item, str) and item for item in checks)
    ):
        raise EvalError(
            f"waiver {source} requires a non-empty checks list of gate ids", path=str(source)
        )
    expires = payload["expires"].strip()
    if expires != "pr":
        try:
            day = date.fromisoformat(expires)
        except ValueError as exc:
            raise EvalError(
                f"waiver {source} expires must be an ISO date or 'pr', got {expires!r}",
                path=str(source),
            ) from exc
        if day < date.today():
            payload["expired"] = True
    payload["expired"] = bool(payload.get("expired")) or expires == ""
    payload["path"] = str(source)
    return payload


def compare_against_baseline(
    candidate: EvalReport | dict[str, Any] | str | Path,
    baseline: EvalReport | dict[str, Any] | str | Path,
    *,
    policy: dict[str, Any] | str | Path | None = None,
    waivers: Sequence[str | Path] | None = None,
    reports_dir: str | Path | None = None,
    persist: bool = True,
    command: str | None = None,
    bootstrap_resamples: int = 2000,
    seed: int = 1234,
) -> EvalReport:
    """Compare a candidate report against a committed baseline and apply contribution gates."""
    candidate_data = _coerce_report(candidate, "candidate")
    if isinstance(baseline, (str, Path)):
        baseline_data, baseline_path = load_baseline(baseline)
    else:
        baseline_data = _coerce_report(baseline, "baseline")
        baseline_path = Path("baseline/report.json")
    if isinstance(policy, dict):
        policy_data = policy
        policy_path = Path("inline-policy.json")
    elif isinstance(policy, (str, Path)):
        policy_data, policy_path = load_policy(policy)
    else:
        policy_data, policy_path = load_policy()
    level = str(candidate_data.get("level", ""))
    if level not in ("node", "routing", "e2e"):
        raise EvalError(f"cannot gate a {level!r} report; expected a node, routing, or e2e report")
    level_policy = (policy_data.get("levels") or {}).get(level)
    if not isinstance(level_policy, dict):
        raise EvalError(f"gate policy has no level configuration for {level!r}")
    baseline_path_text = str(baseline_path)
    if command is None:
        command = f"mg eval compare --baseline {baseline_path_text} --candidate <candidate-report>"
    base = compare_reports(
        candidate_data,
        baseline_data,
        reports_dir=None,
        persist=False,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
        command=command,
    )
    candidate_cases = {str(case.get("case_id")): case for case in candidate_data["cases"]}
    baseline_cases = {str(case.get("case_id")): case for case in baseline_data["cases"]}
    common = sorted(set(candidate_cases) & set(baseline_cases))
    checks = _metric_checks(
        level,
        level_policy,
        candidate_data,
        baseline_data,
        candidate_cases,
        baseline_cases,
        common,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    hard_gates = _hard_gates(candidate_data, baseline_data, policy_data)
    waived, waiver_records = _apply_waivers(
        checks,
        hard_gates,
        candidate_data,
        waivers or [],
    )
    checks = waived
    soft_failures = [check["id"] for check in checks if check["status"] == "fail"]
    hard_failures = [
        name
        for name in HARD_GATE_NAMES
        if isinstance(hard_gates.get(name), dict) and hard_gates[name]["status"] == "fail"
    ]
    passed = not soft_failures and not hard_failures
    gate = {
        "passed": passed,
        "level": level,
        "policy": {
            "path": str(policy_path),
            "hash": sha256_text(canonical_json(policy_data)),
            "schema": policy_data.get("schema"),
            "source": policy_data.get("source"),
            "thresholds": level_policy,
        },
        "checks": checks,
        "hard_gates": hard_gates,
        "hard_failures": hard_failures,
        "failures": [*soft_failures, *[f"hard.{name}" for name in hard_failures]],
        "waived": [check["id"] for check in checks if check["status"] == "waived"],
        "waivers": waiver_records,
    }
    primary_metric = str(level_policy.get("primary") or candidate_data["primary_metric"]["name"])
    primary = next((check for check in checks if check["metric"] == primary_metric), None)
    for check in checks:
        report_key = check["metric"]
        base.metrics[f"{report_key}_delta"] = check["delta"]
        base.metrics[f"{report_key}_delta_ci95"] = check["ci95"]
    if primary is None:
        raise EvalError(f"gate policy level {level!r} has no primary metric {primary_metric!r}")
    base.primary_metric = {
        "name": f"{primary_metric}_delta",
        "value": primary["delta"],
        "ci95": primary["ci95"],
        "threshold": None,
        "direction": primary["direction"],
        "passed": primary["status"] in ("pass", "waived"),
    }
    base.comparisons = [
        {
            "name": check["id"],
            "baseline": check["baseline"],
            "candidate": check["candidate"],
            "delta": check["delta"],
            "improvement": check["improvement"],
            "ci95": check["ci95"],
            "direction": check["direction"],
            "tolerance": check.get("tolerance"),
            "relative_tolerance": check.get("relative_tolerance"),
            "passed": check["status"] in ("pass", "waived"),
            "status": check["status"],
        }
        for check in checks
    ]
    base.gate = gate
    base.thresholds = [
        {
            "metric": check["metric"],
            "threshold": check.get("tolerance"),
            "value": check["delta"],
            "direction": check["direction"],
            "passed": check["status"] in ("pass", "waived"),
        }
        for check in checks
    ]
    limitations = list(base.limitations)
    if not common:
        limitations.append("No case ids were shared; the gate is based on aggregate metrics only.")
    limitations.append(
        "Gate decisions use point estimates against the reviewed policy; confidence intervals "
        "are reported but are not a second significance test."
    )
    not_measured = [
        name
        for name in HARD_GATE_NAMES
        if isinstance(hard_gates.get(name), dict) and hard_gates[name]["status"] == "not_measured"
    ]
    if not_measured:
        limitations.append(
            "Hard gates not measured by this suite: " + ", ".join(not_measured) + "."
        )
    if gate["waived"]:
        limitations.append("Waived checks are listed in the report; hard gates are never waived.")
    base.limitations = limitations
    cast(Any, base).__class__ = GateComparisonReport
    if persist:
        base.write(reports_dir)
    return base


def gate_summary(report: EvalReport) -> list[str]:
    """Return compact human-readable lines for the CLI."""
    gate = report.gate or {}
    hard = gate.get("hard_gates") or {}
    lines = [f"gate:     {'PASS' if gate.get('passed') else 'FAIL'}"]
    policy = gate.get("policy") or {}
    if policy:
        lines.append(f"policy:   {policy.get('path')} ({policy.get('hash')})")
    for check in gate.get("checks") or []:
        lines.append(
            f"check:    {check.get('id')} {check.get('status')} "
            f"delta={_fmt_signed(check.get('delta'))} "
            f"ci95={_fmt_ci(check.get('ci95'))} "
            f"tolerance={_fmt_tolerance(check)}"
        )
    for name in HARD_GATE_NAMES:
        entry = hard.get(name)
        if isinstance(entry, dict):
            lines.append(f"hard:     {name} {entry.get('status')} - {entry.get('detail')}")
    for waiver in gate.get("waivers") or []:
        lines.append(
            f"waiver:   {waiver.get('path')} by {waiver.get('approved_by')} "
            f"(expires {waiver.get('expires')}) applied={waiver.get('applied')} "
            f"rejected={waiver.get('rejected')}"
        )
    if report.report_dir is not None:
        lines.append(f"report:   {report.report_dir}")
    return lines


def _metric_checks(
    level: str,
    level_policy: dict[str, Any],
    candidate_data: dict[str, Any],
    baseline_data: dict[str, Any],
    candidate_cases: dict[str, dict[str, Any]],
    baseline_cases: dict[str, dict[str, Any]],
    common: list[str],
    *,
    bootstrap_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rules = level_policy.get("metrics")
    if not isinstance(rules, dict) or not rules:
        raise EvalError(f"gate policy for {level!r} has no metrics")
    checks: list[dict[str, Any]] = []
    for metric in sorted(rules):
        rule = rules[metric]
        if not isinstance(rule, dict):
            raise EvalError(f"gate rule {level}.{metric} must be an object")
        direction = str(rule.get("direction", "max"))
        if direction not in ("min", "max"):
            raise EvalError(f"gate rule {level}.{metric} has invalid direction {direction!r}")
        baseline_value = _optional_float(baseline_data.get("metrics", {}).get(metric))
        candidate_value = _optional_float(candidate_data.get("metrics", {}).get(metric))
        deltas: list[float] = []
        for case_id in common:
            candidate_value_case = _case_metric_value(
                level, metric, candidate_data, candidate_cases[case_id]
            )
            baseline_value_case = _case_metric_value(
                level, metric, baseline_data, baseline_cases[case_id]
            )
            if candidate_value_case is None or baseline_value_case is None:
                continue
            deltas.append(candidate_value_case - baseline_value_case)
        delta = (
            candidate_value - baseline_value
            if candidate_value is not None and baseline_value is not None
            else None
        )
        ci = bootstrap_ci(deltas, resamples=bootstrap_resamples, seed=seed) if deltas else None
        tolerance = _optional_float(rule.get("tolerance")) or 0.0
        relative_tolerance = _optional_float(rule.get("relative_tolerance"))
        absolute_tolerance = _optional_float(rule.get("absolute_tolerance")) or 0.0
        passed: bool | None
        detail: str
        if delta is None:
            # A metric listed in the reviewed policy is a required measurement. Treating a
            # missing value as "not measured" would let a candidate drop a gate metric and
            # pass, so a gate metric that cannot be read is a failure, not a silent pass.
            passed = False
            status = "fail"
            detail = "gate metric is absent or null in one or both reports"
        else:
            if direction == "max":
                allowance = tolerance + (
                    abs(baseline_value) * relative_tolerance
                    if relative_tolerance is not None and baseline_value is not None
                    else 0.0
                )
                passed = delta >= -allowance
            else:
                allowance = (
                    abs(baseline_value) * relative_tolerance
                    if relative_tolerance is not None and baseline_value is not None
                    else 0.0
                ) + absolute_tolerance
                if relative_tolerance is None and rule.get("tolerance") is not None:
                    allowance += tolerance
                passed = delta <= allowance
            status = "pass" if passed else "fail"
            detail = f"delta {_fmt_signed(delta)} vs tolerance {_fmt_tolerance(rule)}"
        improvement = delta if delta is None or direction == "max" else -delta
        checks.append(
            {
                "id": f"{level}.{metric}",
                "metric": metric,
                "level": level,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta,
                "improvement": improvement,
                "ci95": ci,
                "direction": direction,
                "tolerance": tolerance if rule.get("tolerance") is not None else None,
                "relative_tolerance": relative_tolerance,
                "absolute_tolerance": absolute_tolerance
                if rule.get("absolute_tolerance")
                else None,
                "paired_cases": len(deltas),
                "passed": passed,
                "status": status,
                "detail": detail,
            }
        )
    return checks


def _case_metric_value(
    level: str,
    metric: str,
    report: dict[str, Any],
    case: dict[str, Any],
) -> float | None:
    # Oracle agreement excludes ambiguous labels, matching the routing harness. All other
    # metrics delegate to the paired-comparison extractor so the two paths cannot drift.
    if level == "routing" and metric == "oracle_agreement":
        oracle = case.get("oracle")
        if isinstance(oracle, dict) and oracle.get("ambiguous"):
            return None
        if case.get("correct") is not None:
            return 1.0 if case.get("correct") else 0.0
        return None
    return _case_value({"primary_metric": {"name": metric}}, case)


def _hard_gates(
    candidate_data: dict[str, Any],
    baseline_data: dict[str, Any],
    policy_data: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    config = policy_data.get("hard_gates") or {}
    validity_config = config.get("validity") or {}
    safety_config = config.get("safety") or {}
    secret_config = config.get("secret_scan") or {}
    hard: dict[str, dict[str, Any]] = {}

    report_issues = validate_report(candidate_data)
    hashes_ok = _hashes_ok(candidate_data)
    schema_passed = not report_issues and hashes_ok
    hard["schema"] = {
        "passed": schema_passed,
        "status": "pass" if schema_passed else "fail",
        "detail": (
            "candidate report validates and suite/graph hashes are present"
            if schema_passed
            else (report_issues[0] if report_issues else "missing suite or graph hash")
        ),
    }

    validity_value = _optional_float(candidate_data.get("metrics", {}).get("validity"))
    validity_min = _optional_float(validity_config.get("min"))
    validity_baseline = _optional_float(baseline_data.get("metrics", {}).get("validity"))
    if validity_value is None:
        hard["validity"] = {
            "passed": True,
            "status": "not_measured",
            "detail": "this suite does not report validity",
            "value": None,
        }
    else:
        problems: list[str] = []
        if validity_min is not None and validity_value + 1e-12 < validity_min:
            problems.append(f"validity {validity_value:.4f} < minimum {validity_min:.4f}")
        if (
            validity_config.get("compare_to_baseline", True)
            and validity_baseline is not None
            and validity_value + 1e-12 < validity_baseline
        ):
            problems.append(
                f"validity regressed from {validity_baseline:.4f} to {validity_value:.4f}"
            )
        hard["validity"] = {
            "passed": not problems,
            "status": "fail" if problems else "pass",
            "detail": "; ".join(problems) if problems else f"validity {validity_value:.4f}",
            "value": validity_value,
        }

    safety_value, safety_detail = _safety_value(candidate_data)
    if safety_value is None:
        hard["safety"] = {
            "passed": True,
            "status": "not_measured",
            "detail": f"this suite does not report safety violations ({safety_detail})",
            "value": None,
        }
    else:
        maximum = _optional_float(safety_config.get("max_violations")) or 0.0
        hard["safety"] = {
            "passed": safety_value <= maximum,
            "status": "pass" if safety_value <= maximum else "fail",
            "detail": f"{_fmt(safety_value)} safety violations (maximum {_fmt(maximum)})",
            "value": safety_value,
        }

    stopped = candidate_data.get("budget", {}).get("stopped") is True
    gate_budget = candidate_data.get("gate", {}).get("budget_stopped")
    budget_problem = stopped or gate_budget is not None
    if not budget_problem:
        budget_problem = any(_case_budget_stopped(case) for case in candidate_data.get("cases", []))
    hard["budget"] = {
        "passed": not budget_problem,
        "status": "fail" if budget_problem else "pass",
        "detail": (
            "a run exceeded or stopped on its declared budget"
            if budget_problem
            else "no run stopped on its declared budget"
        ),
    }

    matches = _secret_matches(candidate_data) if secret_config.get("enabled", True) else []
    hard["secret_scan"] = {
        "passed": not matches,
        "status": "fail" if matches else "pass",
        "detail": (
            "possible secrets: " + ", ".join(sorted(set(matches)))
            if matches
            else "no credential-shaped strings found in the candidate report"
        ),
    }
    return hard


def _apply_waivers(
    checks: list[dict[str, Any]],
    hard_gates: dict[str, dict[str, Any]],
    candidate_data: dict[str, Any],
    waiver_paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    waivers = [load_waiver(path) for path in waiver_paths]
    hard_ids = {f"hard.{name}" for name in HARD_GATE_NAMES}
    for waiver in waivers:
        applied: list[str] = []
        rejected: list[str] = []
        suite_hash = candidate_data.get("suite", {}).get("hash")
        graph_hash = (candidate_data.get("graph") or {}).get("hash")
        scoped_suite = waiver.get("suite_hash")
        scoped_graph = waiver.get("graph_hash")
        scope_error: str | None = None
        if scoped_suite is not None and scoped_suite != suite_hash:
            scope_error = "waiver suite_hash does not match the candidate suite"
        if scoped_graph is not None and scoped_graph != graph_hash:
            scope_error = "waiver graph_hash does not match the candidate graph"
        if waiver.get("expired"):
            scope_error = "waiver is expired"
        for check_id in waiver["checks"]:
            if check_id in hard_ids or check_id in HARD_GATE_NAMES:
                rejected.append(check_id)
                continue
            if scope_error is not None:
                rejected.append(check_id)
                continue
            matching = next((check for check in checks if check["id"] == check_id), None)
            if matching is None:
                rejected.append(check_id)
                continue
            if matching["status"] != "fail":
                rejected.append(check_id)
                continue
            matching["status"] = "waived"
            matching["waived"] = True
            matching["waiver"] = {
                "path": waiver["path"],
                "approved_by": waiver["approved_by"],
                "reason": waiver["reason"],
                "expires": waiver["expires"],
            }
            matching["passed"] = True
            applied.append(check_id)
        records.append(
            {
                "path": waiver["path"],
                "schema": waiver.get("schema"),
                "approved_by": waiver["approved_by"],
                "reason": waiver["reason"],
                "expires": waiver["expires"],
                "checks": waiver["checks"],
                "applied": applied,
                "rejected": rejected,
                "scope_error": scope_error,
            }
        )
    return checks, records


def _default_policy_path() -> Path:
    override = os.environ.get("MG_GATE_POLICY")
    if override:
        return Path(override).expanduser()
    repo_default = Path(__file__).resolve().parents[3] / DEFAULT_POLICY_PATH
    if repo_default.is_file():
        return repo_default
    cwd_default = Path.cwd() / DEFAULT_POLICY_PATH
    if cwd_default.is_file():
        return cwd_default
    return repo_default


def _report_path(path: str | Path, label: str) -> Path:
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "report.json"
    if not source.is_file():
        raise EvalError(f"{label} report not found: {source}", path=str(source))
    return source


def _clean_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Drop CLI output locators that are outside the report schema.

    ``mg eval ... --json`` adds ``report_dir`` and ``manifest`` so a human can find the files;
    a saved stdout capture is still a report, so comparisons accept it.
    """
    cleaned = dict(data)
    cleaned.pop("report_dir", None)
    cleaned.pop("manifest", None)
    return cleaned


def _coerce_report(
    value: EvalReport | dict[str, Any] | str | Path,
    label: str,
) -> dict[str, Any]:
    if isinstance(value, EvalReport):
        return value.to_dict()
    data = value if isinstance(value, dict) else load_report(_report_path(value, label))
    data = _clean_payload(data)
    issues = validate_report(data)
    if issues:
        raise EvalError(
            f"{label} report is invalid: {issues[0]}",
            details={"issues": issues},
        )
    return data


def _hashes_ok(report: dict[str, Any]) -> bool:
    suite_hash = report.get("suite", {}).get("hash")
    graph_hash = (report.get("graph") or {}).get("hash")
    pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
    return bool(
        isinstance(suite_hash, str)
        and pattern.match(suite_hash)
        and isinstance(graph_hash, str)
        and pattern.match(graph_hash)
    )


def _safety_value(report: dict[str, Any]) -> tuple[float | None, str]:
    metrics = report.get("metrics") or {}
    for key in ("safety_violations", "safety_violation_rate"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value), f"metrics.{key}"
    total = 0.0
    found = False
    for case in report.get("cases", []):
        value = case.get("safety_violations")
        if isinstance(value, list):
            total += len(value)
            found = True
        elif isinstance(value, (int, float)):
            total += float(value)
            found = True
    if found:
        return total, "case safety_violations"
    return None, "no safety metric or safety_violations field"


def _case_budget_stopped(case: dict[str, Any]) -> bool:
    outcome = str(case.get("outcome") or "").lower()
    if "budget" in outcome:
        return True
    error = case.get("error")
    if isinstance(error, dict):
        text = f"{error.get('code', '')} {error.get('message', '')}".lower()
        if "budget" in text:
            return True
    return False


def _secret_matches(report: dict[str, Any]) -> list[str]:
    text = canonical_json(report)
    return [label for label, pattern in SECRET_PATTERNS if pattern.search(text)]


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_signed(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:+.4f}"
    return "—"


def _fmt_ci(ci: Any) -> str:
    if isinstance(ci, list) and len(ci) == 2:
        return f"[{_fmt_signed(ci[0])}, {_fmt_signed(ci[1])}]"
    return "—"


def _fmt_tolerance(rule: dict[str, Any]) -> str:
    if rule.get("relative_tolerance") is not None:
        parts = [f"{float(rule['relative_tolerance']):.0%} relative"]
        if rule.get("absolute_tolerance"):
            parts.append(f"${float(rule['absolute_tolerance']):.6f} absolute")
        return " + ".join(parts)
    tolerance = _optional_float(rule.get("tolerance"))
    return f"{tolerance:.4f}" if tolerance is not None else "—"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
