"""Machine-readable and human-readable evaluation reports.

JSON is the contract; Markdown is a rendering of the same report for review. Reports always
reference the suite hash, graph hash, exact commands, environment, and the eval run manifest
written next to them.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from modelgraph import __version__
from modelgraph.evaluation.errors import SuiteError
from modelgraph.evaluation.metrics import threshold_result
from modelgraph.paths import runs_dir as default_runs_dir
from modelgraph.util import canonical_json, new_id, sha256_text, utc_now

REPORT_SCHEMA = "modelgraph/eval-report/v0"
MANIFEST_SCHEMA = "modelgraph/eval-manifest/v0"
MANIFEST_FILENAME = "manifest.json"


@dataclass
class EvalReport:
    report_id: str
    level: str
    suite: dict[str, Any]
    graph: dict[str, Any] | None
    target: dict[str, Any]
    provider: dict[str, Any]
    command: str
    commands: list[str]
    reproduce: str
    environment: dict[str, Any]
    selection: dict[str, Any]
    budget: dict[str, Any]
    metrics: dict[str, Any]
    primary_metric: dict[str, Any]
    thresholds: list[dict[str, Any]]
    cases: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    slices: dict[str, dict[str, Any]]
    protocol_errors: dict[str, Any]
    costs: dict[str, Any]
    latency: dict[str, Any]
    usage: dict[str, Any]
    determinism: dict[str, Any]
    limitations: list[str]
    claim: str | None = None
    gate: dict[str, Any] | None = None
    oracle: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    confusion: dict[str, Any] | None = None
    abstain: dict[str, Any] | None = None
    judge: dict[str, Any] | None = None
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    baseline: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    e2e: dict[str, Any] | None = None
    generated_at: str = field(default_factory=utc_now)
    schema: str = REPORT_SCHEMA
    report_dir: Path | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema": self.schema,
            "report_id": self.report_id,
            "level": self.level,
            "claim": self.claim,
            "generated_at": self.generated_at,
            "suite": self.suite,
            "graph": self.graph,
            "target": self.target,
            "provider": self.provider,
            "command": self.command,
            "commands": self.commands,
            "reproduce": self.reproduce,
            "run_manifest": MANIFEST_FILENAME,
            "environment": self.environment,
            "selection": self.selection,
            "budget": self.budget,
            "metrics": self.metrics,
            "primary_metric": self.primary_metric,
            "thresholds": self.thresholds,
            "gate": self.gate,
            "cases": self.cases,
            "failures": self.failures,
            "slices": self.slices,
            "protocol_errors": self.protocol_errors,
            "costs": self.costs,
            "latency": self.latency,
            "usage": self.usage,
            "determinism": self.determinism,
            "limitations": self.limitations,
            "oracle": self.oracle,
            "calibration": self.calibration,
            "confusion": self.confusion,
            "abstain": self.abstain,
            "judge": self.judge,
            "comparisons": self.comparisons,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "e2e": self.e2e,
        }
        return data

    def manifest(self) -> dict[str, Any]:
        files = {
            "report": "report.json",
            "markdown": "report.md",
            "cases": "cases/",
        }
        if self.e2e is not None:
            files["e2e"] = "e2e.json"
            oracle = self.e2e.get("oracle") or {}
            if oracle.get("labels_file"):
                files["oracle_labels"] = str(oracle["labels_file"])
        return {
            "schema": MANIFEST_SCHEMA,
            "report_id": self.report_id,
            "created_at": self.generated_at,
            "level": self.level,
            "claim": self.claim,
            "suite": self.suite,
            "graph": self.graph,
            "target": self.target,
            "provider": self.provider,
            "command": self.command,
            "commands": self.commands,
            "reproduce": self.reproduce,
            "environment": self.environment,
            "selection": self.selection,
            "budget": self.budget,
            "usage": self.usage,
            "gate": self.gate,
            "determinism": self.determinism,
            "oracle": self.oracle,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "e2e": self.e2e,
            "files": files,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def validate(self) -> list[str]:
        return validate_report(self.to_dict())

    def write(self, reports_dir: str | Path | None = None) -> Path:
        return write_report(self, reports_dir=reports_dir)

    def to_markdown(self) -> str:
        lines: list[str] = []
        suite = self.suite
        lines.append(f"# Evaluation report — {suite.get('id', '?')} @ {suite.get('version', '?')}")
        lines.append("")
        if self.claim:
            lines.append(f"**Claim:** {self.claim}")
            lines.append("")
        lines.append(f"- **Level:** `{self.level}`")
        lines.append(f"- **Report id:** `{self.report_id}`")
        lines.append(f"- **Generated:** {self.generated_at}")
        lines.append(f"- **Suite hash:** `{suite.get('hash', '?')}`")
        if self.graph:
            lines.append(
                f"- **Graph:** `{self.graph.get('id', '?')}@{self.graph.get('version', '?')}` "
                f"(`{self.graph.get('hash', '?')}`)"
            )
        lines.append(f"- **Command:** `{self.command}`")
        lines.append(f"- **Reproduce (current API):** `{self.reproduce}`")
        lines.append(f"- **Run manifest:** `{MANIFEST_FILENAME}`")
        lines.append(f"- **Determinism fingerprint:** `{self.determinism.get('fingerprint', '?')}`")
        lines.append("")

        if self.gate is not None:
            status = "PASS" if self.gate.get("passed") else "FAIL"
            lines.append(f"## Gate: {status}")
            lines.append("")
            if self.gate.get("failures"):
                lines.append(f"Failed thresholds: {', '.join(self.gate['failures'])}")
                lines.append("")

        lines.append("## Primary metric")
        lines.append("")
        primary = self.primary_metric
        lines.append(
            f"`{primary.get('name')}` = **{_fmt(primary.get('value'))}**"
            + _ci_suffix(primary.get("ci95"))
            + (
                f" (threshold {_fmt(primary.get('threshold'))}, "
                f"{'pass' if primary.get('passed') else 'fail'})"
                if primary.get("threshold") is not None
                else ""
            )
        )
        lines.append("")
        lines.append("## Metrics")
        lines.append("")
        lines.extend(_metric_table(self.metrics))
        lines.append("")

        lines.append("## Cost, latency, and usage")
        lines.append("")
        if self.level == "comparison":
            lines.append(
                f"- Paired cases: {self.selection.get('paired_cases', 0)} / "
                f"{self.selection.get('total_cases', 0)} candidate cases"
            )
            lines.append(
                f"- Cost/case: baseline ${_fmt(self.costs.get('baseline_per_task_usd'), 6)}, "
                f"candidate ${_fmt(self.costs.get('candidate_per_task_usd'), 6)}, "
                f"delta ${_fmt(self.costs.get('delta_per_task_usd'), 6)}"
                + _ci_suffix(self.costs.get("delta_ci95"))
            )
            lines.append(
                f"- Latency p95: baseline "
                f"{_fmt(self.latency.get('baseline_p95_seconds'), 3)}s, candidate "
                f"{_fmt(self.latency.get('candidate_p95_seconds'), 3)}s, "
                f"paired delta p50 {_fmt(self.latency.get('delta_p50_seconds'), 3)}s"
            )
        else:
            lines.append(
                f"- Cases run: {self.usage.get('cases', 0)} / selected "
                f"{self.selection.get('selected_cases', 0)}"
            )
            lines.append(
                f"- Cost: **${_fmt(self.costs.get('total_usd'), 6)}** total, "
                f"${_fmt(self.costs.get('per_task_usd'), 6)} per case"
                + _ci_suffix(self.costs.get("per_task_usd_ci95"))
            )
            lines.append(
                f"- Tokens: input={self.usage.get('input_tokens', 0)}, "
                f"output={self.usage.get('output_tokens', 0)}, "
                f"total={self.usage.get('total_tokens', 0)}"
            )
            lines.append(
                f"- Latency (wall): mean {_fmt(self.latency.get('mean_seconds'), 3)}s, "
                f"p50 {_fmt(self.latency.get('p50_seconds'), 3)}s, "
                f"p95 {_fmt(self.latency.get('p95_seconds'), 3)}s"
            )
            if self.costs.get("note"):
                lines.append(f"- Cost note: {self.costs['note']}")
        lines.append("")

        if self.slices:
            lines.append("## Slices")
            lines.append("")
            if self.level == "comparison":
                lines.append(
                    "| slice | cases | baseline | candidate | delta | 95% CI | "
                    "W/L/T | cost delta/case | cost 95% CI | latency p50 delta | "
                    "latency 95% CI |"
                )
                lines.append(
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
                )
                for name in sorted(self.slices):
                    entry = self.slices[name]
                    lines.append(
                        f"| {_escape(name)} | {entry.get('cases', 0)} | "
                        f"{_fmt(entry.get('baseline_primary'))} | "
                        f"{_fmt(entry.get('candidate_primary'))} | "
                        f"{_fmt(entry.get('primary_delta'))} | "
                        f"{_ci_text(entry.get('primary_delta_ci95'))} | "
                        f"{entry.get('wins', 0)}/{entry.get('losses', 0)}/"
                        f"{entry.get('ties', 0)} | "
                        f"${_fmt(entry.get('cost_delta_per_task_usd'), 6)} | "
                        f"{_ci_text(entry.get('cost_delta_ci95'))} | "
                        f"{_fmt(entry.get('latency_delta_p50_seconds'), 3)}s | "
                        f"{_ci_text(entry.get('latency_delta_ci95'))} |"
                    )
            elif self.level == "routing":
                lines.append(
                    "| slice | cases | labeled | oracle agreement | abstain rate | "
                    "cost/case | latency p95 |"
                )
                lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
                for name in sorted(self.slices):
                    entry = self.slices[name]
                    lines.append(
                        f"| {_escape(name)} | {entry.get('cases', 0)} | "
                        f"{entry.get('labeled_cases', 0)} | "
                        f"{_fmt(entry.get('oracle_agreement'))} | "
                        f"{_fmt(entry.get('abstain_rate'))} | "
                        f"${_fmt(entry.get('cost_per_task_usd'), 6)} | "
                        f"{_fmt(entry.get('latency_p95_seconds'), 3)}s |"
                    )
            else:
                if self.level == "e2e":
                    lines.append(
                        "| slice | cases | success | validity | failures | cost/case | "
                        "latency p95 | hops p95 |"
                    )
                    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                    for name in sorted(self.slices):
                        entry = self.slices[name]
                        lines.append(
                            f"| {_escape(name)} | {entry.get('cases', 0)} | "
                            f"{_fmt(entry.get('task_success'))} | "
                            f"{_fmt(entry.get('validity'))} | "
                            f"{entry.get('failure_count', 0)} | "
                            f"${_fmt(entry.get('cost_per_task_usd'), 6)} | "
                            f"{_fmt(entry.get('latency_p95_seconds'), 3)}s | "
                            f"{_fmt(entry.get('hop_count_p95'), 2)} |"
                        )
                else:
                    lines.append("| slice | cases | success | validity | cost/case | latency p95 |")
                    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
                    for name in sorted(self.slices):
                        entry = self.slices[name]
                        lines.append(
                            f"| {_escape(name)} | {entry.get('cases', 0)} | "
                            f"{_fmt(entry.get('task_success'))} | "
                            f"{_fmt(entry.get('validity'))} | "
                            f"${_fmt(entry.get('cost_per_task_usd'), 6)} | "
                            f"{_fmt(entry.get('latency_p95_seconds'), 3)}s |"
                        )
            lines.append("")

        if self.oracle:
            lines.append("## Oracle labels")
            lines.append("")
            labeling = self.oracle.get("labeling", {})
            lines.append(f"- Labeled cases: {self.oracle.get('labeled_cases', 0)}")
            lines.append(f"- Ambiguous cases: {self.oracle.get('ambiguous_cases', 0)}")
            lines.append(f"- Oracle agreement: {_fmt(self.oracle.get('oracle_agreement'))}")
            lines.append(
                f"- Cost-weighted accuracy: {_fmt(self.oracle.get('cost_weighted_accuracy'))}"
            )
            lines.append(f"- Oracle gap: {_fmt(self.oracle.get('oracle_gap'))}")
            lines.append(
                f"- Labeling method: {labeling.get('labeling_method') or 'not documented'}"
            )
            lines.append(
                f"- Labeling cost: ${_fmt(labeling.get('total_usd'), 6)} "
                f"(${_fmt(labeling.get('per_case_usd'), 6)} per case)"
            )
            lines.append("")

        if self.confusion:
            lines.append("## Confusion matrix")
            lines.append("")
            lines.extend(_confusion_table(self.confusion))
            lines.append("")

        if self.abstain:
            lines.append("## Abstain")
            lines.append("")
            lines.append("| metric | value |")
            lines.append("| --- | ---: |")
            for key in (
                "ambiguous",
                "abstained",
                "precision",
                "recall",
                "f1",
                "abstain_rate",
            ):
                lines.append(f"| {key} | {_fmt(self.abstain.get(key))} |")
            lines.append("")

        if self.calibration:
            lines.append("## Calibration")
            lines.append("")
            lines.append(
                f"- ECE: **{_fmt(self.calibration.get('ece'))}** over "
                f"{self.calibration.get('n', 0)} answered decisions"
            )
            lines.append(f"- Max bin gap: {_fmt(self.calibration.get('max_calibration_error'))}")
            lines.append("")
            lines.append("| bin | n | confidence | accuracy | gap |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for entry in self.calibration.get("bins", []):
                lines.append(
                    f"| [{_fmt(entry.get('lower'), 2)}, {_fmt(entry.get('upper'), 2)}) | "
                    f"{entry.get('count', 0)} | {_fmt(entry.get('confidence'))} | "
                    f"{_fmt(entry.get('accuracy'))} | {_fmt(entry.get('gap'))} |"
                )
            lines.append("")
            if self.calibration.get("sweep"):
                lines.append("### Threshold sweep")
                lines.append("")
                lines.append("| threshold | coverage | selective accuracy | abstain rate |")
                lines.append("| ---: | ---: | ---: | ---: |")
                for entry in self.calibration["sweep"]:
                    lines.append(
                        f"| {_fmt(entry.get('threshold'), 2)} | "
                        f"{_fmt(entry.get('coverage'))} | "
                        f"{_fmt(entry.get('selective_accuracy'))} | "
                        f"{_fmt(entry.get('abstain_rate'))} |"
                    )
                lines.append("")

        if self.judge:
            lines.append("## Judge")
            lines.append("")
            lines.append(f"- Configured: {self.judge.get('configured')}")
            lines.append(f"- Model: {self.judge.get('model') or 'not set'}")
            lines.append(f"- Calls: {self.judge.get('calls', 0)}")
            if self.judge.get("error"):
                lines.append(f"- Error: {self.judge['error']}")
            pair = self.judge.get("pair")
            if pair:
                lines.append(
                    f"- Blinded paired cases: {pair.get('cases', 0)} "
                    f"(order swapped: {pair.get('order_swapped')})"
                )
                lines.append(
                    f"- Paired judge delta: {_fmt(pair.get('delta'))}"
                    f"{_ci_suffix(pair.get('delta_ci95'))}"
                )
            calibration = self.judge.get("calibration")
            if calibration:
                lines.append(
                    f"- Human calibration: {calibration.get('n', 0)} labels, "
                    f"MAE {_fmt(calibration.get('mean_absolute_error'))}"
                )
            lines.append("")

        if self.e2e:
            lines.extend(self._e2e_markdown())

        if self.comparisons or self.baseline:
            lines.append("## Paired comparison")
            lines.append("")
            if self.baseline:
                lines.append(
                    f"- Baseline: `{self.baseline.get('report_id')}` "
                    f"(`{self.baseline.get('report_hash')}`)"
                )
            if self.candidate:
                lines.append(
                    f"- Candidate: `{self.candidate.get('report_id')}` "
                    f"(`{self.candidate.get('report_hash')}`)"
                )
            for comparison in self.comparisons:
                label = str(comparison.get("name", "comparison"))
                lines.append(
                    f"- {label.title()}: delta "
                    f"{_fmt(comparison.get('delta'))}{_ci_suffix(comparison.get('ci95'))}"
                )
            lines.append("")

        lines.append("## Failures")
        lines.append("")
        if self.failures:
            lines.append("| case | reason |")
            lines.append("| --- | --- |")
            for failure in self.failures:
                lines.append(
                    f"| {_escape(str(failure.get('case_id')))} | "
                    f"{_escape(str(failure.get('reason')))} |"
                )
        else:
            lines.append("No failing cases.")
        lines.append("")

        lines.append("## Limitations")
        lines.append("")
        for limitation in self.limitations:
            lines.append(f"- {limitation}")
        lines.append("")

        lines.append("## Reproducibility")
        lines.append("")
        lines.append(f"- Suite: `{self.suite.get('path')}`")
        lines.append(f"- Command: `{self.command}`")
        lines.append(f"- Manifest: `{MANIFEST_FILENAME}`")
        lines.append(f"- Environment: {json.dumps(self.environment, sort_keys=True)}")
        lines.append(
            "- Determinism excludes wall-clock latency and run identifiers; the fingerprint "
            "covers scripted quality, validity, cost, token, and artifact outcomes."
        )
        lines.append("")
        return "\n".join(lines)

    def _e2e_markdown(self) -> list[str]:
        assert self.e2e is not None
        data = self.e2e
        lines: list[str] = ["## E2E baselines", ""]
        baselines = data.get("baselines") or {}
        if baselines:
            lines.append("| baseline | task success | cost/case | latency p95 | validity | cases |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for name in sorted(baselines):
                entry = baselines[name]
                if not isinstance(entry, dict) or not entry.get("available", True):
                    reason = entry.get("error") if isinstance(entry, dict) else None
                    lines.append(
                        f"| {_escape(name)} | — | — | — | — | {_escape(str(reason or ''))} |"
                    )
                    continue
                lines.append(
                    f"| {_escape(name)} | {_fmt(entry.get('task_success'))} | "
                    f"${_fmt(entry.get('cost_per_task_usd'), 6)} | "
                    f"{_fmt(entry.get('latency_p95_seconds'), 3)}s | "
                    f"{_fmt(entry.get('validity'))} | {entry.get('cases', 0)} |"
                )
        else:
            lines.append("No baselines were requested.")
        lines.append("")
        notes = data.get("baseline_notes")
        if notes:
            lines.append(str(notes))
            lines.append("")

        pareto = data.get("pareto") or {}
        configurations = pareto.get("configurations") or []
        if configurations:
            lines.append("## Pareto frontier")
            lines.append("")
            lines.append("| configuration | kind | quality | cost/case | latency p95 | frontier |")
            lines.append("| --- | --- | ---: | ---: | ---: | :---: |")
            for entry in configurations:
                lines.append(
                    f"| {_escape(str(entry.get('name')))} | {_escape(str(entry.get('kind')))} | "
                    f"{_fmt(entry.get('quality'))} | "
                    f"${_fmt(entry.get('cost_per_task_usd'), 6)} | "
                    f"{_fmt(entry.get('latency_p95_seconds'), 3)}s | "
                    f"{'yes' if entry.get('frontier') else ''} |"
                )
            lines.append("")
            frontier = pareto.get("frontier") or []
            lines.append(
                "Frontier: " + (", ".join(f"`{name}`" for name in frontier) if frontier else "none")
            )
            lines.append("")
            bands = pareto.get("bands") or []
            if bands:
                lines.append("### Cost bands")
                lines.append("")
                lines.append("| cost ceiling | winner | quality | cost/case |")
                lines.append("| ---: | --- | ---: | ---: |")
                for band in bands:
                    lines.append(
                        f"| ${_fmt(band.get('max_cost_usd'), 6)} | "
                        f"{_escape(str(band.get('winner')))} | "
                        f"{_fmt(band.get('quality'))} | "
                        f"${_fmt(band.get('cost_per_task_usd'), 6)} |"
                    )
                lines.append("")
            if pareto.get("notes"):
                lines.append(str(pareto["notes"]))
                lines.append("")

        failure_slices = data.get("failure_slices") or []
        if failure_slices:
            lines.append("## Worst slices")
            lines.append("")
            lines.append("| slice | cases | failures | task success |")
            lines.append("| --- | ---: | ---: | ---: |")
            for entry in failure_slices:
                lines.append(
                    f"| {_escape(str(entry.get('name')))} | {entry.get('cases', 0)} | "
                    f"{entry.get('failure_count', 0)} | "
                    f"{_fmt(entry.get('task_success'))} |"
                )
            lines.append("")

        run_manifests = data.get("run_manifests") or {}
        if run_manifests:
            lines.append("## Run manifests")
            lines.append("")
            lines.append("| case | manifest | hash |")
            lines.append("| --- | --- | --- |")
            for case_id in sorted(run_manifests):
                entry = run_manifests[case_id]
                lines.append(
                    f"| {_escape(case_id)} | `{_escape(str(entry.get('path')))}` | "
                    f"`{_escape(str(entry.get('hash')))}` |"
                )
            lines.append("")
        return lines


@cache
def report_schema() -> dict[str, Any]:
    path = Path(__file__).with_name("eval-report.schema.json")
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - packaging bug
        raise SuiteError(f"cannot read eval report schema {path}: {exc}") from exc


@cache
def _report_validator() -> Draft202012Validator:
    schema = report_schema()
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:  # pragma: no cover - repo bug
        raise SuiteError(f"invalid eval report schema: {exc.message}") from exc
    return Draft202012Validator(schema)


def validate_report(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for error in sorted(
        _report_validator().iter_errors(payload),
        key=lambda item: [str(part) for part in item.path],
    ):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        issues.append(f"{location}: {error.message}")
    return issues


def write_report(report: EvalReport, *, reports_dir: str | Path | None = None) -> Path:
    issues = report.validate()
    if issues:
        raise SuiteError(
            f"refusing to write invalid eval report: {issues[0]}"
            + (f" (+{len(issues) - 1} more)" if len(issues) > 1 else ""),
            details={"issues": issues},
        )
    base = Path(reports_dir) if reports_dir else default_report_dir()
    directory = base / report.report_id
    directory.mkdir(parents=True, exist_ok=True)
    cases_dir = directory / "cases"
    cases_dir.mkdir(exist_ok=True)
    for case in report.cases:
        case_id = _safe_name(str(case.get("case_id") or case.get("id") or "case"))
        (cases_dir / f"{case_id}.json").write_text(
            json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(report.manifest(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    report.report_dir = directory
    return directory


def default_report_dir() -> Path:
    return default_runs_dir() / "evals"


def build_primary_metric(
    metric: str,
    metrics: dict[str, Any],
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    value = metrics.get(metric)
    threshold = thresholds.get(metric)
    result = (
        threshold_result(metric, threshold, value)
        if threshold is not None
        else {
            "metric": metric,
            "threshold": None,
            "value": value,
            "direction": threshold_result(metric, 0.0, value)["direction"],
            "passed": None,
        }
    )
    primary = {
        "name": metric,
        "value": value,
        "ci95": metrics.get(f"{metric}_ci95"),
        "threshold": threshold,
        "direction": result["direction"],
        "passed": result["passed"],
    }
    threshold_results = [
        threshold_result(name, bound, metrics.get(name))
        for name, bound in sorted(thresholds.items())
    ]
    failures = [result["metric"] for result in threshold_results if result["passed"] is not True]
    gate = {
        "passed": not failures,
        "checks": threshold_results,
        "failures": failures,
    }
    return primary, threshold_results, gate


def environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modelgraph_version": __version__,
        "cwd": str(Path.cwd()),
        "git_commit": _git_commit(),
    }


def determinism_fingerprint(payload: Any) -> str:
    return sha256_text(canonical_json(payload))


def new_report_id() -> str:
    return new_id("eval")


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    value = completed.stdout.strip()
    return value or None


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )


def _metric_table(metrics: dict[str, Any]) -> list[str]:
    rows = ["| metric | value |", "| --- | ---: |"]
    for key in sorted(metrics):
        if key.endswith("_ci95"):
            continue
        value = metrics[key]
        if isinstance(value, (dict, list)):
            continue
        rendered = _fmt(value)
        ci = metrics.get(f"{key}_ci95")
        if isinstance(ci, list) and len(ci) == 2:
            rendered += f" (95% CI {_fmt(ci[0])}–{_fmt(ci[1])})"
        rows.append(f"| `{key}` | {rendered} |")
    return rows


def _confusion_table(confusion: dict[str, Any]) -> list[str]:
    labels = list(confusion.get("labels", []))
    matrix = confusion.get("matrix", [])
    if not labels:
        return ["No confusion matrix data."]
    header = "| oracle \\ predicted | " + " | ".join(_escape(label) for label in labels) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in labels) + " |"
    lines = [header, separator]
    for index, label in enumerate(labels):
        row = matrix[index] if index < len(matrix) else []
        cells = " | ".join(
            str(row[column]) if column < len(row) else "0" for column in range(len(labels))
        )
        lines.append(f"| {_escape(label)} | {cells} |")
    return lines


def _ci_suffix(ci: Any) -> str:
    if isinstance(ci, list) and len(ci) == 2:
        return f" (95% CI {_fmt(ci[0])}–{_fmt(ci[1])})"
    return ""


def _ci_text(ci: Any) -> str:
    if isinstance(ci, list) and len(ci) == 2:
        return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"
    return "—"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
