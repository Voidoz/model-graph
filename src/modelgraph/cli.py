"""Command line interface for Model Graph."""

from __future__ import annotations

import json
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import typer
import yaml

from modelgraph import __version__
from modelgraph.compiler import compile_graph
from modelgraph.compiler.ir import CompiledGraph
from modelgraph.contracts.graph import Defaults
from modelgraph.contracts.loader import load_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.data import export_dataset
from modelgraph.errors import ModelGraphError
from modelgraph.evaluation import (
    E2EConfiguration,
    EvalReport,
    run_e2e_suite,
    run_node_suite,
    run_routing_suite,
)
from modelgraph.evaluation.compare import compare_against_baseline, gate_summary
from modelgraph.evaluation.report import default_report_dir
from modelgraph.inspector import build_inspection, render_inspector_html
from modelgraph.paths import cache_dir as default_cache_dir
from modelgraph.paths import runs_dir as default_runs_dir
from modelgraph.providers import build_provider
from modelgraph.providers.base import Provider
from modelgraph.providers.mock import MockProvider, MockScenario, scenario_from_dict
from modelgraph.pruning import (
    LazyModelLoader,
    Profile,
    PruneError,
    PruneReport,
    apply_budget_overrides,
    format_prune_report,
    load_profile,
    prune_graph,
    resolve_profile_path,
)
from modelgraph.registry import Registry, default_registry
from modelgraph.registry_store import (
    add_contribution,
    default_registry_root,
    inspect_entry,
    list_entries,
    verify_registry,
)
from modelgraph.revision import (
    ChangeSetError,
    RevisionStore,
    analyse_impact,
    annotate_run_trace,
    bootstrap_revision,
    default_store_dir,
    diff_graphs,
    evaluate_coverage,
    format_change_summary,
    format_diff,
    format_impact,
    load_evidence,
    mark_promoted,
    promotion_plan,
    propose_graph_change,
    propose_swap,
    revision_timeline,
    status_for_change,
    status_report,
)
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.cache import CACHE_MODES, ResponseCache
from modelgraph.runtime.engine import DEFAULT_MAX_CONCURRENCY, run_graph
from modelgraph.runtime.manifest import read_run_manifest, recover_interrupted_runs
from modelgraph.runtime.replay import replay_run
from modelgraph.runtime.trace import read_events
from modelgraph.security import profile_security_warnings
from modelgraph.watch import WatchSession, format_watch_report, watch_loop

app = typer.Typer(
    name="mg",
    help="Compose small language models into a routed, observable graph.",
    no_args_is_help=True,
    add_completion=False,
)
trace_app = typer.Typer(help="Inspect run traces.", no_args_is_help=True)
runs_app = typer.Typer(help="List and inspect runs.", no_args_is_help=True)
models_app = typer.Typer(help="Inspect the model registry.", no_args_is_help=True)
eval_app = typer.Typer(
    help="Run node, routing, and e2e evaluation suites, and compare against baselines.",
    no_args_is_help=True,
)
data_app = typer.Typer(
    help="Export recorded traces into consented, redacted training datasets.",
    no_args_is_help=True,
)
training_app = typer.Typer(
    help="Build router datasets, train/calibrate a router, and rehearse specialist training.",
    no_args_is_help=True,
)
registry_app = typer.Typer(
    help="Add, list, inspect, and verify versioned model contributions.",
    no_args_is_help=True,
)
graph_app = typer.Typer(help="Compare graph revisions and deployments.", no_args_is_help=True)
swap_app = typer.Typer(help="Propose model swaps as candidate change sets.", no_args_is_help=True)
change_app = typer.Typer(help="List, show, and explain change sets.", no_args_is_help=True)
app.add_typer(trace_app, name="trace")
app.add_typer(runs_app, name="runs")
app.add_typer(models_app, name="models")
app.add_typer(registry_app, name="registry")
app.add_typer(eval_app, name="eval")
app.add_typer(data_app, name="data")
app.add_typer(training_app, name="train")
app.add_typer(graph_app, name="graph")
app.add_typer(swap_app, name="swap")
app.add_typer(change_app, name="change")

DEMO_TASK = "Write a debounce function in TypeScript."


@app.callback()
def main() -> None:
    """Model Graph command group."""


@app.command()
def version() -> None:
    """Print the installed Model Graph version."""
    typer.echo(__version__)


@eval_app.command("node")
def eval_node(
    suite: Path = typer.Option(..., "--suite", help="Node suite directory or suite.yaml."),
    graph: Path | None = typer.Option(None, "--graph", help="Override the suite target graph."),
    provider_name: str | None = typer.Option(
        None, "--provider", help="mock, auto, ollama, or openai-compatible."
    ),
    live: bool = typer.Option(
        False, "--live", help="Allow a live provider (same as MG_EVAL_LIVE=1)."
    ),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    reports_directory: Path = typer.Option(
        default_report_dir(), "--reports-dir", help="Directory for evaluation reports."
    ),
    case: list[str] = typer.Option([], "--case", help="Select a case id (repeatable)."),
    tag: list[str] = typer.Option([], "--tag", help="Select cases by tag (repeatable)."),
    limit: int | None = typer.Option(None, "--limit", help="Limit selected cases."),
    allow_commands: bool = typer.Option(
        True, "--allow-commands/--no-commands", help="Allow suite command checks."
    ),
    fail_on_gate: bool = typer.Option(
        False, "--fail-on-gate", help="Exit non-zero when a threshold fails."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable report."),
) -> None:
    """Run a node suite and write JSON + Markdown reports."""
    try:
        report = run_node_suite(
            suite,
            graph_path=graph,
            provider=provider_name,
            allow_live=live,
            registry=_registry(registry_dir) if registry_dir else None,
            reports_dir=reports_directory,
            case_ids=case or None,
            tags=tag or None,
            limit=limit,
            allow_commands=allow_commands,
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    _emit_eval_report(report, as_json=as_json, fail_on_gate=fail_on_gate)


@eval_app.command("routing")
def eval_routing(
    suite: Path = typer.Option(..., "--suite", help="Routing suite directory or suite.yaml."),
    graph: Path | None = typer.Option(None, "--graph", help="Override the suite target graph."),
    provider_name: str | None = typer.Option(
        None,
        "--provider",
        help="mock, trained-router, auto, ollama, or openai-compatible.",
    ),
    live: bool = typer.Option(
        False, "--live", help="Allow a live provider (same as MG_EVAL_LIVE=1)."
    ),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    reports_directory: Path = typer.Option(
        default_report_dir(), "--reports-dir", help="Directory for evaluation reports."
    ),
    case: list[str] = typer.Option([], "--case", help="Select a case id (repeatable)."),
    tag: list[str] = typer.Option([], "--tag", help="Select cases by tag (repeatable)."),
    limit: int | None = typer.Option(None, "--limit", help="Limit selected cases."),
    fail_on_gate: bool = typer.Option(
        False, "--fail-on-gate", help="Exit non-zero when a threshold fails."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable report."),
) -> None:
    """Run a routing suite and write JSON + Markdown reports."""
    try:
        report = run_routing_suite(
            suite,
            graph_path=graph,
            provider=provider_name,
            allow_live=live,
            registry=_registry(registry_dir) if registry_dir else None,
            reports_dir=reports_directory,
            case_ids=case or None,
            tags=tag or None,
            limit=limit,
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    _emit_eval_report(report, as_json=as_json, fail_on_gate=fail_on_gate)


@eval_app.command("e2e")
def eval_e2e(
    suite: Path = typer.Option(..., "--suite", help="E2E suite directory or suite.yaml."),
    graph: Path | None = typer.Option(None, "--graph", help="Override the suite target graph."),
    profile_ref: str | None = typer.Option(
        None, "--profile", help="Run profile file or profile name."
    ),
    provider_name: str | None = typer.Option(
        None, "--provider", help="mock, auto, ollama, or openai-compatible."
    ),
    live: bool = typer.Option(
        False, "--live", help="Allow a live provider (same as MG_EVAL_LIVE=1)."
    ),
    baseline: Path | None = typer.Option(
        None, "--baseline", help="Incumbent e2e report JSON or report directory."
    ),
    config: list[str] = typer.Option(
        [],
        "--config",
        help="Additional Pareto configuration as NAME=GRAPH[@PROFILE] (repeatable).",
    ),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    reports_directory: Path = typer.Option(
        default_report_dir(), "--reports-dir", help="Directory for evaluation reports."
    ),
    case: list[str] = typer.Option([], "--case", help="Select a case id (repeatable)."),
    tag: list[str] = typer.Option([], "--tag", help="Select cases by tag (repeatable)."),
    limit: int | None = typer.Option(None, "--limit", help="Limit selected cases."),
    allow_budget_override: bool = typer.Option(
        False,
        "--allow-budget-override",
        help="Allow a profile to raise graph budget defaults.",
    ),
    allow_commands: bool = typer.Option(
        True, "--allow-commands/--no-commands", help="Allow suite command checks."
    ),
    fail_on_gate: bool = typer.Option(
        False, "--fail-on-gate", help="Exit non-zero when a threshold fails."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable report."),
) -> None:
    """Run an end-to-end suite over a graph/profile with baselines and a Pareto report."""
    try:
        configurations = [_parse_configuration(value) for value in config]
        report = run_e2e_suite(
            suite,
            graph_path=graph,
            profile=profile_ref,
            provider=provider_name,
            allow_live=live,
            registry=_registry(registry_dir) if registry_dir else None,
            reports_dir=reports_directory,
            case_ids=case or None,
            tags=tag or None,
            limit=limit,
            baseline=baseline,
            allow_commands=allow_commands,
            allow_budget_override=allow_budget_override,
            configurations=configurations,
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    _emit_eval_report(report, as_json=as_json, fail_on_gate=fail_on_gate)


def _load_report_document(path: Path) -> dict[str, Any] | None:
    """Load a report JSON or a report directory's ``report.json`` for a reference view."""
    source = path / "report.json" if path.is_dir() else path
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _quality_reference(baseline: Path, candidate: Path) -> dict[str, Any] | None:
    """Candidate/baseline/oracle quality rows when the reports carry e2e baselines."""
    candidate_data = _load_report_document(candidate)
    baseline_data = _load_report_document(baseline)
    if candidate_data is None:
        return None
    candidate_e2e = candidate_data.get("e2e")
    baseline_e2e = baseline_data.get("e2e") if baseline_data else None
    if not isinstance(candidate_e2e, dict) or not isinstance(baseline_e2e, dict):
        return None
    baselines = candidate_e2e.get("baselines")
    if not isinstance(baselines, dict):
        return None

    def row(data: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        metrics = data.get("metrics") or {}
        return {
            "task_success": metrics.get("task_success"),
            "task_success_ci95": metrics.get("task_success_ci95"),
            "validity": metrics.get("validity"),
            "cost_per_task_usd": metrics.get("cost_per_task_usd"),
            "latency_p50_seconds": metrics.get("latency_p50_seconds"),
        }

    reference: dict[str, Any] = {
        "candidate": row(candidate_data),
        "baseline": row(baseline_data),
    }
    found_reference = False
    for name in ("oracle_routing", "best_single"):
        entry = baselines.get(name)
        if isinstance(entry, dict) and entry.get("available", True):
            reference[name] = {
                "task_success": entry.get("task_success"),
                "task_success_ci95": entry.get("task_success_ci95"),
                "validity": entry.get("validity"),
                "cost_per_task_usd": entry.get("cost_per_task_usd"),
                "latency_p50_seconds": entry.get("latency_p50_seconds"),
            }
            found_reference = True
    if not found_reference:
        # Printing a header that promises "vs oracle" when no oracle/best-single row exists
        # is worse than omitting the optional table.
        return None
    return reference


def _print_quality_reference(reference: dict[str, Any]) -> None:
    typer.echo("reference: candidate vs incumbent vs oracle (e2e)")
    for name in ("candidate", "baseline", "oracle_routing", "best_single"):
        row = reference.get(name)
        if not isinstance(row, dict) or row.get("task_success") is None:
            continue
        ci = row.get("task_success_ci95")
        ci_text = f" {_format_ci(ci)}" if isinstance(ci, list) and len(ci) == 2 else ""
        typer.echo(
            f"  {name:<15} tasks={_format_rate(row.get('task_success'))}{ci_text} "
            f"validity={_format_rate(row.get('validity'))} "
            f"cost/case=${_format_number(row.get('cost_per_task_usd'), 6)} "
            f"p50={_format_number(row.get('latency_p50_seconds'), 3)}s"
        )


@eval_app.command("compare")
def eval_compare(
    baseline: Path = typer.Option(
        ..., "--baseline", help="Committed baseline report JSON or report directory."
    ),
    candidate: Path = typer.Option(
        ..., "--candidate", help="Candidate report JSON or report directory."
    ),
    policy: Path | None = typer.Option(
        None, "--policy", help="Gate policy JSON (default: evals/baselines/policy.json)."
    ),
    waiver: list[Path] = typer.Option([], "--waiver", help="Reviewed waiver file (repeatable)."),
    reports_directory: Path = typer.Option(
        default_report_dir(), "--reports-dir", help="Directory for comparison reports."
    ),
    fail_on_gate: bool = typer.Option(
        True,
        "--fail-on-gate/--no-fail-on-gate",
        help="Exit non-zero when a contribution gate fails.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable comparison report."),
) -> None:
    """Compare a candidate report against a baseline and apply contribution gates."""
    command_parts = [
        "mg",
        "eval",
        "compare",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
    ]
    if policy is not None:
        command_parts.extend(["--policy", str(policy)])
    for waiver_path in waiver:
        command_parts.extend(["--waiver", str(waiver_path)])
    try:
        report = compare_against_baseline(
            candidate,
            baseline,
            policy=policy,
            waivers=waiver,
            reports_dir=reports_directory,
            command=" ".join(command_parts),
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    reference = _quality_reference(baseline, candidate)
    if as_json:
        payload = report.to_dict()
        if report.report_dir is not None:
            payload["report_dir"] = str(report.report_dir)
        if reference is not None:
            payload["reference"] = reference
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_eval_report(report)
        for line in gate_summary(report):
            if line.startswith(("gate:", "report:")):
                continue
            typer.echo(line)
        if reference is not None:
            _print_quality_reference(reference)
    if fail_on_gate and (report.gate is None or report.gate.get("passed") is not True):
        raise typer.Exit(code=1)


def _parse_configuration(value: str) -> E2EConfiguration:
    if "=" not in value:
        raise typer.BadParameter(f"expected NAME=GRAPH[@PROFILE], got {value!r}")
    name, target = value.split("=", 1)
    graph, _, profile = target.partition("@")
    if not name.strip() or not graph.strip():
        raise typer.BadParameter(f"expected NAME=GRAPH[@PROFILE], got {value!r}")
    return E2EConfiguration(
        name=name.strip(),
        graph=graph.strip(),
        profile=profile.strip() or None,
    )


def _emit_eval_report(report: EvalReport, *, as_json: bool, fail_on_gate: bool) -> None:
    if as_json:
        payload = report.to_dict()
        if report.report_dir is not None:
            payload["report_dir"] = str(report.report_dir)
            payload["manifest"] = str(report.report_dir / "manifest.json")
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_eval_report(report)
    if fail_on_gate and report.gate is not None and report.gate.get("passed") is not True:
        raise typer.Exit(code=1)


def _print_eval_report(report: EvalReport) -> None:
    suite = report.suite
    typer.echo(f"level:    {report.level}")
    typer.echo(
        f"suite:    {suite.get('id', '?')}@{suite.get('version', '?')} ({suite.get('hash', '?')})"
    )
    if report.graph:
        typer.echo(
            f"graph:    {report.graph.get('id', '?')}@{report.graph.get('version', '?')} "
            f"({report.graph.get('hash', '?')})"
        )
    typer.echo(f"report:   {report.report_id}")
    if report.report_dir is not None:
        typer.echo(f"dir:      {report.report_dir}")
        typer.echo(f"manifest: {report.report_dir / 'manifest.json'}")
    if report.gate is not None:
        typer.echo(f"gate:     {'PASS' if report.gate.get('passed') else 'FAIL'}")
        if report.gate.get("failures"):
            typer.echo(f"failed:   {', '.join(report.gate['failures'])}")
    primary = report.primary_metric
    ci = primary.get("ci95")
    ci_text = f" (95% CI {ci[0]:.4f}-{ci[1]:.4f})" if isinstance(ci, list) and len(ci) == 2 else ""
    typer.echo(f"metric:   {primary.get('name')}={primary.get('value')}{ci_text}")
    metrics = report.metrics
    if report.level == "comparison":
        typer.echo(
            f"paired:   {report.selection.get('paired_cases', 0)} cases; "
            f"cost delta ${_format_number(report.costs.get('delta_per_task_usd'), 6)}/case; "
            f"latency delta p50 {_format_number(report.latency.get('delta_p50_seconds'), 3)}s"
        )
        typer.echo(
            f"cost CI:  {_format_ci(report.costs.get('delta_ci95'))} per case (95% bootstrap)"
        )
        typer.echo(
            "latency CI: "
            f"{_format_ci(report.latency.get('delta_p50_ci95'))} seconds paired p50 "
            "(95% bootstrap)"
        )
    else:
        parts = [
            f"tasks={_format_rate(metrics.get('task_success'))}",
            f"validity={_format_rate(metrics.get('validity'))}",
            f"cost/case=${_format_number(metrics.get('cost_per_task_usd'), 6)}",
            f"p95={_format_number(metrics.get('latency_p95_seconds'), 3)}s",
        ]
        if metrics.get("hop_count_mean") is not None:
            parts.append(f"hops={_format_number(metrics.get('hop_count_mean'), 2)}")
        typer.echo("metrics:  " + " ".join(parts))
    if report.level == "e2e" and report.e2e:
        baselines = report.e2e.get("baselines") or {}
        for name in sorted(baselines):
            entry = baselines[name]
            if not isinstance(entry, dict):
                continue
            if not entry.get("available", True):
                typer.echo(f"baseline: {name}=unavailable")
                continue
            typer.echo(
                f"baseline: {name} tasks={_format_rate(entry.get('task_success'))} "
                f"cost/case=${_format_number(entry.get('cost_per_task_usd'), 6)}"
            )
        pareto = report.e2e.get("pareto") or {}
        if pareto.get("frontier"):
            typer.echo(f"frontier: {', '.join(pareto['frontier'])}")
        for band in pareto.get("bands", []):
            typer.echo(
                f"band:     <={_format_number(band.get('max_cost_usd'), 6)} USD -> "
                f"{band.get('winner')}"
            )
    if report.level == "comparison" and report.slices:
        typer.echo("slices:   paired delta and 95% bootstrap CI (W/L/T per slice)")
        for name in sorted(report.slices):
            entry = report.slices[name]
            typer.echo(
                f"  {name}: n={entry.get('cases', 0)} "
                f"baseline={_format_rate(entry.get('baseline_primary'))} "
                f"candidate={_format_rate(entry.get('candidate_primary'))} "
                f"delta={_format_signed(entry.get('primary_delta'))} "
                f"ci={_format_ci(entry.get('primary_delta_ci95'))} "
                f"W/L/T={entry.get('wins', 0)}/{entry.get('losses', 0)}/"
                f"{entry.get('ties', 0)} "
                f"cost_delta=${_format_number(entry.get('cost_delta_per_task_usd'), 6)} "
                f"cost_ci={_format_ci(entry.get('cost_delta_ci95'))} "
                f"latency_p50_delta={_format_number(entry.get('latency_delta_p50_seconds'), 3)}s "
                f"latency_ci={_format_ci(entry.get('latency_delta_ci95'))}"
            )
    typer.echo(f"failures: {len(report.failures)}")
    for failure in report.failures[:5]:
        typer.echo(f"  - {failure.get('case_id')}: {failure.get('reason')}")
    if len(report.failures) > 5:
        typer.echo(f"  ... {len(report.failures) - 5} more")


def _format_rate(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def _format_number(value: Any, digits: int) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def _format_signed(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:+.4f}"


def _format_ci(value: Any) -> str:
    if isinstance(value, list) and len(value) == 2:
        return f"[{_format_signed(value[0])}, {_format_signed(value[1])}]"
    return "—"


def _registry(model_dirs: list[Path]) -> Registry:
    if model_dirs:
        return Registry.from_paths(list(model_dirs))
    return default_registry()


@data_app.command("export")
def data_export(
    run: list[Path] = typer.Option(..., "--run", help="Recorded run directory (repeatable)."),
    out: Path = typer.Option(..., "--out", help="Dataset output directory."),
    dataset_id: str = typer.Option("trace-export", "--dataset-id", help="Dataset id."),
    dataset_name: str = typer.Option("Trace export", "--dataset-name"),
    dataset_version: str = typer.Option("0.1.0", "--dataset-version"),
    dataset_description: str = typer.Option(
        "Trace-derived examples exported by mg data export.", "--dataset-description"
    ),
    task: str | None = typer.Option(None, "--task", help="Dataset task description."),
    license_id: str = typer.Option("unknown", "--license", help="Dataset license id."),
    license_uri: str | None = typer.Option(None, "--license-uri"),
    consent: str | None = typer.Option(
        None,
        "--consent",
        help="Recorded consent statement/id. Required unless --no-require-consent is explicit.",
    ),
    consent_scope: str = typer.Option("local-training", "--consent-scope"),
    require_consent: bool = typer.Option(
        True,
        "--require-consent/--no-require-consent",
        help="Fail closed when consent is not recorded (default: true).",
    ),
    include_failures: bool = typer.Option(
        False, "--include-failures", help="Include non-succeeded runs; label them failure."
    ),
    redact: str = typer.Option(
        "default", "--redact", help="Redaction preset: default, none, or a YAML/JSON config path."
    ),
    redact_action: str = typer.Option(
        "hash", "--redact-action", help="Default action: hash, mask, or drop."
    ),
    labels: list[Path] = typer.Option([], "--labels", help="Human/oracle label file (repeatable)."),
    checks: list[Path] = typer.Option(
        [], "--checks", help="Node check file using the eval CheckSpec shape (repeatable)."
    ),
    allow_command_checks: bool = typer.Option(
        False, "--allow-command-checks", help="Execute local command checks from --checks."
    ),
    eval_suite: list[Path] = typer.Option(
        [], "--eval-suite", help="Eval suite used for contamination checking (repeatable)."
    ),
    dedupe: str = typer.Option(
        "exact+near", "--dedupe", help="Deduplication: none, exact, near, or exact+near."
    ),
    near_duplicate_distance: int = typer.Option(
        3, "--near-duplicate-distance", help="SimHash Hamming-distance threshold (0-64)."
    ),
    group_by: str = typer.Option(
        "source", "--group-by", help="Split grouping: source, graph, run, or license."
    ),
    split_ratio: str = typer.Option(
        "0.8,0.1,0.1", "--split-ratio", help="TRAIN,VALIDATION,TEST ratios."
    ),
    split_seed: str = typer.Option("modelgraph-v0", "--split-seed"),
    split_plan: Path | None = typer.Option(
        None, "--split-plan", help="Persist/reuse group-to-split assignments (JSON)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable summary."),
) -> None:
    """Export accepted traces into a redacted, labeled, split dataset."""
    try:
        result = export_dataset(
            run,
            out,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            dataset_description=dataset_description,
            task=task,
            license_id=license_id,
            license_uri=license_uri,
            consent=consent,
            consent_scope=consent_scope,
            require_consent=require_consent,
            include_failures=include_failures,
            redact=redact,
            redact_action=redact_action,
            labels=labels,
            checks=checks,
            allow_command_checks=allow_command_checks,
            eval_suites=eval_suite,
            dedupe=dedupe,
            near_duplicate_distance=near_duplicate_distance,
            group_by=group_by,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_plan=split_plan,
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return
    examples = result.manifest["spec"]["examples"]
    contamination = result.manifest["spec"]["eval_contamination"]
    typer.echo(f"dataset:   {result.dataset_dir}")
    typer.echo(f"manifest:  {result.dataset_dir / 'dataset.json'}")
    typer.echo(
        f"examples:  {result.examples} (train={result.counts['train']} "
        f"validation={result.counts['validation']} test={result.counts['test']})"
    )
    by_node = examples.get("by_node") or {}
    by_route = examples.get("by_route") or {}
    node_counts = ", ".join(f"{name}={count}" for name, count in sorted(by_node.items())) or "none"
    route_counts = (
        ", ".join(f"{name}={count}" for name, count in sorted(by_route.items())) or "none"
    )
    typer.echo(f"node:      {examples['node']} (by source node: {node_counts})")
    typer.echo(f"routing:   {examples['routing']} (per route: {route_counts})")
    typer.echo(
        f"consent:   {'recorded' if result.manifest['spec']['consent']['recorded'] else 'not recorded'}"
    )
    redaction = result.redaction
    matched = [item["name"] for item in redaction["patterns"] if item["count"]]
    typer.echo(
        f"redaction: {redaction['status']} "
        f"({redaction['total_redactions']} redaction(s)"
        + (f"; {', '.join(matched)}" if matched else "")
        + ")"
    )
    typer.echo(f"contamination: {contamination['status']} (excluded={contamination['excluded']})")


@training_app.command("dataset")
def train_dataset(
    dataset: Path = typer.Option(..., "--dataset", help="T-0023 dataset directory or JSONL file."),
    out: Path = typer.Option(..., "--out", help="Output directory for router examples + summary."),
    strict_labels: bool = typer.Option(
        False, "--strict-labels", help="Fail instead of excluding routing examples without a label."
    ),
    allow_unrendered: bool = typer.Option(
        False, "--allow-unrendered", help="Allow examples whose rendered prompt was not captured."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable summary."),
) -> None:
    """Build router examples from a T-0023 dataset export."""
    from modelgraph.training.dataset import load_router_dataset

    try:
        loaded = load_router_dataset(
            dataset, strict_labels=strict_labels, allow_unrendered=allow_unrendered
        )
        destination = loaded.write(out)
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    payload = {
        "dataset": loaded.summary["dataset"],
        "examples": loaded.summary["examples"],
        "counts": loaded.summary["counts"],
        "class_balance": loaded.summary["class_balance"],
        "per_route": loaded.summary["per_route"],
        "excluded": loaded.summary["excluded"],
        "output": str(destination),
        "examples_file": str(destination / "router-examples.jsonl"),
        "summary_file": str(destination / "router-dataset.json"),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    typer.echo(f"router dataset: {destination}")
    typer.echo("examples: " + ", ".join(f"{name}={count}" for name, count in loaded.counts.items()))
    typer.echo(
        "classes: "
        + ", ".join(f"{name}={count}" for name, count in loaded.summary["per_route"].items())
    )
    typer.echo(f"excluded: {loaded.summary['excluded']}")
    typer.echo(f"summary: {destination / 'router-dataset.json'}")


@training_app.command("router")
def train_router_command(
    config: Path = typer.Option(..., "--config", help="Router training config YAML."),
    dataset: Path | None = typer.Option(None, "--dataset", help="Override config dataset path."),
    artifact: Path | None = typer.Option(None, "--artifact", help="Override artifact output path."),
    report: Path | None = typer.Option(None, "--report", help="Override Markdown report path."),
    write: bool = typer.Option(
        True, "--write/--no-write", help="Write the artifact and report (default: write)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable summary."),
) -> None:
    """Train and calibrate a router, then emit an artifact and report."""
    from modelgraph.training.config import load_training_config
    from modelgraph.training.dataset import load_router_dataset
    from modelgraph.training.model import train_router

    try:
        loaded_config = load_training_config(config)
        updates: dict[str, Any] = {}
        if dataset is not None:
            updates["dataset"] = str(dataset)
        if artifact is not None:
            updates["artifact"] = str(artifact)
        if report is not None:
            updates["report"] = str(report)
        if updates:
            loaded_config = loaded_config.model_copy(update=updates)
        training_data = load_router_dataset(loaded_config.dataset)
        command = (
            f"mg train router --config {config}"
            + (f" --dataset {dataset}" if dataset is not None else "")
            + (f" --artifact {artifact}" if artifact is not None else "")
            + (f" --report {report}" if report is not None else "")
        )
        result = train_router(training_data, loaded_config, command=command, write=write)
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    payload = {
        "artifact": str(result.artifact_path) if result.artifact_path else None,
        "artifact_sha256": result.predictor.artifact.sha256(),
        "artifact_schema": result.artifact["schema"],
        "report": str(result.report_path) if result.report_path else None,
        "seed": loaded_config.seed,
        "classes": result.artifact["classes"],
        "calibration": result.report["calibration"],
        "abstain": result.report["abstain"],
        "counts": result.dataset.counts,
        "metrics": result.report["metrics"],
        "recommendation": result.report["recommendation"],
        "limitations": result.report["limitations"],
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    typer.echo(f"artifact:  {result.artifact_path}")
    typer.echo(f"sha256:    {result.predictor.artifact.sha256()}")
    typer.echo(f"report:    {result.report_path}")
    typer.echo(
        f"temperature: {result.report['calibration']['temperature']:.3f} "
        f"(fit on {result.report['calibration']['fit_split']})"
    )
    typer.echo(
        f"abstain threshold: {result.report['abstain']['threshold']:.3f} (fit on validation)"
    )
    for split in ("validation", "test"):
        trained = result.report["metrics"]["trained"].get(split, {})
        typer.echo(
            f"{split}: agreement={trained.get('oracle_agreement')} "
            f"coverage={trained.get('coverage')} ece={trained.get('ece')} "
            f"cost/case={trained.get('expected_cost_per_case')}"
        )


@training_app.command("specialist")
def train_specialist_command(
    config: Path = typer.Option(..., "--config", help="Specialist training config YAML."),
    dataset: Path | None = typer.Option(None, "--dataset", help="Override config dataset path."),
    output: Path | None = typer.Option(None, "--output", help="Override output directory."),
    node: str | None = typer.Option(None, "--node", help="Override the source node filter."),
    seed: int | None = typer.Option(None, "--seed", help="Override the training seed."),
    run_eval: bool = typer.Option(
        True,
        "--eval/--no-eval",
        help="Run the offline node-suite + graph-regression hook (default: run).",
    ),
    eval_report: Path | None = typer.Option(
        None,
        "--eval-report",
        help="Use a supplied passing registry eval report instead of running the hook.",
    ),
    promote: bool = typer.Option(
        False,
        "--promote",
        help="Require a passing eval report and mark a real artifact promoted (refused for rehearsals).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable summary."),
) -> None:
    """Run the deterministic specialist rehearsal: artifact, manifest, card, eval, repro."""
    from modelgraph.training.specialist import load_specialist_config, train_specialist

    try:
        loaded_config = load_specialist_config(config)
        updates: dict[str, Any] = {}
        if dataset is not None:
            updates["dataset"] = str(dataset)
        if output is not None:
            updates["output_dir"] = str(output)
        if node is not None:
            updates["node"] = node
        if seed is not None:
            updates["seed"] = seed
        if updates:
            loaded_config = loaded_config.model_copy(update=updates)
        command_parts = [f"mg train specialist --config {config}"]
        if dataset is not None:
            command_parts.append(f"--dataset {dataset}")
        if output is not None:
            command_parts.append(f"--output {output}")
        if node is not None:
            command_parts.append(f"--node {node}")
        if seed is not None:
            command_parts.append(f"--seed {seed}")
        if promote:
            command_parts.append("--promote")
        result = train_specialist(
            loaded_config,
            run_eval=run_eval,
            eval_report_path=eval_report,
            promote=promote,
            command=" ".join(command_parts),
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    payload = result.summary()
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    typer.echo(f"model:     {payload['model']} ({payload['status']})")
    typer.echo(f"artifact:  {payload['artifact']}")
    typer.echo(f"sha256:    {payload['artifact_sha256']}")
    typer.echo(f"manifest:  {payload['manifest']}")
    typer.echo(f"card:      {payload['card']}")
    typer.echo(f"repro:     {payload['reproducibility']}")
    typer.echo(f"eval:      {payload['eval_report'] or 'not run'}")
    dataset_info = payload["dataset"]
    typer.echo(
        f"dataset:   {dataset_info['id']}@{dataset_info['version']} "
        f"({dataset_info['hash']}) license={dataset_info['license']}"
    )
    typer.echo(f"splits:    {dataset_info['counts']}")
    typer.echo(f"excluded:  {dataset_info['excluded']}")
    if payload["eval_passed"] is not None:
        typer.echo(f"gate:      {'PASS' if payload['eval_passed'] else 'FAIL'}")
    typer.echo("quality:   not claimed (synthetic rehearsal)")


# --------------------------------------------------------------------------------------
# Registry (T-0026)
# --------------------------------------------------------------------------------------


def _registry_profile(profile: str | None) -> Any | None:
    if profile is None:
        return None
    return load_profile(profile)


@registry_app.command("add")
def registry_add_command(
    path: Path = typer.Argument(..., help="Contribution directory or manifest file."),
    root: Path = typer.Option(
        default_registry_root(), "--root", help="Versioned registry root (default: registry/)."
    ),
    manifest: Path | None = typer.Option(None, "--manifest", help="Override manifest path."),
    card: Path | None = typer.Option(None, "--card", help="Override card.md path."),
    report: Path | None = typer.Option(None, "--report", help="Override eval-report.json path."),
    profile: str | None = typer.Option(None, "--profile", help="License policy profile."),
    license_policy: str = typer.Option(
        "distributable",
        "--license-policy",
        help="distributable (default) or local.",
    ),
    allow_unknown_license: bool = typer.Option(
        False, "--allow-unknown-license", help="Explicitly allow an unknown license declaration."
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Add a verified contribution: manifest, card, eval report, and evidence."""
    try:
        entry, changed = add_contribution(
            root,
            path,
            manifest_path=manifest,
            card_path=card,
            report_path=report,
            profile=_registry_profile(profile),
            allow_unknown=allow_unknown_license,
            policy=license_policy,
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    payload = {
        "model": entry.ref,
        "path": str(entry.path),
        "changed": changed,
        "hashes": entry.hashes,
        "warnings": entry.warnings,
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    typer.echo(f"{'added' if changed else 'already present'}: {entry.ref}")
    typer.echo(f"path:    {entry.path}")
    typer.echo(f"hashes:  {entry.hashes}")
    for warning in entry.warnings:
        typer.echo(f"warning: {warning}", err=True)


@registry_app.command("list")
def registry_list_command(
    root: Path = typer.Option(default_registry_root(), "--root"),
    profile: str | None = typer.Option(None, "--profile", help="License policy profile."),
    license_policy: str = typer.Option("distributable", "--license-policy"),
    allow_unknown_license: bool = typer.Option(False, "--allow-unknown-license"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List versioned registry entries."""
    try:
        entries = list_entries(
            root,
            profile=_registry_profile(profile),
            allow_unknown=allow_unknown_license,
            policy=license_policy,
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    rows = [
        {
            "id": entry.model_id,
            "version": entry.version,
            "provider": entry.manifest.get("spec", {}).get("provider"),
            "license": entry.manifest.get("spec", {}).get("license", {}).get("name"),
            "passed": entry.passed,
            "warnings": entry.warnings,
            "errors": entry.errors,
            "path": str(entry.path),
        }
        for entry in entries
    ]
    if as_json:
        typer.echo(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    if not rows:
        typer.echo("no versioned registry entries")
        typer.echo("flat starter manifests under registry/models/*.yaml still resolve at runtime")
        return
    for row in rows:
        status = "ok" if row["passed"] else "FAIL"
        typer.echo(
            f"{row['id']}@{row['version']:<10} {row['provider']:<18} {row['license']:<14} {status}"
        )
        for error in row["errors"]:
            typer.echo(f"  error: {error}", err=True)


@registry_app.command("inspect")
def registry_inspect_command(
    ref: str = typer.Argument(..., help="ID or ID@VERSION."),
    root: Path = typer.Option(default_registry_root(), "--root"),
    profile: str | None = typer.Option(None, "--profile", help="License policy profile."),
    license_policy: str = typer.Option("distributable", "--license-policy"),
    allow_unknown_license: bool = typer.Option(False, "--allow-unknown-license"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect one versioned entry: manifest, card, report, and hashes."""
    try:
        entry = inspect_entry(
            root,
            ref,
            profile=_registry_profile(profile),
            allow_unknown=allow_unknown_license,
            policy=license_policy,
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    payload = {
        "model": entry.ref,
        "path": str(entry.path),
        "manifest": entry.manifest,
        "card": entry.card,
        "eval_report": entry.report,
        "hashes": entry.hashes,
        "passed": entry.passed,
        "errors": entry.errors,
        "warnings": entry.warnings,
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    typer.echo(yaml.safe_dump(entry.manifest, sort_keys=False, allow_unicode=True))
    typer.echo(f"path:    {entry.path}")
    typer.echo(f"hashes:  {entry.hashes}")
    typer.echo(f"passed:  {entry.passed}")
    for warning in entry.warnings:
        typer.echo(f"warning: {warning}", err=True)
    for error in entry.errors:
        typer.echo(f"error:   {error}", err=True)
    typer.echo("")
    typer.echo(entry.card)
    typer.echo("")
    typer.echo("Eval evidence:")
    typer.echo(json.dumps(entry.report.get("evidence") or {}, indent=2, sort_keys=True))


@registry_app.command("verify")
def registry_verify_command(
    root: Path = typer.Option(default_registry_root(), "--root"),
    profile: str | None = typer.Option(None, "--profile", help="License policy profile."),
    license_policy: str = typer.Option(
        "distributable",
        "--license-policy",
        help="distributable (default) rejects unknown licenses; local permits them.",
    ),
    allow_unknown_license: bool = typer.Option(False, "--allow-unknown-license"),
    write_index: bool = typer.Option(
        False,
        "--write-index",
        help="Write/refresh the deterministic index.yaml after a clean check.",
    ),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Verify manifests, cards, reports, evidence hashes, licenses, and the index."""
    try:
        selected_profile = _registry_profile(profile)
        result = verify_registry(
            root,
            profile=selected_profile,
            allow_unknown=allow_unknown_license,
            policy=license_policy,
            write_index=write_index,
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=1) from exc
    payload = {
        "root": str(result.root),
        "passed": result.passed,
        "index": str(result.index_path),
        "content_hash": result.index_document.get("content_hash"),
        "index_current": result.index_current,
        "entries": [
            {
                "model": entry.ref,
                "passed": entry.passed,
                "hashes": entry.hashes,
                "warnings": entry.warnings,
                "errors": entry.errors,
            }
            for entry in result.entries
        ],
        "errors": result.errors,
        "warnings": result.warnings,
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for entry in result.entries:
            typer.echo(f"{entry.ref}: {'ok' if entry.passed else 'FAIL'}")
            for warning in entry.warnings:
                typer.echo(f"  warning: {warning}", err=True)
            for error in entry.errors:
                typer.echo(f"  error: {error}", err=True)
        typer.echo(f"index: {result.index_path} ({payload['content_hash']})")
        if result.index_current:
            typer.echo("index: current")
        elif write_index and result.passed:
            typer.echo("index: written")
        else:
            typer.echo("index: stale/missing (run --write-index after fixing entries)")
        if result.passed:
            typer.echo("registry: OK")
        else:
            typer.echo("registry: FAILED", err=True)
    if not result.passed:
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------------------


@app.command()
def validate(
    graphs: list[Path] = typer.Argument(..., help="Graph YAML file(s)."),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    strict: bool = typer.Option(False, "--strict", help="Require all model and subgraph refs."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Validate graph structure, references, and budgets."""
    registry = _registry(registry_dir)
    results: list[dict[str, Any]] = []
    failed = False
    for graph_path in graphs:
        result = compile_graph(graph_path, registry=registry, strict_models=strict)
        entry = {
            "path": str(graph_path),
            "valid": result.ok and not result.errors,
            "graph": (
                {
                    "id": result.graph.id,
                    "version": result.graph.version,
                    "hash": result.graph.graph_hash,
                }
                if result.graph
                else None
            ),
            "errors": list(result.errors),
            "warnings": list(result.warnings),
        }
        results.append(entry)
        if not entry["valid"]:
            failed = True
    if as_json:
        typer.echo(json.dumps({"valid": not failed, "graphs": results}, indent=2))
    else:
        for entry in results:
            status = "valid" if entry["valid"] else "INVALID"
            graph = cast(dict[str, Any], entry["graph"]) if entry["graph"] else {}
            typer.echo(f"{status}: {entry['path']} ({graph.get('id', '?')})")
            for message in cast(list[str], entry["errors"]):
                typer.echo(f"  error: {message}")
            for message in cast(list[str], entry["warnings"]):
                typer.echo(f"  warning: {message}")
    if failed:
        raise typer.Exit(code=1)


@app.command("compile")
def compile_command(
    graph: Path = typer.Argument(..., help="Graph YAML file."),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Require all references."),
    as_json: bool = typer.Option(False, "--json", help="Emit canonical IR JSON."),
) -> None:
    """Compile a graph into the canonical IR."""
    result = compile_graph(
        graph,
        registry=_registry(registry_dir),
        strict_models=strict,
        graph_paths=[graph.parent],
    )
    if not result.ok or result.graph is None:
        for message in result.errors:
            typer.echo(message, err=True)
        raise typer.Exit(code=1)
    if as_json:
        typer.echo(json.dumps(result.graph.to_dict(), indent=2, ensure_ascii=False))
    else:
        typer.echo(
            f"{result.graph.id}@{result.graph.version}  {result.graph.graph_hash}\n"
            f"entry={result.graph.entry} exit={', '.join(result.graph.exits)} "
            f"nodes={len(result.graph.nodes)} edges={len(result.graph.edges)}"
        )
        for warning in result.warnings:
            typer.echo(f"warning: {warning}")


@app.command()
def run(
    graph: Path = typer.Argument(..., help="Graph YAML file."),
    task: str = typer.Option(DEMO_TASK, "--task", "-t", help="Request text."),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    provider_name: str = typer.Option(
        "auto", "--provider", help="auto, mock, ollama, or openai-compatible."
    ),
    mock_scenario: Path | None = typer.Option(
        None, "--mock-scenario", help="JSON/YAML mock scenario with responses and failures."
    ),
    runs_directory: Path = typer.Option(
        default_runs_dir(), "--runs-dir", help="Directory for run artifacts."
    ),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    max_tokens: int | None = typer.Option(None, "--max-tokens"),
    max_cost: float | None = typer.Option(None, "--max-cost"),
    seed: int | None = typer.Option(
        None, "--seed", help="Deterministic sampling seed passed to every provider request."
    ),
    cache_mode: str = typer.Option(
        "on", "--cache-mode", help="Exact-match response cache: on, off, or readonly."
    ),
    cache_directory: Path | None = typer.Option(
        None, "--cache-dir", help="Local response-cache directory (default: cache/responses)."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Bypass cache reads and writes (same as --cache-mode off)."
    ),
    profile_ref: str | None = typer.Option(
        None, "--profile", help="Run profile file or profile name."
    ),
    allow_budget_override: bool = typer.Option(
        False,
        "--allow-budget-override",
        help="Allow a profile to raise graph budget defaults.",
    ),
    max_concurrency: int = typer.Option(
        DEFAULT_MAX_CONCURRENCY,
        "--max-concurrency",
        min=1,
        help="Maximum independent fan-out branches in flight at once.",
    ),
    join_timeout: float | None = typer.Option(
        None,
        "--join-timeout",
        help="Seconds a joiner may wait for declared inputs before failing (default: wall budget).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Compile and run a graph, writing a trace and run manifest."""
    try:
        response_cache = _response_cache(
            cache_mode,
            cache_directory if cache_directory is not None else default_cache_dir(),
            no_cache=no_cache,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    profile: Profile | None = None
    profile_path: Path | None = None
    prune_report: PruneReport | None = None
    try:
        from modelgraph.observability import build_logger_from_env, build_span_exporter

        log = build_logger_from_env()
        span_exporter = build_span_exporter()
        if profile_ref:
            profile_path = resolve_profile_path(profile_ref)
            profile = load_profile(profile_path)
            for warning in profile_security_warnings(profile):
                if not as_json:
                    typer.echo(f"warning: {warning}", err=True)
        result = compile_graph(
            graph,
            registry=_registry(registry_dir),
            strict_models=True,
            graph_paths=[graph.parent],
        )
        if not result.ok or result.graph is None:
            for message in result.errors:
                typer.echo(message, err=True)
            raise typer.Exit(code=1)
        compiled = result.graph
        if profile is not None:
            try:
                pruned = prune_graph(compiled, profile)
            except PruneError as exc:
                _print_prune_failure(exc, profile, profile_path, as_json=as_json)
                raise typer.Exit(code=3) from exc
            compiled = pruned.graph
            prune_report = pruned.report
            if not as_json:
                location = f" ({profile_path})" if profile_path else ""
                typer.echo(f"profile:  {profile.name}{location}")
                for line in format_prune_report(prune_report):
                    typer.echo(line)
        scenario = MockScenario()
        if mock_scenario:
            scenario = scenario_from_dict(load_document(mock_scenario))
        budget = _budget(compiled.defaults, None, None, None)
        if profile is not None:
            warnings = apply_budget_overrides(
                budget,
                profile.budget_overrides,
                defaults=compiled.defaults,
                allow_raise=allow_budget_override,
            )
            if not as_json:
                for warning in warnings:
                    typer.echo(f"warning: {warning}", err=True)
        # CLI budget flags are explicit user intent, so they apply after profile defaults
        # (a profile must not silently raise a tighter --max-steps/--max-tokens/--max-cost).
        budget = _budget(compiled.defaults, max_steps, max_tokens, max_cost, base=budget)
        loader = LazyModelLoader(_provider_factory(provider_name, scenario))
        outcome = run_graph(
            compiled,
            task,
            loader.provider_for,
            runs_dir=runs_directory,
            budget=budget,
            profile=profile.to_dict() if profile is not None else None,
            prune_report=prune_report.to_dict() if prune_report is not None else None,
            model_loads=loader.records,
            seed=seed,
            cache=response_cache,
            logger=log,
            span_exporter=span_exporter,
            max_concurrency=max_concurrency,
            join_timeout_seconds=join_timeout,
        )
        if outcome.run_dir is not None and outcome.manifest is not None:
            annotate_run_trace(outcome.run_dir, outcome.manifest)
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc

    if as_json:
        payload = outcome.to_dict()
        payload["manifest"] = str(outcome.run_dir / "run.json") if outcome.run_dir else None
        if profile is not None:
            payload["profile"] = profile.to_dict()
        if prune_report is not None:
            payload["prune"] = prune_report.to_dict()
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        typer.echo(f"run:      {outcome.run_id}")
        typer.echo(f"outcome:  {outcome.outcome}")
        if outcome.error:
            typer.echo(f"error:    [{outcome.error['code']}] {outcome.error['message']}")
        typer.echo(
            "usage:    "
            f"steps={outcome.usage.get('steps')} "
            f"tokens={outcome.usage.get('total_tokens')} "
            f"cost=${outcome.usage.get('cost_usd', 0):.6f} "
            f"elapsed={outcome.usage.get('elapsed_seconds')}s"
        )
        cache_info = outcome.cache
        typer.echo(
            "cache:    "
            f"mode={cache_info.get('mode', 'off')} "
            f"hits={cache_info.get('hits', 0)} "
            f"misses={cache_info.get('misses', 0)} "
            f"avoided_tokens={cache_info.get('avoided_tokens', 0)} "
            f"avoided_cost=${cache_info.get('avoided_cost_usd', 0.0):.6f}"
        )
        for branch in outcome.branches:
            if branch.get("kind") == "root":
                continue
            typer.echo(
                "branch:   "
                f"{branch.get('branch_id')} "
                f"outcome={branch.get('outcome')} "
                f"tokens={branch.get('total_tokens')} "
                f"cost=${branch.get('cost_usd', 0.0):.6f}"
                f"{' contributed' if branch.get('contributed') else ''}"
            )
        for entry in outcome.path:
            transition = f" -> {entry.to}" if entry.to else ""
            decision = f" {entry.decision}" if entry.decision else ""
            typer.echo(f"node:     {entry.node} ({entry.kind}){decision}{transition}")
        for artifact in outcome.artifacts:
            typer.echo(f"\n--- {artifact.name} ({artifact.type.value}) ---")
            typer.echo(artifact.content)
        if outcome.run_dir:
            typer.echo(f"\nrun dir:  {outcome.run_dir}")
    if not outcome.ok:
        raise typer.Exit(code=3)


def _emit_watch_report(report: dict[str, Any], *, as_json: bool, compact: bool) -> None:
    if as_json:
        typer.echo(
            json.dumps(
                report,
                indent=None if compact else 2,
                separators=(",", ":") if compact else None,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    for line in format_watch_report(report):
        typer.echo(line)


@app.command()
def watch(
    graph: Path = typer.Argument(..., help="Graph YAML file to validate, compile, and mock-run."),
    once: bool = typer.Option(
        False, "--once", help="Run one cycle and exit; non-zero on validation or run failure."
    ),
    interval: float = typer.Option(
        1.0, "--interval", min=0.01, help="Seconds between graph-file content checks."
    ),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    runs_directory: Path = typer.Option(
        default_runs_dir(), "--runs-dir", help="Directory for watch run artifacts."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable cycle result."),
) -> None:
    """Watch a graph and rerun it with the mock provider on every save.

    The watch loop never selects a real provider, never loads model weights, and adds no
    filesystem-watch dependency: it polls the graph file's content with the standard library.
    """
    session = WatchSession(
        graph,
        registry=_registry(registry_dir),
        runs_dir=runs_directory,
    )
    if once:
        report = session.poll_once()
        if report is None:  # pragma: no cover - the first poll always runs
            typer.echo("watch: no cycle was run", err=True)
            raise typer.Exit(code=1)
        _emit_watch_report(report, as_json=as_json, compact=False)
        if report["status"] != "ok":
            raise typer.Exit(code=1 if report["status"] == "invalid" else 3)
        return
    try:
        watch_loop(
            session,
            interval=interval,
            emit=lambda report: _emit_watch_report(report, as_json=as_json, compact=True),
        )
    except KeyboardInterrupt:
        if not as_json:
            typer.echo("\nwatch: stopped")


@app.command()
def replay(
    run: Path = typer.Argument(..., help="Run directory."),
    allow_ir_mismatch: bool = typer.Option(
        False, "--allow-ir-mismatch", help="Replay even if the graph hash changed."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Replay a recorded run without calling a provider."""
    try:
        result = replay_run(run, allow_ir_mismatch=allow_ir_mismatch)
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2))
    else:
        typer.echo(f"replay: {'OK' if result.ok else 'MISMATCH'}")
        typer.echo(f"run:    {result.run_id}")
        if result.outcome:
            typer.echo(f"outcome:{result.outcome.outcome}")
        for mismatch in result.mismatches:
            typer.echo(f"  - {mismatch}")
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Loopback interface to bind."),
    port: int = typer.Option(8765, "--port", help="Port to bind (0 picks a free port)."),
    open_browser: bool = typer.Option(False, "--open", help="Open the GUI in the default browser."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Never open a browser."),
    runs_directory: Path = typer.Option(
        default_runs_dir(), "--runs-dir", help="Directory for run artifacts."
    ),
) -> None:
    """Serve the local GUI (loopback only)."""
    from modelgraph.web import create_server, is_loopback_host

    if not is_loopback_host(host):
        typer.echo(f"refusing to bind {host!r}: mg ui is loopback-only", err=True)
        raise typer.Exit(code=2)
    if port < 0 or port > 65535:
        typer.echo(f"invalid port {port}", err=True)
        raise typer.Exit(code=2)
    try:
        server = create_server(host=host, port=port, runs_dir=runs_directory)
    except OSError as exc:
        typer.echo(f"cannot bind {host}:{port}: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    bound = cast(tuple[Any, ...], server.server_address)
    actual_host = str(bound[0])
    actual_port = int(bound[1])
    url_host = f"[{actual_host}]" if ":" in actual_host else actual_host
    url = f"http://{url_host}:{actual_port}/"
    typer.echo(f"Model Graph UI: {url}")
    typer.echo(f"runs dir:        {runs_directory}")
    if open_browser and not no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nstopping...")
    finally:
        server.server_close()


@app.command()
def inspect(
    run: Path = typer.Argument(..., help="Recorded run directory."),
    graph: Path = typer.Option(..., "--graph", help="Authored graph YAML (may have been pruned)."),
    out: Path = typer.Option(..., "--out", help="Output HTML file."),
    html: bool = typer.Option(
        False, "--html", help="Write a self-contained, offline HTML inspector."
    ),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    change_id: str | None = typer.Option(
        None, "--change", help="Change set id to show as a read-only impact panel."
    ),
    store_path: Path | None = typer.Option(
        None, "--store", help="Revision store directory for --change."
    ),
    compare_run: Path | None = typer.Option(
        None, "--compare", help="Second recorded run directory for a side-by-side comparison."
    ),
) -> None:
    """Write a static, self-contained HTML inspector for a recorded run.

    ``--change`` adds the stored impact report, hash-based stale badges, and revision
    timeline. ``--compare`` adds a side-by-side run comparison. Both are read-only: the
    inspector never promotes, rolls back, recompiles, or rewrites a revision or run.
    """
    if not html:
        typer.echo("error: mg inspect currently requires --html", err=True)
        raise typer.Exit(code=2)
    if store_path is not None and change_id is None:
        typer.echo("error: --store requires --change", err=True)
        raise typer.Exit(code=2)
    if not run.is_dir():
        typer.echo(f"run directory not found: {run}", err=True)
        raise typer.Exit(code=1)
    if compare_run is not None and not compare_run.is_dir():
        typer.echo(f"comparison run directory not found: {compare_run}", err=True)
        raise typer.Exit(code=1)
    try:
        manifest = read_run_manifest(run)
        if compare_run is not None:
            read_run_manifest(compare_run)
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=1) from exc

    result = compile_graph(
        graph,
        registry=_registry(registry_dir),
        strict_models=True,
        graph_paths=[graph.parent],
    )
    if not result.ok or result.graph is None:
        for message in result.errors:
            typer.echo(message, err=True)
        raise typer.Exit(code=1)

    if not out.parent.exists() or not out.parent.is_dir():
        typer.echo(f"output directory not found: {out.parent}", err=True)
        raise typer.Exit(code=2)
    if out.exists() and out.is_dir():
        typer.echo(f"output path is a directory: {out}", err=True)
        raise typer.Exit(code=2)

    graph_info = manifest.get("graph")
    graph_info = graph_info if isinstance(graph_info, dict) else {}
    if graph_info.get("hash") != result.graph.graph_hash:
        typer.echo(
            "note: authored graph hash differs from the recorded run; "
            "the inspector will show the prune report from the run manifest",
            err=True,
        )

    change: dict[str, Any] | None = None
    impact: dict[str, Any] | None = None
    timeline: dict[str, Any] | None = None
    if change_id is not None:
        store_root = store_path if store_path is not None else default_store_dir()
        if not Path(store_root).is_dir():
            typer.echo(
                f"revision store not found: {store_root}; run `mg status --revisions` or pass "
                "--store",
                err=True,
            )
            raise typer.Exit(code=1)
        try:
            # create=False keeps this inspection truly read-only: a missing store is an error
            # above, and opening an existing store must not add revisions/ or changes/.
            store = RevisionStore(store_root, create=False)
            change = store.load_change(change_id)
            impact = store.load_impact(change_id)
            timeline = revision_timeline(store, change=change)
        except ModelGraphError as exc:
            typer.echo(f"[{exc.code}] {exc.message}", err=True)
            raise typer.Exit(code=1) from exc

    try:
        model = build_inspection(
            run,
            result.graph,
            change=change,
            impact=impact,
            timeline=timeline,
            compare_run=compare_run,
        )
        rendered = render_inspector_html(model)
        out.write_text(rendered, encoding="utf-8")
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        typer.echo(f"cannot write inspector: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    totals = model["totals"]
    typer.echo(f"inspector: {out}")
    typer.echo(f"run:       {model['summary']['run_id']} ({model['summary']['outcome']})")
    typer.echo(
        f"content:   nodes={len(model['nodes'])} artifacts={len(model['artifacts'])} "
        f"tokens={totals['total_tokens']} cost=${totals['cost_usd']:.6f} "
        f"elapsed={totals['elapsed_seconds']:.3f}s"
    )
    iteration = model.get("iteration")
    if isinstance(iteration, dict):
        change_record = cast(dict[str, Any], iteration.get("change") or {})
        impact_record = cast(dict[str, Any], iteration.get("impact") or {})
        typer.echo(
            f"impact:    change={change_record.get('change_id')} "
            f"class={impact_record.get('change_class')} "
            f"status={change_record.get('status')}"
        )
    comparison = model.get("compare")
    if isinstance(comparison, dict):
        summary = cast(dict[str, Any], comparison.get("summary") or {})
        typer.echo(
            f"compare:   same_inputs={comparison.get('same_inputs')} "
            f"same_graph={comparison.get('same_graph')} "
            f"node_differences={summary.get('node_differences')} "
            f"artifact_differences={summary.get('artifact_differences')}"
        )


@trace_app.command("show")
def trace_show(
    run: Path = typer.Argument(..., help="Run directory."),
    as_json: bool = typer.Option(False, "--json", help="Emit all events as JSON."),
    limit: int = typer.Option(50, "--limit", help="Maximum events in human output."),
) -> None:
    """Show the events recorded for a run."""
    trace_path = run / "trace.jsonl"
    if not trace_path.exists():
        typer.echo(f"trace not found: {trace_path}", err=True)
        raise typer.Exit(code=1)
    events = read_events(trace_path)
    if as_json:
        typer.echo(json.dumps(events, indent=2, ensure_ascii=False))
        return
    for event in events[:limit]:
        detail = ""
        for key in ("node", "decision", "to", "code", "outcome", "name", "prompt_hash"):
            if key in event:
                detail += f" {key}={event[key]}"
        typer.echo(f"{event.get('ts', '')} {event.get('event', '')}{detail}")
    if len(events) > limit:
        typer.echo(f"... {len(events) - limit} more events (use --limit)")


@runs_app.command("list")
def runs_list(
    runs_directory: Path = typer.Option(default_runs_dir(), "--runs-dir"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List recorded runs."""
    entries: list[dict[str, Any]] = []
    if runs_directory.exists():
        # Inspecting the runs directory is also the recovery point for a crashed run: a
        # provisional manifest whose process is dead is marked interrupted here, so `runs list`
        # never keeps reporting a crashed run as "running" forever.
        recover_interrupted_runs(runs_directory)
        for run_dir in sorted(
            (path for path in runs_directory.iterdir() if (path / "run.json").exists()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            entries.append(
                {
                    "run_id": manifest.get("run_id"),
                    "outcome": manifest.get("outcome"),
                    "graph": manifest.get("graph", {}).get("id"),
                    "started_at": manifest.get("started_at"),
                    "tokens": manifest.get("usage", {}).get("total_tokens"),
                    "dir": str(run_dir),
                }
            )
    if as_json:
        typer.echo(json.dumps(entries, indent=2))
        return
    if not entries:
        typer.echo(f"no runs in {runs_directory}")
        return
    for entry in entries:
        typer.echo(
            f"{entry['run_id']}  {entry['outcome']:<12} {entry['graph']:<24} "
            f"tokens={entry['tokens']}  {entry['dir']}"
        )


@runs_app.command("show")
def runs_show(
    run: Path = typer.Argument(..., help="Run directory."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Show a run manifest, recovering a crashed provisional run first."""
    try:
        manifest = read_run_manifest(run, recover=True)
    except ModelGraphError as exc:
        message = (
            f"manifest not found: {run / 'run.json'}"
            if not (run / "run.json").exists()
            else exc.message
        )
        typer.echo(message, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(manifest, indent=2, ensure_ascii=False))


@models_app.command("list")
def models_list(
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List models in the local registry."""
    registry = _registry(registry_dir)
    rows = []
    for manifest in sorted(registry.manifests.values(), key=lambda item: item.metadata.id):
        rows.append(
            {
                "id": manifest.metadata.id,
                "version": manifest.metadata.version,
                "provider": manifest.spec.provider,
                "model": manifest.spec.model,
                "license": manifest.spec.license.name,
            }
        )
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        typer.echo("no models in the registry")
        return
    for row in rows:
        typer.echo(
            f"{row['id']:<24} {row['version']:<8} {row['provider']:<18} "
            f"{row['model']:<32} {row['license']}"
        )


# --------------------------------------------------------------------------------------
# Revision and change-set commands
# --------------------------------------------------------------------------------------


def _revision_store(path: Path | None) -> RevisionStore:
    return RevisionStore(path if path is not None else default_store_dir())


def _resolve_graph_ref(ref: str, store: RevisionStore, registry: Registry) -> CompiledGraph:
    if ref.startswith("rev_"):
        return store.compile_revision(ref)
    candidate = Path(ref).expanduser()
    source: Path | None = candidate if candidate.is_file() else registry.resolve_graph(ref)
    if source is None:
        raise ModelGraphError(
            f"graph {ref!r} is not a path, a registered graph ref, or a revision id in {store.root}"
        )
    result = compile_graph(
        source,
        registry=registry,
        strict_models=True,
        graph_paths=[source.parent],
    )
    if not result.ok or result.graph is None:
        raise ModelGraphError(
            "graph failed to compile: " + "; ".join(result.errors), path=str(source)
        )
    return result.graph


def _revision_target(ref: str, store: RevisionStore) -> str | None:
    """Read a revision's deployment target; graph paths have no target metadata."""
    if not ref.startswith("rev_"):
        return None
    value = store.load_revision(ref).get("target")
    return str(value) if value is not None else None


def _change_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "dry_run": result.get("dry_run"),
        "active_revision": result.get("active_revision"),
        "change": result.get("change"),
        "impact": result.get("impact"),
        "revision": result.get("revision"),
    }


@graph_app.command("diff")
def graph_diff(
    before: str = typer.Argument(..., help="Graph path, graph id@version, or revision id."),
    after: str = typer.Argument(..., help="Graph path, graph id@version, or revision id."),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    store_path: Path | None = typer.Option(None, "--store", help="Revision store directory."),
    before_target: str | None = typer.Option(
        None,
        "--before-target",
        help="Deployment target for the left side (defaults for revisions).",
    ),
    after_target: str | None = typer.Option(
        None,
        "--after-target",
        help="Deployment target for the right side (defaults for revisions).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable diff."),
) -> None:
    """Show topology, prompt, binding, policy, and deployment-target differences."""
    try:
        store = _revision_store(store_path)
        registry = _registry(registry_dir)
        before_graph = _resolve_graph_ref(before, store, registry)
        after_graph = _resolve_graph_ref(after, store, registry)
        diff = diff_graphs(
            before_graph,
            after_graph,
            before_target=before_target or _revision_target(before, store),
            after_target=after_target or _revision_target(after, store),
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    if as_json:
        typer.echo(json.dumps(diff, indent=2, ensure_ascii=False))
        return
    for line in format_diff(diff):
        typer.echo(line)


@swap_app.command("node")
def swap_node(
    node: str = typer.Argument(..., help="Node id to rebind."),
    to_model: str = typer.Option(
        ..., "--to", help="Model ref (id@version) or a model manifest path."
    ),
    graph_path: Path | None = typer.Option(
        None, "--graph", help="Base graph source (default: the active revision)."
    ),
    target: str = typer.Option("portable", "--target", help="Deployment target."),
    base_revision: str | None = typer.Option(
        None, "--base-revision", help="Explicit base revision (default: active)."
    ),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    profile_ref: str | None = typer.Option(
        None, "--profile", help="Profile for VRAM admission preview."
    ),
    store_path: Path | None = typer.Option(None, "--store", help="Revision store directory."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compile and explain the swap without persisting anything."
    ),
    summary: str | None = typer.Option(None, "--summary", help="Human summary for the change."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable result."),
) -> None:
    """Create a candidate revision that swaps one node's model.

    The active revision is never changed here. Promotion is a separate evidence-gated step.
    """
    try:
        store = _revision_store(store_path)
        result = propose_swap(
            store,
            node=node,
            to_model=to_model,
            graph_path=graph_path,
            target=target,
            base_revision=base_revision,
            registry=_registry(registry_dir),
            model_dirs=registry_dir,
            profile=profile_ref,
            dry_run=dry_run,
            summary=summary,
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    if as_json:
        typer.echo(json.dumps(_change_payload(result), indent=2, ensure_ascii=False))
        return
    if dry_run:
        typer.echo("dry run: no candidate revision or change set was persisted")
    for line in format_change_summary(result["change"]):
        typer.echo(line)
    typer.echo("")
    for line in format_impact(result["impact"]):
        typer.echo(line)


@change_app.command("create")
def change_create(
    graph_path: Path = typer.Option(..., "--graph", help="Candidate graph source file."),
    base_revision: str | None = typer.Option(
        None, "--base-revision", help="Base revision (default: active)."
    ),
    base_graph: Path | None = typer.Option(
        None, "--base-graph", help="Base graph source when no revision exists."
    ),
    target: str | None = typer.Option(
        None, "--target", help="Deployment target (default: base revision target or portable)."
    ),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    profile_ref: str | None = typer.Option(
        None, "--profile", help="Profile for VRAM admission preview."
    ),
    store_path: Path | None = typer.Option(None, "--store", help="Revision store directory."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compile and explain without persisting anything."
    ),
    summary: str | None = typer.Option(None, "--summary"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable result."),
) -> None:
    """Create a change set from an edited graph (prompt, topology, routing, or binding)."""
    try:
        store = _revision_store(store_path)
        result = propose_graph_change(
            store,
            candidate_graph_path=graph_path,
            base_revision=base_revision,
            graph_path=base_graph,
            target=target,
            registry=_registry(registry_dir),
            profile=profile_ref,
            dry_run=dry_run,
            summary=summary,
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    if as_json:
        typer.echo(json.dumps(_change_payload(result), indent=2, ensure_ascii=False))
        return
    if dry_run:
        typer.echo("dry run: no candidate revision or change set was persisted")
    for line in format_change_summary(result["change"]):
        typer.echo(line)
    typer.echo("")
    for line in format_impact(result["impact"]):
        typer.echo(line)


@change_app.command("list")
def change_list(
    store_path: Path | None = typer.Option(None, "--store", help="Revision store directory."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable list."),
) -> None:
    """List change sets in the local revision store."""
    try:
        store = _revision_store(store_path)
        changes = store.list_changes()
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    if as_json:
        typer.echo(json.dumps(changes, indent=2, ensure_ascii=False))
        return
    if not changes:
        typer.echo(f"no change sets in {store.root}")
        return
    for change in changes:
        typer.echo(
            f"{change.get('change_id')}  {str(change.get('status')):<12} "
            f"{str(change.get('change_class')):<14} {str(change.get('target')):<10} "
            f"{change.get('candidate_revision')}  {change.get('summary')}"
        )


@change_app.command("show")
def change_show(
    change_id: str = typer.Argument(..., help="Change set id."),
    store_path: Path | None = typer.Option(None, "--store", help="Revision store directory."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable change set."),
) -> None:
    """Show one change set, its impact report, and its evidence."""
    try:
        store = _revision_store(store_path)
        change = store.load_change(change_id)
        impact = store.load_impact(change_id)
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    if as_json:
        typer.echo(json.dumps({"change": change, "impact": impact}, indent=2, ensure_ascii=False))
        return
    for line in format_change_summary(change):
        typer.echo(line)
    typer.echo("")
    for line in format_impact(impact):
        typer.echo(line)
    evidence = change.get("evidence") or []
    if evidence:
        typer.echo("")
        typer.echo("Evidence")
        for item in evidence:
            typer.echo(
                f"  {item.get('level'):<8} valid={item.get('valid')} "
                f"gate={item.get('gate_passed')} {item.get('path')}"
            )


@change_app.command("explain")
def change_explain(
    change_id: str = typer.Argument(..., help="Change set id."),
    evidence: list[Path] = typer.Option(
        [], "--evidence", help="Candidate eval report to include in coverage (repeatable)."
    ),
    store_path: Path | None = typer.Option(None, "--store", help="Revision store directory."),
    profile_ref: str | None = typer.Option(
        None, "--profile", help="Profile for VRAM admission preview."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable explanation."),
) -> None:
    """Explain compatibility, invalidation, VRAM, evidence coverage, and rollback."""
    try:
        store = _revision_store(store_path)
        change = store.load_change(change_id)
        impact = store.load_impact(change_id)
        revision = store.load_revision(str(change["candidate_revision"]))
        records = {
            str(item.get("path")): dict(item)
            for item in cast(list[dict[str, Any]], change.get("evidence") or [])
        }
        for path in evidence:
            record = load_evidence(
                path,
                candidate_graph_hash=str(revision.get("graph", {}).get("hash")),
                candidate_graph_id=str(revision.get("graph", {}).get("id")),
            )
            records[str(record["path"])] = record
        evidence_records = list(records.values())
        base_revision = change.get("base_revision")
        if profile_ref and base_revision and store.revision_exists(str(base_revision)):
            # Recompute the report with the requested profile instead of overwriting one field
            # of a stale report. The stored change and impact files are not modified.
            loaded = load_profile(profile_ref)
            impact = analyse_impact(
                store.compile_revision(str(base_revision)),
                store.compile_revision(str(change["candidate_revision"])),
                target=str(change.get("target") or "portable"),
                base_target=cast(str | None, change.get("base_target")),
                base_revision=str(base_revision),
                candidate_revision=str(change["candidate_revision"]),
                profile=loaded,
                evidence=evidence_records,
                store=store,
            )
        required = cast(list[str], impact.get("eval", {}).get("required") or [])
        coverage = evaluate_coverage(required, evidence_records)
        impact.setdefault("eval", {})["coverage"] = coverage
        current = store.active_revision_id()
        promotion = cast(dict[str, Any], change.get("promotion") or {})
        rollback_target = promotion.get("rollback_target") or current
        rollback = impact.setdefault("rollback", {})
        rollback["target"] = rollback_target
        rollback["previous"] = rollback_target
        rollback["retained"] = bool(rollback_target and store.revision_exists(str(rollback_target)))
        impact["status"] = status_for_change(change, impact, coverage)
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    if as_json:
        typer.echo(
            json.dumps(
                {"change": change, "impact": impact, "evidence": evidence_records},
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    for line in format_change_summary(change):
        typer.echo(line)
    typer.echo("")
    for line in format_impact(impact):
        typer.echo(line)


@app.command()
def promote(
    revision: str | None = typer.Option(
        None, "--revision", help="Candidate deployment revision to promote."
    ),
    graph_path: Path | None = typer.Option(
        None, "--graph", help="Bootstrap an initial revision from this graph."
    ),
    target: str = typer.Option("portable", "--target", help="Target for --graph bootstrap."),
    evidence: list[Path] = typer.Option(
        [], "--evidence", help="Candidate eval report used as promotion evidence (repeatable)."
    ),
    registry_dir: list[Path] = typer.Option([], "--registry", help="Model registry directory."),
    store_path: Path | None = typer.Option(None, "--store", help="Revision store directory."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate evidence and show the switch without applying it."
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable result."),
) -> None:
    """Atomically promote an evidence-gated revision, or bootstrap the first revision."""
    if (revision is None) == (graph_path is None):
        raise typer.BadParameter("pass exactly one of --revision or --graph")
    try:
        store = _revision_store(store_path)
        if graph_path is not None:
            bootstrap = bootstrap_revision(
                store,
                graph_path=graph_path,
                target=target,
                registry=_registry(registry_dir),
                dry_run=dry_run,
            )
            if dry_run:
                if as_json:
                    typer.echo(json.dumps(bootstrap, indent=2, ensure_ascii=False))
                else:
                    typer.echo(f"dry run: would create and promote {bootstrap['revision_id']}")
                return
            revision_id = str(cast(dict[str, Any], bootstrap["revision"])["revision_id"])
            plan = promotion_plan(store, revision_id, evidence_paths=evidence)
            pointer = store.promote(
                revision_id, reason="mg promote --graph", evidence=plan["evidence"]
            )
            payload = {
                "dry_run": False,
                "bootstrap": True,
                "pointer": pointer,
                "revision": plan["revision"],
            }
        else:
            assert revision is not None
            plan = promotion_plan(
                store,
                revision,
                evidence_paths=evidence,
                persist_evidence=not dry_run,
            )
            if dry_run:
                payload = {
                    "dry_run": True,
                    "revision": plan["revision"],
                    "change": plan.get("change"),
                    "coverage": plan["coverage"],
                    "would_promote": True,
                }
            else:
                pointer = store.promote(revision, reason="mg promote", evidence=plan["evidence"])
                mark_promoted(store, plan, reason="mg promote")
                payload = {
                    "dry_run": False,
                    "pointer": pointer,
                    "revision": plan["revision"],
                    "change": plan.get("change"),
                    "coverage": plan["coverage"],
                }
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if payload.get("dry_run"):
        typer.echo(f"dry run: promotion of {payload['revision'].get('revision_id')} is allowed")
        coverage = cast(dict[str, Any], payload.get("coverage") or {})
        typer.echo(
            f"evidence: satisfied={coverage.get('satisfied')} missing={coverage.get('missing')}"
        )
        return
    pointer = cast(dict[str, Any], payload["pointer"])
    typer.echo(f"active:   {pointer.get('revision_id')}")
    typer.echo(f"previous: {pointer.get('previous_revision') or 'none'}")
    typer.echo(f"target:   {pointer.get('target')}")


@app.command()
def rollback(
    to: str = typer.Option(..., "--to", help="Retained revision to make active."),
    store_path: Path | None = typer.Option(None, "--store", help="Revision store directory."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the pointer switch without applying it."
    ),
    reason: str = typer.Option("mg rollback", "--reason", help="Audit reason for the rollback."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable result."),
) -> None:
    """Point active back at a retained revision; no recompile is performed."""
    try:
        store = _revision_store(store_path)
        revision = store.load_revision(to)
        current = store.active_revision_id()
        if current == to:
            raise ChangeSetError(f"revision {to!r} is already active")
        if dry_run:
            payload = {
                "dry_run": True,
                "current": current,
                "to": to,
                "target": revision.get("target"),
                "graph": revision.get("graph"),
            }
        else:
            pointer = store.rollback(to, reason=reason)
            payload = {"dry_run": False, "pointer": pointer}
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if dry_run:
        typer.echo(f"dry run: would switch active {current or 'none'} -> {to}")
        typer.echo(
            f"graph:   {revision.get('graph', {}).get('id')} target={revision.get('target')}"
        )
        return
    pointer = cast(dict[str, Any], payload["pointer"])
    typer.echo(f"active:   {pointer.get('revision_id')}")
    typer.echo(f"previous: {pointer.get('previous_revision') or 'none'}")
    typer.echo(f"target:   {pointer.get('target')}")


@app.command()
def status(
    revisions: bool = typer.Option(
        False, "--revisions", help="Include retained revisions and previous revisions."
    ),
    store_path: Path | None = typer.Option(None, "--store", help="Revision store directory."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable status."),
) -> None:
    """Show the active revision, open change sets, and revision retention."""
    try:
        store = _revision_store(store_path)
        report = status_report(store, include_revisions=revisions)
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc
    if as_json:
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
        return
    typer.echo(f"store:    {report['store']}")
    if report.get("active_revision"):
        graph = cast(dict[str, Any], report.get("graph") or {})
        typer.echo(
            f"active:   {report['active_revision']} ({graph.get('id')} target={report.get('target')})"
        )
        typer.echo(f"previous: {report.get('previous_revision') or 'none'}")
    else:
        typer.echo("active:   none (bootstrap with mg promote --graph <graph.yaml>)")
    changes = cast(list[dict[str, Any]], report.get("changes") or [])
    if changes:
        typer.echo("changes:")
        for change in changes:
            typer.echo(
                f"  {change.get('change_id')}  {str(change.get('status')):<12} "
                f"{change.get('change_class')}  {change.get('candidate_revision')}"
            )
    else:
        typer.echo("changes:  none")
    typer.echo(f"students: {report.get('distilled_student_status')} (not tracked in v0)")
    if revisions:
        retained = cast(list[dict[str, Any]], report.get("revisions") or [])
        typer.echo("revisions:")
        if not retained:
            typer.echo("  none")
        for item in retained:
            marker = "*" if item.get("active") else " "
            typer.echo(
                f" {marker}{item.get('revision_id')}  {str(item.get('target')):<9} "
                f"age={item.get('age_days')}d  {item.get('graph')}  models={','.join(item.get('models') or [])}"
            )
        previous = cast(list[str], report.get("previous_revisions") or [])
        if previous:
            typer.echo("previous revisions: " + ", ".join(previous))
    typer.echo(f"note:     {report.get('note')}")


def _response_cache(mode: str, directory: Path, *, no_cache: bool = False) -> ResponseCache | None:
    """Resolve ``--cache-mode``/``--no-cache`` into a cache or ``None`` (off)."""

    normalized = mode.strip().lower()
    if no_cache:
        if normalized == "readonly":
            raise ValueError("--no-cache conflicts with --cache-mode readonly")
        normalized = "off"
    if normalized not in CACHE_MODES:
        raise ValueError(f"unknown --cache-mode {mode!r}; expected one of {', '.join(CACHE_MODES)}")
    if normalized == "off":
        return None
    return ResponseCache(directory, mode=normalized)


def _provider_factory(
    provider_name: str, scenario: MockScenario
) -> Callable[[ModelManifest], Provider]:
    def factory(manifest: ModelManifest) -> Provider:
        if provider_name == "mock":
            return MockProvider(scenario, max_context=manifest.spec.context_window)
        if provider_name == "auto":
            return build_provider(manifest, scenario=scenario)
        if provider_name not in {"ollama", "openai-compatible", "llamacpp"}:
            raise ModelGraphError(f"unknown provider {provider_name!r}")
        spec = manifest.spec.model_copy(update={"provider": provider_name})
        forced = manifest.model_copy(update={"spec": spec})
        return build_provider(forced)

    return factory


def _print_prune_failure(
    exc: PruneError, profile: Profile, profile_path: Path | None, *, as_json: bool
) -> None:
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "outcome": "failed",
                    "profile": profile.to_dict(),
                    "prune": exc.report.to_dict(),
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                        "details": exc.details,
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    location = f" ({profile_path})" if profile_path else ""
    typer.echo(f"profile:  {profile.name}{location}")
    for line in format_prune_report(exc.report):
        typer.echo(line)
    typer.echo(f"[{exc.code}] {exc.message}", err=True)


def _budget(
    defaults: Defaults,
    max_steps: int | None,
    max_tokens: int | None,
    max_cost: float | None,
    *,
    base: Budget | None = None,
) -> Budget:
    budget = base if base is not None else Budget.from_defaults(defaults)
    if max_steps is not None:
        budget.max_steps = max_steps
    if max_tokens is not None:
        budget.max_total_tokens = max_tokens
    if max_cost is not None:
        budget.max_cost_usd = max_cost
    return budget


@app.command()
def stats(
    runs_directory: Path = typer.Option(
        default_runs_dir(), "--runs-dir", help="Directory containing recorded runs."
    ),
    last: int = typer.Option(20, "--last", help="Summarize the N most recent runs."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable summary."),
) -> None:
    """Summarize recent recorded runs from their traces, without calling a provider."""
    if last < 1:
        typer.echo("error: --last must be at least 1", err=True)
        raise typer.Exit(code=2)
    # Imported here so the stats command stays a read-only projection and cannot pull in a
    # provider or registry construction path.
    from modelgraph.observability import collect_stats

    try:
        report = collect_stats(runs_directory, last=last)
    except OSError as exc:
        typer.echo(f"error: cannot read runs in {runs_directory}: {exc}", err=True)
        raise typer.Exit(code=3) from exc
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return
    typer.echo(f"runs dir: {report.runs_dir}")
    if report.available_runs == 0:
        typer.echo("no runs recorded")
        return
    snapshot = report.metrics
    typer.echo(f"runs:     {report.selected_runs} of {report.available_runs} (last {last})")
    outcomes = (
        " ".join(f"{name}={count}" for name, count in sorted(snapshot.outcomes.items())) or "none"
    )
    typer.echo(f"outcomes: {outcomes}")
    success = report.success_rate
    typer.echo(f"success:  {f'{success * 100:.1f}%' if success is not None else '—'}")
    typer.echo(
        f"tokens:   input={snapshot.total_input_tokens} "
        f"output={snapshot.total_output_tokens} total={snapshot.total_tokens}"
    )
    typer.echo(f"cost:     ${snapshot.total_cost_usd:.6f}")
    latency = snapshot.latency
    typer.echo(
        f"latency:  p50={_format_number(latency.get('p50_seconds'), 3)}s "
        f"p95={_format_number(latency.get('p95_seconds'), 3)}s (n={latency.get('count', 0)})"
    )
    typer.echo(
        f"cache:    hits={snapshot.cache.get('hits', 0)} misses={snapshot.cache.get('misses', 0)}"
    )
    if report.top_failure_codes:
        typer.echo(
            "failures: "
            + ", ".join(f"{entry['code']}={entry['count']}" for entry in report.top_failure_codes)
        )
    else:
        typer.echo("failures: none")


if __name__ == "__main__":
    app()
