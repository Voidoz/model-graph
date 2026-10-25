"""Command line interface for Model Graph."""

from __future__ import annotations

import json
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import typer

from modelgraph import __version__
from modelgraph.compiler import compile_graph
from modelgraph.contracts.graph import Defaults
from modelgraph.contracts.loader import load_document
from modelgraph.contracts.manifest import ModelManifest
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
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.engine import run_graph
from modelgraph.runtime.replay import replay_run
from modelgraph.runtime.trace import read_events

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
app.add_typer(trace_app, name="trace")
app.add_typer(runs_app, name="runs")
app.add_typer(models_app, name="models")
app.add_typer(eval_app, name="eval")

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
    if as_json:
        payload = report.to_dict()
        if report.report_dir is not None:
            payload["report_dir"] = str(report.report_dir)
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_eval_report(report)
        for line in gate_summary(report):
            if line.startswith(("gate:", "report:")):
                continue
            typer.echo(line)
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
    typer.echo(f"failures: {len(report.failures)}")
    for failure in report.failures[:5]:
        typer.echo(f"  - {failure.get('case_id')}: {failure.get('reason')}")
    if len(report.failures) > 5:
        typer.echo(f"  ... {len(report.failures) - 5} more")


def _format_rate(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def _format_number(value: Any, digits: int) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def _registry(model_dirs: list[Path]) -> Registry:
    if model_dirs:
        return Registry.from_paths(list(model_dirs))
    return default_registry()


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
    profile_ref: str | None = typer.Option(
        None, "--profile", help="Run profile file or profile name."
    ),
    allow_budget_override: bool = typer.Option(
        False,
        "--allow-budget-override",
        help="Allow a profile to raise graph budget defaults.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Compile and run a graph, writing a trace and run manifest."""
    profile: Profile | None = None
    profile_path: Path | None = None
    prune_report: PruneReport | None = None
    try:
        if profile_ref:
            profile_path = resolve_profile_path(profile_ref)
            profile = load_profile(profile_path)
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
        )
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
    """Show a run manifest."""
    manifest_path = run / "run.json"
    if not manifest_path.exists():
        typer.echo(f"manifest not found: {manifest_path}", err=True)
        raise typer.Exit(code=1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    app()
