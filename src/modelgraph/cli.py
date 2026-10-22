"""Command line interface for Model Graph."""

from __future__ import annotations

import json
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
from modelgraph.paths import runs_dir as default_runs_dir
from modelgraph.providers import build_provider
from modelgraph.providers.base import Provider
from modelgraph.providers.mock import MockProvider, MockScenario, scenario_from_dict
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
app.add_typer(trace_app, name="trace")
app.add_typer(runs_app, name="runs")
app.add_typer(models_app, name="models")

DEMO_TASK = "Write a debounce function in TypeScript."


@app.callback()
def main() -> None:
    """Model Graph command group."""


@app.command()
def version() -> None:
    """Print the installed Model Graph version."""
    typer.echo(__version__)


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
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Compile and run a graph, writing a trace and run manifest."""
    try:
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
        scenario = MockScenario()
        if mock_scenario:
            scenario = scenario_from_dict(load_document(mock_scenario))
        budget = _budget(result.graph.defaults, max_steps, max_tokens, max_cost)
        outcome = run_graph(
            result.graph,
            task,
            _provider_factory(provider_name, scenario),
            runs_dir=runs_directory,
            budget=budget,
        )
    except ModelGraphError as exc:
        typer.echo(f"[{exc.code}] {exc.message}", err=True)
        raise typer.Exit(code=3) from exc

    if as_json:
        payload = outcome.to_dict()
        payload["manifest"] = str(outcome.run_dir / "run.json") if outcome.run_dir else None
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


def _budget(
    defaults: Defaults, max_steps: int | None, max_tokens: int | None, max_cost: float | None
) -> Budget:
    budget = Budget.from_defaults(defaults)
    if max_steps is not None:
        budget.max_steps = max_steps
    if max_tokens is not None:
        budget.max_total_tokens = max_tokens
    if max_cost is not None:
        budget.max_cost_usd = max_cost
    return budget


if __name__ == "__main__":
    app()
