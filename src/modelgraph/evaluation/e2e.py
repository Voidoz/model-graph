"""End-to-end evaluation over a compiled graph and an optional run profile.

E2E evaluation runs the real runtime engine (compiler, renderer, executors, budgets, traces)
against the configured provider. The deterministic mock is the offline default; live
providers are gated by ``MG_EVAL_LIVE=1`` or ``allow_live=True``.

Baselines are measurements, not synthetic formulas:

* best single worker node (same suite cases, same checks)
* cheapest capable worker node
* oracle routing over the first router's candidate branches
* deterministic random routing over those branches
* an incumbent report supplied with ``baseline=``

Oracle labels are produced by forcing the first router through each candidate branch and
grading the same exit checks. For a live provider the forcing wrapper answers only the router
call; workers still run against the real model. Ambiguous labels are stored and excluded from
the oracle gap.
"""

from __future__ import annotations

import json
import random
import statistics
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelgraph.compiler import compile_graph
from modelgraph.compiler.ir import CompiledGraph, CompiledNode
from modelgraph.contracts.graph import NodeKind
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.errors import ModelGraphError
from modelgraph.evaluation.checks import JudgeHook, JudgeRequest, evaluate_checks
from modelgraph.evaluation.errors import EvalError, SuiteError
from modelgraph.evaluation.metrics import (
    bootstrap_ci,
    latency_summary,
    mean_or_none,
    percentile,
    safe_div,
    wilson_interval,
)
from modelgraph.evaluation.providers import (
    build_live_provider,
    normalize_mock_script,
    provider_summary,
    require_live,
)
from modelgraph.evaluation.report import (
    EvalReport,
    build_primary_metric,
    default_report_dir,
    determinism_fingerprint,
    environment_info,
    new_report_id,
)
from modelgraph.evaluation.suite import E2ETarget, EvalCase, EvalSuite, load_suite
from modelgraph.evaluation.support import (
    artifact_record,
    build_state,
    case_budget,
    case_fingerprint,
    commands_for,
    costs_summary,
    graph_summary,
    limitations_for,
    node_validity_issues,
    protocol_errors_from_issues,
    protocol_summary,
    provider_calls,
    provider_responses,
    render_hashes,
    usage_snapshot,
    usage_totals,
)
from modelgraph.paths import project_root
from modelgraph.providers.base import (
    Capabilities,
    ChatRequest,
    ChatResponse,
    Health,
    Provider,
    Usage,
)
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.pruning import (
    Profile,
    PruneReport,
    apply_budget_overrides,
    load_profile,
    prune_graph,
    resolve_profile_path,
)
from modelgraph.registry import Registry, default_registry
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.engine import RunOutcome, run_graph
from modelgraph.runtime.executors import Executor, WorkerExecutor
from modelgraph.runtime.trace import TraceRecorder, read_events
from modelgraph.util import estimate_tokens, new_id, sha256_text, utc_now

ProviderFactory = Callable[[ModelManifest], Provider]


@dataclass(frozen=True)
class E2EConfiguration:
    """An additional graph/profile configuration for the Pareto table."""

    name: str
    graph: str
    profile: str | None = None


@dataclass
class _BaselineSet:
    name: str
    kind: str
    cases: list[dict[str, Any]]
    summary: dict[str, Any]
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class _OracleMeasurement:
    available: bool
    labels: list[dict[str, Any]] = field(default_factory=list)
    cases: list[dict[str, Any]] = field(default_factory=list)
    branches: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    labeler: str | None = None
    measured_cost_usd: float = 0.0


class _ForcedRouteProvider:
    """Answer one router call with a fixed decision, delegating everything else."""

    name = "forced-route"

    def __init__(self, delegate: Provider, router_id: str, route: str) -> None:
        self._delegate = delegate
        self._router_id = router_id
        self._route = route
        self.name = str(getattr(delegate, "name", "forced-route"))

    def capabilities(self) -> Capabilities:
        return self._delegate.capabilities()

    def health(self) -> Health:
        return self._delegate.health()

    def chat(self, request: ChatRequest) -> ChatResponse:
        if request.node_id == self._router_id:
            text = json.dumps(
                {
                    "decision": "route",
                    "to": self._route,
                    "confidence": 1.0,
                    "reason": "evaluation oracle branch probe",
                }
            )
            return ChatResponse(
                text=text,
                usage=Usage(
                    input_tokens=estimate_tokens(request.prompt_text()),
                    output_tokens=estimate_tokens(text),
                    approximated=True,
                ),
                finish_reason="stop",
                provider_meta={"forced_route": self._route},
            )
        return self._delegate.chat(request)


def run_e2e_suite(
    suite: EvalSuite | str | Path,
    *,
    graph_path: str | Path | None = None,
    profile: str | Path | Profile | None = None,
    provider: str | None = None,
    allow_live: bool = False,
    registry: Registry | None = None,
    judge: JudgeHook | None = None,
    reports_dir: str | Path | None = None,
    case_ids: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
    baseline: EvalReport | dict[str, Any] | str | Path | None = None,
    command: str | None = None,
    persist: bool = True,
    allow_commands: bool = True,
    allow_budget_override: bool = False,
    bootstrap_resamples: int = 2000,
    seed: int = 1234,
    configurations: list[E2EConfiguration] | None = None,
) -> EvalReport:
    """Run one end-to-end suite and return (and optionally persist) its report."""
    loaded = suite if isinstance(suite, EvalSuite) else load_suite(suite)
    if loaded.level != "e2e":
        raise SuiteError(
            f"suite {loaded.metadata.id!r} is a {loaded.level!r} suite; e2e evaluation "
            "requires level: e2e"
        )
    if not isinstance(loaded.spec.target, E2ETarget):
        raise SuiteError(f"suite {loaded.metadata.id!r} has no e2e target")

    provider_name = provider or loaded.provider
    require_live(provider_name, allow_live=allow_live)
    registry = registry or default_registry()
    report_root = Path(reports_dir) if reports_dir is not None else default_report_dir()
    report_id = new_report_id()
    runtime_root = report_root / report_id / "runs"

    graph_file = (
        Path(graph_path).expanduser().resolve()
        if graph_path is not None
        else loaded.resolve_graph_path()
    )
    if not graph_file.is_file():
        raise SuiteError(f"target graph not found: {graph_file}", path=str(graph_file))
    compiled = compile_graph(
        graph_file,
        registry=registry,
        strict_models=True,
        graph_paths=[graph_file.parent],
    )
    if not compiled.ok or compiled.graph is None:
        raise SuiteError(
            f"graph {graph_file} failed to compile: " + "; ".join(compiled.errors),
            path=str(graph_file),
        )
    graph = compiled.graph
    _validate_mock_nodes(loaded, graph)

    profile_obj, profile_path = _resolve_profile(profile, loaded)
    prune_report: PruneReport | None = None
    if profile_obj is not None:
        pruned = prune_graph(graph, profile_obj)
        graph = pruned.graph
        prune_report = pruned.report

    selected = loaded.select_cases(case_ids=case_ids, tags=tags, limit=limit)
    selection = loaded.selection_summary(list(selected), case_ids=case_ids, tags=tags, limit=limit)
    report_root_path = report_root
    cli_command, commands, reproduce = commands_for(
        level="e2e",
        suite_path=loaded.source_path,
        graph_path=graph_file,
        provider=provider_name,
        reports_dir=report_root_path,
        case_ids=case_ids,
        tags=tags,
        limit=limit,
        profile=str(profile_path) if profile_path is not None else None,
    )
    if command is not None:
        cli_command = command
        commands = [cli_command, reproduce]

    warnings: list[str] = []
    cases: list[dict[str, Any]] = []
    total_cost = 0.0
    total_tokens = 0
    stopped: str | None = None
    budget_config = loaded.spec.budget
    for case in selected:
        if budget_config.max_cost_usd is not None and total_cost > budget_config.max_cost_usd:
            stopped = (
                f"suite cost budget exceeded (${total_cost:.6f} > "
                f"${budget_config.max_cost_usd:.6f})"
            )
            break
        if budget_config.max_tokens is not None and total_tokens > budget_config.max_tokens:
            stopped = f"suite token budget exceeded ({total_tokens} > {budget_config.max_tokens})"
            break
        budget = _case_budget(
            compiled.graph,
            loaded,
            profile_obj,
            allow_budget_override=allow_budget_override,
            warnings=warnings,
        )
        result = _run_graph_case(
            case,
            graph,
            loaded,
            provider_kind=provider_name,
            scenario=_scenario(case),
            forced_route=None,
            budget=budget,
            allow_commands=allow_commands,
            judge=judge,
            persist=persist,
            runs_dir=runtime_root,
            profile_payload=profile_obj.to_dict() if profile_obj is not None else None,
            prune_payload=prune_report.to_dict() if prune_report is not None else None,
        )
        cases.append(result)
        total_cost += float(result["cost_usd"])
        total_tokens += int(result["usage"]["total_tokens"])
        if budget_config.max_cost_usd is not None and total_cost > budget_config.max_cost_usd:
            stopped = (
                f"suite cost budget exceeded (${total_cost:.6f} > "
                f"${budget_config.max_cost_usd:.6f})"
            )
            break
        if budget_config.max_tokens is not None and total_tokens > budget_config.max_tokens:
            stopped = f"suite token budget exceeded ({total_tokens} > {budget_config.max_tokens})"
            break
    if not cases:
        raise EvalError("suite budget stopped before the first case ran")

    metrics = _metrics(cases, seed=seed, resamples=bootstrap_resamples)
    costs = costs_summary(cases)
    costs["per_task_usd_ci95"] = bootstrap_ci(
        [float(case["cost_usd"]) for case in cases],
        resamples=bootstrap_resamples,
        seed=seed,
    )
    latencies = [float(case["latency_seconds"]) for case in cases]
    latency = latency_summary(latencies)
    latency["p95_ci95"] = bootstrap_ci(
        latencies,
        lambda values: percentile(values, 0.95),
        resamples=bootstrap_resamples,
        seed=seed,
    )
    usage = usage_totals(cases)
    protocol = protocol_summary(cases)
    slices = _slices(cases)
    failure_slices = [
        {"name": name, **entry}
        for name, entry in slices.items()
        if entry.get("failure_count", 0) > 0
    ]
    failure_slices.sort(key=lambda entry: (float(entry.get("task_success") or 0.0), entry["name"]))

    baseline_config = loaded.spec.baselines
    best_single, cheapest_capable = _single_node_baselines(
        graph,
        selected,
        loaded,
        provider_kind=provider_name,
        profile=profile_obj,
        allow_budget_override=allow_budget_override,
        warnings=warnings,
        allow_commands=allow_commands,
        judge=judge,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    oracle = _measure_oracle(
        graph,
        selected,
        cases,
        loaded,
        provider_kind=provider_name,
        profile=profile_obj,
        allow_budget_override=allow_budget_override,
        warnings=warnings,
        allow_commands=allow_commands,
        judge=judge,
    )
    random_routing = _random_baseline(
        oracle, seed=seed, random_seed=loaded.spec.baselines.random_seed
    )
    oracle_summary = _oracle_summary(loaded, oracle)
    oracle_baseline = _oracle_set(oracle, oracle_summary)
    _attach_oracle_labels(cases, oracle.labels)
    failures = _failures(cases)
    if stopped:
        failures.append({"case_id": "$suite", "reason": stopped, "error": "MG-4000"})

    comparisons: list[dict[str, Any]] = []
    for name, baseline_set in (
        ("best_single", best_single),
        ("cheapest_capable", cheapest_capable),
        ("oracle_routing", oracle_baseline),
        ("random_routing", random_routing),
    ):
        if baseline_set is None or not baseline_set.cases:
            continue
        comparison = _paired_comparison(
            name,
            cases,
            baseline_set.cases,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        comparisons.append(comparison)

    pareto_entries = [_configuration_entry("graph", "graph", metrics)]
    for baseline_set in (best_single, cheapest_capable, oracle_baseline, random_routing):
        if baseline_set is not None and baseline_set.summary.get("available", True):
            pareto_entries.append(
                _configuration_entry(
                    baseline_set.name,
                    baseline_set.kind,
                    baseline_set.summary,
                    extra=baseline_set.info,
                )
            )
    extra_configurations: list[dict[str, Any]] = []
    for extra in configurations or []:
        entry = _evaluate_configuration(
            extra,
            loaded,
            selected,
            provider_kind=provider_name,
            allow_budget_override=allow_budget_override,
            allow_commands=allow_commands,
            judge=judge,
            registry=registry,
        )
        extra_configurations.append(entry)
        if entry.get("available", True):
            pareto_entries.append(entry)

    baseline_payload, baseline_source = _load_baseline(baseline)
    incumbent_entry: dict[str, Any] | None = None
    if baseline_payload is not None:
        incumbent_entry = _incumbent_summary(baseline_payload, baseline_source)
        if incumbent_entry.get("available", True):
            pareto_entries.append(
                _configuration_entry(
                    incumbent_entry["name"],
                    "incumbent",
                    incumbent_entry,
                    extra={"report_id": incumbent_entry.get("report_id")},
                )
            )

    pareto = _apply_pareto(pareto_entries)
    if persist and oracle.labels:
        oracle_summary["labels_file"] = "oracle-labels.json"
    run_manifests = {
        case["case_id"]: {
            "path": case.get("run_manifest"),
            "hash": case.get("run_manifest_hash"),
            "run_id": case.get("run_id"),
        }
        for case in cases
        if case.get("run_manifest")
    }
    judge_summary = _judge_summary(loaded, cases, baseline_payload, judge)
    e2e_data: dict[str, Any] = {
        "profile": profile_obj.to_dict() if profile_obj is not None else None,
        "profile_path": str(profile_path) if profile_path is not None else None,
        "prune": prune_report.to_dict() if prune_report is not None else None,
        "baselines": {
            "best_single": best_single.summary if best_single is not None else {"available": False},
            "cheapest_capable": (
                cheapest_capable.summary if cheapest_capable is not None else {"available": False}
            ),
            "oracle_routing": (
                oracle_baseline.summary if oracle_baseline is not None else {"available": False}
            ),
            "random_routing": (
                random_routing.summary if random_routing is not None else {"available": False}
            ),
            "incumbent": incumbent_entry if incumbent_entry is not None else {"available": False},
        },
        "baseline_notes": baseline_config.notes,
        "oracle": oracle_summary,
        "failure_slices": failure_slices,
        "pareto": pareto,
        "configurations": pareto["configurations"],
        "extra_configurations": extra_configurations,
        "run_manifests": run_manifests,
        "judge": judge_summary,
        "input_artifacts_supported": False,
        "baseline_config": baseline_config.model_dump(mode="json", exclude_none=True),
    }
    # Replace the sentinel wrapper used above with a plain mapping before serialization.
    if incumbent_entry is not None:
        e2e_data["baselines"]["incumbent"] = incumbent_entry

    limitations = limitations_for(
        level="e2e",
        provider=provider_name,
        case_count=len(cases),
        oracle_labeling_method=(
            loaded.spec.oracle.labeling_method if loaded.spec.oracle is not None else "measured"
        ),
        command_checks=any(check.type == "command" for case in selected for check in case.checks),
        judge_errors=[str(case["judge_error"]) for case in cases if case.get("judge_error")],
    )
    if any(case.input.artifacts for case in selected):
        limitations.append(
            "E2E v0 runs the graph with the task text only; suite input artifacts are recorded "
            "but the current run_graph API does not inject them."
        )
    limitations.append(
        "Best-single and cheapest-capable baselines run each worker node directly on the full "
        "task. They exclude routing and handoff, which is the point of the comparison, but they "
        "also ignore the graph's node view and any intermediate artifacts."
    )
    limitations.append(
        "Oracle labels force only the first router's branch. Nested routing is measured through "
        "the case's authored mock scripts or live provider behavior, not exhaustively labeled."
    )
    limitations.append(
        "Oracle labeling runs every selected case; stratification is the suite author's "
        "selection (tags/limit), not an automatic sample design."
    )
    limitations.append(
        "Costs come from manifest pricing. Free local model manifests report zero marginal cost, "
        "so a cost band can collapse to a single $0 point."
    )
    if oracle.labels and oracle.labeler == "measured":
        limitations.append(
            "Measured oracle labels are produced by the same checks used for the graph, so a "
            "broken check under-labels every branch equally."
        )
    if judge is None and loaded.spec.judge.enabled:
        limitations.append(
            "The suite configures a judge but no judge hook was supplied; rubric checks fail closed."
        )
    if judge is not None and not _human_calibration_labels(cases):
        limitations.append("Judge scores are not calibrated against human labels for this suite.")
    if baseline_payload is None:
        limitations.append(
            "No incumbent report was supplied, so the incumbent baseline is unavailable."
        )
    if stopped:
        limitations.append(stopped)
    limitations.extend(loaded.warnings)
    for warning in dict.fromkeys(warnings):
        if warning not in limitations:
            limitations.append(warning)
    if any(not entry.get("available", True) for entry in extra_configurations):
        limitations.append(
            "One or more extra Pareto configurations failed to compile or run; see "
            "e2e.extra_configurations for the error."
        )

    primary, thresholds, gate = build_primary_metric(
        loaded.spec.metric, metrics, loaded.spec.thresholds
    )
    if stopped:
        gate["passed"] = False
        gate.setdefault("failures", []).append("budget")
        gate["budget_stopped"] = stopped
    determinism_payload = {
        "suite_hash": loaded.content_hash,
        "graph_hash": graph.graph_hash,
        "level": "e2e",
        "provider": provider_name,
        "selected_case_ids": [case["case_id"] for case in cases],
        "metrics": {key: value for key, value in metrics.items() if not key.startswith("latency")},
        "cases": [case_fingerprint(case) for case in cases],
        "baselines": {
            name: {
                key: value
                for key, value in summary.items()
                if not key.startswith("latency")
                and (isinstance(value, (str, int, float, bool)) or value is None)
            }
            for name, summary in e2e_data["baselines"].items()
            if isinstance(summary, dict)
        },
    }
    determinism = {
        "scripted_provider": provider_name == "mock",
        "fingerprint": determinism_fingerprint(determinism_payload),
        "excludes": [
            "generated_at",
            "report_id",
            "latency_seconds",
            "latency_*_seconds",
            "report_dir",
            "run_id",
            "run_manifest",
            "run_manifests",
        ],
    }
    target = _target(loaded, graph, profile_obj, prune_report, selection)
    report = EvalReport(
        report_id=report_id,
        level="e2e",
        claim=loaded.spec.claim or loaded.metadata.description,
        suite=loaded.summary(),
        graph=graph_summary(graph, graph_file),
        target=target,
        provider=provider_summary(provider_name, loaded),
        command=cli_command,
        commands=commands,
        reproduce=reproduce,
        environment=environment_info(),
        selection={**selection, "baseline_cases": _baseline_case_count(e2e_data)},
        budget={
            "suite": loaded.spec.budget.model_dump(mode="json", exclude_none=True),
            "runtime_defaults": graph.defaults.model_dump(mode="json"),
            "profile": profile_obj.to_dict() if profile_obj is not None else None,
            "profile_path": str(profile_path) if profile_path is not None else None,
            "stopped": stopped is not None,
            "stop_reason": stopped,
        },
        metrics=metrics,
        primary_metric=primary,
        thresholds=thresholds,
        gate=gate,
        cases=cases,
        failures=failures,
        slices=slices,
        protocol_errors=protocol,
        costs=costs,
        latency=latency,
        usage=usage,
        determinism=determinism,
        limitations=limitations,
        oracle=oracle_summary if oracle_summary.get("labeled_cases") else None,
        judge=judge_summary if judge_summary.get("configured") else None,
        comparisons=comparisons,
        e2e=e2e_data,
    )

    if baseline_payload is not None:
        report, incumbent_comparison = _compare_incumbent(
            report,
            baseline_payload,
            baseline_source,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        if incumbent_comparison is not None:
            report.e2e = e2e_data
            report.e2e["baselines"]["incumbent"] = _incumbent_result(
                incumbent_entry or {}, incumbent_comparison, baseline_source
            )
            pareto = _apply_pareto(pareto_entries)
            report.e2e["pareto"] = pareto
            report.e2e["configurations"] = pareto["configurations"]
            report.comparisons = [
                *comparisons,
                _comparison_entry("incumbent", incumbent_comparison),
            ]
        else:
            report.e2e = e2e_data
            report.e2e["baselines"]["incumbent"] = {
                "available": False,
                "name": "incumbent",
                "kind": "incumbent",
                "report_path": str(baseline_source) if baseline_source is not None else None,
            }
            pareto_entries = [entry for entry in pareto_entries if entry.get("kind") != "incumbent"]
            pareto = _apply_pareto(pareto_entries)
            report.e2e["pareto"] = pareto
            report.e2e["configurations"] = pareto["configurations"]

    if persist:
        report.write(report_root)
        labels = oracle.labels
        if labels and report.report_dir is not None:
            report_dir = report.report_dir
            (report_dir / "oracle-labels.json").write_text(
                json.dumps(
                    {
                        "schema": "modelgraph/oracle-labels/v0",
                        "suite_hash": loaded.content_hash,
                        "graph_hash": graph.graph_hash,
                        "labeler": oracle.labeler,
                        "labels": labels,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            if report.e2e is not None:
                report.e2e["oracle"]["labels_file"] = "oracle-labels.json"
        if report.report_dir is not None:
            (report.report_dir / "e2e.json").write_text(
                json.dumps(report.e2e, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    return report


# ---------------------------------------------------------------------------
# Case execution


def _validate_mock_nodes(suite: EvalSuite, graph: CompiledGraph) -> None:
    known = _node_ids(graph)
    for case in suite.cases:
        missing = sorted(set(case.mock_nodes) - known)
        if missing:
            raise SuiteError(
                f"e2e case {case.id!r} scripts unknown node(s): {missing}",
                details={"known_nodes": sorted(known)},
            )


def _node_ids(graph: CompiledGraph, seen: set[str] | None = None) -> set[str]:
    seen = seen if seen is not None else set()
    if graph.id in seen:
        return set()
    seen.add(graph.id)
    ids = set(graph.nodes)
    for node in graph.nodes.values():
        if node.kind == NodeKind.subgraph:
            inner = graph.subgraphs.get(node.graph_ref or "")
            if inner is not None:
                ids.update(_node_ids(inner, seen))
    return ids


def _scenario(case: EvalCase) -> MockScenario:
    return MockScenario(
        responses={
            node_id: normalize_mock_script(script) for node_id, script in case.mock_nodes.items()
        }
    )


def _resolve_profile(
    value: str | Path | Profile | None, suite: EvalSuite
) -> tuple[Profile | None, Path | None]:
    ref: str | Path | Profile | None = value
    target = suite.spec.target
    if ref is None and isinstance(target, E2ETarget) and target.profile:
        raw = Path(target.profile)
        local = (suite.directory / raw).resolve()
        ref = local if local.is_file() else target.profile
    if ref is None:
        return None, None
    if isinstance(ref, Profile):
        return ref, None
    path = resolve_profile_path(ref)
    return load_profile(path), path


def _case_budget(
    graph: CompiledGraph,
    suite: EvalSuite,
    profile: Profile | None,
    *,
    allow_budget_override: bool,
    warnings: list[str],
) -> Budget:
    budget = case_budget(graph, suite)
    if profile is not None:
        warnings.extend(
            apply_budget_overrides(
                budget,
                profile.budget_overrides,
                defaults=graph.defaults,
                allow_raise=allow_budget_override,
            )
        )
    return budget


def _make_provider(
    kind: str,
    manifest: ModelManifest,
    scenario: MockScenario,
    forced_route: tuple[str, str] | None,
) -> Provider:
    provider: Provider
    if kind == "mock":
        provider = MockProvider(scenario, max_context=manifest.spec.context_window)
    else:
        provider = build_live_provider(kind, manifest)
    if forced_route is not None:
        router_id, route = forced_route
        return _ForcedRouteProvider(provider, router_id, route)
    return provider


def _provider_factory(
    kind: str, scenario: MockScenario, forced_route: tuple[str, str] | None
) -> ProviderFactory:
    def factory(manifest: ModelManifest) -> Provider:
        return _make_provider(kind, manifest, scenario, forced_route)

    return factory


def _run_graph_case(
    case: EvalCase,
    graph: CompiledGraph,
    suite: EvalSuite,
    *,
    provider_kind: str,
    scenario: MockScenario,
    forced_route: tuple[str, str] | None,
    budget: Budget,
    allow_commands: bool,
    judge: JudgeHook | None,
    persist: bool,
    runs_dir: Path | None,
    profile_payload: dict[str, Any] | None,
    prune_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    # Even when the caller does not want to keep the run directory, the trace is required:
    # responses, decisions, chosen route, provider-call counts, and hop counts all come from
    # it. Persist into a temporary directory and discard it, rather than silently returning an
    # eventless result (which used to make reports generated with ``persist=False`` lose their
    # decisions and oracle agreement).
    temporary: tempfile.TemporaryDirectory[str] | None = None
    runtime_persist = persist
    runtime_runs_dir = runs_dir
    if not persist:
        temporary = tempfile.TemporaryDirectory(prefix="mg-eval-trace-")
        runtime_persist = True
        runtime_runs_dir = Path(temporary.name)
    try:
        started = time.monotonic()
        outcome = run_graph(
            graph,
            case.input.task,
            _provider_factory(provider_kind, scenario, forced_route),
            runs_dir=runtime_runs_dir,
            persist=runtime_persist,
            budget=budget,
            profile=profile_payload,
            prune_report=prune_payload,
        )
        latency = time.monotonic() - started
        events: list[dict[str, Any]] = []
        run_manifest: str | None = None
        run_manifest_hash: str | None = None
        if runtime_persist and outcome.run_dir is not None:
            trace_path = outcome.run_dir / "trace.jsonl"
            if trace_path.is_file():
                events = read_events(trace_path)
            manifest_path = outcome.run_dir / "run.json"
            if persist and manifest_path.is_file():
                run_manifest = str(manifest_path)
                run_manifest_hash = sha256_text(manifest_path.read_text(encoding="utf-8"))
    finally:
        if temporary is not None:
            temporary.cleanup()
    trace = TraceRecorder(outcome.run_id)
    trace.events = list(events)
    responses = provider_responses(trace)
    validity_issues = _graph_validity_issues(outcome, graph, responses)
    outcome_checks = evaluate_checks(
        case.checks,
        outcome.artifacts,
        artifact_names=list(graph.exits),
        task=case.input.task,
        case_id=case.id,
        rubric_text=suite.rubric_text,
        judge=judge,
        suite_dir=suite.directory,
        metadata=case.metadata,
        allow_commands=allow_commands,
    )
    if outcome_checks.judge_error:
        validity_issues.append(
            {
                "code": "EVAL-JUDGE",
                "message": outcome_checks.judge_error,
                "source": "judge",
            }
        )
    valid = not validity_issues
    success = valid and outcome.ok and outcome_checks.passed
    decisions = [
        {
            "node": event.get("node"),
            "decision": event.get("decision"),
            "to": event.get("to"),
            "confidence": event.get("confidence"),
            "reason": event.get("reason"),
        }
        for event in events
        if event.get("event") == "decision.made"
    ]
    first = decisions[0] if decisions else None
    usage = outcome.usage
    check_results = [result.to_dict() for result in outcome_checks.results]
    return {
        "case_id": case.id,
        "tags": list(case.tags),
        "task": case.input.task,
        "input_artifacts": [
            {
                "name": artifact.name,
                "type": artifact.type.value,
                "hash": sha256_text(artifact.content),
                "bytes": len(artifact.content.encode("utf-8")),
            }
            for artifact in case.input.artifacts
        ],
        "weight": case.weight,
        "metadata": case.metadata,
        "outcome": outcome.outcome,
        "valid": valid,
        "success": success,
        "run_success": outcome.ok,
        "quality": outcome_checks.judge_score,
        "checks_passed": outcome_checks.passed,
        "check_results": check_results,
        "checks_total": len(check_results),
        "checks_passed_count": sum(1 for result in outcome_checks.results if result.passed),
        "judge_called": outcome_checks.judge_called,
        "judge_error": outcome_checks.judge_error,
        "artifacts": [artifact_record(artifact) for artifact in outcome.artifacts],
        "responses": responses,
        "finish_reasons": [response.get("finish_reason") for response in responses],
        "path": [entry.model_dump(mode="json", exclude_none=True) for entry in outcome.path],
        "hop_count": _hop_count(events, outcome.path, int(usage.get("steps", 0))),
        "decisions": decisions,
        "first_router": first.get("node") if first else (forced_route[0] if forced_route else None),
        "chosen_route": first.get("to") if first else (forced_route[1] if forced_route else None),
        "run_id": outcome.run_id,
        "run_manifest": run_manifest,
        "run_manifest_hash": run_manifest_hash,
        "usage": usage,
        "cost_usd": float(usage.get("cost_usd", 0.0)),
        "latency_seconds": latency,
        "provider_calls": (provider_calls(trace) if events else int(usage.get("steps", 0))),
        "render_hashes": render_hashes(trace),
        "protocol_errors": protocol_errors_from_issues(validity_issues),
        "validity_issues": validity_issues,
        "error": outcome.error,
        "trial": "graph" if forced_route is None else f"oracle:{forced_route[1]}",
    }


def _graph_validity_issues(
    outcome: RunOutcome,
    graph: CompiledGraph,
    responses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not outcome.ok:
        error = outcome.error or {"code": "MG-2000", "message": "graph run failed"}
        issues.append(
            {
                "code": str(error.get("code", "MG-2000")),
                "message": str(error.get("message", "graph run failed")),
                "source": "runtime",
            }
        )
        return issues
    produced = {artifact.name: artifact for artifact in outcome.artifacts}
    for exit_name in graph.exits:
        artifact = produced.get(exit_name)
        if artifact is None:
            issues.append(
                {
                    "code": "MG-2202",
                    "message": f"exit artifact {exit_name!r} was not produced",
                    "source": "contract",
                }
            )
            continue
        if not artifact.content.strip():
            issues.append(
                {
                    "code": "MG-2202",
                    "message": f"exit artifact {exit_name!r} is empty",
                    "source": "contract",
                }
            )
            continue
        write = _exit_write(graph, exit_name)
        if write is not None and (write.type.value == "json" or write.strict_json):
            try:
                json.loads(artifact.content)
            except json.JSONDecodeError as exc:
                issues.append(
                    {
                        "code": "MG-2202",
                        "message": (
                            f"exit artifact {exit_name!r} is not valid JSON: {exc.msg} "
                            f"(line {exc.lineno}, column {exc.colno})"
                        ),
                        "source": "contract",
                    }
                )
    if responses and responses[-1].get("finish_reason") == "length":
        issues.append(
            {
                "code": "MG-2203",
                "message": "provider output was truncated (finish_reason=length)",
                "source": "runtime",
            }
        )
    return issues


def _exit_write(graph: CompiledGraph, name: str) -> Any:
    for node_id in sorted(graph.nodes):
        for write in graph.nodes[node_id].writes:
            if write.name == name:
                return write
    return None


def _hop_count(events: list[dict[str, Any]], path: list[Any], steps: int) -> int:
    if events:
        return sum(1 for event in events if event.get("event") == "node.entered")
    return steps or len(path)


def _run_node_probe(
    containing: CompiledGraph,
    node: CompiledNode,
    case: EvalCase,
    suite: EvalSuite,
    *,
    provider_kind: str,
    budget: Budget,
    allow_commands: bool,
    judge: JudgeHook | None,
) -> dict[str, Any]:
    if node.model is None:
        raise EvalError(f"worker {node.id!r} has no model manifest")
    state = build_state(containing, case.input)
    trace = TraceRecorder(new_id("evalnode"))
    provider = _make_provider(provider_kind, node.model, _scenario(case), None)
    started = time.monotonic()
    error: ModelGraphError | None = None
    try:
        WorkerExecutor(Executor(provider, budget=budget, trace=trace)).run(node, containing, state)
    except ModelGraphError as exc:
        error = exc
    latency = time.monotonic() - started
    close = getattr(provider, "close", None)
    if callable(close):
        close()
    responses = provider_responses(trace)
    produced = [artifact for artifact in state.artifacts if artifact.producer == node.id]
    validity_issues = node_validity_issues(node, state, error, responses)
    outcome_checks = evaluate_checks(
        case.checks,
        produced,
        artifact_names=[write.name for write in node.writes],
        task=case.input.task,
        case_id=case.id,
        rubric_text=suite.rubric_text,
        judge=judge,
        suite_dir=suite.directory,
        metadata=case.metadata,
        allow_commands=allow_commands,
    )
    if outcome_checks.judge_error:
        validity_issues.append(
            {
                "code": "EVAL-JUDGE",
                "message": outcome_checks.judge_error,
                "source": "judge",
            }
        )
    valid = not validity_issues
    success = valid and error is None and outcome_checks.passed
    usage = usage_snapshot(budget)
    check_results = [result.to_dict() for result in outcome_checks.results]
    return {
        "case_id": case.id,
        "tags": list(case.tags),
        "task": case.input.task,
        "weight": case.weight,
        "metadata": case.metadata,
        "outcome": "succeeded" if error is None else "failed",
        "valid": valid,
        "success": success,
        "run_success": error is None,
        "quality": outcome_checks.judge_score,
        "checks_passed": outcome_checks.passed,
        "check_results": check_results,
        "checks_total": len(check_results),
        "checks_passed_count": sum(1 for result in outcome_checks.results if result.passed),
        "judge_called": outcome_checks.judge_called,
        "judge_error": outcome_checks.judge_error,
        "artifacts": [artifact_record(artifact) for artifact in produced],
        "responses": responses,
        "finish_reasons": [response.get("finish_reason") for response in responses],
        "hop_count": 1 if error is None else 0,
        "decisions": [],
        "first_router": None,
        "chosen_route": None,
        "usage": usage,
        "cost_usd": float(usage.get("cost_usd", 0.0)),
        "latency_seconds": latency,
        "provider_calls": provider_calls(trace),
        "render_hashes": render_hashes(trace),
        "protocol_errors": protocol_errors_from_issues(validity_issues),
        "validity_issues": validity_issues,
        "error": (
            {
                "code": str(error.details.get("code", error.code)),
                "message": error.message,
            }
            if error is not None
            else None
        ),
        "trial": "single-node",
    }


# ---------------------------------------------------------------------------
# Baselines


def _single_node_baselines(
    graph: CompiledGraph,
    selected: tuple[EvalCase, ...],
    suite: EvalSuite,
    *,
    provider_kind: str,
    profile: Profile | None,
    allow_budget_override: bool,
    warnings: list[str],
    allow_commands: bool,
    judge: JudgeHook | None,
    resamples: int,
    seed: int,
) -> tuple[_BaselineSet | None, _BaselineSet | None]:
    config = suite.spec.baselines
    if not (config.best_single or config.cheapest_capable):
        return None, None
    workers = list(_worker_nodes(graph))
    if not workers:
        return None, None
    counts: dict[str, int] = {}
    for _, node in workers:
        counts[node.id] = counts.get(node.id, 0) + 1
    candidates: list[_BaselineSet] = []
    for containing, node in workers:
        label = node.id if counts.get(node.id, 0) == 1 else f"{containing.id}/{node.id}"
        results: list[dict[str, Any]] = []
        for case in selected:
            budget = _case_budget(
                containing,
                suite,
                profile,
                allow_budget_override=allow_budget_override,
                warnings=warnings,
            )
            results.append(
                _run_node_probe(
                    containing,
                    node,
                    case,
                    suite,
                    provider_kind=provider_kind,
                    budget=budget,
                    allow_commands=allow_commands,
                    judge=judge,
                )
            )
        summary = _metrics(results, seed=seed, resamples=resamples)
        summary.update(
            {
                "name": label,
                "kind": "single-node",
                "node": node.id,
                "graph": containing.id,
                "model": node.model.spec.model if node.model else None,
                "model_ref": node.model_ref,
                "cases": len(results),
                "available": True,
            }
        )
        candidates.append(
            _BaselineSet(
                name=label,
                kind="single-node",
                cases=results,
                summary=summary,
                info={"node": node.id, "graph": containing.id, "model_ref": node.model_ref},
            )
        )
    envelope = (
        config.cost_envelope_usd
        if config.cost_envelope_usd is not None
        else (
            suite.spec.budget.max_case_cost_usd
            if suite.spec.budget.max_case_cost_usd is not None
            else graph.defaults.max_cost_usd
        )
    )
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate.summary.get("cost_per_task_usd") or 0.0) <= envelope
    ]
    excluded = len(candidates) - len(eligible)
    if not eligible:
        eligible = candidates
        excluded = 0
    best = None
    if config.best_single:
        chosen = max(
            eligible,
            key=lambda item: (
                float(item.summary.get("task_success") or 0.0),
                float(item.summary.get("validity") or 0.0),
                -float(item.summary.get("cost_per_task_usd") or 0.0),
                item.name,
            ),
        )
        best = _clone_baseline(chosen, "best_single")
        best.summary["candidates"] = [_candidate_summary(item) for item in candidates]
        best.summary["cost_envelope_usd"] = envelope
        best.summary["excluded_by_envelope"] = excluded
        best.info["cost_envelope_usd"] = envelope
    cheapest = None
    if config.cheapest_capable:
        capable = [
            item
            for item in eligible
            if float(item.summary.get("task_success") or 0.0) >= config.capability_threshold
            and float(item.summary.get("task_success") or 0.0) > 0.0
        ]
        pool = capable or eligible
        chosen_cheap = min(
            pool,
            key=lambda item: (
                float(item.summary.get("cost_per_task_usd") or 0.0),
                -float(item.summary.get("task_success") or 0.0),
                item.name,
            ),
        )
        cheapest = _clone_baseline(chosen_cheap, "cheapest_capable")
        cheapest.summary["capability_threshold"] = config.capability_threshold
        cheapest.summary["cost_envelope_usd"] = envelope
        cheapest.summary["candidates"] = [_candidate_summary(item) for item in candidates]
    return best, cheapest


def _worker_nodes(
    graph: CompiledGraph, seen: set[str] | None = None
) -> Iterator[tuple[CompiledGraph, CompiledNode]]:
    seen = seen if seen is not None else set()
    if graph.id in seen:
        return
    seen.add(graph.id)
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if node.kind in {NodeKind.worker, NodeKind.joiner}:
            yield graph, node
        elif node.kind == NodeKind.subgraph:
            inner = graph.subgraphs.get(node.graph_ref or "")
            if inner is not None:
                yield from _worker_nodes(inner, seen)


def _clone_baseline(source: _BaselineSet, kind: str) -> _BaselineSet:
    return _BaselineSet(
        name=kind,
        kind=kind,
        cases=list(source.cases),
        summary={**source.summary, "name": kind, "kind": kind, "winner": source.name},
        info={**source.info, "winner": source.name},
    )


def _candidate_summary(baseline: _BaselineSet) -> dict[str, Any]:
    return {
        "name": baseline.name,
        "node": baseline.summary.get("node"),
        "graph": baseline.summary.get("graph"),
        "model": baseline.summary.get("model"),
        "task_success": baseline.summary.get("task_success"),
        "validity": baseline.summary.get("validity"),
        "cost_per_task_usd": baseline.summary.get("cost_per_task_usd"),
        "latency_p95_seconds": baseline.summary.get("latency_p95_seconds"),
        "cases": baseline.summary.get("cases"),
    }


def _measure_oracle(
    graph: CompiledGraph,
    selected: tuple[EvalCase, ...],
    primary_cases: list[dict[str, Any]],
    suite: EvalSuite,
    *,
    provider_kind: str,
    profile: Profile | None,
    allow_budget_override: bool,
    warnings: list[str],
    allow_commands: bool,
    judge: JudgeHook | None,
) -> _OracleMeasurement:
    config = suite.spec.baselines
    if not config.oracle_routing:
        return _OracleMeasurement(available=False, reason="oracle routing baseline disabled")
    entry = graph.nodes.get(graph.entry)
    if entry is None or entry.kind != NodeKind.router:
        return _OracleMeasurement(
            available=False,
            reason="the graph entry is not a router; first-hop oracle labeling is unavailable",
        )
    routes = sorted({edge.to for edge in graph.outgoing(entry.id)})
    if not routes:
        return _OracleMeasurement(available=False, reason="entry router has no candidate routes")
    if not config.measure_oracle:
        return _authored_oracle(selected, primary_cases, suite, reason=None)

    primary_by_case = {case["case_id"]: case for case in primary_cases}
    labels: list[dict[str, Any]] = []
    oracle_cases: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    measured_cost = 0.0
    for case in selected:
        branch_results: dict[str, dict[str, Any]] = {}
        for route in routes:
            budget = _case_budget(
                graph,
                suite,
                profile,
                allow_budget_override=allow_budget_override,
                warnings=warnings,
            )
            result = _run_graph_case(
                case,
                graph,
                suite,
                provider_kind=provider_kind,
                scenario=_scenario(case),
                forced_route=(entry.id, route),
                budget=budget,
                allow_commands=allow_commands,
                judge=judge,
                persist=False,
                runs_dir=None,
                profile_payload=profile.to_dict() if profile is not None else None,
                prune_payload=None,
            )
            branch_results[route] = result
            measured_cost += float(result["cost_usd"])
        quality_by_route = {route: _case_score(result) for route, result in branch_results.items()}
        best_route = max(
            routes,
            key=lambda route: (
                quality_by_route[route],
                -float(branch_results[route]["cost_usd"]),
                route,
            ),
        )
        ordered = sorted(quality_by_route.values(), reverse=True)
        ambiguous = bool(
            len(ordered) >= 2 and ordered[0] - ordered[1] <= config.oracle_ambiguity_delta
        )
        explicit = case.oracle.ambiguous if case.oracle is not None else False
        ambiguous = explicit or ambiguous
        primary = primary_by_case.get(case.id, {})
        label = {
            "case_id": case.id,
            "tags": list(case.tags),
            "oracle_route": best_route,
            "quality_by_route": quality_by_route,
            "ambiguous": ambiguous,
            "labeler": "measured",
            "labeled_at": utc_now(),
            "labeling_cost_usd": sum(float(item["cost_usd"]) for item in branch_results.values()),
            "chosen_route": primary.get("chosen_route"),
            "chosen_quality": quality_by_route.get(str(primary.get("chosen_route"))),
            "source": "first-hop-route-probes",
        }
        labels.append(label)
        chosen = dict(branch_results[best_route])
        chosen["case_id"] = case.id
        chosen["oracle_route"] = best_route
        chosen["oracle_quality_by_route"] = quality_by_route
        chosen["oracle_ambiguous"] = ambiguous
        chosen["chosen_route"] = best_route
        chosen["route_correct"] = primary.get("chosen_route") == best_route
        chosen["trial"] = f"oracle:{best_route}"
        oracle_cases.append(chosen)
        branches.append(
            {
                "case_id": case.id,
                "routes": routes,
                "oracle_route": best_route,
                "ambiguous": ambiguous,
                "results": {route: _case_probe(result) for route, result in branch_results.items()},
            }
        )
    return _OracleMeasurement(
        available=True,
        labels=labels,
        cases=oracle_cases,
        branches=branches,
        labeler="measured",
        measured_cost_usd=measured_cost,
    )


def _authored_oracle(
    selected: tuple[EvalCase, ...],
    primary_cases: list[dict[str, Any]],
    suite: EvalSuite,
    *,
    reason: str | None,
) -> _OracleMeasurement:
    primary_by_case = {case["case_id"]: case for case in primary_cases}
    labels: list[dict[str, Any]] = []
    for case in selected:
        if case.oracle is None:
            continue
        quality_by_route = dict(case.oracle.quality_by_route)
        if case.oracle.quality is not None:
            quality_by_route.setdefault(case.oracle.route, float(case.oracle.quality))
        primary = primary_by_case.get(case.id, {})
        chosen = primary.get("chosen_route")
        labels.append(
            {
                "case_id": case.id,
                "tags": list(case.tags),
                "oracle_route": case.oracle.route,
                "quality_by_route": quality_by_route,
                "ambiguous": case.oracle.ambiguous,
                "labeler": case.oracle.labeler or "authored",
                "labeled_at": case.oracle.labeled_at or utc_now(),
                "labeling_cost_usd": case.oracle.labeling_cost_usd,
                "chosen_route": chosen,
                "chosen_quality": quality_by_route.get(str(chosen)),
                "source": "suite-case-labels",
            }
        )
    if not labels:
        return _OracleMeasurement(
            available=False,
            reason=reason or "no authored oracle labels and measured labeling is disabled",
        )
    return _OracleMeasurement(
        available=False,
        labels=labels,
        reason="oracle routing baseline needs measured branch results; only labels are available",
        labeler="authored",
    )


def _oracle_set(measurement: _OracleMeasurement, summary: dict[str, Any]) -> _BaselineSet | None:
    if not measurement.available or not measurement.cases:
        return None
    metrics = _metrics(measurement.cases, seed=1234, resamples=2000)
    metrics.update(
        {
            "name": "oracle_routing",
            "kind": "oracle-routing",
            "available": True,
            "labeler": measurement.labeler,
            "ambiguous_cases": summary.get("ambiguous_cases"),
            "oracle_agreement": summary.get("oracle_agreement"),
            "cases": len(measurement.cases),
        }
    )
    return _BaselineSet(
        name="oracle_routing",
        kind="oracle-routing",
        cases=measurement.cases,
        summary=metrics,
        info={
            "labeler": measurement.labeler,
            "routes": measurement.labels[0].get("quality_by_route") if measurement.labels else {},
        },
    )


def _random_baseline(
    oracle: _OracleMeasurement, *, seed: int, random_seed: int
) -> _BaselineSet | None:
    if not oracle.available or not oracle.branches:
        return None
    effective_seed = seed + random_seed
    rng = random.Random(effective_seed)
    cases: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    for entry in oracle.branches:
        routes = sorted(entry["results"])
        if not routes:
            continue
        route = routes[rng.randrange(len(routes))]
        result = dict(entry["results"][route])
        result["case_id"] = entry["case_id"]
        result["chosen_route"] = route
        result["oracle_route"] = entry["oracle_route"]
        result["oracle_ambiguous"] = entry["ambiguous"]
        result["trial"] = f"random:{route}"
        cases.append(result)
        choices.append({"case_id": entry["case_id"], "route": route})
    if not cases:
        return None
    summary = _metrics(cases, seed=seed, resamples=2000)
    summary.update(
        {
            "name": "random_routing",
            "kind": "random-routing",
            "available": True,
            "seed": effective_seed,
            "cases": len(cases),
            "routes": choices,
        }
    )
    return _BaselineSet(
        name="random_routing",
        kind="random-routing",
        cases=cases,
        summary=summary,
        info={"seed": effective_seed},
    )


def _case_probe(result: dict[str, Any]) -> dict[str, Any]:
    """Compact a branch run for oracle labels and the random baseline."""
    return {
        key: result.get(key)
        for key in (
            "case_id",
            "tags",
            "task",
            "weight",
            "metadata",
            "outcome",
            "valid",
            "success",
            "run_success",
            "quality",
            "checks_passed",
            "check_results",
            "checks_total",
            "checks_passed_count",
            "judge_called",
            "judge_error",
            "artifacts",
            "responses",
            "finish_reasons",
            "hop_count",
            "decisions",
            "first_router",
            "chosen_route",
            "usage",
            "cost_usd",
            "latency_seconds",
            "provider_calls",
            "render_hashes",
            "protocol_errors",
            "validity_issues",
            "error",
            "trial",
        )
    }


def _oracle_summary(suite: EvalSuite, measurement: _OracleMeasurement) -> dict[str, Any]:
    labels = measurement.labels
    labeled = [label for label in labels if not label.get("ambiguous")]
    deltas = [
        float(label["quality_by_route"][label["oracle_route"]])
        - float(label["quality_by_route"][label["chosen_route"]])
        for label in labeled
        if label.get("chosen_route") in (label.get("quality_by_route") or {})
        and label.get("oracle_route") in (label.get("quality_by_route") or {})
    ]
    agreement = sum(
        1 for label in labeled if label.get("chosen_route") == label.get("oracle_route")
    )
    config = suite.spec.oracle
    suite_cost = config.labeling_cost_usd if config is not None else 0.0
    measured_cost = measurement.measured_cost_usd
    total = suite_cost + measured_cost
    return {
        "available": measurement.available or bool(labels),
        "reason": measurement.reason,
        "labeler": measurement.labeler,
        "labeled_cases": len(labeled),
        "ambiguous_cases": len(labels) - len(labeled),
        "oracle_agreement": safe_div(agreement, len(labeled)),
        "oracle_agreement_ci95": wilson_interval(agreement, len(labeled)),
        "oracle_gap": mean_or_none(deltas),
        "oracle_gap_ci95": bootstrap_ci(deltas, seed=1234),
        "quality_loss_cases": len(deltas),
        "labels": labels,
        "labeling": {
            "labeling_method": (
                config.labeling_method
                if config is not None and config.labeling_method
                else (
                    "measured-first-hop-route-probes"
                    if measurement.labeler == "measured"
                    else "authored-suite-labels"
                )
            ),
            "sampled_from": config.sampled_from if config is not None else None,
            "ambiguity_rule": (
                config.ambiguity_rule
                if config is not None and config.ambiguity_rule
                else "Explicit case flags plus a quality delta at or below the suite threshold."
            ),
            "notes": config.notes if config is not None else None,
            "suite_cost_usd": suite_cost,
            "measured_cost_usd": measured_cost,
            "total_usd": total,
            "per_case_usd": safe_div(total, len(labels)),
        },
    }


def _attach_oracle_labels(cases: list[dict[str, Any]], labels: list[dict[str, Any]]) -> None:
    by_case = {label["case_id"]: label for label in labels}
    for case in cases:
        label = by_case.get(case["case_id"])
        if label is None:
            case["oracle_route"] = None
            case["oracle_ambiguous"] = None
            case["oracle_quality_by_route"] = {}
            case["route_correct"] = None
            continue
        case["oracle_route"] = label["oracle_route"]
        case["oracle_ambiguous"] = label["ambiguous"]
        case["oracle_quality_by_route"] = label["quality_by_route"]
        case["route_correct"] = case.get("chosen_route") == label["oracle_route"]


# ---------------------------------------------------------------------------
# Metrics, slices, and failures


def _metrics(cases: list[dict[str, Any]], *, seed: int, resamples: int) -> dict[str, Any]:
    n = len(cases)
    successes = sum(1 for case in cases if case["success"])
    valid = sum(1 for case in cases if case["valid"])
    run_success = sum(1 for case in cases if case["run_success"])
    checks_total = sum(int(case["checks_total"]) for case in cases)
    checks_passed = sum(int(case["checks_passed_count"]) for case in cases)
    weights = [float(case.get("weight", 1.0)) for case in cases]
    weighted_success = safe_div(
        sum(weight for weight, case in zip(weights, cases, strict=True) if case["success"]),
        sum(weights),
    )
    quality = [float(case["quality"]) for case in cases if case.get("quality") is not None]
    costs = [float(case["cost_usd"]) for case in cases]
    latencies = [float(case["latency_seconds"]) for case in cases]
    hops = [float(case.get("hop_count", 0)) for case in cases]
    protocol = protocol_summary(cases)
    cost_per_task = mean_or_none(costs)
    return {
        "cases": n,
        "task_success": successes / n,
        "task_success_ci95": wilson_interval(successes, n),
        "weighted_task_success": weighted_success,
        "validity": valid / n,
        "validity_ci95": wilson_interval(valid, n),
        "run_success_rate": run_success / n,
        "run_success_rate_ci95": wilson_interval(run_success, n),
        "check_pass_rate": checks_passed / checks_total if checks_total else None,
        "quality_score": mean_or_none(quality),
        "quality_score_ci95": bootstrap_ci(quality, resamples=resamples, seed=seed),
        "judged_cases": len(quality),
        "cost_total_usd": sum(costs),
        "cost_per_task_usd": cost_per_task,
        "cost_per_task_usd_ci95": bootstrap_ci(costs, resamples=resamples, seed=seed),
        "quality_per_dollar": (
            safe_div(successes / n, cost_per_task)
            if cost_per_task is not None and cost_per_task > 0
            else None
        ),
        "input_tokens": sum(int(case["usage"]["input_tokens"]) for case in cases),
        "output_tokens": sum(int(case["usage"]["output_tokens"]) for case in cases),
        "total_tokens": sum(int(case["usage"]["total_tokens"]) for case in cases),
        "latency_mean_seconds": mean_or_none(latencies),
        "latency_p50_seconds": percentile(latencies, 0.5),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "latency_p95_seconds_ci95": bootstrap_ci(
            latencies,
            lambda values: percentile(values, 0.95),
            resamples=resamples,
            seed=seed,
        ),
        "hop_count_mean": mean_or_none(hops),
        "hop_count_p50": percentile(hops, 0.5),
        "hop_count_p95": percentile(hops, 0.95),
        "hop_count_max": max(hops) if hops else None,
        "protocol_error_rate": protocol["rate"],
        "protocol_errors_per_100_calls": protocol["per_100_calls"],
        "provider_calls": protocol["provider_calls"],
        "failure_count": n - successes,
    }


def _slices(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys: set[str] = set()
    for case in cases:
        for tag in case.get("tags", []):
            keys.add(str(tag))
        metadata = case.get("metadata") or {}
        for dimension in ("domain", "difficulty"):
            value = metadata.get(dimension)
            if value is not None:
                keys.add(f"{dimension}:{value}")
    slices: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        subset = [case for case in cases if _in_slice(case, key)]
        if not subset:
            continue
        metrics = _metrics(subset, seed=1234, resamples=200)
        slices[key] = {
            "cases": len(subset),
            "task_success": metrics["task_success"],
            "validity": metrics["validity"],
            "quality_score": metrics["quality_score"],
            "cost_per_task_usd": metrics["cost_per_task_usd"],
            "latency_p95_seconds": metrics["latency_p95_seconds"],
            "hop_count_p95": metrics["hop_count_p95"],
            "failure_count": metrics["failure_count"],
        }
    return slices


def _in_slice(case: dict[str, Any], key: str) -> bool:
    if key in (case.get("tags") or []):
        return True
    if ":" not in key:
        return False
    dimension, value = key.split(":", 1)
    metadata = case.get("metadata") or {}
    return str(metadata.get(dimension)) == value


def _failures(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for case in cases:
        if case["success"]:
            continue
        reasons: list[str] = []
        if case.get("error"):
            reasons.append(f"{case['error']['code']}: {case['error']['message']}")
        for issue in case.get("validity_issues", []):
            reason = f"{issue.get('code')}: {issue.get('message')}"
            if reason not in reasons:
                reasons.append(reason)
        failed_checks = [
            f"{result['type']}: {result['detail']}"
            for result in case.get("check_results", [])
            if not result.get("passed")
        ]
        reasons.extend(failed_checks)
        if (
            case.get("oracle_route")
            and case.get("chosen_route")
            and case["chosen_route"] != case["oracle_route"]
        ):
            reasons.append(
                f"oracle route {case['oracle_route']!r}, graph chose {case['chosen_route']!r}"
            )
        failures.append(
            {
                "case_id": case["case_id"],
                "tags": list(case.get("tags", [])),
                "reason": ("; ".join(dict.fromkeys(reasons)) or "checks failed")[:1000],
                "error": (case.get("error") or {}).get("code"),
                "failed_checks": failed_checks,
                "valid": case["valid"],
                "chosen_route": case.get("chosen_route"),
                "oracle_route": case.get("oracle_route"),
                "oracle_ambiguous": case.get("oracle_ambiguous"),
            }
        )
    return failures


def _case_score(case: dict[str, Any]) -> float:
    quality = case.get("quality")
    if isinstance(quality, (int, float)):
        return float(quality)
    return 1.0 if case.get("success") else 0.0


def _paired_comparison(
    name: str,
    candidate_cases: list[dict[str, Any]],
    baseline_cases: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    candidate_by = {case["case_id"]: case for case in candidate_cases}
    baseline_by = {case["case_id"]: case for case in baseline_cases}
    common = sorted(set(candidate_by) & set(baseline_by))
    deltas: list[float] = []
    cost_deltas: list[float] = []
    latency_deltas: list[float] = []
    wins = losses = ties = 0
    for case_id in common:
        candidate = candidate_by[case_id]
        base = baseline_by[case_id]
        delta = _case_score(candidate) - _case_score(base)
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
        cost_deltas.append(float(candidate["cost_usd"]) - float(base["cost_usd"]))
        latency_deltas.append(float(candidate["latency_seconds"]) - float(base["latency_seconds"]))
    paired = len(deltas)
    mean_delta = mean_or_none(deltas)
    spread = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    effect_size = mean_delta / spread if mean_delta is not None and spread > 0 else None
    return {
        "name": name,
        "baseline": name,
        "direction": "max",
        "paired_cases": paired,
        "missing_cases": len(set(candidate_by) ^ set(baseline_by)),
        "delta": mean_delta,
        "ci95": bootstrap_ci(deltas, resamples=resamples, seed=seed),
        "effect_size_cohens_d": effect_size,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": safe_div(wins, paired),
        "loss_rate": safe_div(losses, paired),
        "cost_delta_per_task_usd": mean_or_none(cost_deltas),
        "cost_delta_per_task_usd_ci95": bootstrap_ci(cost_deltas, resamples=resamples, seed=seed),
        "latency_delta_p50_seconds": percentile(latency_deltas, 0.5),
        "latency_delta_mean_seconds": mean_or_none(latency_deltas),
        "latency_delta_mean_seconds_ci95": bootstrap_ci(
            latency_deltas, resamples=resamples, seed=seed
        ),
    }


# ---------------------------------------------------------------------------
# Pareto and extra configurations


def _configuration_entry(
    name: str,
    kind: str,
    summary: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "name": name,
        "kind": kind,
        "quality": summary.get("task_success", summary.get("quality")),
        "quality_ci95": summary.get("task_success_ci95"),
        "validity": summary.get("validity"),
        "cost_per_task_usd": summary.get("cost_per_task_usd"),
        "cost_per_task_usd_ci95": summary.get("cost_per_task_usd_ci95"),
        "latency_p50_seconds": summary.get("latency_p50_seconds"),
        "latency_p95_seconds": summary.get("latency_p95_seconds"),
        "hop_count_mean": summary.get("hop_count_mean"),
        "cases": summary.get("cases"),
        "available": summary.get("available", True),
    }
    if extra:
        entry.update({key: value for key, value in extra.items() if value is not None})
    return entry


def _apply_pareto(entries: list[dict[str, Any]]) -> dict[str, Any]:
    configurations = [dict(entry) for entry in entries if entry.get("quality") is not None]
    configurations.sort(
        key=lambda entry: (
            -float(entry.get("quality") or 0.0),
            float(entry.get("cost_per_task_usd") or 0.0),
            str(entry.get("name")),
        )
    )
    frontier: list[str] = []
    for entry in configurations:
        quality = float(entry.get("quality") or 0.0)
        cost = float(entry.get("cost_per_task_usd") or 0.0)
        dominated = False
        for other in configurations:
            if other is entry:
                continue
            other_quality = float(other.get("quality") or 0.0)
            other_cost = float(other.get("cost_per_task_usd") or 0.0)
            if (
                other_quality >= quality
                and other_cost <= cost
                and (other_quality > quality or other_cost < cost)
            ):
                dominated = True
                break
        entry["frontier"] = not dominated
        if not dominated:
            frontier.append(str(entry["name"]))
    costs = sorted(
        {
            float(entry["cost_per_task_usd"])
            for entry in configurations
            if entry.get("cost_per_task_usd") is not None
        }
    )
    bands: list[dict[str, Any]] = []
    for ceiling in costs:
        eligible = [
            entry
            for entry in configurations
            if entry.get("cost_per_task_usd") is not None
            and float(entry["cost_per_task_usd"]) <= ceiling
        ]
        if not eligible:
            continue
        winner = max(
            eligible,
            key=lambda entry: (
                float(entry.get("quality") or 0.0),
                -float(entry["cost_per_task_usd"]),
            ),
        )
        if bands and bands[-1]["winner"] == winner["name"]:
            bands[-1]["max_cost_usd"] = ceiling
            continue
        bands.append(
            {
                "max_cost_usd": ceiling,
                "winner": winner["name"],
                "quality": winner.get("quality"),
                "cost_per_task_usd": winner.get("cost_per_task_usd"),
            }
        )
    return {
        "configurations": configurations,
        "frontier": frontier,
        "bands": bands,
        "notes": (
            "Quality is task success; cost is manifest-priced cost per task. Frontier entries are "
            "not dominated on quality and cost. With free local manifests every configuration "
            "sits in the same $0 cost band, so the highest-quality configuration wins that band. "
            "Do not compare configurations at different budgets and call the difference quality."
        ),
    }


def _evaluate_configuration(
    configuration: E2EConfiguration,
    suite: EvalSuite,
    selected: tuple[EvalCase, ...],
    *,
    provider_kind: str,
    allow_budget_override: bool,
    allow_commands: bool,
    judge: JudgeHook | None,
    registry: Registry,
) -> dict[str, Any]:
    raw = Path(configuration.graph).expanduser()
    local = (suite.directory / raw).resolve()
    path = raw if raw.is_absolute() else (local if local.is_file() else (project_root() / raw))
    compiled = compile_graph(
        path,
        registry=registry,
        strict_models=True,
        graph_paths=[path.parent],
    )
    if not compiled.ok or compiled.graph is None:
        return {
            "name": configuration.name,
            "kind": "graph-profile",
            "available": False,
            "error": "; ".join(compiled.errors),
            "graph": str(path),
            "profile": configuration.profile,
        }
    graph = compiled.graph
    profile_obj: Profile | None = None
    prune_report: PruneReport | None = None
    try:
        if configuration.profile:
            profile_obj = load_profile(
                resolve_profile_path(
                    (suite.directory / configuration.profile)
                    if not Path(configuration.profile).is_absolute()
                    and (suite.directory / configuration.profile).is_file()
                    else configuration.profile
                )
            )
            pruned = prune_graph(graph, profile_obj)
            graph = pruned.graph
            prune_report = pruned.report
    except ModelGraphError as exc:
        return {
            "name": configuration.name,
            "kind": "graph-profile",
            "available": False,
            "error": exc.message,
            "graph": str(path),
            "profile": configuration.profile,
        }
    warnings: list[str] = []
    cases: list[dict[str, Any]] = []
    for case in selected:
        budget = _case_budget(
            compiled.graph,
            suite,
            profile_obj,
            allow_budget_override=allow_budget_override,
            warnings=warnings,
        )
        cases.append(
            _run_graph_case(
                case,
                graph,
                suite,
                provider_kind=provider_kind,
                scenario=_scenario(case),
                forced_route=None,
                budget=budget,
                allow_commands=allow_commands,
                judge=judge,
                persist=False,
                runs_dir=None,
                profile_payload=profile_obj.to_dict() if profile_obj is not None else None,
                prune_payload=prune_report.to_dict() if prune_report is not None else None,
            )
        )
    summary = _metrics(cases, seed=1234, resamples=200)
    entry = _configuration_entry(configuration.name, "graph-profile", summary)
    entry["graph"] = graph_summary(graph, path)
    entry["profile"] = profile_obj.name if profile_obj is not None else None
    entry["prune"] = prune_report.to_dict() if prune_report is not None else None
    entry["warnings"] = list(dict.fromkeys(warnings))
    return entry


# ---------------------------------------------------------------------------
# Judge and incumbent


def score_blinded_pair(
    *,
    case_id: str,
    task: str,
    candidate: str,
    baseline: str,
    rubric: str,
    judge: JudgeHook,
    model: str | None = None,
) -> dict[str, Any]:
    """Score two outputs with the outputs blinded and their A/B position swapped.

    The hook sees neutral slot names (``A``/``B``) and a ``target: A`` instruction in
    metadata. The first call scores the candidate in slot A; the second scores the baseline
    in slot A. Returning the same score in both orders is agreement; a large difference is
    position sensitivity and must be reported, not averaged away.
    """
    requests = [
        JudgeRequest(
            case_id=f"{case_id}:ab",
            task=task,
            artifacts={"A": candidate, "B": baseline},
            rubric=rubric,
            metadata={
                "target": "A",
                "blinded": True,
                "order": "AB",
                "pair_case_id": case_id,
            },
        ),
        JudgeRequest(
            case_id=f"{case_id}:ba",
            task=task,
            artifacts={"A": baseline, "B": candidate},
            rubric=rubric,
            metadata={
                "target": "A",
                "blinded": True,
                "order": "BA",
                "pair_case_id": case_id,
            },
        ),
    ]
    verdicts = []
    for request in requests:
        try:
            verdict = judge(request)
        except Exception as exc:  # fail closed
            raise EvalError(f"blinded judge failed for case {case_id!r}: {exc}") from exc
        if not 0.0 <= float(verdict.score) <= 1.0:
            raise EvalError(
                f"blinded judge returned score {verdict.score!r} outside [0, 1] for "
                f"case {case_id!r}"
            )
        verdicts.append(verdict)
    candidate_score = float(verdicts[0].score)
    baseline_score = float(verdicts[1].score)
    return {
        "case_id": case_id,
        "candidate_score": candidate_score,
        "baseline_score": baseline_score,
        "delta": candidate_score - baseline_score,
        "agreement": abs(candidate_score - baseline_score),
        "order_swapped": True,
        "blinded": True,
        "calls": 2,
        "model": verdicts[0].model or verdicts[1].model or model,
    }


def _judge_summary(
    suite: EvalSuite,
    cases: list[dict[str, Any]],
    baseline_payload: dict[str, Any] | None,
    judge: JudgeHook | None,
) -> dict[str, Any]:
    config = suite.spec.judge
    errors = [
        f"{case['case_id']}: {case['judge_error']}" for case in cases if case.get("judge_error")
    ]
    rubric_path = suite.spec.rubric or config.rubric
    summary: dict[str, Any] = {
        "configured": config.enabled,
        "model": config.model,
        "rubric": rubric_path,
        "rubric_hash": (sha256_text(suite.rubric_text) if suite.rubric_text is not None else None),
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "suite_hash": suite.content_hash,
        "calls": sum(1 for case in cases if case.get("judge_called")),
        "judged_cases": sum(1 for case in cases if case.get("quality") is not None),
        "errors": errors,
        "error": errors[0] if errors else None,
        "blinded": True,
        "order_swapped": True,
        "calibration": _judge_calibration(cases),
    }
    if judge is None or not config.enabled or baseline_payload is None:
        return summary
    baseline_cases = {
        case.get("case_id"): case
        for case in baseline_payload.get("cases", [])
        if isinstance(case, dict)
    }
    pair_rows: list[dict[str, Any]] = []
    pair_errors: list[str] = []
    for case in cases:
        baseline_case = baseline_cases.get(case["case_id"])
        if baseline_case is None or not case.get("judge_called"):
            continue
        candidate_text = _artifact_text(case.get("artifacts") or [])
        baseline_text = _artifact_text(baseline_case.get("artifacts") or [])
        if not candidate_text or not baseline_text:
            continue
        rubric = suite.rubric_text or ""
        try:
            pair_rows.append(
                score_blinded_pair(
                    case_id=case["case_id"],
                    task=case["task"],
                    candidate=candidate_text,
                    baseline=baseline_text,
                    rubric=rubric,
                    judge=judge,
                    model=config.model,
                )
            )
        except EvalError as exc:
            pair_errors.append(str(exc))
    if pair_rows:
        deltas = [float(row["delta"]) for row in pair_rows]
        summary["pair"] = {
            "cases": len(pair_rows),
            "blinded": True,
            "order_swapped": True,
            "delta": mean_or_none(deltas),
            "delta_ci95": bootstrap_ci(deltas),
            # This is the mean absolute candidate/baseline gap, not a position-bias
            # estimate: ``score_blinded_pair`` scores each output once, in slot A.
            "mean_absolute_pair_delta": mean_or_none(
                [float(row["agreement"]) for row in pair_rows]
            ),
            "model": pair_rows[0].get("model"),
            "rows": pair_rows,
        }
    if pair_errors:
        summary["pair_errors"] = pair_errors
    return summary


def _judge_calibration(cases: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = _human_calibration_labels(cases)
    if not pairs:
        return {"available": False, "n": 0}
    errors = [abs(judge - human) for judge, human in pairs]
    within = sum(1 for error in errors if error <= 0.1)
    return {
        "available": True,
        "n": len(pairs),
        "mean_absolute_error": mean_or_none(errors),
        "within_0_1": safe_div(within, len(pairs)),
    }


def _human_calibration_labels(cases: list[dict[str, Any]]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for case in cases:
        quality = case.get("quality")
        metadata = case.get("metadata") or {}
        human = metadata.get("human_score", metadata.get("human_quality"))
        if isinstance(quality, (int, float)) and isinstance(human, (int, float)):
            pairs.append((float(quality), float(human)))
    return pairs


def _artifact_text(artifacts: list[dict[str, Any]]) -> str:
    parts = []
    for artifact in artifacts:
        content = artifact.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n\n".join(parts)


def _load_baseline(
    baseline: EvalReport | dict[str, Any] | str | Path | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    if baseline is None:
        return None, None
    if isinstance(baseline, EvalReport):
        path = baseline.report_dir / "report.json" if baseline.report_dir else None
        return baseline.to_dict(), path
    if isinstance(baseline, dict):
        return baseline, None
    path = Path(baseline)
    if path.is_dir():
        path = path / "report.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read incumbent report {path}: {exc}", path=str(path)) from exc
    if not isinstance(payload, dict):
        raise EvalError(f"incumbent report {path} must be a JSON object", path=str(path))
    return payload, path


def _incumbent_summary(payload: dict[str, Any], source: Path | None) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    primary = payload.get("primary_metric") or {}
    costs = payload.get("costs") or {}
    latency = payload.get("latency") or {}
    suite = payload.get("suite") or {}
    graph = payload.get("graph") or {}
    return {
        "available": True,
        "name": "incumbent",
        "kind": "incumbent",
        "report_id": payload.get("report_id"),
        "report_path": str(source) if source is not None else None,
        "suite_hash": suite.get("hash"),
        "graph_hash": graph.get("hash"),
        "task_success": primary.get("value", metrics.get("task_success")),
        "task_success_ci95": metrics.get("task_success_ci95"),
        "validity": metrics.get("validity"),
        "cost_per_task_usd": costs.get("per_task_usd", metrics.get("cost_per_task_usd")),
        "latency_p50_seconds": latency.get("p50_seconds"),
        "latency_p95_seconds": latency.get("p95_seconds"),
        "cases": len(payload.get("cases") or []),
    }


def _incumbent_result(
    summary: dict[str, Any], comparison: EvalReport, source: Path | None
) -> dict[str, Any]:
    return {
        **summary,
        "available": True,
        "name": "incumbent",
        "kind": "incumbent",
        "report_path": str(source) if source is not None else None,
        "delta": comparison.metrics.get("primary_delta"),
        "ci95": comparison.metrics.get("primary_delta_ci95"),
        "wins": comparison.metrics.get("wins"),
        "losses": comparison.metrics.get("losses"),
        "ties": comparison.metrics.get("ties"),
        "cost_delta_per_task_usd": comparison.metrics.get("cost_delta_per_task_usd"),
        "candidate_report_id": comparison.candidate.get("report_id")
        if comparison.candidate
        else None,
        "comparison_report_id": comparison.report_id,
    }


def _compare_incumbent(
    report: EvalReport,
    payload: dict[str, Any],
    source: Path | None,
    *,
    resamples: int,
    seed: int,
) -> tuple[EvalReport, EvalReport | None]:
    from modelgraph.evaluation.comparison import compare_reports

    try:
        comparison = compare_reports(
            report,
            payload,
            persist=False,
            bootstrap_resamples=resamples,
            seed=seed,
        )
    except EvalError as exc:
        if report.e2e is not None:
            report.e2e["baselines"]["incumbent"] = {
                "available": False,
                "name": "incumbent",
                "kind": "incumbent",
                "report_path": str(source) if source is not None else None,
                "error": exc.message,
            }
        report.limitations.append(f"Incumbent comparison unavailable: {exc.message}")
        return report, None
    report.baseline = comparison.baseline
    report.candidate = comparison.candidate
    return report, comparison


def _comparison_entry(name: str, comparison: EvalReport) -> dict[str, Any]:
    metrics = comparison.metrics
    return {
        "name": name,
        "baseline": name,
        "baseline_report_id": comparison.baseline.get("report_id") if comparison.baseline else None,
        "candidate_report_id": comparison.candidate.get("report_id")
        if comparison.candidate
        else None,
        "delta": metrics.get("primary_delta"),
        "ci95": metrics.get("primary_delta_ci95"),
        "wins": metrics.get("wins"),
        "losses": metrics.get("losses"),
        "ties": metrics.get("ties"),
        "direction": metrics.get("primary_improvement_direction", "max"),
    }


def _target(
    suite: EvalSuite,
    graph: CompiledGraph,
    profile: Profile | None,
    prune: PruneReport | None,
    selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "level": "e2e",
        "graph": graph.id,
        "entry": graph.entry,
        "nodes": len(graph.nodes),
        "exits": list(graph.exits),
        "suite_graph": suite.spec.target.graph,
        "profile": profile.name if profile is not None else None,
        "pruned": prune is not None,
        "prune_removed_nodes": len(prune.removed_nodes) if prune is not None else 0,
        "selection": selection,
    }


def _baseline_case_count(e2e_data: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, summary in e2e_data["baselines"].items():
        if isinstance(summary, dict) and isinstance(summary.get("cases"), int):
            counts[name] = summary["cases"]
    return counts
