"""Routing evaluation runner.

Uses the runtime's own router executor so thresholds, fallback policy, repair, and decision
parsing are exactly what a run would do. Oracle labels are authored data; this module only
compares the router's decision to them and reports agreement, confusion, abstention, and
calibration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from modelgraph.compiler import compile_graph
from modelgraph.compiler.ir import CompiledGraph, CompiledNode
from modelgraph.contracts.graph import NodeKind
from modelgraph.errors import ModelGraphError
from modelgraph.evaluation.errors import EvalError, SuiteError
from modelgraph.evaluation.metrics import (
    abstain_metrics,
    bootstrap_ci,
    confusion_matrix,
    latency_summary,
    mean_or_none,
    percentile,
    reliability_bins,
    safe_div,
    threshold_sweep,
    wilson_interval,
)
from modelgraph.evaluation.providers import ProviderPool, provider_summary
from modelgraph.evaluation.report import (
    EvalReport,
    build_primary_metric,
    default_report_dir,
    determinism_fingerprint,
    environment_info,
    new_report_id,
)
from modelgraph.evaluation.suite import EvalCase, EvalSuite, RoutingTarget, load_suite
from modelgraph.evaluation.support import (
    build_state,
    case_budget,
    case_fingerprint,
    commands_for,
    costs_summary,
    error_payload,
    graph_summary,
    limitations_for,
    protocol_error,
    protocol_summary,
    provider_calls,
    provider_responses,
    render_hashes,
    suite_usage_note,
    target_summary,
    usage_snapshot,
    usage_totals,
)
from modelgraph.providers.base import Provider
from modelgraph.registry import Registry, default_registry
from modelgraph.runtime.executors import Executor, RouterExecutor
from modelgraph.runtime.renderer import render_prompt
from modelgraph.runtime.trace import TraceRecorder
from modelgraph.util import new_id


def run_routing_suite(
    suite: EvalSuite | str | Path,
    *,
    graph_path: str | Path | None = None,
    provider: str | None = None,
    allow_live: bool = False,
    registry: Registry | None = None,
    reports_dir: str | Path | None = None,
    case_ids: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
    command: str | None = None,
    persist: bool = True,
    bootstrap_resamples: int = 2000,
    seed: int = 1234,
) -> EvalReport:
    """Run one routing suite and return (and optionally persist) its report."""
    loaded = suite if isinstance(suite, EvalSuite) else load_suite(suite)
    provider_name = provider or loaded.provider
    pool = ProviderPool(provider_name, allow_live=allow_live)
    try:
        graph_file = (
            Path(graph_path).expanduser().resolve()
            if graph_path is not None
            else loaded.resolve_graph_path()
        )
        if not graph_file.is_file():
            raise SuiteError(f"target graph not found: {graph_file}", path=str(graph_file))
        compiled = compile_graph(
            graph_file,
            registry=registry or default_registry(),
            strict_models=True,
            graph_paths=[graph_file.parent],
        )
        if not compiled.ok or compiled.graph is None:
            raise SuiteError(
                f"graph {graph_file} failed to compile: " + "; ".join(compiled.errors),
                path=str(graph_file),
            )
        graph = compiled.graph
        if not isinstance(loaded.spec.target, RoutingTarget):
            raise SuiteError(f"suite {loaded.metadata.id!r} is not a routing suite")
        router_id = loaded.spec.target.router or graph.entry
        router = _target_router(graph, router_id, loaded)

        selected = loaded.select_cases(case_ids=case_ids, tags=tags, limit=limit)
        selection = loaded.selection_summary(
            list(selected), case_ids=case_ids, tags=tags, limit=limit
        )
        report_root = Path(reports_dir) if reports_dir is not None else default_report_dir()
        report_id = new_report_id()
        cli_command, commands, reproduce = commands_for(
            level="routing",
            suite_path=loaded.source_path,
            graph_path=graph_file,
            provider=provider_name,
            reports_dir=report_root,
            case_ids=case_ids,
            tags=tags,
            limit=limit,
        )
        if command is not None:
            cli_command = command
            commands = [cli_command, reproduce]

        results: list[dict[str, Any]] = []
        total_cost = 0.0
        total_tokens = 0
        stopped: str | None = None
        budget_config = loaded.spec.budget
        try:
            for case in selected:
                if (
                    budget_config.max_cost_usd is not None
                    and total_cost > budget_config.max_cost_usd
                ):
                    stopped = (
                        f"suite cost budget exceeded (${total_cost:.6f} > "
                        f"${budget_config.max_cost_usd:.6f})"
                    )
                    break
                if budget_config.max_tokens is not None and total_tokens > budget_config.max_tokens:
                    stopped = (
                        f"suite token budget exceeded ({total_tokens} > {budget_config.max_tokens})"
                    )
                    break
                provider_instance = pool.for_case(case, router)
                result = _run_routing_case(case, graph, router, provider_instance, loaded)
                results.append(result)
                total_cost += float(result["cost_usd"])
                total_tokens += int(result["usage"]["total_tokens"])
                if (
                    budget_config.max_cost_usd is not None
                    and total_cost > budget_config.max_cost_usd
                ):
                    stopped = (
                        f"suite cost budget exceeded (${total_cost:.6f} > "
                        f"${budget_config.max_cost_usd:.6f})"
                    )
                    break
                if budget_config.max_tokens is not None and total_tokens > budget_config.max_tokens:
                    stopped = (
                        f"suite token budget exceeded ({total_tokens} > {budget_config.max_tokens})"
                    )
                    break
        finally:
            pool.close()
        if not results:
            raise EvalError("suite budget stopped before the first case ran")

        metrics = _routing_metrics(results, loaded, seed=seed, resamples=bootstrap_resamples)
        costs = costs_summary(results, pricing_note=suite_usage_note(loaded, provider_name))
        costs["per_task_usd_ci95"] = bootstrap_ci(
            [float(result["cost_usd"]) for result in results],
            resamples=bootstrap_resamples,
            seed=seed,
        )
        latencies = [float(result["latency_seconds"]) for result in results]
        latency = latency_summary(latencies)
        latency["p95_ci95"] = bootstrap_ci(
            latencies,
            lambda values: percentile(values, 0.95),
            resamples=bootstrap_resamples,
            seed=seed,
        )
        usage = usage_totals(results)
        protocol = protocol_summary(results)
        failures = _failures(results)
        slices = _slices(results)
        oracle = _oracle_summary(results, loaded)
        calibration = _calibration(results, loaded)
        confusion = confusion_matrix(
            [str(result["oracle"]["route"]) for result in results],
            [str(result["chosen"]) for result in results],
        )
        abstain = abstain_metrics(
            [result["chosen"] == "abstain" for result in results],
            [bool(result["oracle"]["ambiguous"]) for result in results],
        )
        primary, thresholds, gate = build_primary_metric(
            loaded.spec.metric, metrics, loaded.spec.thresholds
        )
        if stopped:
            gate["passed"] = False
            gate.setdefault("failures", []).append("budget")
            gate["budget_stopped"] = stopped
        limitations = limitations_for(
            level="routing",
            provider=provider_name,
            case_count=len(results),
            oracle_labeling_method=loaded.spec.oracle.labeling_method
            if loaded.spec.oracle
            else None,
            command_checks=False,
        )
        if stopped:
            limitations.append(stopped)
        determinism_payload = {
            "suite_hash": loaded.content_hash,
            "graph_hash": graph.graph_hash,
            "level": "routing",
            "provider": provider_name,
            "selected_case_ids": [result["case_id"] for result in results],
            "metrics": {
                key: value for key, value in metrics.items() if not key.startswith("latency")
            },
            "cases": [case_fingerprint(result) for result in results],
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
            ],
        }
        report = EvalReport(
            report_id=report_id,
            level="routing",
            claim=loaded.spec.claim or loaded.metadata.description,
            suite=loaded.summary(),
            graph=graph_summary(graph, graph_file),
            target=target_summary(loaded, router.id, router.kind.value),
            provider=provider_summary(provider_name, loaded, pool),
            command=cli_command or "mg eval routing (CLI wiring lands in T-0021)",
            commands=commands,
            reproduce=reproduce,
            environment=environment_info(),
            selection=selection,
            budget={
                "suite": loaded.spec.budget.model_dump(mode="json", exclude_none=True),
                "runtime_defaults": graph.defaults.model_dump(mode="json"),
                "stopped": stopped is not None,
                "stop_reason": stopped,
            },
            metrics=metrics,
            primary_metric=primary,
            thresholds=thresholds,
            gate=gate,
            cases=results,
            failures=failures,
            slices=slices,
            protocol_errors=protocol,
            costs=costs,
            latency=latency,
            usage=usage,
            determinism=determinism,
            limitations=limitations,
            oracle=oracle,
            calibration=calibration,
            confusion=confusion,
            abstain=abstain,
        )
        if persist:
            report.write(report_root)
        return report
    finally:
        pool.close()


def _target_router(graph: CompiledGraph, router_id: str, suite: EvalSuite) -> CompiledNode:
    try:
        router = graph.node(router_id)
    except KeyError as exc:
        raise SuiteError(
            f"router {router_id!r} is not in graph {graph.id}@{graph.version}"
        ) from exc
    if router.kind != NodeKind.router:
        raise SuiteError(
            f"node {router_id!r} is a {router.kind.value}; routing evaluation requires a router"
        )
    if suite.spec.oracle is None:  # loader already enforces this; keep the invariant visible
        raise SuiteError(f"routing suite {suite.metadata.id!r} has no oracle configuration")
    return router


def _run_routing_case(
    case: EvalCase,
    graph: CompiledGraph,
    router: CompiledNode,
    provider: Provider,
    suite: EvalSuite,
) -> dict[str, Any]:
    assert case.oracle is not None
    state = build_state(graph, case.input)
    trace = TraceRecorder(new_id("evalcase"))
    budget = case_budget(graph, suite)
    rendered = render_prompt(router, graph, state)
    started = time.monotonic()
    error: ModelGraphError | None = None
    decision: Any = None
    try:
        decision = RouterExecutor(Executor(provider, budget=budget, trace=trace)).run(
            router, graph, state
        )
    except ModelGraphError as exc:
        error = exc
    latency = time.monotonic() - started
    responses = provider_responses(trace)
    usage = usage_snapshot(budget)
    action = "route" if decision is not None and decision.action == "route" else "abstain"
    if error is not None:
        chosen = "error"
        action = "abstain"
    elif decision is not None and decision.action == "route" and decision.to is not None:
        chosen = decision.to
    else:
        chosen = "abstain"
    confidence = float(decision.confidence) if decision is not None else None
    fallback_to = decision.to if decision is not None and decision.action != "route" else None
    if fallback_to is None and action == "abstain" and error is None:
        fallback_to = graph.fallback
    correct = bool(action == "route" and chosen == case.oracle.route)
    success = bool(chosen == "abstain" and error is None) if case.oracle.ambiguous else correct
    quality_loss, quality_available = _quality_loss(case, action, chosen)
    weight = float(case.weight) * (
        case.oracle.cost_usd if case.oracle.cost_usd and case.oracle.cost_usd > 0 else 1.0
    )
    protocol_errors: list[dict[str, Any]] = []
    if error is not None:
        payload = error_payload(error) or {}
        code = str(payload.get("code", ""))
        if code.startswith("MG-2"):
            protocol_errors.append(
                protocol_error(code, str(payload.get("message", "")), source="router")
            )
    return {
        "case_id": case.id,
        "tags": list(case.tags),
        "task": case.input.task,
        "weight": case.weight,
        "oracle": {
            "route": case.oracle.route,
            "ambiguous": case.oracle.ambiguous,
            "quality": case.oracle.quality,
            "quality_by_route": case.oracle.quality_by_route,
            "cost_usd": case.oracle.cost_usd,
            "misroute_loss": case.oracle.misroute_loss,
            "labeler": case.oracle.labeler,
            "labeled_at": case.oracle.labeled_at,
            "labeling_cost_usd": case.oracle.labeling_cost_usd,
            "note": case.oracle.note,
        },
        "decision": (
            {
                "action": decision.action,
                "to": decision.to,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "note": decision.note,
            }
            if decision is not None
            else None
        ),
        "action": action,
        "chosen": chosen,
        "fallback_to": fallback_to,
        "confidence": confidence,
        "correct": correct,
        "ambiguous": case.oracle.ambiguous,
        "quality_loss": quality_loss,
        "quality_loss_available": quality_available,
        "weight_for_accuracy": weight,
        "outcome": "succeeded" if error is None else "failed",
        "run_success": error is None,
        "valid": error is None,
        "success": success,
        "responses": responses,
        "finish_reasons": [response.get("finish_reason") for response in responses],
        "prompt": rendered.to_dict(),
        "prompt_hash": rendered.prompt_hash,
        "render_hashes": render_hashes(trace),
        "usage": usage,
        "cost_usd": usage["cost_usd"],
        "latency_seconds": latency,
        "provider_calls": provider_calls(trace),
        "protocol_errors": protocol_errors,
        "error": error_payload(error),
    }


def _quality_loss(case: EvalCase, action: str, chosen: str) -> tuple[float | None, bool]:
    assert case.oracle is not None
    oracle = case.oracle
    quality_map = oracle.quality_by_route
    if quality_map:
        oracle_quality = quality_map.get(oracle.route)
        chosen_quality = quality_map.get(chosen) if action == "route" else None
        if oracle_quality is not None and chosen_quality is not None:
            return max(0.0, oracle_quality - chosen_quality), True
    if oracle.misroute_loss is not None and action != "route":
        return float(oracle.misroute_loss), True
    return None, False


def _routing_metrics(
    cases: list[dict[str, Any]],
    suite: EvalSuite,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    n = len(cases)
    labeled = [case for case in cases if not case["oracle"]["ambiguous"]]
    correct_labeled = sum(1 for case in labeled if case["correct"])
    correct_all = sum(1 for case in cases if case["correct"])
    agreement = safe_div(correct_labeled, len(labeled))
    weighted_total = sum(float(case["weight_for_accuracy"]) for case in labeled)
    weighted_correct = sum(
        float(case["weight_for_accuracy"]) for case in labeled if case["correct"]
    )
    losses = [
        float(case["quality_loss"]) for case in labeled if case.get("quality_loss") is not None
    ]
    abstain = abstain_metrics(
        [case["chosen"] == "abstain" for case in cases],
        [bool(case["oracle"]["ambiguous"]) for case in cases],
    )
    calibration = _calibration(cases, suite)
    latencies = [float(case["latency_seconds"]) for case in cases]
    protocol = protocol_summary(cases)
    return {
        "cases": n,
        "labeled_cases": len(labeled),
        "ambiguous_cases": n - len(labeled),
        "oracle_agreement": agreement,
        "oracle_agreement_ci95": wilson_interval(correct_labeled, len(labeled)),
        "oracle_agreement_all_cases": safe_div(correct_all, n),
        "cost_weighted_accuracy": safe_div(weighted_correct, weighted_total),
        "cost_weighted_error": mean_or_none(losses),
        "quality_loss_cases": len(losses),
        "abstain_rate": abstain["abstain_rate"],
        "abstain_precision": abstain["precision"],
        "abstain_recall": abstain["recall"],
        "abstain_f1": abstain["f1"],
        "ece": calibration["ece"],
        "max_calibration_error": calibration["max_calibration_error"],
        "calibration_cases": calibration["n"],
        "run_success_rate": sum(1 for case in cases if case["run_success"]) / n,
        "run_success_rate_ci95": wilson_interval(
            sum(1 for case in cases if case["run_success"]), n
        ),
        "protocol_error_rate": protocol["rate"],
        "protocol_errors_per_100_calls": protocol["per_100_calls"],
        "failure_count": sum(1 for case in cases if not case["success"]),
        "mean_confidence": mean_or_none(
            [
                float(case["confidence"])
                for case in cases
                if isinstance(case.get("confidence"), (int, float))
            ]
        ),
        "cost_total_usd": sum(float(case["cost_usd"]) for case in cases),
        "cost_per_task_usd": mean_or_none([float(case["cost_usd"]) for case in cases]),
        "cost_per_task_usd_ci95": bootstrap_ci(
            [float(case["cost_usd"]) for case in cases],
            resamples=resamples,
            seed=seed,
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
        "oracle_gap": _oracle_gap(labeled),
    }


def _oracle_gap(labeled: list[dict[str, Any]]) -> float | None:
    routed = [case for case in labeled if case["action"] == "route"]
    if not routed:
        return None
    oracle_values: list[float] = []
    routed_values: list[float] = []
    for case in routed:
        quality_map = case["oracle"].get("quality_by_route") or {}
        oracle_quality = quality_map.get(case["oracle"]["route"])
        chosen_quality = quality_map.get(case["chosen"])
        if oracle_quality is None or chosen_quality is None:
            return None
        oracle_values.append(float(oracle_quality))
        routed_values.append(float(chosen_quality))
    oracle_mean = mean_or_none(oracle_values)
    routed_mean = mean_or_none(routed_values)
    if oracle_mean is None or routed_mean is None:
        return None
    return oracle_mean - routed_mean


def _oracle_summary(cases: list[dict[str, Any]], suite: EvalSuite) -> dict[str, Any]:
    n = len(cases)
    labeled = [case for case in cases if not case["oracle"]["ambiguous"]]
    config = suite.spec.oracle
    case_labeling = sum(float(case["oracle"].get("labeling_cost_usd", 0.0)) for case in cases)
    suite_labeling = config.labeling_cost_usd if config else 0.0
    total = suite_labeling + case_labeling
    correct = sum(1 for case in labeled if case["correct"])
    weighted_total = sum(float(case["weight_for_accuracy"]) for case in labeled)
    weighted_correct = sum(
        float(case["weight_for_accuracy"]) for case in labeled if case["correct"]
    )
    losses = [
        float(case["quality_loss"]) for case in labeled if case.get("quality_loss") is not None
    ]
    return {
        "labeling": {
            "labeling_method": config.labeling_method if config else None,
            "sampled_from": config.sampled_from if config else None,
            "ambiguity_rule": config.ambiguity_rule if config else None,
            "notes": config.notes if config else None,
            "suite_cost_usd": suite_labeling,
            "case_cost_usd": case_labeling,
            "total_usd": total,
            "per_case_usd": safe_div(total, n),
        },
        "labeled_cases": len(labeled),
        "ambiguous_cases": n - len(labeled),
        "oracle_agreement": safe_div(correct, len(labeled)),
        "oracle_agreement_all_cases": safe_div(sum(1 for case in cases if case["correct"]), n),
        "cost_weighted_accuracy": safe_div(weighted_correct, weighted_total),
        "cost_weighted_error": mean_or_none(losses),
        "quality_loss_cases": len(losses),
        "oracle_gap": _oracle_gap(labeled),
    }


def _calibration(cases: list[dict[str, Any]], suite: EvalSuite) -> dict[str, Any]:
    labeled = [case for case in cases if not case["oracle"]["ambiguous"]]
    answered = [
        case
        for case in labeled
        if case["action"] == "route" and isinstance(case.get("confidence"), (int, float))
    ]
    config = suite.spec.calibration
    bins = reliability_bins(
        [float(case["confidence"]) for case in answered],
        [bool(case["correct"]) for case in answered],
        n_bins=config.bins,
    )
    sweep = threshold_sweep(
        [
            {
                "action": case["action"],
                "confidence": case["confidence"],
                "correct": case["correct"],
            }
            for case in labeled
        ],
        steps=config.threshold_steps,
    )
    bins["sweep"] = sweep
    bins["notes"] = config.notes
    bins["partition"] = (
        "descriptive: all non-ambiguous cases for the threshold sweep; answered (routed) "
        "cases for reliability bins. No fit/eval split or temperature scaling in v0."
    )
    return bins


def _slices(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tags = sorted({tag for case in cases for tag in case.get("tags", [])})
    slices: dict[str, dict[str, Any]] = {}
    for tag in tags:
        subset = [case for case in cases if tag in case.get("tags", [])]
        if not subset:
            continue
        labeled = [case for case in subset if not case["oracle"]["ambiguous"]]
        abstain = abstain_metrics(
            [case["chosen"] == "abstain" for case in subset],
            [bool(case["oracle"]["ambiguous"]) for case in subset],
        )
        protocol = protocol_summary(subset)
        slices[tag] = {
            "cases": len(subset),
            "labeled_cases": len(labeled),
            "oracle_agreement": safe_div(
                sum(1 for case in labeled if case["correct"]), len(labeled)
            ),
            "abstain_rate": abstain["abstain_rate"],
            "abstain_precision": abstain["precision"],
            "abstain_recall": abstain["recall"],
            "cost_per_task_usd": mean_or_none([float(case["cost_usd"]) for case in subset]),
            "latency_p95_seconds": percentile(
                [float(case["latency_seconds"]) for case in subset], 0.95
            ),
            "protocol_error_rate": protocol["rate"],
        }
    return slices


def _failures(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for case in cases:
        if case["success"]:
            continue
        ambiguous = bool(case["oracle"]["ambiguous"])
        reasons: list[str] = []
        if case["error"]:
            reasons.append(f"{case['error']['code']}: {case['error']['message']}")
        if ambiguous:
            reasons.append(
                f"ambiguous request was routed to {case['chosen']!r} instead of abstaining"
            )
        elif not case["correct"]:
            reasons.append(
                f"oracle route {case['oracle']['route']!r}, router chose {case['chosen']!r} "
                f"(confidence {case['confidence']})"
            )
        failures.append(
            {
                "case_id": case["case_id"],
                "tags": list(case.get("tags", [])),
                "reason": ("; ".join(reasons) or "routing mismatch")[:1000],
                "error": (case.get("error") or {}).get("code"),
                "oracle_route": case["oracle"]["route"],
                "chosen": case["chosen"],
                "confidence": case["confidence"],
                "ambiguous": ambiguous,
            }
        )
    return failures
