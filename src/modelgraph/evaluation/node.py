"""Node-level evaluation runner.

The runner uses the compiler for the graph, the runtime renderer for the exact prompt, the
runtime executors for provider calls/artifact parsing/repair, and the deterministic mock for
offline tests. It never re-implements those contracts; it measures them.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from modelgraph.compiler import compile_graph
from modelgraph.compiler.ir import CompiledGraph, CompiledNode
from modelgraph.contracts.graph import NodeKind
from modelgraph.errors import ModelGraphError
from modelgraph.evaluation.checks import JudgeHook, evaluate_checks
from modelgraph.evaluation.errors import EvalError, SuiteError
from modelgraph.evaluation.metrics import (
    bootstrap_ci,
    latency_summary,
    mean_or_none,
    percentile,
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
from modelgraph.evaluation.suite import EvalCase, EvalSuite, NodeTarget, load_suite
from modelgraph.evaluation.support import (
    artifact_record,
    build_state,
    case_budget,
    case_fingerprint,
    commands_for,
    costs_summary,
    error_payload,
    graph_summary,
    limitations_for,
    node_validity_issues,
    protocol_errors_from_issues,
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
from modelgraph.runtime.executors import Executor, WorkerExecutor
from modelgraph.runtime.renderer import render_prompt
from modelgraph.runtime.trace import TraceRecorder
from modelgraph.util import new_id, sha256_text


def run_node_suite(
    suite: EvalSuite | str | Path,
    *,
    graph_path: str | Path | None = None,
    provider: str | None = None,
    allow_live: bool = False,
    registry: Registry | None = None,
    judge: JudgeHook | None = None,
    reports_dir: str | Path | None = None,
    case_ids: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
    command: str | None = None,
    persist: bool = True,
    allow_commands: bool = True,
    bootstrap_resamples: int = 2000,
    seed: int = 1234,
) -> EvalReport:
    """Run one node suite and return (and optionally persist) its report."""
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
        if not isinstance(loaded.spec.target, NodeTarget):
            raise SuiteError(f"suite {loaded.metadata.id!r} is not a node suite")
        node = _target_node(graph, loaded.spec.target.node)

        selected = loaded.select_cases(case_ids=case_ids, tags=tags, limit=limit)
        selection = loaded.selection_summary(
            list(selected), case_ids=case_ids, tags=tags, limit=limit
        )
        report_root = Path(reports_dir) if reports_dir is not None else default_report_dir()
        report_id = new_report_id()
        cli_command, commands, reproduce = commands_for(
            level="node",
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
                provider_instance = pool.for_case(case, node)
                result = _run_node_case(
                    case,
                    graph,
                    node,
                    provider_instance,
                    loaded,
                    allow_commands=allow_commands,
                    judge=judge,
                )
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

        metrics = _node_metrics(results, seed=seed, resamples=bootstrap_resamples)
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
        primary, thresholds, gate = build_primary_metric(
            loaded.spec.metric, metrics, loaded.spec.thresholds
        )
        if stopped:
            gate["passed"] = False
            gate.setdefault("failures", []).append("budget")
            gate["budget_stopped"] = stopped
        judge_summary = _judge_summary(loaded, results)
        limitations = limitations_for(
            level="node",
            provider=provider_name,
            case_count=len(results),
            oracle_labeling_method=None,
            command_checks=any(
                check.type == "command" for case in loaded.cases for check in case.checks
            ),
            judge_errors=[
                str(result["judge_error"]) for result in results if result.get("judge_error")
            ],
        )
        if stopped:
            limitations.append(stopped)
        determinism_payload = {
            "suite_hash": loaded.content_hash,
            "graph_hash": graph.graph_hash,
            "level": "node",
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
            level="node",
            claim=loaded.spec.claim or loaded.metadata.description,
            suite=loaded.summary(),
            graph=graph_summary(graph, graph_file),
            target=target_summary(loaded, node.id, node.kind.value),
            provider=provider_summary(provider_name, loaded, pool),
            command=cli_command or "mg eval node (CLI wiring lands in T-0021)",
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
            judge=judge_summary,
        )
        if persist:
            report.write(report_root)
        return report
    finally:
        pool.close()


def _target_node(graph: CompiledGraph, node_id: str) -> CompiledNode:
    try:
        node = graph.node(node_id)
    except KeyError as exc:
        raise SuiteError(
            f"target node {node_id!r} is not in graph {graph.id}@{graph.version}"
        ) from exc
    if node.kind == NodeKind.router:
        raise SuiteError(
            f"node {node_id!r} is a router; use a routing suite for routing evaluation"
        )
    if node.kind == NodeKind.subgraph:
        raise SuiteError(
            f"node {node_id!r} is a subgraph; node evaluation requires a worker or joiner"
        )
    return node


def _run_node_case(
    case: EvalCase,
    graph: CompiledGraph,
    node: CompiledNode,
    provider: Provider,
    suite: EvalSuite,
    *,
    allow_commands: bool,
    judge: JudgeHook | None,
) -> dict[str, Any]:
    state = build_state(graph, case.input)
    trace = TraceRecorder(new_id("evalcase"))
    budget = case_budget(graph, suite)
    rendered = render_prompt(node, graph, state)
    started = time.monotonic()
    error: ModelGraphError | None = None
    try:
        WorkerExecutor(Executor(provider, budget=budget, trace=trace)).run(node, graph, state)
    except ModelGraphError as exc:
        error = exc
    latency = time.monotonic() - started
    responses = provider_responses(trace)
    produced = [artifact for artifact in state.artifacts if artifact.producer == node.id]
    validity_issues = node_validity_issues(node, state, error, responses)
    outcome = evaluate_checks(
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
    if outcome.judge_error:
        validity_issues.append(
            {
                "code": "EVAL-JUDGE",
                "message": outcome.judge_error,
                "source": "judge",
            }
        )
    valid = not validity_issues
    success = valid and outcome.passed
    usage = usage_snapshot(budget)
    protocol_errors = protocol_errors_from_issues(validity_issues)
    check_results = [result.to_dict() for result in outcome.results]
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
        "outcome": "succeeded" if error is None else "failed",
        "valid": valid,
        "success": success,
        "run_success": error is None,
        "quality": outcome.judge_score,
        "checks_passed": outcome.passed,
        "check_results": check_results,
        "checks_total": len(check_results),
        "checks_passed_count": sum(1 for result in outcome.results if result.passed),
        "judge_called": outcome.judge_called,
        "judge_error": outcome.judge_error,
        "artifacts": [artifact_record(artifact) for artifact in produced],
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
        "validity_issues": validity_issues,
        "error": error_payload(error),
    }


def _node_metrics(
    cases: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    n = len(cases)
    successes = sum(1 for case in cases if case["success"])
    valid = sum(1 for case in cases if case["valid"])
    run_success = sum(1 for case in cases if case["run_success"])
    checks_total = sum(int(case["checks_total"]) for case in cases)
    checks_passed = sum(int(case["checks_passed_count"]) for case in cases)
    weights = [float(case.get("weight", 1.0)) for case in cases]
    weighted_success = (
        sum(weight for weight, case in zip(weights, cases, strict=True) if case["success"])
        / sum(weights)
        if sum(weights)
        else None
    )
    quality = [float(case["quality"]) for case in cases if case.get("quality") is not None]
    costs = [float(case["cost_usd"]) for case in cases]
    latencies = [float(case["latency_seconds"]) for case in cases]
    protocol = protocol_summary(cases)
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
        "cost_total_usd": sum(costs),
        "cost_per_task_usd": mean_or_none(costs),
        "cost_per_task_usd_ci95": bootstrap_ci(costs, resamples=resamples, seed=seed),
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
        "protocol_error_rate": protocol["rate"],
        "protocol_errors_per_100_calls": protocol["per_100_calls"],
        "failure_count": n - successes,
        "judged_cases": len(quality),
    }


def _slices(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tags = sorted({tag for case in cases for tag in case.get("tags", [])})
    slices: dict[str, dict[str, Any]] = {}
    for tag in tags:
        subset = [case for case in cases if tag in case.get("tags", [])]
        if not subset:
            continue
        total = len(subset)
        checks_total = sum(int(case["checks_total"]) for case in subset)
        checks_passed = sum(int(case["checks_passed_count"]) for case in subset)
        protocol = protocol_summary(subset)
        slices[tag] = {
            "cases": total,
            "task_success": sum(1 for case in subset if case["success"]) / total,
            "validity": sum(1 for case in subset if case["valid"]) / total,
            "run_success_rate": sum(1 for case in subset if case["run_success"]) / total,
            "check_pass_rate": checks_passed / checks_total if checks_total else None,
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
        failures.append(
            {
                "case_id": case["case_id"],
                "tags": list(case.get("tags", [])),
                "reason": ("; ".join(dict.fromkeys(reasons)) or "checks failed")[:1000],
                "error": (case.get("error") or {}).get("code"),
                "failed_checks": failed_checks,
                "valid": case["valid"],
            }
        )
    return failures


def _judge_summary(suite: EvalSuite, cases: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        f"{case['case_id']}: {case['judge_error']}" for case in cases if case.get("judge_error")
    ]
    return {
        "configured": suite.spec.judge.enabled,
        "model": suite.spec.judge.model,
        "rubric": suite.spec.rubric or suite.spec.judge.rubric,
        "calls": sum(1 for case in cases if case.get("judge_called")),
        "judged_cases": sum(1 for case in cases if case.get("quality") is not None),
        "errors": errors,
        "error": errors[0] if errors else None,
    }
