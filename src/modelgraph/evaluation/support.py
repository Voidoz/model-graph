"""Shared runner plumbing: runtime state, budgets, commands, and determinism payloads."""

from __future__ import annotations

import json
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

from modelgraph.compiler.ir import CompiledGraph, CompiledNode
from modelgraph.contracts.envelope import Artifact
from modelgraph.contracts.graph import ArtifactType
from modelgraph.errors import ModelGraphError
from modelgraph.evaluation.suite import CaseInput, EvalSuite
from modelgraph.runtime.budget import Budget
from modelgraph.runtime.envelope import RuntimeState
from modelgraph.runtime.trace import TraceRecorder
from modelgraph.util import sha256_text, utc_now

RUN_FUNCTION = {
    "node": "run_node_suite",
    "routing": "run_routing_suite",
    "e2e": "run_e2e_suite",
}


def build_state(graph: CompiledGraph, case_input: CaseInput) -> RuntimeState:
    state = RuntimeState.create(graph, case_input.task)
    if not case_input.artifacts:
        return state
    inputs = {artifact.name: artifact for artifact in state.inputs}
    for item in case_input.artifacts:
        inputs[item.name] = Artifact(
            name=item.name,
            type=item.type,
            media_type=item.media_type,
            content=item.content,
            producer="$case",
            created_at=utc_now(),
            hash=sha256_text(item.content),
        )
    state.inputs = list(inputs.values())
    return state


def case_budget(graph: CompiledGraph, suite: EvalSuite) -> Budget:
    budget = Budget.from_defaults(graph.defaults)
    config = suite.spec.budget
    if config.max_case_cost_usd is not None:
        budget.max_cost_usd = config.max_case_cost_usd
    if config.max_case_tokens is not None:
        budget.max_total_tokens = config.max_case_tokens
    if config.max_case_wall_seconds is not None:
        budget.max_wall_seconds = config.max_case_wall_seconds
    return budget


def usage_snapshot(budget: Budget) -> dict[str, Any]:
    return {
        "steps": budget.used_steps,
        "input_tokens": budget.used_input_tokens,
        "output_tokens": budget.used_output_tokens,
        "total_tokens": budget.used_tokens,
        "cost_usd": budget.used_cost_usd,
    }


def provider_responses(trace: TraceRecorder) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for event in trace.events:
        if event.get("event") != "provider.response":
            continue
        responses.append(
            {
                "text": event.get("text", ""),
                "finish_reason": event.get("finish_reason", "stop"),
                "usage": event.get("usage", {}),
                "latency_seconds": event.get("latency_seconds"),
                "attempt": event.get("attempt"),
            }
        )
    return responses


def provider_calls(trace: TraceRecorder) -> int:
    return sum(1 for event in trace.events if event.get("event") == "provider.request")


def render_hashes(trace: TraceRecorder) -> list[str]:
    return [
        str(event.get("prompt_hash"))
        for event in trace.events
        if event.get("event") == "render.completed"
    ]


def error_payload(error: BaseException | None) -> dict[str, Any] | None:
    if error is None:
        return None
    details = getattr(error, "details", {}) or {}
    code = details.get("code", getattr(error, "code", "MG-0000"))
    return {
        "code": str(code),
        "message": str(getattr(error, "message", error)),
        "path": str(getattr(error, "path", "")),
    }


def error_code(error: BaseException | None) -> str | None:
    payload = error_payload(error)
    return payload["code"] if payload else None


def protocol_error(code: str, message: str, *, source: str) -> dict[str, Any]:
    return {"code": code, "message": message, "source": source}


def protocol_errors_from_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep runtime/protocol issues (``MG-2xxx``) for protocol-error accounting."""
    seen: set[tuple[str, str]] = set()
    errors: list[dict[str, Any]] = []
    for issue in issues:
        code = str(issue.get("code"))
        if not code.startswith("MG-2"):
            continue
        key = (code, str(issue.get("message")))
        if key in seen:
            continue
        seen.add(key)
        errors.append(
            protocol_error(code, str(issue.get("message")), source=str(issue.get("source", "")))
        )
    return errors


def artifact_record(artifact: Artifact) -> dict[str, Any]:
    return {
        "name": artifact.name,
        "type": artifact.type.value,
        "media_type": artifact.media_type,
        "producer": artifact.producer,
        "content": artifact.content,
        "hash": artifact.hash or sha256_text(artifact.content),
        "bytes": len(artifact.content.encode("utf-8")),
        "metadata": artifact.metadata,
    }


def node_validity_issues(
    node: CompiledNode,
    state: RuntimeState,
    error: ModelGraphError | None,
    responses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Contract-level validity for one node result, independent of suite checks."""
    issues: list[dict[str, Any]] = []
    if error is not None:
        payload = error_payload(error) or {}
        issues.append(
            {
                "code": payload.get("code", "MG-2000"),
                "message": payload.get("message", str(error)),
                "source": "runtime",
            }
        )
        return issues
    produced = {
        artifact.name: artifact for artifact in state.artifacts if artifact.producer == node.id
    }
    for write in node.writes:
        artifact = produced.get(write.name)
        if artifact is None:
            issues.append(
                {
                    "code": "MG-2202",
                    "message": f"artifact {write.name!r} was not produced",
                    "source": "contract",
                }
            )
            continue
        if not artifact.content.strip():
            issues.append(
                {
                    "code": "MG-2202",
                    "message": f"artifact {write.name!r} is empty",
                    "source": "contract",
                }
            )
            continue
        if write.type == ArtifactType.json or write.strict_json:
            try:
                json.loads(artifact.content)
            except json.JSONDecodeError as exc:
                issues.append(
                    {
                        "code": "MG-2202",
                        "message": (
                            f"artifact {write.name!r} is not valid JSON: {exc.msg} "
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


def protocol_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    cases_with_errors = 0
    calls = 0
    for case in cases:
        errors = case.get("protocol_errors") or []
        if errors:
            cases_with_errors += 1
            events.extend(errors)
        calls += int(case.get("provider_calls", 0))
    codes = Counter(str(event.get("code")) for event in events)
    return {
        "total_events": len(events),
        "cases_with_errors": cases_with_errors,
        "rate": cases_with_errors / len(cases) if cases else None,
        "per_100_calls": (len(events) / calls * 100) if calls else None,
        "by_code": dict(sorted(codes.items())),
        "provider_calls": calls,
    }


def usage_totals(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(cases),
        "input_tokens": sum(int(case.get("usage", {}).get("input_tokens", 0)) for case in cases),
        "output_tokens": sum(int(case.get("usage", {}).get("output_tokens", 0)) for case in cases),
        "total_tokens": sum(int(case.get("usage", {}).get("total_tokens", 0)) for case in cases),
        "cost_usd": sum(float(case.get("usage", {}).get("cost_usd", 0.0)) for case in cases),
        "provider_calls": sum(int(case.get("provider_calls", 0)) for case in cases),
    }


def graph_summary(graph: CompiledGraph, path: Path) -> dict[str, Any]:
    return {
        "id": graph.id,
        "version": graph.version,
        "hash": graph.graph_hash,
        "path": str(path),
    }


def target_summary(suite: EvalSuite, node_id: str | None, kind: str) -> dict[str, Any]:
    return {
        "level": suite.level,
        "node": node_id,
        "kind": kind,
        "suite_graph": suite.spec.target.graph,
        "selection": suite.spec.selection.model_dump(mode="json", exclude_none=True),
    }


def commands_for(
    *,
    level: str,
    suite_path: Path,
    graph_path: Path,
    provider: str,
    reports_dir: Path,
    case_ids: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
    profile: str | None = None,
) -> tuple[str, list[str], str]:
    cli_parts = [
        "mg",
        "eval",
        level,
        "--suite",
        str(suite_path),
        "--graph",
        str(graph_path),
        "--provider",
        provider,
    ]
    if provider != "mock":
        # Live providers are gated at run time, so a reproduction command is only complete
        # when it carries the same explicit gate. API callers may have used MG_EVAL_LIVE=1;
        # ``--live`` is equivalent and makes the recorded command runnable as written.
        cli_parts.append("--live")
    cli_parts.extend(["--reports-dir", str(reports_dir)])
    for case_id in case_ids or []:
        cli_parts.extend(["--case", case_id])
    for tag in tags or []:
        cli_parts.extend(["--tag", tag])
    if limit is not None:
        cli_parts.extend(["--limit", str(limit)])
    if profile is not None:
        cli_parts.extend(["--profile", profile])
    command = " ".join(shlex.quote(part) for part in cli_parts)
    function = RUN_FUNCTION[level]
    code = (
        f"from modelgraph.evaluation import {function}; "
        f"{function}({str(suite_path)!r}, graph_path={str(graph_path)!r}, "
        f"provider={provider!r}, reports_dir={str(reports_dir)!r}"
    )
    if profile is not None:
        code += f", profile={profile!r}"
    if provider != "mock":
        code += ", allow_live=True"
    code += ")"
    reproduce = f'python -c "{code}"'
    return command, [command, reproduce], reproduce


def limitations_for(
    *,
    level: str,
    provider: str,
    case_count: int,
    oracle_labeling_method: str | None,
    command_checks: bool,
    judge_errors: list[str] | None = None,
) -> list[str]:
    limitations: list[str] = []
    if provider == "mock":
        limitations.append(
            "The mock provider serves scripted responses; these metrics validate the harness "
            "and suite, not model quality."
        )
    else:
        limitations.append(
            "Live provider output is non-deterministic; report model versions and hardware "
            "with the result and repeat runs before making a claim."
        )
    if case_count < 20:
        limitations.append(
            f"Small suite (n={case_count}); treat this as a development smoke result, not a "
            "quality claim."
        )
    if level == "routing" and case_count < 200:
        limitations.append(
            f"Routing n={case_count} is below the 200-case guidance; per-class and calibration "
            "numbers are indicative only."
        )
    limitations.append(
        "Wall-clock latency includes harness overhead and varies between runs; it is excluded "
        "from the determinism fingerprint."
    )
    if level == "routing":
        limitations.append(
            "Calibration bins and the threshold sweep are descriptive on the evaluated "
            "partition; v0 does not fit temperature scaling or hold out a calibration split."
        )
        if oracle_labeling_method:
            if (
                "synthetic" in oracle_labeling_method.lower()
                or "authored" in oracle_labeling_method.lower()
            ):
                limitations.append(
                    f"Oracle labels came from {oracle_labeling_method!r}; they are authored "
                    "labels and were not independently re-labeled."
                )
        else:
            limitations.append("Oracle labeling method is not documented in the suite.")
    if command_checks:
        limitations.append(
            "Command checks execute local suite-authored processes; suites are trusted test "
            "code and should be reviewed before running."
        )
    if judge_errors:
        limitations.append(
            "One or more rubric checks failed closed because the judge was unavailable or its "
            "output was invalid."
        )
    return limitations


# Timing fields are measurements, not outcomes. ``usage.elapsed_seconds`` comes from the
# runtime budget clock and varies between identical scripted runs, so it must never enter the
# determinism fingerprint. Match by name rather than by an allowlist so a future timing field
# cannot silently make the fingerprint flaky again.
_TIMING_MARKERS = ("elapsed", "latency", "duration", "wall", "timestamp", "started_at", "ended_at")


def _deterministic_usage(usage: Any) -> Any:
    if not isinstance(usage, dict):
        return usage
    return {
        key: value
        for key, value in usage.items()
        if not (
            isinstance(key, str)
            and (
                key.endswith("_seconds")
                or key.endswith("_ms")
                or any(marker in key.lower() for marker in _TIMING_MARKERS)
            )
        )
    }


def case_fingerprint(case: dict[str, Any]) -> dict[str, Any]:
    """Deterministic subset of a case result for the report fingerprint."""
    payload: dict[str, Any] = {
        "case_id": case.get("case_id"),
        "valid": case.get("valid"),
        "success": case.get("success"),
        "run_success": case.get("run_success"),
        "cost_usd": case.get("cost_usd"),
        "usage": _deterministic_usage(case.get("usage")),
        "prompt_hash": case.get("prompt_hash"),
        "error": (case.get("error") or {}).get("code"),
        "protocol_error_codes": sorted(
            str(error.get("code")) for error in case.get("protocol_errors", [])
        ),
        "artifacts": [
            {
                "name": artifact.get("name"),
                "type": artifact.get("type"),
                "hash": artifact.get("hash"),
                "bytes": artifact.get("bytes"),
            }
            for artifact in case.get("artifacts", [])
        ],
        "checks": [
            {
                "type": check.get("type"),
                "artifact": check.get("artifact"),
                "passed": check.get("passed"),
                "score": check.get("score"),
            }
            for check in case.get("check_results", [])
        ],
        "responses": [response.get("text") for response in case.get("responses", [])],
        "finish_reasons": case.get("finish_reasons"),
        "hop_count": case.get("hop_count"),
    }
    if "decision" in case:
        payload["decision"] = case.get("decision")
        payload["chosen"] = case.get("chosen")
        payload["confidence"] = case.get("confidence")
        payload["correct"] = case.get("correct")
        payload["oracle_route"] = (case.get("oracle") or {}).get("route")
        payload["oracle_ambiguous"] = (case.get("oracle") or {}).get("ambiguous")
    # E2E cases carry the routing outcome at the top level rather than in a ``decision``
    # object. Include it when present so a changed route or oracle label cannot leave the
    # fingerprint unchanged.
    for key in ("chosen_route", "route_correct", "oracle_route", "oracle_ambiguous"):
        if key in case and key not in payload:
            payload[key] = case.get(key)
    return payload


def costs_summary(
    cases: list[dict[str, Any]], *, pricing_note: str | None = None
) -> dict[str, Any]:
    costs = [float(case.get("cost_usd", 0.0)) for case in cases]
    total = sum(costs)
    per_case = total / len(costs) if costs else None
    return {
        "total_usd": total,
        "per_task_usd": per_case,
        "pricing": "manifest-declared",
        "note": pricing_note,
    }


def suite_usage_note(suite: EvalSuite, provider: str) -> str | None:
    if provider == "mock":
        return (
            "Mock usage uses the mock provider's token estimate; cost follows the resolved "
            "model manifest, which is free for the example models."
        )
    return None
