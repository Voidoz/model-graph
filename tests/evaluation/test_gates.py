"""Contribution-gate tests: tolerance failures, hard gates, waivers, and CLI exit codes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from modelgraph.cli import app
from modelgraph.evaluation import (
    EvalError,
    run_e2e_suite,
    run_node_suite,
    run_routing_suite,
    validate_report,
)
from modelgraph.evaluation.compare import (
    GATE_WAIVER_SCHEMA,
    compare_against_baseline,
    load_policy,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "evals" / "baselines"
NODE_SUITE = ROOT / "evals" / "mock" / "js-specialist"
ROUTING_SUITE = ROOT / "evals" / "mock" / "triage"
E2E_SUITE = ROOT / "evals" / "e2e" / "composed-assistant"
E2E_GRAPH = ROOT / "examples" / "composed-assistant.graph.yaml"
E2E_CASES = [
    "js-debounce",
    "js-types",
    "general-rust",
    "story-lighthouse",
    "story-misrouted",
]
runner = CliRunner()


@pytest.fixture(scope="module")
def node_report() -> dict[str, Any]:
    return run_node_suite(NODE_SUITE, provider="mock", persist=False).to_dict()


@pytest.fixture(scope="module")
def routing_report() -> dict[str, Any]:
    return run_routing_suite(ROUTING_SUITE, provider="mock", persist=False).to_dict()


@pytest.fixture(scope="module")
def e2e_report() -> dict[str, Any]:
    return run_e2e_suite(
        E2E_SUITE,
        graph_path=E2E_GRAPH,
        profile="local-8gb",
        provider="mock",
        case_ids=E2E_CASES,
        persist=False,
    ).to_dict()


def _baseline(level: str) -> dict[str, Any]:
    return json.loads((BASELINES / f"{level}.json").read_text(encoding="utf-8"))


def _write_report(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _waiver(path: Path, **overrides: Any) -> Path:
    payload: dict[str, Any] = {
        "schema": GATE_WAIVER_SCHEMA,
        "approved_by": "test-maintainer",
        "reason": "Known fixture artifact; tracked in test.",
        "checks": ["node.task_success"],
        "expires": "pr",
    }
    payload.update(overrides)
    return _write_report(path, payload)


def test_unchanged_mock_reports_pass_every_level(
    tmp_path: Path,
    node_report: dict[str, Any],
    routing_report: dict[str, Any],
    e2e_report: dict[str, Any],
) -> None:
    for level, candidate in (
        ("node", node_report),
        ("routing", routing_report),
        ("e2e", e2e_report),
    ):
        comparison = compare_against_baseline(
            candidate,
            BASELINES / f"{level}.json",
            reports_dir=tmp_path,
            persist=False,
        )
        assert comparison.gate is not None
        assert comparison.gate["passed"] is True, level
        assert comparison.gate["hard_failures"] == [], level
        assert comparison.metrics["paired_cases"] == len(candidate["cases"])
        assert comparison.primary_metric["value"] == 0.0
        assert comparison.report_id


def test_unchanged_graph_report_is_schema_valid_with_metric_deltas(
    node_report: dict[str, Any],
) -> None:
    comparison = compare_against_baseline(
        node_report,
        BASELINES / "node.json",
        persist=False,
    )
    payload = comparison.to_dict()
    assert validate_report(payload) == []
    assert payload["gate"]["passed"] is True
    for check in payload["gate"]["checks"]:
        assert check["ci95"] is not None and len(check["ci95"]) == 2
        assert f"{check['metric']}_delta" in payload["metrics"]
        assert f"{check['metric']}_delta_ci95" in payload["metrics"]
    assert payload["primary_metric"]["name"] == "task_success_delta"


def test_deliberately_regressing_e2e_report_fails(
    tmp_path: Path,
    e2e_report: dict[str, Any],
) -> None:
    candidate = copy.deepcopy(e2e_report)
    regressed = next(case for case in candidate["cases"] if case["success"] is True)
    regressed["success"] = False
    regressed["checks_passed"] = False
    regressed["checks_passed_count"] = 0
    candidate["metrics"]["task_success"] = 0.6
    candidate["primary_metric"]["value"] = 0.6
    comparison = compare_against_baseline(
        candidate,
        BASELINES / "e2e.json",
        reports_dir=tmp_path,
    )
    assert comparison.gate is not None
    assert comparison.gate["passed"] is False
    assert "e2e.task_success" in comparison.gate["failures"]
    check = next(entry for entry in comparison.gate["checks"] if entry["id"] == "e2e.task_success")
    assert check["status"] == "fail"
    assert check["delta"] == pytest.approx(-0.2)
    assert comparison.report_dir is not None
    markdown = (comparison.report_dir / "report.md").read_text(encoding="utf-8")
    assert "**Result:** FAIL" in markdown
    assert "e2e.task_success" in markdown


def test_missing_gate_metric_fails_closed(
    tmp_path: Path,
    node_report: dict[str, Any],
) -> None:
    candidate = copy.deepcopy(node_report)
    del candidate["metrics"]["task_success"]
    candidate["primary_metric"]["value"] = None
    candidate["primary_metric"]["passed"] = None
    comparison = compare_against_baseline(
        candidate,
        BASELINES / "node.json",
        reports_dir=tmp_path,
        persist=False,
    )
    assert comparison.gate is not None
    assert comparison.gate["passed"] is False
    assert "node.task_success" in comparison.gate["failures"]
    check = next(entry for entry in comparison.gate["checks"] if entry["id"] == "node.task_success")
    assert check["status"] == "fail"
    assert check["delta"] is None
    assert "absent" in check["detail"]
    # The other gate metrics are still measured, so only the missing one fails.
    assert [entry["id"] for entry in comparison.gate["checks"] if entry["status"] == "fail"] == [
        "node.task_success"
    ]


def test_improving_e2e_report_passes(
    tmp_path: Path,
    e2e_report: dict[str, Any],
) -> None:
    candidate = copy.deepcopy(e2e_report)
    for case in candidate["cases"]:
        case["success"] = True
        case["valid"] = True
        case["checks_passed"] = True
        case["checks_passed_count"] = int(case.get("checks_total") or 0)
    candidate["metrics"]["task_success"] = 1.0
    candidate["primary_metric"]["value"] = 1.0
    comparison = compare_against_baseline(
        candidate,
        BASELINES / "e2e.json",
        reports_dir=tmp_path,
        persist=False,
    )
    assert comparison.gate is not None
    assert comparison.gate["passed"] is True
    check = next(entry for entry in comparison.gate["checks"] if entry["id"] == "e2e.task_success")
    assert check["delta"] == pytest.approx(0.2)
    assert check["status"] == "pass"


def test_hard_gate_failure_is_not_offset_by_a_quality_gain(
    node_report: dict[str, Any],
    tmp_path: Path,
) -> None:
    candidate = copy.deepcopy(node_report)
    # Keep the quality score perfect, but break the validity contract. A waiver must not hide it.
    candidate["metrics"]["validity"] = 0.5
    candidate["cases"][0]["valid"] = False
    waiver = _waiver(tmp_path / "waiver.json", checks=["hard.validity"])
    comparison = compare_against_baseline(
        candidate,
        BASELINES / "node.json",
        waivers=[waiver],
        reports_dir=tmp_path,
        persist=False,
    )
    assert comparison.gate is not None
    assert comparison.gate["passed"] is False
    assert "hard.validity" in comparison.gate["failures"]
    assert comparison.gate["hard_gates"]["validity"]["status"] == "fail"
    assert comparison.gate["waived"] == []
    assert comparison.gate["waivers"][0]["rejected"] == ["hard.validity"]
    markdown = comparison.to_markdown()
    assert "Hard-gate failures: hard.validity" in markdown
    assert "hard.validity" in markdown
    assert "test-maintainer" in markdown


def test_quality_waiver_is_visible_and_expires(
    node_report: dict[str, Any],
    tmp_path: Path,
) -> None:
    candidate = copy.deepcopy(node_report)
    candidate["metrics"]["task_success"] = 0.5
    candidate["cases"][0]["success"] = False
    candidate["cases"][1]["success"] = False
    candidate["primary_metric"]["value"] = 0.5
    waiver = _waiver(tmp_path / "waiver.json")
    comparison = compare_against_baseline(
        candidate,
        BASELINES / "node.json",
        waivers=[waiver],
        reports_dir=tmp_path,
        persist=False,
    )
    assert comparison.gate is not None
    assert comparison.gate["passed"] is True
    assert comparison.gate["waived"] == ["node.task_success"]
    assert comparison.gate["waivers"][0]["applied"] == ["node.task_success"]
    assert comparison.gate["waivers"][0]["reason"].startswith("Known fixture")
    markdown = comparison.to_markdown()
    assert "Waived checks: node.task_success" in markdown
    assert "Known fixture artifact" in markdown

    expired = _waiver(tmp_path / "expired.json", expires="2000-01-01")
    failed = compare_against_baseline(
        candidate,
        BASELINES / "node.json",
        waivers=[expired],
        reports_dir=tmp_path,
        persist=False,
    )
    assert failed.gate is not None
    assert failed.gate["passed"] is False
    assert failed.gate["waived"] == []
    assert failed.gate["waivers"][0]["rejected"] == ["node.task_success"]
    assert failed.gate["waivers"][0]["scope_error"] == "waiver is expired"


def test_secret_scan_hard_gate_fails(node_report: dict[str, Any], tmp_path: Path) -> None:
    candidate = copy.deepcopy(node_report)
    candidate["cases"][0]["responses"] = [{"content": "here is a key " + "AKIA" + "A" * 16}]
    comparison = compare_against_baseline(
        candidate,
        BASELINES / "node.json",
        reports_dir=tmp_path,
        persist=False,
    )
    assert comparison.gate is not None
    assert comparison.gate["passed"] is False
    assert comparison.gate["hard_gates"]["secret_scan"]["status"] == "fail"
    assert "aws-access-key" in comparison.gate["hard_gates"]["secret_scan"]["detail"]


def test_safety_and_budget_hard_gates(node_report: dict[str, Any], tmp_path: Path) -> None:
    candidate = copy.deepcopy(node_report)
    candidate["metrics"]["safety_violations"] = 2
    candidate["budget"]["stopped"] = True
    candidate["budget"]["stop_reason"] = "max_cost_usd"
    comparison = compare_against_baseline(
        candidate,
        BASELINES / "node.json",
        reports_dir=tmp_path,
        persist=False,
    )
    assert comparison.gate is not None
    assert comparison.gate["passed"] is False
    assert comparison.gate["hard_gates"]["safety"]["status"] == "fail"
    assert comparison.gate["hard_gates"]["budget"]["status"] == "fail"
    assert set(comparison.gate["hard_failures"]) == {"safety", "budget"}


def test_missing_baseline_is_an_explicit_error(
    tmp_path: Path,
    node_report: dict[str, Any],
) -> None:
    with pytest.raises(EvalError, match="baseline report not found"):
        compare_against_baseline(
            node_report,
            tmp_path / "does-not-exist.json",
            persist=False,
        )


def test_report_directories_are_accepted(
    tmp_path: Path,
    node_report: dict[str, Any],
) -> None:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    _write_report(candidate_dir / "report.json", node_report)
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    _write_report(baseline_dir / "report.json", _baseline("node"))
    comparison = compare_against_baseline(
        candidate_dir,
        baseline_dir,
        reports_dir=tmp_path / "comparisons",
    )
    assert comparison.gate is not None and comparison.gate["passed"] is True
    assert comparison.report_dir is not None
    assert (comparison.report_dir / "report.json").exists()


def test_policy_is_loaded_from_one_reviewed_file() -> None:
    policy, source = load_policy()
    assert source.name == "policy.json"
    assert policy["levels"]["node"]["metrics"]["task_success"]["tolerance"] == 0.02
    assert policy["hard_gates"]["validity"]["min"] == 1.0


def test_cli_exit_codes_and_readable_summary(
    tmp_path: Path,
    node_report: dict[str, Any],
) -> None:
    passing = _write_report(tmp_path / "passing.json", node_report)
    result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline",
            str(BASELINES / "node.json"),
            "--candidate",
            str(passing),
            "--reports-dir",
            str(tmp_path / "comparisons"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "gate:     PASS" in result.output
    assert "check:    node.task_success pass" in result.output
    assert "hard:     validity pass" in result.output

    regressed = copy.deepcopy(node_report)
    regressed["metrics"]["task_success"] = 0.5
    regressed["cases"][0]["success"] = False
    regressed["cases"][1]["success"] = False
    regressed_path = _write_report(tmp_path / "regressed.json", regressed)
    result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline",
            str(BASELINES / "node.json"),
            "--candidate",
            str(regressed_path),
            "--reports-dir",
            str(tmp_path / "comparisons"),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "gate:     FAIL" in result.output

    result = runner.invoke(
        app,
        [
            "eval",
            "compare",
            "--baseline",
            str(tmp_path / "missing.json"),
            "--candidate",
            str(passing),
        ],
    )
    assert result.exit_code == 3
    assert "baseline report not found" in result.output
