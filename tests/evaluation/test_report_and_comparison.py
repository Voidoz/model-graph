import copy
import json
from pathlib import Path

import pytest

from modelgraph.evaluation import (
    EvalError,
    SuiteError,
    compare_reports,
    run_node_suite,
    validate_report,
    write_report,
)

ROOT = Path(__file__).resolve().parents[2]
JS_SUITE = ROOT / "evals" / "mock" / "js-specialist"


def test_report_schema_rejects_incomplete_reports() -> None:
    assert validate_report({})
    report = run_node_suite(JS_SUITE, reports_dir=None, persist=False)
    payload = report.to_dict()
    assert validate_report(payload) == []
    payload["level"] = "unknown"
    assert validate_report(payload)


def test_write_report_refuses_invalid_payload(tmp_path: Path) -> None:
    report = run_node_suite(JS_SUITE, persist=False, bootstrap_resamples=10)
    report.report_id = ""  # invalid report id, still a dataclass
    with pytest.raises(SuiteError, match="invalid eval report"):
        write_report(report, reports_dir=tmp_path)


def test_paired_comparison_of_identical_runs(tmp_path: Path) -> None:
    baseline = run_node_suite(JS_SUITE, reports_dir=tmp_path)
    candidate = run_node_suite(JS_SUITE, reports_dir=tmp_path)
    comparison = compare_reports(candidate, baseline, reports_dir=tmp_path)
    assert comparison.level == "comparison"
    assert comparison.metrics["paired_cases"] == 6
    assert comparison.metrics["primary_delta"] == 0.0
    assert comparison.metrics["wins"] == 0
    assert comparison.metrics["losses"] == 0
    assert comparison.metrics["ties"] == 6
    assert comparison.metrics["primary_delta_ci95"] == [0.0, 0.0]
    assert validate_report(comparison.to_dict()) == []
    assert comparison.baseline is not None
    assert comparison.candidate is not None
    markdown = comparison.to_markdown()
    assert "Paired comparison" in markdown
    assert "baseline" in markdown
    assert comparison.report_dir is not None
    assert (comparison.report_dir / "report.json").exists()


def test_paired_comparison_detects_a_regression(tmp_path: Path) -> None:
    baseline = run_node_suite(JS_SUITE, reports_dir=tmp_path, persist=False)
    original = baseline.to_dict()
    candidate = copy.deepcopy(original)
    candidate["report_id"] = "eval_candidate"
    candidate["cases"][0]["success"] = False
    candidate["cases"][0]["checks_passed"] = False
    candidate["cases"][0]["checks_passed_count"] = 0
    candidate["primary_metric"]["value"] = 5 / 6
    candidate["primary_metric"]["passed"] = False
    candidate["gate"]["passed"] = False
    comparison = compare_reports(candidate, original, reports_dir=tmp_path, persist=False)
    assert comparison.metrics["primary_delta"] == pytest.approx(-1 / 6)
    assert comparison.metrics["wins"] == 0
    assert comparison.metrics["losses"] == 1
    assert comparison.metrics["ties"] == 5
    assert comparison.metrics["primary_delta_ci95"] is not None
    assert comparison.failures
    assert comparison.failures[0]["case_id"] == "js-001"


def test_comparison_rejects_suite_mismatch(tmp_path: Path) -> None:
    baseline = run_node_suite(JS_SUITE, reports_dir=tmp_path, persist=False)
    candidate = baseline.to_dict()
    candidate["report_id"] = "eval_candidate"
    candidate["suite"] = dict(candidate["suite"])
    candidate["suite"]["hash"] = "sha256:" + "0" * 64
    with pytest.raises(EvalError, match="different suite versions"):
        compare_reports(candidate, baseline.to_dict(), persist=False)
    allowed = compare_reports(
        candidate,
        baseline.to_dict(),
        allow_suite_mismatch=True,
        persist=False,
    )
    assert any("Suite hashes differ" in item for item in allowed.limitations)


def test_report_files_are_machine_readable(tmp_path: Path) -> None:
    report = run_node_suite(JS_SUITE, reports_dir=tmp_path)
    assert report.report_dir is not None
    payload = json.loads((report.report_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["determinism"]["fingerprint"].startswith("sha256:")
    assert payload["suite"]["hash"].startswith("sha256:")
    assert payload["graph"]["hash"].startswith("sha256:")
    assert payload["environment"]["python"]
    assert payload["commands"]
