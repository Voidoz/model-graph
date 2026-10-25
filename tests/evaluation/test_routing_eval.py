import json
from pathlib import Path

import pytest

from modelgraph.compiler import compile_graph
from modelgraph.evaluation import load_suite, run_routing_suite, validate_report
from modelgraph.runtime.envelope import RuntimeState
from modelgraph.runtime.renderer import render_prompt

ROOT = Path(__file__).resolve().parents[2]
TRIAGE_SUITE = ROOT / "evals" / "mock" / "triage"


def deterministic_metrics(report) -> dict:
    return {key: value for key, value in report.metrics.items() if not key.startswith("latency")}


def test_routing_metrics_confusion_abstain_and_calibration(tmp_path: Path) -> None:
    report = run_routing_suite(TRIAGE_SUITE, reports_dir=tmp_path)
    assert validate_report(report.to_dict()) == []
    assert report.metrics["cases"] == 12
    assert report.metrics["labeled_cases"] == 10
    assert report.metrics["ambiguous_cases"] == 2
    assert report.metrics["oracle_agreement"] == pytest.approx(0.6)
    assert report.metrics["oracle_agreement_ci95"] is not None
    assert report.metrics["cost_weighted_accuracy"] == pytest.approx(9 / 14)
    assert report.metrics["cost_weighted_error"] == pytest.approx(0.075)
    assert report.metrics["oracle_gap"] == pytest.approx(0.075)
    assert report.metrics["abstain_precision"] == pytest.approx(0.5)
    assert report.metrics["abstain_recall"] == pytest.approx(0.5)
    assert report.metrics["ece"] is not None
    assert report.gate is not None and report.gate["passed"] is True

    assert report.confusion is not None
    labels = report.confusion["labels"]
    assert labels == ["coding", "story", "abstain", "error"]
    matrix = report.confusion["matrix"]
    assert len(matrix) == len(labels)
    assert sum(sum(row) for row in matrix) == 12
    assert report.abstain is not None
    assert report.abstain["true_positive"] == 1
    assert report.abstain["false_positive"] == 1
    assert report.abstain["false_negative"] == 1
    assert report.calibration is not None
    assert report.calibration["n"] == 8
    assert len(report.calibration["bins"]) == 5
    assert len(report.calibration["sweep"]) == 11
    assert report.oracle is not None
    assert report.oracle["labeling"]["labeling_method"] == "synthetic-authored"
    assert report.oracle["labeling"]["total_usd"] == 0.0
    assert report.oracle["labeled_cases"] == 10


def test_routing_report_files_and_markdown(tmp_path: Path) -> None:
    report = run_routing_suite(TRIAGE_SUITE, reports_dir=tmp_path)
    assert report.report_dir is not None
    report_dir = report.report_dir
    payload = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["level"] == "routing"
    assert payload["command"].startswith("mg eval routing")
    assert "run_routing_suite" in payload["reproduce"]
    markdown = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "Confusion matrix" in markdown
    assert "Abstain" in markdown
    assert "Calibration" in markdown
    assert "Threshold sweep" in markdown
    assert "Oracle labels" in markdown
    assert len(list((report_dir / "cases").glob("*.json"))) == 12
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["level"] == "routing"
    assert manifest["oracle"]["labeling"]["labeling_method"] == "synthetic-authored"


def test_routing_prompt_is_the_exact_runtime_render(tmp_path: Path) -> None:
    report = run_routing_suite(TRIAGE_SUITE, reports_dir=tmp_path)
    suite = load_suite(TRIAGE_SUITE)
    case = next(case for case in suite.cases if case.id == "tri-001")
    compiled = compile_graph(ROOT / "examples" / "hello-triage.graph.yaml", strict_models=True)
    assert compiled.ok and compiled.graph is not None
    graph = compiled.graph
    state = RuntimeState.create(graph, case.input.task)
    expected = render_prompt(graph.node("triage"), graph, state)
    actual = next(item for item in report.cases if item["case_id"] == "tri-001")
    assert actual["prompt_hash"] == expected.prompt_hash
    assert actual["prompt"]["messages"] == [message.to_dict() for message in expected.messages]
    assert actual["prompt_hash"] in actual["render_hashes"]


def test_routing_report_is_deterministic_with_scripted_decisions(tmp_path: Path) -> None:
    first = run_routing_suite(TRIAGE_SUITE, reports_dir=tmp_path)
    second = run_routing_suite(TRIAGE_SUITE, reports_dir=tmp_path)
    assert first.determinism["fingerprint"] == second.determinism["fingerprint"]
    assert deterministic_metrics(first) == deterministic_metrics(second)
    first_decisions = [
        {
            "case_id": case["case_id"],
            "chosen": case["chosen"],
            "confidence": case["confidence"],
            "correct": case["correct"],
            "prompt_hash": case["prompt_hash"],
            "cost_usd": case["cost_usd"],
        }
        for case in first.cases
    ]
    second_decisions = [
        {
            "case_id": case["case_id"],
            "chosen": case["chosen"],
            "confidence": case["confidence"],
            "correct": case["correct"],
            "prompt_hash": case["prompt_hash"],
            "cost_usd": case["cost_usd"],
        }
        for case in second.cases
    ]
    assert first_decisions == second_decisions


def test_low_confidence_fallback_is_an_abstain(tmp_path: Path) -> None:
    report = run_routing_suite(TRIAGE_SUITE, reports_dir=tmp_path)
    case = next(item for item in report.cases if item["case_id"] == "tri-011")
    assert case["confidence"] == pytest.approx(0.45)
    assert case["action"] == "abstain"
    assert case["chosen"] == "abstain"
    assert case["fallback_to"] == "coding"
    assert case["decision"]["action"] == "fallback"
    assert case["correct"] is False


def test_protocol_error_is_counted_and_raw_response_preserved(tmp_path: Path) -> None:
    report = run_routing_suite(TRIAGE_SUITE, reports_dir=tmp_path)
    case = next(item for item in report.cases if item["case_id"] == "tri-012")
    assert case["error"]["code"] == "MG-2101"
    assert case["protocol_errors"][0]["code"] == "MG-2101"
    assert case["responses"][0]["text"] == "This looks like a coding question."
    assert len(case["responses"]) == 2  # original + repair attempt
    assert report.metrics["protocol_error_rate"] == pytest.approx(1 / 12)
    assert report.protocol_errors["by_code"] == {"MG-2101": 1}
    assert any(failure["case_id"] == "tri-012" for failure in report.failures)


def test_ambiguous_requests_are_reported_separately(tmp_path: Path) -> None:
    report = run_routing_suite(TRIAGE_SUITE, reports_dir=tmp_path)
    ambiguous = {case["case_id"] for case in report.cases if case["oracle"]["ambiguous"]}
    assert ambiguous == {"tri-009", "tri-010"}
    assert report.failures
    assert all(
        failure["ambiguous"] for failure in report.failures if failure["case_id"] in ambiguous
    )
    assert not any(failure["case_id"] == "tri-009" for failure in report.failures)
