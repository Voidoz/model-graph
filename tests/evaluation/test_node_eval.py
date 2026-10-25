import json
from pathlib import Path

from modelgraph.compiler import compile_graph
from modelgraph.evaluation import (
    load_suite,
    run_node_suite,
    validate_report,
)
from modelgraph.runtime.envelope import RuntimeState
from modelgraph.runtime.renderer import render_prompt

ROOT = Path(__file__).resolve().parents[2]
JS_SUITE = ROOT / "evals" / "mock" / "js-specialist"
STRUCTURED_SUITE = ROOT / "evals" / "mock" / "structured-contract"


def deterministic_metrics(report) -> dict:
    return {key: value for key, value in report.metrics.items() if not key.startswith("latency")}


def test_js_specialist_suite_produces_schema_valid_reports(tmp_path: Path) -> None:
    report = run_node_suite(JS_SUITE, reports_dir=tmp_path)
    assert report.metrics["task_success"] == 1.0
    assert report.metrics["validity"] == 1.0
    assert report.metrics["protocol_error_rate"] == 0.0
    assert report.metrics["total_tokens"] > 0
    assert report.metrics["cost_total_usd"] == 0.0
    assert report.gate is not None
    assert report.gate["passed"] is True
    assert report.determinism["scripted_provider"] is True
    assert report.provider["kind"] == "mock"
    assert validate_report(report.to_dict()) == []

    assert report.report_dir is not None
    report_dir = report.report_dir
    assert (report_dir / "report.json").exists()
    assert (report_dir / "report.md").exists()
    assert (report_dir / "manifest.json").exists()
    assert len(list((report_dir / "cases").glob("*.json"))) == 6
    payload = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["run_manifest"] == "manifest.json"
    assert payload["suite"]["hash"] == report.suite["hash"]
    assert payload["command"].startswith("mg eval node")
    assert "run_node_suite" in payload["reproduce"]
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite"]["hash"] == report.suite["hash"]
    assert manifest["files"]["report"] == "report.json"
    markdown = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "Primary metric" in markdown
    assert "Cost, latency, and usage" in markdown
    assert "No failing cases." in markdown


def test_node_prompt_is_the_exact_runtime_render(tmp_path: Path) -> None:
    report = run_node_suite(JS_SUITE, reports_dir=tmp_path)
    suite = load_suite(JS_SUITE)
    case = next(case for case in suite.cases if case.id == "js-001")
    compiled = compile_graph(ROOT / "examples" / "coding-specialist.graph.yaml", strict_models=True)
    assert compiled.ok and compiled.graph is not None
    graph = compiled.graph
    state = RuntimeState.create(graph, case.input.task)
    expected = render_prompt(graph.node("js-specialist"), graph, state)
    actual = next(item for item in report.cases if item["case_id"] == "js-001")
    assert actual["prompt_hash"] == expected.prompt_hash
    assert actual["prompt"]["messages"] == [message.to_dict() for message in expected.messages]
    assert actual["prompt_hash"] in actual["render_hashes"]


def test_node_report_is_deterministic_with_scripted_responses(tmp_path: Path) -> None:
    first = run_node_suite(JS_SUITE, reports_dir=tmp_path)
    second = run_node_suite(JS_SUITE, reports_dir=tmp_path)
    assert first.determinism["fingerprint"] == second.determinism["fingerprint"]
    assert deterministic_metrics(first) == deterministic_metrics(second)
    first_cases = [
        {
            "case_id": case["case_id"],
            "valid": case["valid"],
            "success": case["success"],
            "cost_usd": case["cost_usd"],
            "usage": case["usage"],
            "prompt_hash": case["prompt_hash"],
            "artifacts": [artifact["hash"] for artifact in case["artifacts"]],
        }
        for case in first.cases
    ]
    second_cases = [
        {
            "case_id": case["case_id"],
            "valid": case["valid"],
            "success": case["success"],
            "cost_usd": case["cost_usd"],
            "usage": case["usage"],
            "prompt_hash": case["prompt_hash"],
            "artifacts": [artifact["hash"] for artifact in case["artifacts"]],
        }
        for case in second.cases
    ]
    assert first_cases == second_cases


def test_invalid_node_output_is_a_validity_failure_and_inspectable(
    tmp_path: Path,
) -> None:
    report = run_node_suite(STRUCTURED_SUITE, reports_dir=tmp_path)
    assert report.metrics["validity"] == 0.5
    assert report.metrics["task_success"] == 0.5
    assert report.metrics["protocol_error_rate"] == 0.5
    assert report.gate is not None
    assert report.gate["passed"] is False
    assert "validity" in report.gate["failures"]
    invalid = next(case for case in report.cases if case["case_id"] == "structured-invalid")
    assert invalid["valid"] is False
    assert invalid["success"] is False
    assert invalid["responses"][0]["text"] == "not json at all"
    assert invalid["protocol_errors"][0]["code"] == "MG-2202"
    assert "not valid JSON" in invalid["validity_issues"][0]["message"]
    assert invalid["artifacts"][0]["content"] == "not json at all"
    assert report.failures and report.failures[0]["case_id"] == "structured-invalid"
    assert report.report_dir is not None
    case_file = report.report_dir / "cases" / "structured-invalid.json"
    assert case_file.exists()
    assert json.loads(case_file.read_text(encoding="utf-8"))["valid"] is False


def test_command_checks_can_be_disabled(tmp_path: Path) -> None:
    report = run_node_suite(JS_SUITE, reports_dir=tmp_path, allow_commands=False)
    assert report.metrics["task_success"] < 1.0
    failed = next(case for case in report.cases if case["case_id"] == "js-004")
    assert failed["success"] is False
    command_check = next(
        result for result in failed["check_results"] if result["type"] == "command"
    )
    assert "disabled" in command_check["detail"]


def test_node_selection_limit_is_reported(tmp_path: Path) -> None:
    report = run_node_suite(JS_SUITE, reports_dir=tmp_path, limit=2)
    assert len(report.cases) == 2
    assert report.selection["selected_cases"] == 2
    assert report.selection["total_cases"] == 6
    assert report.usage["cases"] == 2


def _rubric_suite(tmp_path: Path) -> Path:
    import shutil

    import yaml

    from modelgraph.contracts.loader import load_document

    copied = Path(shutil.copytree(JS_SUITE, tmp_path / "rubric-suite"))
    document = load_document(copied / "suite.yaml")
    document["spec"]["rubric"] = "rubric.md"
    document["spec"]["judge"] = {
        "enabled": True,
        "model": "judge-test",
        "rubric": "rubric.md",
    }
    (copied / "suite.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    (copied / "rubric.md").write_text(
        "# Rubric\nScore correctness and clarity from 0 to 1.\n", encoding="utf-8"
    )
    cases = [
        json.loads(line)
        for line in (copied / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cases[0]["checks"] = [{"type": "rubric", "artifact": "answer", "min_score": 0.5}]
    with (copied / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    return copied


def test_rubric_judge_hook_populates_quality_score(tmp_path: Path) -> None:
    from modelgraph.evaluation import JudgeRequest, JudgeVerdict

    suite_path = _rubric_suite(tmp_path)
    seen: list[str] = []

    def judge(request: JudgeRequest) -> JudgeVerdict:
        seen.append(request.case_id)
        assert "correctness" in request.rubric
        return JudgeVerdict(score=0.8, rationale="clear", raw="0.8", model="judge-test")

    report = run_node_suite(suite_path, reports_dir=tmp_path, judge=judge)
    assert report.metrics["quality_score"] == 0.8
    assert report.metrics["judged_cases"] == 1
    assert report.judge is not None
    assert report.judge["configured"] is True
    assert report.judge["calls"] == 1
    assert seen == ["js-001"]
    assert report.metrics["validity"] == 1.0


def test_rubric_without_judge_fails_closed_through_the_runner(tmp_path: Path) -> None:
    suite_path = _rubric_suite(tmp_path)
    report = run_node_suite(suite_path, reports_dir=tmp_path)
    assert report.metrics["validity"] < 1.0
    assert report.judge is not None
    assert report.judge["error"]
    judged = next(case for case in report.cases if case["case_id"] == "js-001")
    assert judged["valid"] is False
    assert judged["judge_error"]
    assert any(failure["case_id"] == "js-001" for failure in report.failures)
