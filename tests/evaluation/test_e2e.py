import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from modelgraph.cli import app
from modelgraph.contracts.loader import load_document
from modelgraph.evaluation import (
    E2EConfiguration,
    JudgeRequest,
    JudgeVerdict,
    SuiteError,
    load_suite,
    run_e2e_suite,
    score_blinded_pair,
    validate_report,
)

ROOT = Path(__file__).resolve().parents[2]
E2E_SUITE = ROOT / "evals" / "e2e" / "composed-assistant"
JS_SUITE = ROOT / "evals" / "mock" / "js-specialist"
TRIAGE_SUITE = ROOT / "evals" / "mock" / "triage"
STRUCTURED_SUITE = ROOT / "evals" / "mock" / "structured-contract"
runner = CliRunner()


def deterministic_metrics(report) -> dict:
    return {key: value for key, value in report.metrics.items() if "latency" not in key}


def test_e2e_report_is_schema_valid_and_references_manifests(tmp_path: Path) -> None:
    report = run_e2e_suite(E2E_SUITE, reports_dir=tmp_path)

    assert report.level == "e2e"
    assert validate_report(report.to_dict()) == []
    assert report.metrics["cases"] == 6
    assert report.metrics["task_success"] == pytest.approx(4 / 6)
    assert report.metrics["validity"] == pytest.approx(5 / 6)
    assert report.metrics["cost_per_task_usd"] == 0.0
    assert report.metrics["hop_count_mean"] > 2
    assert report.gate is not None and report.gate["passed"] is True
    assert report.determinism["scripted_provider"] is True

    assert report.report_dir is not None
    report_dir = report.report_dir
    payload = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["level"] == "e2e"
    assert payload["run_manifest"] == "manifest.json"
    assert payload["suite"]["hash"] == report.suite["hash"]
    assert payload["e2e"]["oracle"]["labeled_cases"] == 5
    assert payload["e2e"]["oracle"]["ambiguous_cases"] == 1
    assert payload["e2e"]["oracle"]["labels_file"] == "oracle-labels.json"
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["level"] == "e2e"
    assert manifest["e2e"]["oracle"]["labels_file"] == "oracle-labels.json"
    assert manifest["files"]["e2e"] == "e2e.json"
    assert manifest["files"]["oracle_labels"] == "oracle-labels.json"
    labels = json.loads((report_dir / "oracle-labels.json").read_text(encoding="utf-8"))
    assert labels["labeler"] == "measured"
    assert len(labels["labels"]) == 6
    assert any(label["ambiguous"] for label in labels["labels"])
    assert (report_dir / "e2e.json").exists()

    manifests = [
        Path(case["run_manifest"]) for case in report.cases if case.get("run_manifest") is not None
    ]
    assert len(manifests) == 6
    assert all(path.is_file() for path in manifests)

    markdown = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "## E2E baselines" in markdown
    assert "## Pareto frontier" in markdown
    assert "## Worst slices" in markdown
    assert "## Oracle labels" in markdown


def test_e2e_baselines_oracle_gap_and_pareto(tmp_path: Path) -> None:
    report = run_e2e_suite(E2E_SUITE, reports_dir=tmp_path)
    assert report.e2e is not None
    baselines = report.e2e["baselines"]

    best = baselines["best_single"]
    assert best["available"] is True
    assert best["task_success"] < report.metrics["task_success"]
    assert len(best["candidates"]) == 3
    assert best["winner"] in {"js-specialist", "generic-coder", "story"}

    cheapest = baselines["cheapest_capable"]
    assert cheapest["available"] is True
    assert cheapest["cost_per_task_usd"] == 0.0

    oracle = baselines["oracle_routing"]
    assert oracle["available"] is True
    assert oracle["task_success"] == pytest.approx(5 / 6)
    assert oracle["oracle_agreement"] == pytest.approx(0.8)

    random_routing = baselines["random_routing"]
    assert random_routing["available"] is True
    assert random_routing["seed"] == 1234 + 7
    assert len(random_routing["routes"]) == 6

    summary = report.e2e["oracle"]
    assert summary["oracle_gap"] == pytest.approx(0.2)
    assert summary["oracle_gap_ci95"] is not None
    assert summary["labeler"] == "measured"
    assert summary["labeling"]["measured_cost_usd"] == 0.0

    assert report.comparisons
    names = {entry["name"] for entry in report.comparisons}
    assert {"best_single", "cheapest_capable", "oracle_routing", "random_routing"} <= names
    oracle_comparison = next(
        entry for entry in report.comparisons if entry["name"] == "oracle_routing"
    )
    assert oracle_comparison["delta"] == pytest.approx(-1 / 6)
    assert oracle_comparison["ci95"] is not None

    pareto = report.e2e["pareto"]
    names = {entry["name"] for entry in pareto["configurations"]}
    assert {"graph", "best_single", "cheapest_capable", "oracle_routing", "random_routing"} <= names
    assert pareto["frontier"] == ["oracle_routing"]
    assert pareto["bands"][0]["winner"] == "oracle_routing"
    assert all("frontier" in entry for entry in pareto["configurations"])


def test_e2e_failures_and_slices_show_the_routing_and_validity_losses(
    tmp_path: Path,
) -> None:
    report = run_e2e_suite(E2E_SUITE, reports_dir=tmp_path)
    assert report.e2e is not None
    failures = {failure["case_id"]: failure for failure in report.failures}
    assert set(failures) == {"story-misrouted", "missing-artifact"}
    assert failures["story-misrouted"]["chosen_route"] == "coding"
    assert failures["story-misrouted"]["oracle_route"] == "story"
    assert failures["story-misrouted"]["oracle_ambiguous"] is False
    assert failures["missing-artifact"]["valid"] is False
    assert failures["missing-artifact"]["oracle_ambiguous"] is True

    missing = next(case for case in report.cases if case["case_id"] == "missing-artifact")
    assert missing["success"] is False
    assert missing["valid"] is False
    assert missing["protocol_errors"][0]["code"] == "MG-2202"
    assert missing["oracle_ambiguous"] is True
    misrouted = next(case for case in report.cases if case["case_id"] == "story-misrouted")
    assert misrouted["route_correct"] is False
    assert misrouted["valid"] is True

    assert report.slices["domain:writing"]["failure_count"] == 1
    assert report.slices["domain:general"]["cases"] == 2
    assert report.e2e["failure_slices"]
    assert {entry["name"] for entry in report.e2e["failure_slices"]} >= {
        "domain:writing",
        "domain:general",
    }


def test_e2e_profile_is_applied_and_reported(tmp_path: Path) -> None:
    report = run_e2e_suite(
        E2E_SUITE,
        profile="local-4gb",
        case_ids=["js-debounce"],
        reports_dir=tmp_path,
    )
    assert report.e2e is not None
    assert report.e2e["profile"]["name"] == "local-4gb"
    assert report.e2e["prune"] is not None
    removed = {entry["id"] for entry in report.e2e["prune"]["removed_nodes"]}
    assert {"story", "generic-coder"} <= removed
    assert report.target["nodes"] == 2
    assert report.metrics["task_success"] == 1.0
    assert report.metrics["validity"] == 1.0


def test_e2e_is_deterministic_with_scripted_responses(tmp_path: Path) -> None:
    first = run_e2e_suite(E2E_SUITE, reports_dir=tmp_path)
    second = run_e2e_suite(E2E_SUITE, reports_dir=tmp_path)
    assert first.determinism["fingerprint"] == second.determinism["fingerprint"]
    assert deterministic_metrics(first) == deterministic_metrics(second)

    def core(case: dict) -> dict:
        return {
            "case_id": case["case_id"],
            "success": case["success"],
            "valid": case["valid"],
            "chosen_route": case["chosen_route"],
            "hop_count": case["hop_count"],
            "quality": case["quality"],
            "cost_usd": case["cost_usd"],
            "artifact_hashes": [artifact["hash"] for artifact in case["artifacts"]],
        }

    assert [core(case) for case in first.cases] == [core(case) for case in second.cases]
    assert first.e2e is not None and second.e2e is not None
    assert first.e2e["pareto"]["frontier"] == second.e2e["pareto"]["frontier"]
    assert first.e2e["oracle"]["oracle_gap"] == second.e2e["oracle"]["oracle_gap"]


def test_e2e_report_without_persisted_runs_keeps_trace_evidence(tmp_path: Path) -> None:
    # The committed baselines are generated with persist=False so they do not reference
    # uncommitted run directories. The trace is still required for decisions, responses,
    # provider-call counts, and oracle agreement, so both modes must produce the same report
    # content apart from run identifiers and manifests.
    kept = run_e2e_suite(E2E_SUITE, reports_dir=tmp_path, persist=True)
    ephemeral = run_e2e_suite(E2E_SUITE, reports_dir=tmp_path, persist=False)
    assert ephemeral.determinism["fingerprint"] == kept.determinism["fingerprint"]
    assert deterministic_metrics(ephemeral) == deterministic_metrics(kept)
    assert all(case["run_manifest"] is None for case in ephemeral.cases)
    assert all(case["responses"] for case in ephemeral.cases)
    assert all(case["chosen_route"] for case in ephemeral.cases)
    assert ephemeral.e2e is not None and kept.e2e is not None
    assert ephemeral.e2e["oracle"]["oracle_agreement"] == kept.e2e["oracle"]["oracle_agreement"]
    assert ephemeral.e2e["oracle"]["oracle_gap"] == kept.e2e["oracle"]["oracle_gap"]
    assert ephemeral.e2e["run_manifests"] == {}


def test_e2e_incumbent_report_is_a_paired_baseline(tmp_path: Path) -> None:
    first = run_e2e_suite(E2E_SUITE, reports_dir=tmp_path)
    assert first.report_dir is not None
    second = run_e2e_suite(E2E_SUITE, baseline=first.report_dir, reports_dir=tmp_path)
    assert second.e2e is not None
    incumbent = second.e2e["baselines"]["incumbent"]
    assert incumbent["available"] is True
    assert incumbent["delta"] == 0.0
    assert incumbent["ci95"] == [0.0, 0.0]
    assert incumbent["wins"] == 0
    assert incumbent["losses"] == 0
    assert incumbent["ties"] == 6
    assert any(entry["name"] == "incumbent" for entry in second.comparisons)
    assert any(entry["name"] == "incumbent" for entry in second.e2e["pareto"]["configurations"])
    assert second.baseline is not None


def test_e2e_extra_configuration_appears_in_the_pareto_table(tmp_path: Path) -> None:
    report = run_e2e_suite(
        E2E_SUITE,
        configurations=[E2EConfiguration(name="hello", graph="examples/hello-triage.graph.yaml")],
        reports_dir=tmp_path,
    )
    assert report.e2e is not None
    extra = report.e2e["extra_configurations"]
    assert len(extra) == 1
    assert extra[0]["name"] == "hello"
    assert extra[0]["available"] is True
    assert any(entry["name"] == "hello" for entry in report.e2e["pareto"]["configurations"])


def test_blinded_pair_judge_swaps_order_and_fails_closed() -> None:
    seen: list[JudgeRequest] = []

    def judge(request: JudgeRequest) -> JudgeVerdict:
        seen.append(request)
        score = 0.8 if request.metadata["order"] == "AB" else 0.4
        return JudgeVerdict(score=score, rationale="test", raw=str(score), model="judge-pinned")

    result = score_blinded_pair(
        case_id="c-1",
        task="task",
        candidate="candidate text",
        baseline="baseline text",
        rubric="rubric",
        judge=judge,
        model="judge-pinned",
    )
    assert len(seen) == 2
    assert [request.metadata["order"] for request in seen] == ["AB", "BA"]
    assert seen[0].artifacts == {"A": "candidate text", "B": "baseline text"}
    assert seen[1].artifacts == {"A": "baseline text", "B": "candidate text"}
    assert all(request.metadata["blinded"] is True for request in seen)
    assert all(request.metadata["target"] == "A" for request in seen)
    assert result["candidate_score"] == 0.8
    assert result["baseline_score"] == 0.4
    assert result["delta"] == pytest.approx(0.4)
    assert result["order_swapped"] is True
    assert result["blinded"] is True
    assert result["model"] == "judge-pinned"

    def bad_judge(request: JudgeRequest) -> JudgeVerdict:
        return JudgeVerdict(score=1.4)

    from modelgraph.evaluation import EvalError

    with pytest.raises(EvalError, match="outside"):
        score_blinded_pair(
            case_id="c-2",
            task="task",
            candidate="a",
            baseline="b",
            rubric="r",
            judge=bad_judge,
        )


def test_e2e_judge_pair_and_calibration_are_reported(tmp_path: Path) -> None:
    suite_dir = _judge_suite(tmp_path)
    calls: list[str] = []

    def judge(request: JudgeRequest) -> JudgeVerdict:
        calls.append(request.metadata.get("order", "single"))
        score = 0.8 if request.metadata.get("order") != "BA" else 0.6
        return JudgeVerdict(score=score, rationale="rubric", raw=str(score), model="judge-pinned")

    first = run_e2e_suite(suite_dir, judge=judge, reports_dir=tmp_path)
    assert first.judge is not None
    assert first.judge["configured"] is True
    assert first.judge["model"] == "judge-pinned"
    assert first.judge["rubric_hash"].startswith("sha256:")
    assert first.judge["calibration"]["available"] is True
    assert first.judge["calibration"]["n"] == 1

    second = run_e2e_suite(
        suite_dir,
        judge=judge,
        baseline=first.report_dir,
        reports_dir=tmp_path,
    )
    assert second.judge is not None
    pair = second.judge["pair"]
    assert pair["cases"] == 1
    assert pair["blinded"] is True
    assert pair["order_swapped"] is True
    assert pair["delta"] == pytest.approx(0.2)
    assert "AB" in calls and "BA" in calls


def test_cli_eval_node_routing_and_e2e(tmp_path: Path) -> None:
    node = runner.invoke(
        app,
        ["eval", "node", "--suite", str(JS_SUITE), "--reports-dir", str(tmp_path), "--json"],
    )
    assert node.exit_code == 0, node.stdout + node.stderr
    node_payload = json.loads(node.stdout)
    assert node_payload["level"] == "node"
    assert node_payload["gate"]["passed"] is True

    routing = runner.invoke(
        app,
        [
            "eval",
            "routing",
            "--suite",
            str(TRIAGE_SUITE),
            "--reports-dir",
            str(tmp_path),
            "--json",
        ],
    )
    assert routing.exit_code == 0, routing.stdout + routing.stderr
    routing_payload = json.loads(routing.stdout)
    assert routing_payload["level"] == "routing"
    assert routing_payload["metrics"]["oracle_agreement"] == pytest.approx(0.6)

    e2e = runner.invoke(
        app,
        [
            "eval",
            "e2e",
            "--suite",
            str(E2E_SUITE),
            "--graph",
            str(ROOT / "examples" / "composed-assistant.graph.yaml"),
            "--provider",
            "mock",
            "--reports-dir",
            str(tmp_path),
            "--json",
        ],
    )
    assert e2e.exit_code == 0, e2e.stdout + e2e.stderr
    e2e_payload = json.loads(e2e.stdout)
    assert e2e_payload["level"] == "e2e"
    assert e2e_payload["e2e"]["pareto"]["frontier"]
    assert Path(e2e_payload["report_dir"]).is_dir()
    assert Path(e2e_payload["manifest"]).is_file()


def test_cli_fail_on_gate_exits_non_zero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "node",
            "--suite",
            str(STRUCTURED_SUITE),
            "--reports-dir",
            str(tmp_path),
            "--fail-on-gate",
            "--json",
        ],
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["gate"]["passed"] is False


def test_e2e_unknown_mock_node_is_rejected(tmp_path: Path) -> None:
    copied = Path(shutil.copytree(E2E_SUITE, tmp_path / "suite"))
    cases = [
        json.loads(line)
        for line in (copied / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cases[0]["mock_nodes"]["no-such-node"] = "x"
    _write_cases(copied / "cases.jsonl", cases)
    suite = load_suite(copied)
    with pytest.raises(SuiteError, match="unknown node"):
        run_e2e_suite(suite, reports_dir=tmp_path, persist=False)


def _judge_suite(tmp_path: Path) -> Path:
    copied = Path(shutil.copytree(E2E_SUITE, tmp_path / "judge-suite"))
    document = load_document(copied / "suite.yaml")
    document["spec"]["judge"] = {
        "enabled": True,
        "model": "judge-pinned",
        "rubric": "rubric.md",
        "temperature": 0.0,
        "max_output_tokens": 128,
    }
    document["spec"]["rubric"] = "rubric.md"
    (copied / "suite.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    (copied / "rubric.md").write_text(
        "# Rubric\nScore correctness and clarity from 0 to 1.\n", encoding="utf-8"
    )
    cases = [
        json.loads(line)
        for line in (copied / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cases[0]["checks"] = [
        *cases[0]["checks"],
        {"type": "rubric", "artifact": "answer", "min_score": 0.5},
    ]
    cases[0]["metadata"] = {"human_score": 0.8}
    _write_cases(copied / "cases.jsonl", cases)
    return copied


def _write_cases(path: Path, cases: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")


def test_e2e_live_provider_is_gated(monkeypatch, tmp_path: Path) -> None:
    from modelgraph.evaluation import EvalError

    monkeypatch.delenv("MG_EVAL_LIVE", raising=False)
    with pytest.raises(EvalError, match="gated"):
        run_e2e_suite(
            E2E_SUITE,
            provider="ollama",
            reports_dir=tmp_path,
            persist=False,
        )
