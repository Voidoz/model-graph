import json
import shutil
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from modelgraph.contracts.loader import load_document
from modelgraph.evaluation.errors import SuiteError
from modelgraph.evaluation.suite import (
    EvalSuite,
    eval_suite_schema,
    load_suite,
)

ROOT = Path(__file__).resolve().parents[2]
JS_SUITE = ROOT / "evals" / "mock" / "js-specialist"
TRIAGE_SUITE = ROOT / "evals" / "mock" / "triage"
STRUCTURED_SUITE = ROOT / "evals" / "mock" / "structured-contract"


def copy_suite(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def test_suite_schema_is_valid_json_schema() -> None:
    schema = eval_suite_schema()
    Draft202012Validator.check_schema(schema)
    assert schema["title"] == "Eval Suite"
    assert schema["properties"]["apiVersion"]["const"] == "modelgraph/eval/v0"


def test_fixture_suites_load() -> None:
    js = load_suite(JS_SUITE)
    assert isinstance(js, EvalSuite)
    assert js.metadata.id == "mock/js-specialist"
    assert js.case_ids == ("js-001", "js-002", "js-003", "js-004", "js-005", "js-006")
    assert js.content_hash.startswith("sha256:")
    assert js.file_hashes["cases.jsonl"].startswith("sha256:")
    triage = load_suite(TRIAGE_SUITE)
    assert triage.level == "routing"
    assert len(triage.cases) == 12
    assert all(case.oracle is not None for case in triage.cases)
    structured = load_suite(STRUCTURED_SUITE)
    assert structured.level == "node"


def test_suite_hash_changes_when_cases_change(tmp_path: Path) -> None:
    copied = copy_suite(JS_SUITE, tmp_path / "suite")
    before = load_suite(copied)
    cases = copied / "cases.jsonl"
    text = cases.read_text(encoding="utf-8")
    cases.write_text(text.replace('"js-001"', '"js-001-edited"', 1), encoding="utf-8")
    after = load_suite(copied)
    assert before.content_hash != after.content_hash
    assert before.case_ids[0] == "js-001"
    assert after.case_ids[0] == "js-001-edited"


def test_suite_hash_stable_without_changes(tmp_path: Path) -> None:
    copied = copy_suite(JS_SUITE, tmp_path / "suite")
    assert load_suite(copied).content_hash == load_suite(copied).content_hash


def test_invalid_suite_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "modelgraph/eval/v0",
                "kind": "EvalSuite",
                "metadata": {
                    "id": "bad",
                    "name": "Bad",
                    "version": "1.0.0",
                    "description": "Bad suite",
                },
                "spec": {
                    "level": "node",
                    "target": {"graph": "examples/hello-triage.graph.yaml"},
                    "cases": "cases.jsonl",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SuiteError, match="failed schema validation"):
        load_suite(path)


def test_invalid_case_schema_is_rejected(tmp_path: Path) -> None:
    copied = copy_suite(JS_SUITE, tmp_path / "suite")
    (copied / "cases.jsonl").write_text(
        json.dumps(
            {
                "id": "bad-case",
                "input": {"task": "task"},
                "checks": [{"type": "regex"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteError, match="failed schema validation"):
        load_suite(copied)


def test_node_case_requires_a_check(tmp_path: Path) -> None:
    copied = copy_suite(JS_SUITE, tmp_path / "suite")
    (copied / "cases.jsonl").write_text(
        json.dumps({"id": "no-checks", "input": {"task": "task"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteError, match="requires at least one check"):
        load_suite(copied)


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    copied = copy_suite(JS_SUITE, tmp_path / "suite")
    line = (copied / "cases.jsonl").read_text(encoding="utf-8").splitlines()[0]
    (copied / "cases.jsonl").write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(SuiteError, match="duplicate case id"):
        load_suite(copied)


def test_routing_case_requires_oracle(tmp_path: Path) -> None:
    copied = copy_suite(TRIAGE_SUITE, tmp_path / "suite")
    (copied / "cases.jsonl").write_text(
        json.dumps({"id": "no-oracle", "input": {"task": "task"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteError, match="requires an oracle route"):
        load_suite(copied)


def test_routing_suite_requires_oracle_configuration(tmp_path: Path) -> None:
    copied = copy_suite(TRIAGE_SUITE, tmp_path / "suite")
    document = load_document(copied / "suite.yaml")
    del document["spec"]["oracle"]
    (copied / "suite.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(SuiteError, match="must document spec.oracle"):
        load_suite(copied)


def test_level_target_mismatch_is_rejected(tmp_path: Path) -> None:
    copied = copy_suite(JS_SUITE, tmp_path / "suite")
    path = copied / "suite.yaml"
    text = path.read_text(encoding="utf-8").replace("level: node", "level: routing")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SuiteError, match="must declare target.graph"):
        load_suite(copied)


def test_case_path_cannot_escape_suite_directory(tmp_path: Path) -> None:
    copied = copy_suite(JS_SUITE, tmp_path / "suite")
    document = load_document(copied / "suite.yaml")
    document["spec"]["cases"] = "../outside.jsonl"
    (copied / "suite.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    (tmp_path / "outside.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SuiteError, match="escapes the suite directory"):
        load_suite(copied)


def test_selection_is_explicit_and_deterministic() -> None:
    suite = load_suite(JS_SUITE)
    only = suite.select_cases(case_ids=["js-002"])
    assert [case.id for case in only] == ["js-002"]
    tagged = suite.select_cases(tags=["check:exact"])
    assert [case.id for case in tagged] == ["js-002"]
    limited = suite.select_cases(limit=2)
    assert [case.id for case in limited] == ["js-001", "js-002"]
    both = suite.select_cases(tags=["domain:js", "difficulty:easy"], tag_mode="all")
    assert [case.id for case in both] == ["js-001", "js-002"]
    summary = suite.selection_summary(list(limited), limit=2)
    assert summary["selected_cases"] == 2
    assert summary["total_cases"] == 6
    with pytest.raises(SuiteError, match="unknown case id"):
        suite.select_cases(case_ids=["nope"])


def test_case_schema_can_be_used_directly() -> None:
    schema = eval_suite_schema()
    validator = Draft202012Validator(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "#/$defs/case",
            "$defs": schema["$defs"],
        }
    )
    valid = {
        "id": "case-1",
        "input": {"task": "hello"},
        "checks": [{"type": "contains", "value": "h"}],
    }
    assert list(validator.iter_errors(valid)) == []
    invalid = {
        "id": "case-2",
        "input": {"task": "hello"},
        "checks": [{"type": "command"}],
    }
    assert list(validator.iter_errors(invalid))
