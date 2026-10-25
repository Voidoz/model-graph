import json
from pathlib import Path

import pytest

from modelgraph.contracts.envelope import Artifact
from modelgraph.contracts.graph import ArtifactType
from modelgraph.evaluation.checks import (
    JudgeRequest,
    JudgeVerdict,
    evaluate_checks,
)
from modelgraph.evaluation.suite import CheckSpec
from modelgraph.util import sha256_text

ROOT = Path(__file__).resolve().parents[2]


def artifact(
    content: str = "hello world",
    *,
    name: str = "answer",
    artifact_type: ArtifactType = ArtifactType.text,
    media_type: str | None = None,
) -> Artifact:
    return Artifact(
        name=name,
        type=artifact_type,
        media_type=media_type,
        content=content,
        producer="node",
        created_at="2026-10-22T00:00:00Z",
        hash=sha256_text(content),
    )


def outcome(
    checks: list[CheckSpec],
    *,
    artifacts: list[Artifact] | None = None,
    judge=None,
    rubric: str | None = "# Rubric",
    allow_commands: bool = True,
    artifact_names: list[str] | None = None,
):
    produced = artifacts if artifacts is not None else [artifact()]
    return evaluate_checks(
        checks,
        produced,
        artifact_names=artifact_names or [item.name for item in produced],
        task="task",
        case_id="case-1",
        rubric_text=rubric,
        judge=judge,
        suite_dir=ROOT,
        allow_commands=allow_commands,
    )


def test_artifact_presence_and_min_bytes() -> None:
    passed = outcome([CheckSpec(type="artifact", artifact="answer", min_bytes=5)])
    assert passed.passed
    failed = outcome([CheckSpec(type="artifact", artifact="answer", min_bytes=100)])
    assert not failed.passed
    assert "below min_bytes" in failed.results[0].detail


def test_empty_artifact_fails() -> None:
    result = outcome([CheckSpec(type="artifact", artifact="answer")], artifacts=[artifact("  ")])
    assert not result.passed
    assert "empty" in result.results[0].detail


def test_exact_match_normalization() -> None:
    check = CheckSpec(type="exact_match", artifact="answer", value="hello", normalize="strip")
    assert outcome([check], artifacts=[artifact("  hello \n")]).passed
    mismatch = outcome([check], artifacts=[artifact("hello world")])
    assert not mismatch.passed


def test_contains_and_regex_flags() -> None:
    contains = CheckSpec(type="contains", artifact="answer", value="world")
    assert outcome([contains]).passed
    regex = CheckSpec(type="regex", artifact="answer", pattern=r"^HELLO", flags="i")
    assert outcome([regex]).passed
    missing = CheckSpec(type="regex", artifact="answer", pattern="absent")
    assert not outcome([missing]).passed


def test_invalid_regex_is_a_check_failure() -> None:
    result = outcome([CheckSpec(type="regex", artifact="answer", pattern="[")])
    assert not result.passed
    assert "invalid regex" in result.results[0].detail


def test_json_schema_inline_and_parse_failure() -> None:
    check = CheckSpec.model_validate(
        {
            "type": "json_schema",
            "artifact": "answer",
            "schema": {"type": "object", "required": ["ok"]},
        }
    )
    json_artifact = artifact('{"ok": true}', artifact_type=ArtifactType.json)
    assert outcome([check], artifacts=[json_artifact]).passed
    invalid = outcome([check], artifacts=[artifact("not json", artifact_type=ArtifactType.json)])
    assert not invalid.passed
    assert "not valid JSON" in invalid.results[0].detail
    wrong_shape = outcome(
        [check], artifacts=[artifact('{"no": 1}', artifact_type=ArtifactType.json)]
    )
    assert not wrong_shape.passed
    assert "JSON Schema violation" in wrong_shape.results[0].detail


def test_command_check_runs_offline() -> None:
    command = CheckSpec.model_validate(
        {
            "type": "command",
            "artifact": "answer",
            "command": [
                "{python}",
                "-c",
                (
                    "import sys; data=open(sys.argv[1], encoding='utf-8').read(); "
                    "sys.exit(0 if 'hello' in data else 1)"
                ),
                "{artifact_path}",
            ],
            "timeout_seconds": 10,
        }
    )
    assert outcome([command]).passed
    rejected = outcome([command], artifacts=[artifact("goodbye")])
    assert not rejected.passed
    assert rejected.results[0].data["exit_code"] == 1
    disabled = outcome([command], allow_commands=False)
    assert not disabled.passed
    assert "disabled" in disabled.results[0].detail


def test_rubric_without_judge_fails_closed() -> None:
    check = CheckSpec(type="rubric", min_score=0.7)
    result = outcome([check], judge=None)
    assert not result.passed
    assert result.judge_error is not None
    assert "fail closed" in result.results[0].detail


def test_rubric_with_judge_hook() -> None:
    seen: dict[str, object] = {}

    def judge(request: JudgeRequest) -> JudgeVerdict:
        seen["case_id"] = request.case_id
        seen["rubric"] = request.rubric
        return JudgeVerdict(score=0.9, rationale="fits", raw="0.9", model="judge-test")

    check = CheckSpec(type="rubric", min_score=0.8)
    result = outcome([check], judge=judge)
    assert result.passed
    assert result.judge_score == 0.9
    assert result.judge_called
    assert seen["case_id"] == "case-1"
    assert result.results[0].data["model"] == "judge-test"


def test_rubric_out_of_range_fails_closed() -> None:
    def judge(_request: JudgeRequest) -> JudgeVerdict:
        return JudgeVerdict(score=1.5)

    result = outcome([CheckSpec(type="rubric", min_score=0.5)], judge=judge)
    assert not result.passed
    assert result.judge_error is not None


def test_multi_artifact_check_requires_artifact_name() -> None:
    artifacts = [
        artifact("one", name="first"),
        artifact("two", name="second"),
    ]
    result = outcome(
        [CheckSpec(type="exact_match", value="one")],
        artifacts=artifacts,
        artifact_names=["first", "second"],
    )
    assert not result.passed
    assert "requires 'artifact'" in result.results[0].detail


def test_missing_artifact_is_reported() -> None:
    result = outcome([CheckSpec(type="artifact", artifact="missing")])
    assert not result.passed
    assert "was not produced" in result.results[0].detail


def test_json_schema_path_must_stay_in_suite_directory(tmp_path: Path) -> None:
    check = CheckSpec.model_validate(
        {"type": "json_schema", "artifact": "answer", "schema_path": "../outside.json"}
    )
    result = evaluate_checks(
        [check],
        [artifact("{}", artifact_type=ArtifactType.json)],
        artifact_names=["answer"],
        task="task",
        case_id="case-1",
        rubric_text=None,
        judge=None,
        suite_dir=tmp_path / "suite",
    )
    assert not result.passed
    assert "escapes" in result.results[0].detail


def test_check_schema_requires_conditional_fields() -> None:
    with pytest.raises(ValueError):
        CheckSpec(type="regex")
    with pytest.raises(ValueError):
        CheckSpec(type="command")
    valid = CheckSpec.model_validate({"type": "json_schema", "schema": {"type": "object"}})
    assert valid.schema_ == {"type": "object"}
    assert json.dumps(valid.model_dump(by_alias=True))  # alias stays stable for authors
