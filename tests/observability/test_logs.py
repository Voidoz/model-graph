"""Structured logging: JSONL validity, required fields, and content redaction."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from modelgraph.errors import ModelGraphError
from modelgraph.observability import (
    ConsoleLogger,
    NullLogger,
    StructuredLogger,
    build_logger_from_env,
)

SECRET = "do-not-log-this-prompt"


def test_json_logs_are_one_valid_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "run.jsonl"
    logger = StructuredLogger(run_id="run_1", path=path)
    logger.info("run.started", node="triage", graph_id="hello-triage", depth=0)
    logger.warning("budget.charged", node="triage", tokens=12, cost_usd=0.0)
    logger.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    for record in records:
        assert {"level", "event", "run_id", "ts"} <= set(record)
        assert record["run_id"] == "run_1"
    assert records[0]["event"] == "run.started"
    assert records[0]["node"] == "triage"
    assert records[0]["graph_id"] == "hello-triage"
    assert records[0]["level"] == "info"
    assert records[1]["level"] == "warning"
    assert records[1]["tokens"] == 12


def test_json_logs_redact_prompts_artifacts_and_secrets(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    logger = StructuredLogger(run_id="run_redacted", path=path)
    logger.info(
        "render.completed",
        node="triage",
        prompt=SECRET,
        messages=[{"role": "user", "content": SECRET}],
        artifact="artifact-body",
        content=SECRET,
        text=SECRET,
        task=SECRET,
        api_key="sk-secret-value",
        authorization="Bearer secret-token",
        prompt_hash="sha256:not-a-secret-but-content-shaped",
        nested={"credentials": "hunter2", "usage": {"input_tokens": 3}},
        safe_field="visible",
    )
    logger.close()

    raw = path.read_text(encoding="utf-8")
    assert SECRET not in raw
    assert "artifact-body" not in raw
    assert "sk-secret-value" not in raw
    assert "Bearer secret-token" not in raw
    assert "hunter2" not in raw

    record = json.loads(raw.splitlines()[0])
    assert record["prompt"] == "[redacted]"
    assert record["messages"] == "[redacted]"
    assert record["artifact"] == "[redacted]"
    assert record["api_key"] == "[redacted]"
    assert record["nested"]["credentials"] == "[redacted]"
    assert record["nested"]["usage"]["input_tokens"] == 3
    assert record["safe_field"] == "visible"


def test_json_logs_scrub_secrets_in_innocently_named_fields(tmp_path: Path) -> None:
    """Key-based redaction alone is not enough: values are pattern-scrubbed too.

    A router ``reason`` or a provider ``note`` is not named like content, yet a model or
    provider can echo a credential or address into it.
    """

    path = tmp_path / "run.jsonl"
    logger = StructuredLogger(run_id="run_patterns", path=path)
    logger.info(
        "decision.made",
        node="triage",
        reason="the user wrote sk-proj-abcdefghijklmnop into the request",
        note="contact alice@example.com",
        detail="token sk-zzzzzzzzzzzzzzzz",
        nested={"provider_note": "Bearer abcdefghijklmnopqrstuvwxyz0123456789"},
    )
    logger.close()

    raw = path.read_text(encoding="utf-8")
    for secret in (
        "sk-proj-abcdefghijklmnop",
        "sk-zzzzzzzzzzzzzzzz",
        "alice@example.com",
        "abcdefghijklmnopqrstuvwxyz0123456789",
    ):
        assert secret not in raw
    record = json.loads(raw.splitlines()[0])
    assert "[REDACTED:openai-api-key:sha256:" in record["reason"]
    assert "[REDACTED:email:sha256:" in record["note"]
    assert "[REDACTED:bearer-token:sha256:" in record["nested"]["provider_note"]


def test_console_logs_scrub_secrets_in_innocently_named_fields() -> None:
    stream = io.StringIO()
    logger = ConsoleLogger(run_id="run_console", stream=stream)
    logger.info("decision.made", reason="key sk-proj-abcdefghijklmnop")
    output = stream.getvalue()
    assert "sk-proj-abcdefghijklmnop" not in output
    assert "[REDACTED:openai-api-key:sha256:" in output


def test_fixture_run_logs_are_jsonl_and_content_free(tmp_path: Path, mock_run_dir: Path) -> None:
    events = [
        json.loads(line)
        for line in (mock_run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    task = events[0]["task"]
    artifacts = (mock_run_dir / "artifacts").iterdir()
    artifact_content = next(artifacts).read_text(encoding="utf-8")
    path = tmp_path / "fixture-run.jsonl"
    logger = StructuredLogger(run_id=events[0]["run_id"], path=path)
    for event in events:
        fields = {
            key: value
            for key, value in event.items()
            if key not in {"ts", "level", "event", "run_id", "node"}
        }
        logger.event(str(event["event"]), node=event.get("node"), **fields)
    logger.close()

    raw = path.read_text(encoding="utf-8")
    assert task not in raw
    assert artifact_content.strip() not in raw
    records = [json.loads(line) for line in raw.splitlines()]
    assert len(records) == len(events)
    for record in records:
        assert {"ts", "level", "event", "run_id"} <= set(record)
        assert record["run_id"] == events[0]["run_id"]


def test_level_filter_and_null_logger(tmp_path: Path) -> None:
    path = tmp_path / "filtered.jsonl"
    logger = StructuredLogger(run_id="run_2", level="warning", path=path)
    assert logger.info("should.not.appear") is None
    assert logger.warning("should.appear") is not None
    logger.close()
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == ["should.appear"]

    null = NullLogger()
    assert null.event("anything") is None
    assert null.info("anything") is None
    assert null.warning("anything") is None
    assert null.error("anything") is None


def test_console_mode_is_human_readable() -> None:
    stream = io.StringIO()
    logger = ConsoleLogger(run_id="run_3", stream=stream)
    logger.info("run.started", graph_id="hello-triage", prompt=SECRET)
    output = stream.getvalue()
    assert "run.started" in output
    assert "run=run_3" in output
    assert "INFO" in output
    assert SECRET not in output
    assert "[redacted]" in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output)


def test_build_logger_from_env_is_off_by_default(tmp_path: Path) -> None:
    assert isinstance(build_logger_from_env({}), NullLogger)
    assert isinstance(build_logger_from_env({"MG_LOG_FORMAT": "off"}), NullLogger)

    path = tmp_path / "env.jsonl"
    logger = build_logger_from_env(
        {"MG_LOG_FORMAT": "json", "MG_LOG_FILE": str(path), "MG_LOG_RUN_ID": "run_env"}
    )
    assert isinstance(logger, StructuredLogger)
    logger.info("env.event", safe_field="ok")
    logger.close()
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["run_id"] == "run_env"
    assert record["event"] == "env.event"


def test_build_logger_from_env_rejects_unknown_format() -> None:
    with pytest.raises(ModelGraphError) as excinfo:
        build_logger_from_env({"MG_LOG_FORMAT": "xml"})
    assert excinfo.value.code == "MG-5000"


def test_build_logger_from_env_rejects_unknown_level() -> None:
    with pytest.raises(ModelGraphError) as excinfo:
        build_logger_from_env({"MG_LOG_FORMAT": "json", "MG_LOG_LEVEL": "banana"})
    assert excinfo.value.code == "MG-5000"
    assert "MG_LOG_LEVEL" in excinfo.value.message


def test_build_logger_from_env_wraps_an_unwritable_destination(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(ModelGraphError) as excinfo:
        build_logger_from_env({"MG_LOG_FORMAT": "json", "MG_LOG_FILE": str(blocker / "log.jsonl")})
    assert excinfo.value.code == "MG-5000"
    assert "MG_LOG_FILE" in excinfo.value.message
