"""Redaction tests: planted secrets must not survive in persisted logs or errors.

Artifact blobs are deliberately excluded: they are the run product and are content-addressed.
Rewriting them would break the hash contract and replay. The limitation is documented in
docs/security.md and asserted explicitly below.
"""

from __future__ import annotations

import json
from pathlib import Path

from security_helpers import (
    ROUTING_DECISION,
    SECRET_EMAIL,
    SECRET_KEY,
    SECRET_VALUES,
    compile_example,
    persisted_text,
    read_trace,
    run_hello,
)

from modelgraph.providers.base import (
    Capabilities,
    ChatRequest,
    ChatResponse,
    Health,
    ProviderFailure,
)
from modelgraph.providers.mock import MockScenario
from modelgraph.runtime.engine import run_graph
from modelgraph.security import (
    default_patterns,
    redact_data,
    redact_text,
    scan_text,
)


class _FailingProvider:
    """A provider that leaks a credential in its error message."""

    name = "failing"

    def __init__(self, message: str) -> None:
        self.message = message

    def capabilities(self) -> Capabilities:
        return Capabilities(max_context=32768)

    def health(self) -> Health:
        return Health(ok=True)

    def chat(self, request: ChatRequest) -> ChatResponse:
        raise ProviderFailure(self.message, code="MG-3004", retryable=False)


def test_default_patterns_redact_key_and_email_without_echoing_them() -> None:
    text = f"token={SECRET_KEY} contact={SECRET_EMAIL}"
    redacted, findings = redact_text(text)
    for secret in SECRET_VALUES:
        assert secret not in redacted
        assert secret not in repr(findings)
    names = {finding.pattern for finding in findings}
    assert {"openai-api-key", "email"} <= names
    for finding in findings:
        assert finding.digest.startswith("sha256:")
        assert SECRET_KEY not in finding.digest
        assert SECRET_EMAIL not in finding.digest


def test_redaction_report_contains_names_counts_and_hashes_only() -> None:
    payload = {
        "task": f"Use {SECRET_KEY} and mail {SECRET_EMAIL}",
        "nested": [{"note": f"again {SECRET_EMAIL}"}],
    }
    redacted, report = redact_data(payload)
    dumped = json.dumps(redacted) + json.dumps(report.to_dict())
    for secret in SECRET_VALUES:
        assert secret not in dumped
    assert report.total >= 3
    assert report.status == "redacted"
    report_dict = report.to_dict()
    for pattern in report_dict["patterns"]:
        assert pattern["count"] >= 1
        assert all(digest.startswith("sha256:") for digest in pattern["digests"])


def test_scan_text_finds_but_never_returns_matched_text() -> None:
    findings = scan_text(f"key {SECRET_KEY}, email {SECRET_EMAIL}")
    assert {finding.pattern for finding in findings} == {"openai-api-key", "email"}
    rendered = repr(findings)
    assert SECRET_KEY not in rendered
    assert SECRET_EMAIL not in rendered


def test_planted_secret_is_absent_from_persisted_logs_and_manifest(tmp_path: Path) -> None:
    task = f"Fix /home/oscar/private/demo.ts using {SECRET_KEY} and email {SECRET_EMAIL}."
    response = f"export const fixed = true;\n// key {SECRET_KEY}\n// contact {SECRET_EMAIL}\n"
    outcome = run_hello(
        tmp_path,
        task,
        MockScenario(responses={"triage": [ROUTING_DECISION], "coding": [response]}),
    )
    assert outcome.ok, outcome.error
    assert outcome.run_dir is not None
    run_dir = outcome.run_dir

    logs = persisted_text(run_dir)
    for secret in SECRET_VALUES:
        assert secret not in logs

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert "[REDACTED:openai-api-key" in manifest["request"]["task"]
    assert "[REDACTED:email" in manifest["request"]["task"]

    report = json.loads((run_dir / "redaction.json").read_text(encoding="utf-8"))
    assert report["status"] == "redacted"
    names = {item["name"] for item in report["patterns"]}
    assert {"openai-api-key", "email", "home-path"} <= names
    for item in report["patterns"]:
        assert all(digest.startswith("sha256:") for digest in item["digests"])
    assert SECRET_KEY not in json.dumps(report)
    assert SECRET_EMAIL not in json.dumps(report)


def test_artifacts_are_run_data_and_not_rewritten(tmp_path: Path) -> None:
    # Honest limitation: the trace/envelope/manifest are scrubbed; content-addressed artifact
    # blobs are not rewritten because that would break hashes and replay.
    response = f"const answer = 1; // {SECRET_KEY}\n"
    outcome = run_hello(
        tmp_path,
        "Fix the file.",
        MockScenario(responses={"triage": [ROUTING_DECISION], "coding": [response]}),
    )
    assert outcome.ok, outcome.error
    assert outcome.run_dir is not None
    artifacts = "\n".join(
        path.read_text(encoding="utf-8") for path in (outcome.run_dir / "artifacts").glob("*")
    )
    assert SECRET_KEY in artifacts
    assert SECRET_KEY not in persisted_text(outcome.run_dir)


def test_error_messages_and_details_are_redacted_in_trace_and_manifest(
    tmp_path: Path,
) -> None:
    provider = _FailingProvider(f"auth failed for {SECRET_KEY}; contact {SECRET_EMAIL}")
    outcome = run_graph(
        compile_example(),
        "Do work.",
        lambda _model: provider,
        runs_dir=tmp_path / "runs",
    )
    assert outcome.outcome == "failed"
    assert outcome.error is not None
    assert outcome.error["code"] == "MG-3004"
    for secret in SECRET_VALUES:
        assert secret not in outcome.error["message"]
        assert secret not in json.dumps(outcome.error)
    assert outcome.run_dir is not None
    for secret in SECRET_VALUES:
        assert secret not in persisted_text(outcome.run_dir)
    events = read_trace(outcome.run_dir)
    errors = [event for event in events if event["event"] == "provider.error"]
    assert errors
    for secret in SECRET_VALUES:
        assert secret not in json.dumps(errors)


def test_persisted_redaction_is_the_default_and_opt_out_is_explicit(
    tmp_path: Path,
) -> None:
    task = f"Use {SECRET_KEY}"
    scenario = MockScenario(
        responses={"triage": [ROUTING_DECISION], "coding": [f"ok {SECRET_KEY}\n"]}
    )
    default_run = run_hello(tmp_path / "default", task, scenario)
    assert default_run.ok, default_run.error
    assert SECRET_KEY not in persisted_text(default_run.run_dir)

    # The dataset-export tests use this explicit escape hatch to exercise their own
    # configurable redaction over raw traces. It is never the default.
    raw_run = run_hello(
        tmp_path / "raw",
        task,
        MockScenario(responses={"triage": [ROUTING_DECISION], "coding": [f"ok {SECRET_KEY}\n"]}),
        redact_persisted=False,
    )
    assert raw_run.ok, raw_run.error
    assert SECRET_KEY in persisted_text(raw_run.run_dir)
    assert not (raw_run.run_dir / "redaction.json").exists()


def test_only_default_pattern_names_are_available_to_reporting() -> None:
    names = {pattern.name for pattern in default_patterns()}
    assert {"openai-api-key", "email", "home-path", "private-key"} <= names


def test_secret_in_router_reason_is_redacted_in_manifest_path(tmp_path: Path) -> None:
    decision = json.dumps(
        {
            "decision": "route",
            "to": "coding",
            "confidence": 0.95,
            "reason": f"user token {SECRET_KEY}",
        }
    )
    outcome = run_hello(
        tmp_path,
        "Write code.",
        MockScenario(responses={"triage": [decision], "coding": ["done\n"]}),
    )
    assert outcome.ok, outcome.error
    assert outcome.run_dir is not None
    manifest = json.loads((outcome.run_dir / "run.json").read_text(encoding="utf-8"))
    assert SECRET_KEY not in manifest["path"][0]["reason"]
    assert "[REDACTED:openai-api-key" in manifest["path"][0]["reason"]
