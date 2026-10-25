"""Deterministic checks for node evaluation.

A check is data in the suite, not code in the harness. The harness supports artifact
presence, exact match, substring, regex, JSON Schema, local command/tests, and a rubric-judge
hook. Rubric checks fail closed when no judge is configured or the judge output is unusable.
Command checks execute a trusted local process produced by the suite author; suites are
reviewed like test code.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from modelgraph.contracts.envelope import Artifact
from modelgraph.evaluation.suite import CheckSpec

_MAX_CAPTURE = 4000


@dataclass(frozen=True)
class JudgeRequest:
    case_id: str
    task: str
    artifacts: dict[str, str]
    rubric: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JudgeVerdict:
    score: float
    rationale: str = ""
    raw: str = ""
    model: str | None = None


JudgeHook = Callable[[JudgeRequest], JudgeVerdict]


@dataclass
class CheckResult:
    type: str
    artifact: str | None
    passed: bool
    detail: str
    score: float | None = None
    skipped: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "artifact": self.artifact,
            "passed": self.passed,
            "detail": self.detail,
            "score": self.score,
            "skipped": self.skipped,
            "data": self.data,
        }


@dataclass
class CheckOutcome:
    results: list[CheckResult]
    judge_score: float | None = None
    judge_called: bool = False
    judge_error: str | None = None

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failed(self) -> list[CheckResult]:
        return [result for result in self.results if not result.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [result.to_dict() for result in self.results],
            "passed": self.passed,
            "judge_score": self.judge_score,
            "judge_called": self.judge_called,
            "judge_error": self.judge_error,
        }


def evaluate_checks(
    checks: Sequence[CheckSpec],
    artifacts: Sequence[Artifact],
    *,
    artifact_names: Sequence[str],
    task: str,
    case_id: str,
    rubric_text: str | None,
    judge: JudgeHook | None,
    suite_dir: Path,
    metadata: dict[str, Any] | None = None,
    allow_commands: bool = True,
) -> CheckOutcome:
    by_name = {artifact.name: artifact for artifact in artifacts}
    default_artifact = artifact_names[0] if len(artifact_names) == 1 else None
    needs_judge = any(check.type == "rubric" for check in checks)
    verdict: JudgeVerdict | None = None
    judge_error: str | None = None
    judge_called = False
    if needs_judge:
        if judge is None:
            judge_error = "no judge hook configured (fail closed)"
        elif rubric_text is None:
            judge_error = "no rubric text configured (fail closed)"
        else:
            try:
                verdict = judge(
                    JudgeRequest(
                        case_id=case_id,
                        task=task,
                        artifacts={name: artifact.content for name, artifact in by_name.items()},
                        rubric=rubric_text,
                        metadata=metadata or {},
                    )
                )
                judge_called = True
                if not 0.0 <= float(verdict.score) <= 1.0:
                    judge_error = f"judge score {verdict.score!r} is outside [0, 1] (fail closed)"
                    verdict = None
            except Exception as exc:  # fail closed, but keep the reason inspectable
                judge_error = f"judge failed: {exc}"
                verdict = None

    results: list[CheckResult] = []
    for check in checks:
        if check.type == "rubric":
            artifact_name = check.artifact or default_artifact
            if verdict is None:
                results.append(
                    CheckResult(
                        type=check.type,
                        artifact=artifact_name,
                        passed=False,
                        detail=judge_error or "judge unavailable (fail closed)",
                        data={"judge_error": judge_error},
                    )
                )
                continue
            score = float(verdict.score)
            results.append(
                CheckResult(
                    type=check.type,
                    artifact=artifact_name,
                    passed=score >= check.min_score,
                    detail=(
                        f"rubric score {score:.3f} "
                        f"{'≥' if score >= check.min_score else '<'} {check.min_score:.3f}"
                    ),
                    score=score,
                    data={
                        "rationale": verdict.rationale,
                        "model": verdict.model,
                        "raw": verdict.raw[:_MAX_CAPTURE],
                    },
                )
            )
            continue

        artifact_name = check.artifact or default_artifact
        if artifact_name is None:
            results.append(
                CheckResult(
                    type=check.type,
                    artifact=None,
                    passed=False,
                    detail="check requires 'artifact' because the node writes multiple artifacts",
                )
            )
            continue
        artifact = by_name.get(artifact_name)
        if artifact is None:
            results.append(
                CheckResult(
                    type=check.type,
                    artifact=artifact_name,
                    passed=False,
                    detail=f"artifact {artifact_name!r} was not produced",
                )
            )
            continue
        if check.type == "artifact":
            results.append(_check_artifact(check, artifact))
        elif check.type == "exact_match":
            results.append(_check_exact(check, artifact))
        elif check.type == "contains":
            results.append(_check_contains(check, artifact))
        elif check.type == "regex":
            results.append(_check_regex(check, artifact))
        elif check.type == "json_schema":
            results.append(_check_json_schema(check, artifact, suite_dir))
        elif check.type == "command":
            results.append(_run_command(check, artifact, case_id, allow_commands))
        else:  # pragma: no cover - schema and Pydantic both restrict the enum
            results.append(
                CheckResult(
                    type=check.type,
                    artifact=artifact_name,
                    passed=False,
                    detail=f"unsupported check type {check.type!r}",
                )
            )
    return CheckOutcome(
        results=results,
        judge_score=verdict.score if verdict is not None else None,
        judge_called=judge_called,
        judge_error=judge_error,
    )


def _check_artifact(check: CheckSpec, artifact: Artifact) -> CheckResult:
    size = len(artifact.content.encode("utf-8"))
    problems: list[str] = []
    if check.non_empty and not artifact.content.strip():
        problems.append("artifact is empty")
    if size < check.min_bytes:
        problems.append(f"artifact is {size} bytes, below min_bytes={check.min_bytes}")
    return CheckResult(
        type=check.type,
        artifact=artifact.name,
        passed=not problems,
        detail="; ".join(problems) if problems else f"artifact present ({size} bytes)",
        data={"bytes": size, "artifact_type": artifact.type.value},
    )


def _check_exact(check: CheckSpec, artifact: Artifact) -> CheckResult:
    expected = normalize(check.value or "", check.normalize)
    actual = normalize(artifact.content, check.normalize)
    return CheckResult(
        type=check.type,
        artifact=artifact.name,
        passed=actual == expected,
        detail="exact match" if actual == expected else "content differs from expected value",
        data={"normalize": check.normalize, "actual_preview": artifact.content[:500]},
    )


def _check_contains(check: CheckSpec, artifact: Artifact) -> CheckResult:
    needle = normalize(check.value or "", check.normalize)
    haystack = normalize(artifact.content, check.normalize)
    return CheckResult(
        type=check.type,
        artifact=artifact.name,
        passed=needle in haystack,
        detail=(f"contains {needle!r}" if needle in haystack else f"does not contain {needle!r}"),
        data={"normalize": check.normalize},
    )


def _check_regex(check: CheckSpec, artifact: Artifact) -> CheckResult:
    flags = 0
    for flag in check.flags:
        flags |= {
            "i": re.IGNORECASE,
            "m": re.MULTILINE,
            "s": re.DOTALL,
        }[flag]
    try:
        match = re.search(check.pattern or "", artifact.content, flags)
    except re.error as exc:
        return CheckResult(
            type=check.type,
            artifact=artifact.name,
            passed=False,
            detail=f"invalid regex in suite: {exc}",
        )
    return CheckResult(
        type=check.type,
        artifact=artifact.name,
        passed=match is not None,
        detail=(
            f"matched {match.group(0)[:100]!r}"
            if match is not None
            else f"pattern {check.pattern!r} did not match"
        ),
        data={"match": match.group(0)[:500] if match is not None else None},
    )


def _check_json_schema(check: CheckSpec, artifact: Artifact, suite_dir: Path) -> CheckResult:
    try:
        payload = json.loads(artifact.content)
    except json.JSONDecodeError as exc:
        return CheckResult(
            type=check.type,
            artifact=artifact.name,
            passed=False,
            detail=f"artifact is not valid JSON: {exc}",
        )
    schema = check.schema_
    source = "inline"
    if schema is None and check.schema_path:
        path = (suite_dir / check.schema_path).resolve()
        try:
            path.relative_to(suite_dir.resolve())
        except ValueError:
            return CheckResult(
                type=check.type,
                artifact=artifact.name,
                passed=False,
                detail=f"schema path escapes the suite directory: {check.schema_path!r}",
            )
        if not path.is_file():
            return CheckResult(
                type=check.type,
                artifact=artifact.name,
                passed=False,
                detail=f"schema file not found: {path}",
            )
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return CheckResult(
                type=check.type,
                artifact=artifact.name,
                passed=False,
                detail=f"cannot read schema file {path}: {exc}",
            )
        source = str(path)
    if not isinstance(schema, dict):
        return CheckResult(
            type=check.type,
            artifact=artifact.name,
            passed=False,
            detail="JSON Schema must be an object",
        )
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
    except SchemaError as exc:
        return CheckResult(
            type=check.type,
            artifact=artifact.name,
            passed=False,
            detail=f"invalid JSON Schema in suite: {exc.message}",
        )
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda error: [str(part) for part in error.path],
    )
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "$"
        return CheckResult(
            type=check.type,
            artifact=artifact.name,
            passed=False,
            detail=f"JSON Schema violation at {location}: {first.message}",
            data={"schema_source": source, "violations": len(errors)},
        )
    return CheckResult(
        type=check.type,
        artifact=artifact.name,
        passed=True,
        detail="artifact validates against the JSON Schema",
        data={"schema_source": source},
    )


def _run_command(
    check: CheckSpec,
    artifact: Artifact,
    case_id: str,
    allow_commands: bool,
) -> CheckResult:
    if not allow_commands:
        return CheckResult(
            type=check.type,
            artifact=artifact.name,
            passed=False,
            detail="command checks are disabled for this run",
        )
    assert check.command is not None
    timeout = min(check.timeout_seconds or 10.0, 300.0)
    with tempfile.TemporaryDirectory(prefix="mg-eval-") as temporary:
        root = Path(temporary)
        artifact_path = root / _artifact_filename(artifact)
        artifact_path.write_text(artifact.content, encoding="utf-8")
        replacements = {
            "{artifact_path}": str(artifact_path),
            "{artifact}": str(artifact_path),
            "{case_id}": case_id,
            "{python}": sys.executable,
        }
        argv: list[str] = []
        for argument in check.command:
            expanded = argument
            for token, value in replacements.items():
                expanded = expanded.replace(token, value)
            argv.append(expanded)
        if argv and argv[0] == "python":
            argv[0] = sys.executable
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": temporary,
            "TMPDIR": temporary,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "MG_CASE_ID": case_id,
            "MG_ARTIFACT_NAME": artifact.name,
            "MG_ARTIFACT_PATH": str(artifact_path),
        }
        if os.environ.get("PYTHONPATH"):
            env["PYTHONPATH"] = os.environ["PYTHONPATH"]
        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 - suites are trusted local code
                argv,
                cwd=temporary,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            return CheckResult(
                type=check.type,
                artifact=artifact.name,
                passed=False,
                detail=f"command not found: {argv[0]!r} ({exc})",
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                type=check.type,
                artifact=artifact.name,
                passed=False,
                detail=f"command timed out after {timeout:.1f}s",
                data={"timeout_seconds": timeout, "argv": argv},
            )
        duration = time.monotonic() - started
        stdout = completed.stdout[:_MAX_CAPTURE]
        stderr = completed.stderr[:_MAX_CAPTURE]
        exit_ok = completed.returncode == check.expect_exit_code
        stdout_ok = check.stdout_contains is None or check.stdout_contains in completed.stdout
        passed = exit_ok and stdout_ok
        details = [f"exit code {completed.returncode} (expected {check.expect_exit_code})"]
        if check.stdout_contains is not None:
            details.append(
                "stdout contains the expected text"
                if stdout_ok
                else f"stdout does not contain {check.stdout_contains!r}"
            )
        if not passed and stderr:
            details.append(f"stderr: {stderr.strip()[:300]}")
        return CheckResult(
            type=check.type,
            artifact=artifact.name,
            passed=passed,
            detail="; ".join(details),
            data={
                "argv": argv,
                "exit_code": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": round(duration * 1000, 3),
            },
        )


def normalize(value: str, mode: str) -> str:
    if mode == "strip":
        return value.strip()
    if mode == "casefold":
        return value.casefold()
    if mode == "collapse_whitespace":
        return re.sub(r"\s+", " ", value).strip()
    return value


def _artifact_filename(artifact: Artifact) -> str:
    media_type = (artifact.media_type or "").lower()
    suffix = {
        "application/json": ".json",
        "text/typescript": ".ts",
        "text/javascript": ".js",
        "text/x-python": ".py",
        "text/plain": ".txt",
        "text/markdown": ".md",
    }.get(media_type)
    if suffix is None:
        suffix = {
            "json": ".json",
            "code": ".txt",
            "patch": ".diff",
            "plan": ".md",
            "report": ".md",
            "text": ".txt",
        }[artifact.type.value]
    return f"{artifact.name}{suffix}"
