"""Shared errors and exit codes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Issue:
    """A validation or runtime issue with a stable code and location."""

    code: str
    message: str
    path: str = ""
    severity: str = "error"

    def render(self) -> str:
        location = f" at {self.path}" if self.path else ""
        return f"[{self.code}]{location}: {self.message}"


class ModelGraphError(Exception):
    """Base error for the runtime."""

    code = "MG-0000"
    exit_code = 1

    def __init__(self, message: str, *, path: str = "", details: dict[str, object] | None = None):
        self.message = message
        self.path = path
        self.details = details or {}
        super().__init__(message)


class CompileError(ModelGraphError):
    """One or more validation errors blocked compilation."""

    code = "MG-1001"

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__(f"{len(issues)} validation error(s)")


class ProviderError(ModelGraphError):
    """A model provider failed."""

    code = "MG-3000"


class BudgetExceeded(ModelGraphError):
    """A hard run budget was exhausted."""

    code = "MG-4000"


class RuntimeFailure(ModelGraphError):
    """The run could not complete."""

    code = "MG-2000"


@dataclass
class RunResult:
    """The outcome of one graph execution."""

    run_id: str
    outcome: str
    artifacts: list[object] = field(default_factory=list)
    path: list[object] = field(default_factory=list)
    usage: dict[str, object] = field(default_factory=dict)
    error: ModelGraphError | None = None
