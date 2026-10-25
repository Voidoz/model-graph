"""Configurable redaction for trace-derived datasets.

Redaction is deliberately value-preserving where possible: a matched secret is replaced by a
stable marker (optionally containing a hash of the match) so the redaction is inspectable and
deterministic without writing the secret itself. The report records pattern names, counts, and
field paths, never matched text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelgraph.contracts.loader import load_document
from modelgraph.errors import ModelGraphError

ACTIONS = ("hash", "mask", "drop")


class RedactionError(ModelGraphError):
    """A redaction configuration or report could not be used."""

    code = "MG-5002"


@dataclass(frozen=True)
class RedactionPattern:
    name: str
    pattern: re.Pattern[str]
    action: str


@dataclass(frozen=True)
class FieldRule:
    name: str
    action: str


DEFAULT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("private-key", r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    ("openai-api-key", r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    ("github-token", r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ("aws-access-key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("bearer-token", r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*"),
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("home-path", r"(?:/home|/Users)/[^/\s\"']+"),
)


@dataclass
class Redactor:
    """Apply configured patterns and field rules to a JSON value."""

    patterns: list[RedactionPattern] = field(default_factory=list)
    field_rules: list[FieldRule] = field(default_factory=list)
    enabled: bool = True
    default_action: str = "hash"
    counts: dict[str, int] = field(default_factory=dict)
    fields: dict[str, set[str]] = field(default_factory=dict)
    field_removals: dict[str, list[str]] = field(default_factory=dict)
    records_touched: int = 0
    _record_names: set[str] = field(default_factory=set, init=False, repr=False)
    _record_touched: bool = field(default=False, init=False, repr=False)

    def redact_record(self, value: Any, *, path: str = "") -> tuple[Any, list[str]]:
        """Redact one record, returning the value and the pattern names that fired."""
        self._record_names = set()
        self._record_touched = False
        result = self._walk(value, path)
        if self._record_touched:
            self.records_touched += 1
        return result, sorted(self._record_names)

    def _walk(self, value: Any, path: str) -> Any:
        if not self.enabled:
            return value
        if isinstance(value, str):
            return self._redact_text(value, path)
        if isinstance(value, list):
            return [self._walk(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                rule = self._field_rule(str(key))
                if rule is not None:
                    if rule.action == "drop":
                        self.field_removals.setdefault(rule.name, []).append(child_path)
                        self._record_names.add(rule.name)
                        self._record_touched = True
                        continue
                    if rule.action == "hash" and isinstance(item, str):
                        result[key] = self._marker("hash", rule.name, item)
                        self._count(rule.name, child_path)
                        continue
                    if rule.action == "mask" and isinstance(item, str):
                        result[key] = self._marker("mask", rule.name, item)
                        self._count(rule.name, child_path)
                        continue
                result[key] = self._walk(item, child_path)
            return result
        return value

    def _redact_text(self, value: str, path: str) -> str:
        text = value
        for pattern in self.patterns:
            matched = False

            def replace(match: re.Match[str], current: RedactionPattern = pattern) -> str:
                nonlocal matched
                matched = True
                self._record_names.add(current.name)
                self._record_touched = True
                return self._marker(current.action, current.name, match.group(0))

            text = pattern.pattern.sub(replace, text)
            if matched:
                self._count(pattern.name, path)
        return text

    def _marker(self, action: str, name: str, value: str) -> str:
        if action == "drop":
            return ""
        if action == "mask":
            return f"[REDACTED:{name}]"
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"[REDACTED:{name}:sha256:{digest}]"

    def _count(self, name: str, path: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1
        self.fields.setdefault(name, set()).add(path)

    def _field_rule(self, key: str) -> FieldRule | None:
        for rule in self.field_rules:
            if rule.name == key:
                return rule
        return None

    def report(self) -> dict[str, Any]:
        total = sum(self.counts.values())
        patterns = [
            {
                "name": pattern.name,
                "action": pattern.action,
                "count": self.counts.get(pattern.name, 0),
                "fields": sorted(self.fields.get(pattern.name, set()))[:100],
            }
            for pattern in self.patterns
        ]
        removals: list[dict[str, Any]] = [
            {
                "name": name,
                "count": len(paths),
                "fields": sorted(set(paths))[:100],
            }
            for name, paths in sorted(self.field_removals.items())
        ]
        if not self.enabled:
            status = "disabled"
        elif total or removals:
            status = "redacted"
        else:
            status = "clean"
        return {
            "schema": "modelgraph/redaction-report/v0",
            "enabled": self.enabled,
            "action": self.default_action,
            "status": status,
            "patterns": patterns,
            "field_removals": removals,
            "total_redactions": total + sum(int(item["count"]) for item in removals),
            "records_touched": self.records_touched,
            "clean": status == "clean",
        }


def build_redactor(spec: str | Path, action: str = "hash") -> Redactor:
    """Build a redactor from ``default``, ``none``, or a YAML/JSON config path."""
    if action not in ACTIONS:
        raise RedactionError(f"redaction action must be one of {list(ACTIONS)}, got {action!r}")
    if isinstance(spec, Path):
        return _from_config(spec, action)
    value = str(spec)
    if value == "none":
        return Redactor(enabled=False, default_action=action)
    if value == "default":
        return Redactor(
            patterns=[_pattern(name, regex, action) for name, regex in DEFAULT_PATTERNS],
            enabled=True,
            default_action=action,
        )
    return _from_config(Path(value), action)


def _from_config(path: Path, action: str) -> Redactor:
    if not path.is_file():
        raise RedactionError(f"redaction config not found: {path}", path=str(path))
    document = load_document(path)
    if not isinstance(document, dict):
        raise RedactionError("redaction config must be a mapping", path=str(path))
    default_action = str(document.get("default_action") or action)
    if default_action not in ACTIONS:
        raise RedactionError(
            f"redaction default_action must be one of {list(ACTIONS)}, got {default_action!r}",
            path=str(path),
        )
    include_defaults = bool(document.get("include_defaults", True))
    configured: list[RedactionPattern] = []
    raw_patterns = document.get("patterns") or []
    if not isinstance(raw_patterns, list):
        raise RedactionError("redaction config 'patterns' must be a list", path=str(path))
    for index, item in enumerate(raw_patterns):
        if not isinstance(item, dict):
            raise RedactionError(f"redaction pattern {index} must be a mapping", path=str(path))
        name = str(item.get("name") or f"pattern-{index}")
        regex = item.get("pattern")
        if not isinstance(regex, str) or not regex:
            raise RedactionError(
                f"redaction pattern {name!r} requires a non-empty 'pattern'", path=str(path)
            )
        pattern_action = str(item.get("action") or default_action)
        if pattern_action not in ACTIONS:
            raise RedactionError(
                f"redaction pattern {name!r} has invalid action {pattern_action!r}",
                path=str(path),
            )
        configured.append(_pattern(name, regex, pattern_action))
    if include_defaults:
        overridden = {pattern.name for pattern in configured}
        configured = [
            _pattern(name, regex, default_action)
            for name, regex in DEFAULT_PATTERNS
            if name not in overridden
        ] + configured
    field_rules: list[FieldRule] = []
    raw_fields = document.get("fields") or []
    if not isinstance(raw_fields, list):
        raise RedactionError("redaction config 'fields' must be a list", path=str(path))
    for index, item in enumerate(raw_fields):
        if not isinstance(item, dict):
            raise RedactionError(f"redaction field {index} must be a mapping", path=str(path))
        name = str(item.get("name") or "")
        if not name:
            raise RedactionError(f"redaction field {index} requires a name", path=str(path))
        field_action = str(item.get("action") or default_action)
        if field_action not in ACTIONS:
            raise RedactionError(
                f"redaction field {name!r} has invalid action {field_action!r}", path=str(path)
            )
        field_rules.append(FieldRule(name=name, action=field_action))
    return Redactor(
        patterns=configured,
        field_rules=field_rules,
        enabled=True,
        default_action=default_action,
    )


def _pattern(name: str, regex: str, action: str) -> RedactionPattern:
    try:
        compiled = re.compile(regex)
    except re.error as exc:
        raise RedactionError(f"invalid redaction regex for {name!r}: {exc}") from exc
    return RedactionPattern(name=name, pattern=compiled, action=action)
