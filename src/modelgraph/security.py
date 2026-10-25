"""Security primitives shared by the runtime, persistence, and tooling.

This module is deliberately small and stdlib-only. It provides three things:

1. **Untrusted-content hygiene.** Provenance labels, neutralization of protocol control
   markers, and size caps for text that a model must treat as data.
2. **Default secret/PII detection and redaction.** A conservative pattern set, a recursive
   JSON redactor, and a scanner. Reports contain pattern names, counts, field paths, and
   truncated SHA-256 digests of matches; they never contain the matched text.
3. **Load-time trust checks for model manifests and run profiles.** Manifests may configure
   a provider endpoint and weights/license metadata, but they may not smuggle credentials,
   executable behavior, or a network destination that contradicts their declared
   ``network_access``. Profiles expose allowlist warnings instead of silently running
   unfiltered.

The default patterns are shared with ``modelgraph.data.redaction`` so a secret has one
definition across traces, dataset export, and the scanner. Redaction is pattern-based: it
reduces accidental leakage, it does not guarantee that every secret or every kind of PII is
found. See ``docs/security.md`` for the threat model and limitations.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

#: Conservative default patterns. Names are stable identifiers used in reports.
DEFAULT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("private-key", r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    ("openai-api-key", r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    ("github-token", r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ("aws-access-key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("bearer-token", r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*"),
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("home-path", r"(?:/home|/Users)/[^/\s\"']+"),
)

#: Patterns that describe credentials. Used to reject credentials in manifests without
#: treating an author e-mail or a home path as a manifest defect.
SECRET_PATTERN_NAMES = frozenset(
    {"private-key", "openai-api-key", "github-token", "aws-access-key", "bearer-token"}
)

REDACTION_SCHEMA = "modelgraph/redaction-report/v0"
MAX_REPORTED_DIGESTS = 25
MAX_REPORTED_FIELDS = 100

_PROTOCOL_TAG = re.compile(r"</?(?:request|artifact|routing_note|routes|route)\b", re.IGNORECASE)
_CONTROL_MARKER = re.compile(r"<<<MG:")


@dataclass(frozen=True)
class Pattern:
    """One compiled redaction pattern."""

    name: str
    regex: re.Pattern[str]
    action: str = "hash"


@dataclass(frozen=True)
class Finding:
    """A match report that never carries the matched text itself."""

    pattern: str
    action: str = "hash"
    digest: str = ""
    field: str = ""


@dataclass
class RedactionReport:
    """Counts, fields, and digests of applied redactions (no matched text)."""

    counts: dict[str, int] = field(default_factory=dict)
    digests: dict[str, list[str]] = field(default_factory=dict)
    fields: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def status(self) -> str:
        return "redacted" if self.total else "clean"

    def add(self, finding: Finding) -> None:
        self.counts[finding.pattern] = self.counts.get(finding.pattern, 0) + 1
        if finding.digest:
            bucket = self.digests.setdefault(finding.pattern, [])
            if finding.digest not in bucket and len(bucket) < MAX_REPORTED_DIGESTS:
                bucket.append(finding.digest)
        if finding.field:
            fields = self.fields.setdefault(finding.pattern, [])
            if finding.field not in fields and len(fields) < MAX_REPORTED_FIELDS:
                fields.append(finding.field)

    def merge(self, other: RedactionReport) -> None:
        for name, count in other.counts.items():
            self.counts[name] = self.counts.get(name, 0) + count
        for name, digests in other.digests.items():
            bucket = self.digests.setdefault(name, [])
            for digest in digests:
                if digest not in bucket and len(bucket) < MAX_REPORTED_DIGESTS:
                    bucket.append(digest)
        for name, fields in other.fields.items():
            bucket = self.fields.setdefault(name, [])
            for path in fields:
                if path not in bucket and len(bucket) < MAX_REPORTED_FIELDS:
                    bucket.append(path)

    def to_dict(self) -> dict[str, Any]:
        patterns = [
            {
                "name": name,
                "count": count,
                "digests": list(self.digests.get(name, [])),
                "fields": list(self.fields.get(name, [])),
            }
            for name, count in sorted(self.counts.items())
        ]
        return {
            "schema": REDACTION_SCHEMA,
            "status": self.status,
            "total_redactions": self.total,
            "patterns": patterns,
            "clean": self.total == 0,
        }


def compile_patterns(
    patterns: Sequence[tuple[str, str]] = DEFAULT_PATTERNS,
    *,
    action: str = "hash",
) -> tuple[Pattern, ...]:
    """Compile ``(name, regex)`` pairs; invalid regexes fail at call time."""

    if action not in {"hash", "mask", "drop"}:
        raise ValueError(f"redaction action must be hash, mask, or drop, got {action!r}")
    compiled: list[Pattern] = []
    for name, regex in patterns:
        try:
            compiled.append(Pattern(name=name, regex=re.compile(regex), action=action))
        except re.error as exc:  # pragma: no cover - the defaults are tested
            raise ValueError(f"invalid redaction pattern {name!r}: {exc}") from exc
    return tuple(compiled)


_DEFAULT_COMPILED = compile_patterns()


def default_patterns() -> tuple[Pattern, ...]:
    """The compiled conservative default pattern set."""

    return _DEFAULT_COMPILED


def _patterns_or_default(patterns: Sequence[Pattern] | None) -> Sequence[Pattern]:
    return _DEFAULT_COMPILED if patterns is None else patterns


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _marker(action: str, name: str, value: str) -> str:
    if action == "drop":
        return ""
    if action == "mask":
        return f"[REDACTED:{name}]"
    return f"[REDACTED:{name}:sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}]"


def scan_text(
    text: str,
    *,
    patterns: Sequence[Pattern] | None = None,
    include_pii: bool = True,
) -> list[Finding]:
    """Return findings for ``text`` without modifying it and without copying matches."""

    findings: list[Finding] = []
    for pattern in _patterns_or_default(patterns):
        if not include_pii and pattern.name not in SECRET_PATTERN_NAMES:
            continue
        for match in pattern.regex.finditer(text):
            findings.append(
                Finding(
                    pattern=pattern.name,
                    action=pattern.action,
                    digest=_digest(match.group(0)),
                )
            )
    return findings


def redact_text(
    text: str,
    *,
    patterns: Sequence[Pattern] | None = None,
    action: str | None = None,
    field: str = "",
) -> tuple[str, list[Finding]]:
    """Redact one string; return the safe text and findings (no matched text)."""

    findings: list[Finding] = []
    result = text
    for pattern in _patterns_or_default(patterns):
        selected = action or pattern.action

        def replace(
            match: re.Match[str],
            current: Pattern = pattern,
            selected_action: str = selected,
        ) -> str:
            findings.append(
                Finding(
                    pattern=current.name,
                    action=selected_action,
                    digest=_digest(match.group(0)),
                    field=field,
                )
            )
            return _marker(selected_action, current.name, match.group(0))

        result = pattern.regex.sub(replace, result)
    return result, findings


def redact_text_default(text: str) -> str:
    """Convenience wrapper for persistence hooks that only need the safe text."""

    result, _ = redact_text(text)
    return result


def redact_data(
    value: Any,
    *,
    patterns: Sequence[Pattern] | None = None,
    action: str | None = None,
) -> tuple[Any, RedactionReport]:
    """Recursively redact strings in a JSON-shaped value and report what fired."""

    report = RedactionReport()
    return _redact_value(value, patterns, action, "", report), report


def _redact_value(
    value: Any,
    patterns: Sequence[Pattern] | None,
    action: str | None,
    field_path: str,
    report: RedactionReport,
) -> Any:
    if isinstance(value, str):
        text, findings = redact_text(
            value, patterns=patterns, action=action, field=field_path or "text"
        )
        for finding in findings:
            report.add(finding)
        return text
    if isinstance(value, list):
        return [
            _redact_value(item, patterns, action, f"{field_path}[{index}]", report)
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _redact_value(item, patterns, action, f"{field_path}[{index}]", report)
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            child = f"{field_path}.{key}" if field_path else str(key)
            result[key] = _redact_value(item, patterns, action, child, report)
        return result
    return value


def scan_data(value: Any, *, include_pii: bool = True) -> list[Finding]:
    """Scan a JSON-shaped value and return findings with field paths."""

    findings: list[Finding] = []
    _scan_value(value, "", findings, include_pii)
    return findings


def _scan_value(
    value: Any,
    field_path: str,
    findings: list[Finding],
    include_pii: bool,
) -> None:
    if isinstance(value, str):
        for finding in scan_text(value, include_pii=include_pii):
            findings.append(
                Finding(
                    pattern=finding.pattern,
                    action=finding.action,
                    digest=finding.digest,
                    field=field_path or "text",
                )
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_value(item, f"{field_path}[{index}]", findings, include_pii)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{field_path}.{key}" if field_path else str(key)
            _scan_value(item, child, findings, include_pii)


def neutralize_control_markers(text: str) -> str:
    """Escape protocol delimiters in untrusted text.

    Only the exact protocol tokens are touched: ``<<<MG:`` control markers and the
    request/artifact/route/routing-note tags. Escaping the leading ``<`` keeps the text
    readable while making it impossible for content to open or close a protocol block or
    forge a menu entry. Ordinary prose and code (for example HTML) are unchanged.
    """

    escaped = _PROTOCOL_TAG.sub(lambda match: "&lt;" + match.group(0)[1:], text)
    return _CONTROL_MARKER.sub("&lt;&lt;&lt;MG:", escaped)


def truncate_untrusted(
    text: str,
    limit_chars: int,
    *,
    name: str,
) -> tuple[str, int]:
    """Clip ``text`` to ``limit_chars`` and append a visible truncation marker.

    Returns the clipped text and the number of UTF-8 bytes omitted (0 when untouched).
    """

    if limit_chars < 0:
        limit_chars = 0
    if len(text) <= limit_chars:
        return text, 0
    kept = text[:limit_chars]
    omitted = len(text.encode("utf-8")) - len(kept.encode("utf-8"))
    return f"{kept}\n[[truncated: {name}, {omitted} bytes omitted]]", omitted


# ---------------------------------------------------------------------------
# Load-time trust policy
# ---------------------------------------------------------------------------

#: Manifest keys that would describe executable behavior rather than provider metadata.
DANGEROUS_KEYS = frozenset(
    {
        "command",
        "commands",
        "entrypoint",
        "exec",
        "postinstall",
        "preinstall",
        "script",
        "shell",
        "subprocess",
    }
)

ALLOWED_ENDPOINT_SCHEMES = frozenset({"http", "https"})
ALLOWED_WEIGHTS_SCHEMES = frozenset({"hf", "https", "http", "file", "ollama", "s3"})
_LOCAL_HOSTNAMES = frozenset({"localhost", "localhost.localdomain"})


def _walk_strings(value: Any, path: str = "") -> list[tuple[str, str, str]]:
    """Yield ``(path, key, string_value)`` for every string in a document."""

    found: list[tuple[str, str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            found.extend(_walk_strings(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_strings(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        key = path.rsplit(".", 1)[-1].split("[", 1)[0]
        found.append((path, key, value))
    return found


def _walk_keys(value: Any, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            found.append((child, str(key)))
            found.extend(_walk_keys(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk_keys(item, f"{path}[{index}]"))
    return found


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    normalized = host.strip("[]").lower()
    if normalized in _LOCAL_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _url_scheme(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    return parsed.scheme.lower() or None


def manifest_trust_violations(document: Mapping[str, Any]) -> list[str]:
    """Return load-time trust-policy violations for one model manifest document.

    The checks are intentionally conservative and report paths/pattern names only. They do
    not prove a manifest is safe; they stop the concrete ways a manifest could request
    executable behavior, smuggle a credential, or point a local-only provider somewhere
    else. Registry loading rejects a manifest with any violation.
    """

    violations: list[str] = []
    for path, key in _walk_keys(document):
        if key.lower() in DANGEROUS_KEYS:
            violations.append(
                f"{path or key!r} describes executable behavior; model manifests may only "
                "configure a provider"
            )
    for path, _key, text in _walk_strings(document):
        findings = scan_text(text, include_pii=False)
        if findings:
            names = ", ".join(sorted({finding.pattern for finding in findings}))
            violations.append(
                f"{path or 'manifest'} contains a credential-shaped value ({names}); "
                "reference a keychain or environment variable instead"
            )

    spec = document.get("spec")
    if not isinstance(spec, Mapping):
        return violations

    provider = str(spec.get("provider") or "").lower()
    resources = spec.get("resources")
    raw_network_access = resources.get("network_access") if isinstance(resources, Mapping) else None
    network_access = str(raw_network_access) if raw_network_access is not None else None

    endpoint = spec.get("endpoint")
    if endpoint is not None:
        if not isinstance(endpoint, str) or not endpoint:
            violations.append("spec.endpoint must be a non-empty URL string when present")
        else:
            parsed = urlsplit(endpoint)
            if parsed.scheme.lower() not in ALLOWED_ENDPOINT_SCHEMES:
                violations.append(
                    "spec.endpoint must use http or https; "
                    f"got scheme {parsed.scheme or '<none>'!r}"
                )
            if parsed.username or parsed.password:
                violations.append(
                    "spec.endpoint must not embed credentials; use an environment variable "
                    "or keychain"
                )
            if provider == "mock":
                violations.append("the mock provider must not configure spec.endpoint")
            if network_access == "none":
                violations.append("spec.endpoint is set but resources.network_access is 'none'")
            elif network_access == "loopback" and not _is_loopback(parsed.hostname):
                violations.append(
                    "spec.endpoint host is not loopback but resources.network_access is 'loopback'"
                )

    license_info = spec.get("license")
    if isinstance(license_info, Mapping):
        weights_uri = license_info.get("weights_uri")
        if isinstance(weights_uri, str) and weights_uri:
            scheme = _url_scheme(weights_uri)
            if scheme is not None and scheme not in ALLOWED_WEIGHTS_SCHEMES:
                violations.append(
                    "spec.license.weights_uri has an unsupported scheme "
                    f"{scheme!r}; allowed: {', '.join(sorted(ALLOWED_WEIGHTS_SCHEMES))}"
                )
            parsed_weights = urlsplit(weights_uri)
            if parsed_weights.username or parsed_weights.password:
                violations.append("spec.license.weights_uri must not embed credentials")
        license_url = license_info.get("url")
        if isinstance(license_url, str) and license_url:
            scheme = _url_scheme(license_url)
            if scheme is not None and scheme not in ALLOWED_ENDPOINT_SCHEMES:
                violations.append("spec.license.url must use http or https")

    return violations


def profile_security_warnings(profile: Any) -> list[str]:
    """Warn about a profile that does not declare provider/license allowlists.

    ``None`` historically meant "no filter". That remains the runtime behavior, but a
    release profile should make its allowlists explicit so pruning and the run manifest
    record what was eligible.
    """

    warnings: list[str] = []
    name = getattr(profile, "name", "<profile>")
    if getattr(profile, "providers", None) is None:
        warnings.append(
            f"profile {name!r} has no providers allowlist; every registry provider is eligible"
        )
    if getattr(profile, "licenses", None) is None:
        warnings.append(
            f"profile {name!r} has no licenses allowlist; every model license is eligible"
        )
    if getattr(profile, "network", None) is None:
        warnings.append(
            f"profile {name!r} has no network ceiling; models may declare internet access"
        )
    return warnings
