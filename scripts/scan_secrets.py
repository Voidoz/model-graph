#!/usr/bin/env python3
"""Scan a directory tree for secret/PII-shaped text without printing matches.

Usage::

    uv run python scripts/scan_secrets.py runs/
    uv run python scripts/scan_secrets.py runs/ --json

Exit codes: 0 clean, 1 findings, 2 usage/path error. The scanner is deliberately simple and
offline: it reports the file, line, pattern name, and a truncated SHA-256 digest of the match.
It never prints the matched text, so running it in CI or a terminal does not re-leak a secret.

The default patterns come from :mod:`modelgraph.security`. Credential patterns run by
default; e-mail and home-path patterns are opt-in with ``--include-pii`` because local paths
in historical evidence are noisy. A finding is a prompt to investigate, not proof that the
value is live; pattern-based scanning also misses formats it does not know. See
``docs/security.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from modelgraph.security import scan_text

SCAN_SCHEMA = "modelgraph/secret-scan/v0"
MAX_FILE_BYTES = 10_000_000
SKIP_DIRECTORIES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".uv-python",
    ".venv",
    "__pycache__",
    "node_modules",
}


def iter_files(root: Path) -> list[Path]:
    """Return regular files under ``root`` in deterministic order."""

    if root.is_file():
        return [root]
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def scan_file(path: Path, *, include_pii: bool = False) -> list[dict[str, Any]]:
    """Scan one file line by line; binary and oversized files are skipped."""

    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size > MAX_FILE_BYTES:
        return []
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    if b"\x00" in raw:
        return []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return []
    findings: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for finding in scan_text(line, include_pii=include_pii):
            findings.append(
                {
                    "file": str(path),
                    "line": line_number,
                    "pattern": finding.pattern,
                    "digest": finding.digest,
                }
            )
    return findings


def scan_paths(paths: list[Path], *, include_pii: bool = False) -> tuple[list[dict[str, Any]], int]:
    """Scan all paths and return ``(findings, files_scanned)``."""

    findings: list[dict[str, Any]] = []
    scanned = 0
    for path in paths:
        for file_path in iter_files(path):
            scanned += 1
            findings.extend(scan_file(file_path, include_pii=include_pii))
    return findings, scanned


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to scan (for example runs/).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    parser.add_argument(
        "--include-pii",
        action="store_true",
        help="Also scan e-mail and home-path patterns (noisier).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    missing = [str(path) for path in args.paths if not path.exists()]
    if missing:
        print(f"scan_secrets: path not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    findings, scanned = scan_paths(args.paths, include_pii=args.include_pii)
    if args.json:
        payload = {
            "schema": SCAN_SCHEMA,
            "roots": [str(path) for path in args.paths],
            "files_scanned": scanned,
            "include_pii": args.include_pii,
            "total": len(findings),
            "findings": findings,
        }
        print(json.dumps(payload, indent=2))
    else:
        for finding in findings:
            print(
                f"{finding['file']}:{finding['line']}: {finding['pattern']} ({finding['digest']})"
            )
        if findings:
            print(
                f"\n{len(findings)} finding(s) in {scanned} file(s). "
                "Matched text is intentionally not printed.",
                file=sys.stderr,
            )
        else:
            print(f"clean: no secret/PII patterns in {scanned} file(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
