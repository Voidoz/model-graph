"""The run-directory secret scanner: finds planted values and never echoes them."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from security_helpers import ROOT, SECRET_EMAIL, SECRET_KEY

SCRIPT = ROOT / "scripts" / "scan_secrets.py"


def _scan(
    *paths: Path, as_json: bool = True, include_pii: bool = False
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT), *(str(path) for path in paths)]
    if as_json:
        command.append("--json")
    if include_pii:
        command.append("--include-pii")
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_scanner_finds_planted_values_without_printing_them(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_secret"
    run_dir.mkdir(parents=True)
    (run_dir / "trace.jsonl").write_text(
        json.dumps({"event": "provider.response", "text": f"key {SECRET_KEY}"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps({"request": {"task": f"mail {SECRET_EMAIL}"}}),
        encoding="utf-8",
    )
    result = _scan(tmp_path)
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert SECRET_KEY not in combined
    assert SECRET_EMAIL not in combined
    payload = json.loads(result.stdout)
    assert payload["schema"] == "modelgraph/secret-scan/v0"
    assert payload["include_pii"] is False
    patterns = {finding["pattern"] for finding in payload["findings"]}
    assert patterns == {"openai-api-key"}
    assert all(finding["digest"].startswith("sha256:") for finding in payload["findings"])

    with_pii = _scan(tmp_path, include_pii=True)
    assert with_pii.returncode == 1
    pii_patterns = {finding["pattern"] for finding in json.loads(with_pii.stdout)["findings"]}
    assert {"openai-api-key", "email"} <= pii_patterns


def test_scanner_reports_clean_for_a_clean_tree(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("ordinary notes, no credentials\n", encoding="utf-8")
    result = _scan(tmp_path)
    assert result.returncode == 0
    assert SECRET_KEY not in result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["total"] == 0
    assert payload["files_scanned"] == 1


def test_scanner_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01" + SECRET_KEY.encode())
    (tmp_path / "ok.txt").write_text("nothing to see\n", encoding="utf-8")
    result = _scan(tmp_path)
    assert result.returncode == 0


def test_scanner_exits_two_for_a_missing_path(tmp_path: Path) -> None:
    result = _scan(tmp_path / "missing")
    assert result.returncode == 2
    assert "path not found" in result.stderr


def test_scanner_human_output_never_echoes_matches(tmp_path: Path) -> None:
    (tmp_path / "trace.jsonl").write_text(f"token={SECRET_KEY}\n", encoding="utf-8")
    result = _scan(tmp_path, as_json=False)
    assert result.returncode == 1
    assert SECRET_KEY not in result.stdout + result.stderr
    assert "openai-api-key" in result.stdout
