"""T-0026 local model registry: add/list/inspect/verify, policy, determinism, offline."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest
import yaml
from registry_support import (
    ROOT,
    make_contribution,
    manifest_document,
    write_profile,
)
from typer.testing import CliRunner

from modelgraph.cli import app
from modelgraph.contracts import load_document
from modelgraph.registry import Registry
from modelgraph.registry_store import (
    build_index,
    file_hash,
    index_text,
    iter_entry_directories,
    list_entries,
    load_index,
    verify_registry,
)

runner = CliRunner()


def _add(root: Path, contribution: Path, *extra: str):
    return runner.invoke(
        app,
        ["registry", "add", str(contribution), "--root", str(root), *extra],
    )


def _verify(root: Path, *extra: str):
    return runner.invoke(app, ["registry", "verify", "--root", str(root), *extra])


def test_add_list_inspect_round_trip(tmp_path: Path) -> None:
    contribution = make_contribution(tmp_path / "contribution")
    root = tmp_path / "registry"
    added = _add(root, contribution, "--json")
    assert added.exit_code == 0, added.stdout + added.stderr
    payload = json.loads(added.stdout)
    assert payload["model"] == "js-specialist-test@0.1.0"
    assert Path(payload["path"]).is_dir()
    assert (root / "index.yaml").is_file()
    entry_dir = root / "models" / "js-specialist-test" / "0.1.0"
    for name in ("manifest.yaml", "card.md", "eval-report.json"):
        assert (entry_dir / name).is_file()
    assert (entry_dir / "eval" / "node" / "report.json").is_file()
    listed = runner.invoke(app, ["registry", "list", "--root", str(root), "--json"])
    assert listed.exit_code == 0, listed.stdout + listed.stderr
    rows = json.loads(listed.stdout)
    assert rows == [
        {
            "id": "js-specialist-test",
            "version": "0.1.0",
            "provider": "mock",
            "license": "apache-2.0",
            "passed": True,
            "warnings": [],
            "errors": [],
            "path": str(entry_dir),
        }
    ]
    inspected = runner.invoke(
        app, ["registry", "inspect", "js-specialist-test@0.1.0", "--root", str(root), "--json"]
    )
    assert inspected.exit_code == 0, inspected.stdout + inspected.stderr
    inspected_payload = json.loads(inspected.stdout)
    assert inspected_payload["passed"] is True
    assert inspected_payload["manifest"]["metadata"]["id"] == "js-specialist-test"
    assert "## Limitations" in inspected_payload["card"]
    assert inspected_payload["eval_report"]["evidence"]["node"]["report"]["hash"]
    verified = _verify(root, "--json")
    assert verified.exit_code == 0, verified.stdout + verified.stderr
    verification = json.loads(verified.stdout)
    assert verification["passed"] is True
    assert verification["index_current"] is True
    assert verification["entries"][0]["model"] == "js-specialist-test@0.1.0"


def test_identical_add_is_idempotent(tmp_path: Path) -> None:
    contribution = make_contribution(tmp_path / "contribution")
    root = tmp_path / "registry"
    assert _add(root, contribution).exit_code == 0
    index_before = (root / "index.yaml").read_bytes()
    second = _add(root, contribution, "--json")
    assert second.exit_code == 0, second.stdout + second.stderr
    assert json.loads(second.stdout)["changed"] is False
    assert (root / "index.yaml").read_bytes() == index_before
    assert _verify(root).exit_code == 0


def test_same_version_with_different_content_is_rejected(tmp_path: Path) -> None:
    contribution = make_contribution(tmp_path / "contribution")
    root = tmp_path / "registry"
    assert _add(root, contribution).exit_code == 0
    original = (root / "models/js-specialist-test/0.1.0/card.md").read_bytes()
    (contribution / "card.md").write_text(
        (contribution / "card.md").read_text(encoding="utf-8") + "\nChanged after publication.\n",
        encoding="utf-8",
    )
    conflict = _add(root, contribution, "--json")
    assert conflict.exit_code == 3
    assert "different content" in conflict.stderr
    assert (root / "models/js-specialist-test/0.1.0/card.md").read_bytes() == original


def test_missing_card_or_report_fails_verification(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    missing_card = make_contribution(tmp_path / "missing-card", include_card=False)
    added = _add(root, missing_card)
    assert added.exit_code == 3
    assert "missing card" in added.stderr
    missing_report = make_contribution(tmp_path / "missing-report", include_report=False)
    added = _add(root, missing_report)
    assert added.exit_code == 3
    assert "missing report" in added.stderr
    valid = make_contribution(tmp_path / "valid")
    assert _add(root, valid).exit_code == 0
    (root / "models/js-specialist-test/0.1.0/card.md").unlink()
    verified = _verify(root)
    assert verified.exit_code == 1
    assert "missing card.md" in verified.stdout + verified.stderr
    (root / "models/js-specialist-test/0.1.0/eval-report.json").unlink()
    verified = _verify(root)
    assert verified.exit_code == 1
    assert "missing eval-report.json" in verified.stdout + verified.stderr


def test_card_missing_required_section_fails(tmp_path: Path) -> None:
    card = "# Test\n\n## Base model\n\nx\n\n## Training data\n\nx\n\n## Method\n\nx\n"
    contribution = make_contribution(tmp_path / "bad-card", card_text=card)
    added = _add(tmp_path / "registry", contribution)
    assert added.exit_code == 3
    assert "evaluation" in added.stderr
    assert "limitations" in added.stderr
    assert "license" in added.stderr


def test_evidence_hash_mismatch_fails(tmp_path: Path) -> None:
    contribution = make_contribution(tmp_path / "contribution")
    root = tmp_path / "registry"
    assert _add(root, contribution).exit_code == 0
    evidence = root / "models/js-specialist-test/0.1.0/eval/node/report.json"
    evidence.write_text(evidence.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    verified = _verify(root)
    assert verified.exit_code == 1
    assert "evidence hash mismatch" in verified.stdout + verified.stderr


def test_bad_manifest_reference_hash_fails(tmp_path: Path) -> None:
    contribution = make_contribution(
        tmp_path / "contribution",
        report_overrides={"graph": {"hash": "sha256:" + "d" * 64}},
    )
    added = _add(tmp_path / "registry", contribution)
    assert added.exit_code == 3
    assert "graph hash" in added.stderr


def test_failing_eval_report_is_rejected(tmp_path: Path) -> None:
    contribution = make_contribution(tmp_path / "contribution", passed=False)
    added = _add(tmp_path / "registry", contribution)
    assert added.exit_code == 3
    assert "not passing" in added.stderr


def test_unknown_license_policy(tmp_path: Path) -> None:
    unknown = make_contribution(tmp_path / "unknown", license_name="unknown")
    rejected = _add(tmp_path / "rejected", unknown)
    assert rejected.exit_code == 3
    assert "unknown" in rejected.stderr
    allowed = _add(tmp_path / "allowed", unknown, "--allow-unknown-license", "--json")
    assert allowed.exit_code == 0, allowed.stdout + allowed.stderr

    profile_unknown = write_profile(tmp_path / "unknown.yaml", ["unknown"])
    profiled = _add(tmp_path / "profiled", unknown, "--profile", str(profile_unknown), "--json")
    assert profiled.exit_code == 0, profiled.stdout + profiled.stderr

    profile_apache = write_profile(tmp_path / "apache.yaml", ["apache-2.0"])
    rejected_by_profile = _add(
        tmp_path / "rejected-profile", unknown, "--profile", str(profile_apache)
    )
    assert rejected_by_profile.exit_code == 3
    assert "not allowed by profile" in rejected_by_profile.stderr

    apache_contribution = make_contribution(tmp_path / "apache", license_name="apache-2.0")
    accepted_by_profile = _add(
        tmp_path / "accepted-profile",
        apache_contribution,
        "--profile",
        str(profile_apache),
    )
    assert accepted_by_profile.exit_code == 0, (
        accepted_by_profile.stdout + accepted_by_profile.stderr
    )


def test_weights_hash_is_verified_when_present(tmp_path: Path) -> None:
    contribution = make_contribution(tmp_path / "contribution", weights=True)
    root = tmp_path / "registry"
    assert _add(root, contribution).exit_code == 0
    index = load_index(root)
    assert index is not None
    assert index["models"][0]["hashes"]["weights"].startswith("sha256:")
    weights = root / "models/js-specialist-test/0.1.0/weights/adapter.bin"
    assert weights.is_file()
    weights.write_bytes(b"tampered")
    verified = _verify(root)
    assert verified.exit_code == 1
    assert "weights hash mismatch" in verified.stdout + verified.stderr


def test_index_is_deterministic_and_content_hashed(tmp_path: Path) -> None:
    first = make_contribution(tmp_path / "a", model_id="alpha-model", version="0.1.0")
    second = make_contribution(tmp_path / "b", model_id="beta-model", version="0.1.0")
    root_one = tmp_path / "one"
    root_two = tmp_path / "two"
    assert _add(root_one, first).exit_code == 0
    assert _add(root_one, second).exit_code == 0
    assert _add(root_two, second).exit_code == 0
    assert _add(root_two, first).exit_code == 0
    assert (root_one / "index.yaml").read_bytes() == (root_two / "index.yaml").read_bytes()
    index = load_index(root_one)
    assert index is not None
    payload = {key: value for key, value in index.items() if key != "content_hash"}
    assert (
        index["content_hash"]
        == "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    )
    rebuilt = build_index(list_entries(root_one))
    assert index_text(rebuilt) == (root_one / "index.yaml").read_text(encoding="utf-8")


def test_stale_index_is_detected_and_repairable(tmp_path: Path) -> None:
    contribution = make_contribution(tmp_path / "contribution")
    root = tmp_path / "registry"
    assert _add(root, contribution).exit_code == 0
    card_path = root / "models" / "js-specialist-test" / "0.1.0" / "card.md"
    card_path.write_text(
        card_path.read_text(encoding="utf-8") + "\nEdited card.\n", encoding="utf-8"
    )
    stale = _verify(root)
    assert stale.exit_code == 1
    assert "stale" in stale.stdout + stale.stderr
    repaired = _verify(root, "--write-index")
    assert repaired.exit_code == 0, repaired.stdout + repaired.stderr
    assert _verify(root).exit_code == 0


def test_existing_flat_registry_still_resolves() -> None:
    flat = Registry.from_paths([ROOT / "registry" / "models"])
    assert flat.resolve_model("coder-7b") is not None
    assert flat.resolve_model("coder-7b@1.0.0") is not None
    assert flat.resolve_model("coder-7b@9.9.9") is None


def test_versioned_layout_resolves_with_the_runtime_registry(tmp_path: Path) -> None:
    contribution = make_contribution(tmp_path / "contribution")
    root = tmp_path / "registry"
    assert _add(root, contribution).exit_code == 0
    registry = Registry.from_paths([ROOT / "registry" / "models", root / "models"])
    resolved = registry.resolve_model("js-specialist-test@0.1.0")
    assert resolved is not None
    assert resolved.metadata.id == "js-specialist-test"
    assert registry.manifest_hash("js-specialist-test@0.1.0") is not None
    assert registry.resolve_model("coder-7b") is not None


def test_registry_operations_are_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("registry operations must not open a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    contribution = make_contribution(tmp_path / "contribution")
    root = tmp_path / "registry"
    assert _add(root, contribution).exit_code == 0
    assert _verify(root).exit_code == 0
    listed = runner.invoke(app, ["registry", "list", "--root", str(root), "--json"])
    assert listed.exit_code == 0
    inspected = runner.invoke(
        app, ["registry", "inspect", "js-specialist-test@0.1.0", "--root", str(root)]
    )
    assert inspected.exit_code == 0


def test_index_fails_closed_when_hand_edited(tmp_path: Path) -> None:
    contribution = make_contribution(tmp_path / "contribution")
    root = tmp_path / "registry"
    assert _add(root, contribution).exit_code == 0
    index_path = root / "index.yaml"
    document = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    document["models"][0]["license"] = "not-the-manifest-license"
    index_path.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    verified = _verify(root)
    assert verified.exit_code == 1
    assert "content hash mismatch" in verified.stderr


def test_empty_registry_index_is_current(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    result = verify_registry(root, write_index=True)
    assert result.passed, result.errors
    assert result.index_current is True
    assert (root / "index.yaml").is_file()
    second = verify_registry(root)
    assert second.passed, second.errors
    assert second.index_current is True


def test_release_checklist_includes_registry_verify() -> None:
    security = (ROOT / "docs" / "security.md").read_text(encoding="utf-8")
    assert "mg registry verify" in security


def test_manifest_fixture_is_schema_valid(tmp_path: Path) -> None:
    document = manifest_document()
    (tmp_path / "manifest.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    assert load_document(tmp_path / "manifest.yaml")["kind"] == "Model"


def test_iter_entry_directories_ignores_flat_starter_files(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "coder-7b.yaml").write_text("not a versioned entry", encoding="utf-8")
    assert iter_entry_directories(tmp_path) == []
    make_contribution(tmp_path / "contribution")
    root = tmp_path / "registry"
    assert _add(root, tmp_path / "contribution").exit_code == 0
    assert len(iter_entry_directories(root)) == 1
    assert file_hash(root / "index.yaml").startswith("sha256:")
