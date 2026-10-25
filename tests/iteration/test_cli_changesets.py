"""CLI coverage for the T-0038 revision and change-set commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from modelgraph.cli import app

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_MODELS = Path(__file__).resolve().parent / "fixtures" / "models"
runner = CliRunner()


def _json(result) -> dict:
    return json.loads(result.stdout)


def test_cli_bootstrap_swap_explain_status_and_rollback(tmp_path: Path) -> None:
    store = tmp_path / "store"
    bootstrap = runner.invoke(
        app,
        [
            "promote",
            "--graph",
            str(ROOT / "graphs" / "coding-specialist.graph.yaml"),
            "--store",
            str(store),
            "--json",
        ],
    )
    assert bootstrap.exit_code == 0, bootstrap.stdout + bootstrap.stderr
    base = _json(bootstrap)["pointer"]["revision_id"]

    swap = runner.invoke(
        app,
        [
            "swap",
            "node",
            "js-specialist",
            "--to",
            "js-specialist-3b@0.4.0",
            "--registry",
            str(FIXTURE_MODELS),
            "--store",
            str(store),
            "--json",
        ],
    )
    assert swap.exit_code == 0, swap.stdout + swap.stderr
    payload = _json(swap)
    change = payload["change"]
    candidate = change["candidate_revision"]
    assert payload["active_revision"] == base
    assert change["base_revision"] == base
    assert candidate != base

    listed = runner.invoke(app, ["change", "list", "--store", str(store), "--json"])
    assert listed.exit_code == 0
    assert [item["change_id"] for item in _json(listed)] == [change["change_id"]]

    explained = runner.invoke(
        app,
        [
            "change",
            "explain",
            change["change_id"],
            "--store",
            str(store),
            "--json",
        ],
    )
    assert explained.exit_code == 0
    assert _json(explained)["impact"]["compatibility"]["status"] == "compatible"

    blocked = runner.invoke(
        app,
        ["promote", "--revision", candidate, "--store", str(store), "--dry-run"],
    )
    assert blocked.exit_code == 3
    assert "MG-6004" in blocked.stderr

    rollback = runner.invoke(
        app,
        [
            "rollback",
            "--to",
            base,
            "--store",
            str(store),
            "--dry-run",
            "--json",
        ],
    )
    assert rollback.exit_code == 3
    assert "already active" in rollback.stderr

    # Bootstrap is explicit and must not silently re-promote over an active revision.
    second_bootstrap = runner.invoke(
        app,
        [
            "promote",
            "--graph",
            str(ROOT / "graphs" / "hello-triage.graph.yaml"),
            "--store",
            str(store),
        ],
    )
    assert second_bootstrap.exit_code == 3
    assert "MG-6002" in second_bootstrap.stderr

    status = runner.invoke(app, ["status", "--revisions", "--store", str(store), "--json"])
    assert status.exit_code == 0
    report = _json(status)
    assert report["active_revision"] == base
    assert any(item["revision_id"] == candidate for item in report["revisions"])

    diff = runner.invoke(app, ["graph", "diff", base, candidate, "--store", str(store), "--json"])
    assert diff.exit_code == 0, diff.stdout + diff.stderr
    diff_payload = _json(diff)
    assert diff_payload["graph_changed"] is True
    assert diff_payload["model_bindings"][0]["node"] == "js-specialist"


def test_cli_dry_run_swap_does_not_persist(tmp_path: Path) -> None:
    store = tmp_path / "store"
    bootstrap = runner.invoke(
        app,
        [
            "promote",
            "--graph",
            str(ROOT / "graphs" / "coding-specialist.graph.yaml"),
            "--store",
            str(store),
            "--json",
        ],
    )
    assert bootstrap.exit_code == 0
    swap = runner.invoke(
        app,
        [
            "swap",
            "node",
            "js-specialist",
            "--to",
            str(FIXTURE_MODELS / "js-specialist-3b-0.4.0.yaml"),
            "--store",
            str(store),
            "--dry-run",
            "--json",
        ],
    )
    assert swap.exit_code == 0, swap.stdout + swap.stderr
    assert _json(swap)["dry_run"] is True
    listed = runner.invoke(app, ["change", "list", "--store", str(store), "--json"])
    assert listed.exit_code == 0
    assert _json(listed) == []


def test_cli_fused_tokenizer_rejection_is_actionable(tmp_path: Path) -> None:
    store = tmp_path / "store"
    bootstrap = runner.invoke(
        app,
        [
            "promote",
            "--graph",
            str(ROOT / "graphs" / "coding-specialist.graph.yaml"),
            "--store",
            str(store),
            "--json",
        ],
    )
    assert bootstrap.exit_code == 0
    result = runner.invoke(
        app,
        [
            "swap",
            "node",
            "js-specialist",
            "--to",
            "story-7b",
            "--target",
            "fused",
            "--store",
            str(store),
        ],
    )
    assert result.exit_code == 3
    assert "MG-6003" in result.stderr
    assert "migration" in result.stderr
    status = runner.invoke(app, ["status", "--revisions", "--store", str(store), "--json"])
    assert status.exit_code == 0
    assert _json(status)["active_revision"] == _json(bootstrap)["pointer"]["revision_id"]
