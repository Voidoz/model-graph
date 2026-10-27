"""T-0025 specialist kit: synthetic pipeline, evidence, determinism, honest refusals."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from specialist_support import ROOT, build_dataset, write_config
from typer.testing import CliRunner

from modelgraph.cli import app
from modelgraph.contracts import validate_document
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.registry_store import card_issues, verify_eval_report
from modelgraph.training.errors import SpecialistDataError, SpecialistPromotionError
from modelgraph.training.specialist import (
    ARTIFACT_SCHEMA,
    load_specialist_config,
    train_specialist,
    validate_specialist_artifact,
)

runner = CliRunner()
FORBIDDEN_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".ckpt", ".onnx")


def _load(tmp_path: Path, dataset: Path, **kwargs: object):
    config_path = write_config(tmp_path, dataset, **kwargs)  # type: ignore[arg-type]
    return load_specialist_config(config_path), config_path


@pytest.mark.parametrize(
    ("name", "method", "mode", "tier"),
    [
        ("specialist-synthetic.yaml", "lora", "synthetic", "cpu"),
        ("distill-synthetic.yaml", "distill", "synthetic", "cpu"),
        ("specialist-8gb-qlora.yaml", "qlora", "real", "8gb"),
        ("specialist-24gb-lora.yaml", "lora", "real", "24gb"),
        ("distill-24gb.yaml", "distill", "real", "24gb"),
    ],
)
def test_reference_configs_validate(name: str, method: str, mode: str, tier: str) -> None:
    config = load_specialist_config(ROOT / "training" / "configs" / name)
    assert config.method == method
    assert config.mode == mode
    assert config.hardware.tier == tier
    if mode == "real":
        # The real recipes must state data/hardware expectations and stay unverified.
        assert config.hardware.expected_data
        assert config.hardware.expected_runtime
    if method == "distill":
        assert config.distillation is not None


def test_synthetic_pipeline_end_to_end(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config, config_path = _load(tmp_path, dataset)
    result = train_specialist(
        config, run_eval=True, command=f"mg train specialist --config {config_path}"
    )
    assert result.status == "candidate"
    for path in (
        result.artifact_path,
        result.manifest_path,
        result.card_path,
        result.reproducibility_path,
        result.output_dir / "training-report.md",
        result.output_dir / "dataset-summary.json",
        result.eval_report_path,
    ):
        assert path is not None and Path(path).is_file(), path
    artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert artifact["schema"] == ARTIFACT_SCHEMA
    assert artifact["synthetic"] is True
    assert artifact["quality_claim"] is False
    assert artifact["rehearsal"]["steps"] == 0
    assert artifact["weights"]["format"] == "pseudo-sha256"
    assert validate_specialist_artifact(artifact) == []
    assert validate_document(result.manifest, "manifest") == []
    assert ModelManifest.model_validate(result.manifest).metadata.id == "js-specialist-test"
    assert card_issues(result.card) == []
    assert result.eval_report is not None
    assert result.eval_report["schema"] == "modelgraph/registry-eval-report/v0"
    assert validate_document(result.eval_report, "registry-eval-report") == []
    assert result.eval_report["passed"] is True
    assert result.eval_report["graph"]["pinned_model"] == "js-specialist-test@0.1.0"
    assert result.eval_report["results"]["node"]["cases"] >= 1
    assert result.eval_report["results"]["graph"]["cases"] == 3
    issues, warnings, hashes = verify_eval_report(
        result.eval_report, result.manifest, base=result.output_dir
    )
    assert issues == [], issues
    assert any("integrity.sha256" in warning for warning in warnings)
    assert "node.report" in hashes and "graph.run_manifest" in hashes
    for path in result.output_dir.rglob("*"):
        assert path.suffix not in FORBIDDEN_WEIGHT_SUFFIXES, path
    reproducibility = json.loads(result.reproducibility_path.read_text(encoding="utf-8"))
    assert reproducibility["commit"]
    assert reproducibility["config"]["hash"].startswith("sha256:")
    assert reproducibility["dataset"]["hash"] == result.dataset.dataset_hash
    assert reproducibility["artifact"]["sha256"] == result.artifact_hash
    assert reproducibility["hardware"]["observed"]["python"]


def test_card_has_required_sections_and_propagates_license(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config, _ = _load(tmp_path, dataset)
    result = train_specialist(config, run_eval=False)
    card = result.card.lower()
    for heading in (
        "## base model",
        "## training data",
        "## method",
        "## hyperparameters",
        "## evaluation",
        "## limitations",
        "## license",
        "## reproducibility",
    ):
        assert heading in card
    assert "cc0-1.0" in card
    assert "synthetic test consent" in card
    assert "specialist-test-0001" in card
    assert "apache-2.0" in card
    assert "declaration, not legal advice" in card
    assert "synthetic rehearsal" in card
    notes = result.manifest["spec"]["license"]["notes"]
    assert "CC0-1.0" in notes
    assert result.dataset.consent["recorded"] is True
    assert result.dataset.license_id == "CC0-1.0"


def test_deterministic_artifact_manifest_and_card(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config, config_path = _load(tmp_path, dataset)
    first = train_specialist(
        config, run_eval=False, command=f"mg train specialist --config {config_path}"
    )
    artifact = first.artifact_path.read_bytes()
    manifest = first.manifest_path.read_bytes()
    card = first.card_path.read_bytes()
    second = train_specialist(
        config, run_eval=False, command=f"mg train specialist --config {config_path}"
    )
    assert first.artifact_path.read_bytes() == artifact
    assert second.manifest_path.read_bytes() == manifest
    assert second.card_path.read_bytes() == card
    assert first.artifact_hash == second.artifact_hash


def test_refuses_promotion_without_eval_report(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config, _ = _load(tmp_path, dataset)
    with pytest.raises(SpecialistPromotionError, match="without a passing eval report"):
        train_specialist(config, run_eval=False, promote=True)


def test_refuses_to_promote_synthetic_even_with_passing_eval(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config, config_path = _load(tmp_path, dataset)
    with pytest.raises(SpecialistPromotionError, match="synthetic rehearsal"):
        train_specialist(
            config,
            run_eval=True,
            promote=True,
            command=f"mg train specialist --config {config_path}",
        )


def test_adopts_supplied_passing_eval_report(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config, config_path = _load(tmp_path, dataset)
    first = train_specialist(config, run_eval=True)
    second_config = config.model_copy(update={"output_dir": str(tmp_path / "adopted")})
    second = train_specialist(
        second_config,
        run_eval=False,
        eval_report_path=first.eval_report_path,
        command=f"mg train specialist --config {config_path}",
    )
    assert second.eval_report is not None and second.eval_report["passed"] is True
    assert second.eval_report_path is not None
    assert (second.output_dir / "eval" / "node" / "report.json").is_file()
    assert (second.output_dir / "eval" / "graph" / "manifest.json").is_file()


def test_distillation_rehearsal_filters_recorded_outputs(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config, _ = _load(tmp_path, dataset, method="distill")
    result = train_specialist(config, run_eval=False)
    assert result.artifact["method"] == "distill"
    assert result.artifact["teacher"]["id"] == "recorded-dataset-output"
    assert result.artifact["teacher"]["calls"] == 0
    assert result.dataset.filter_report["teacher"] == "recorded-dataset-output"
    assert result.dataset.filter_report["kept"] == len(result.dataset.examples)
    assert "teacher" in result.card.lower()
    assert "zero teacher calls" in result.card


def test_refuses_dataset_without_recorded_consent(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    manifest_path = dataset / "dataset.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["spec"]["consent"]["recorded"] = False
    manifest_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    config, _ = _load(tmp_path, dataset)
    with pytest.raises(SpecialistDataError, match="consent"):
        train_specialist(config, run_eval=False)


def test_real_recipe_script_refuses_without_stack(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "training" / "sft_lora.py"),
            "--config",
            str(ROOT / "training" / "configs" / "specialist-8gb-qlora.yaml"),
            "--dataset",
            str(dataset),
            "--out",
            str(tmp_path / "real"),
            "--execute",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 3, completed.stdout + completed.stderr
    assert "not installed" in completed.stderr or "Refusing" in completed.stderr
    assert not (tmp_path / "real" / "adapter").exists()


def test_real_recipe_dry_run_is_explicit_about_unverified_environment(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "training" / "sft_lora.py"),
            "--config",
            str(ROOT / "training" / "configs" / "specialist-24gb-lora.yaml"),
            "--dry-run",
            "--json",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["method"] == "lora"
    assert payload["hardware"]["tier"] == "24gb"
    assert payload["missing_modules"]
    assert payload["note"] == "No training is performed by --dry-run."


def test_cli_train_specialist_no_eval(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config_path = write_config(tmp_path, dataset, output=tmp_path / "cli-out")
    result = runner.invoke(
        app,
        [
            "train",
            "specialist",
            "--config",
            str(config_path),
            "--no-eval",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["model"] == "js-specialist-test@0.1.0"
    assert payload["status"] == "candidate"
    assert payload["synthetic"] is True
    assert payload["quality_claim"] is False
    assert Path(payload["artifact"]).is_file()
    assert Path(payload["manifest"]).is_file()
    assert Path(payload["card"]).is_file()
    assert payload["eval_report"] is None


def test_cli_promote_without_eval_fails(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    config_path = write_config(tmp_path, dataset, output=tmp_path / "cli-promote")
    result = runner.invoke(
        app,
        [
            "train",
            "specialist",
            "--config",
            str(config_path),
            "--no-eval",
            "--promote",
        ],
    )
    assert result.exit_code == 3
    assert "without a passing eval report" in result.stderr
