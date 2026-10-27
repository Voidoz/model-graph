"""CLI wiring for `mg train dataset` and `mg train router`."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from training_support import dataset_example, write_jsonl
from typer.testing import CliRunner

from modelgraph.cli import app

runner = CliRunner()


def _dataset(tmp_path: Path) -> Path:
    records = [
        dataset_example(
            example_id="ex_0000000000000011",
            split="train",
            group="g11",
            task="Write a Python function that sorts records.",
            route="coding",
            chosen_route="coding",
        ),
        dataset_example(
            example_id="ex_0000000000000012",
            split="train",
            group="g12",
            task="Tell a story about a lighthouse keeper.",
            route="story",
            chosen_route="story",
        ),
        dataset_example(
            example_id="ex_0000000000000013",
            split="validation",
            group="g13",
            task="Debug a Rust compiler error.",
            route="coding",
            chosen_route="coding",
        ),
        dataset_example(
            example_id="ex_0000000000000014",
            split="validation",
            group="g14",
            task="Write a poem about rain.",
            route="story",
            chosen_route="story",
        ),
        dataset_example(
            example_id="ex_0000000000000015",
            split="test",
            group="g15",
            task="Write a TypeScript function that retries requests.",
            route="coding",
            chosen_route="coding",
        ),
        dataset_example(
            example_id="ex_0000000000000016",
            split="test",
            group="g16",
            task="Tell a tale about a dragon.",
            route="story",
            chosen_route="story",
        ),
    ]
    return write_jsonl(tmp_path / "examples.jsonl", records)


def test_train_dataset_cli(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    out = tmp_path / "router-dataset"
    result = runner.invoke(
        app, ["train", "dataset", "--dataset", str(source), "--out", str(out), "--json"]
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["examples"] == 6
    assert payload["counts"] == {"train": 2, "validation": 2, "test": 2}
    assert (out / "router-examples.jsonl").is_file()
    assert (out / "router-dataset.json").is_file()


def test_train_router_cli_writes_artifact_and_report(tmp_path: Path) -> None:
    source = _dataset(tmp_path)
    config = {
        "schema": "modelgraph/router-training-config/v0",
        "dataset": str(source),
        "artifact": str(tmp_path / "router" / "artifact.json"),
        "report": str(tmp_path / "router" / "report.md"),
        "seed": 5,
        "features": {"dimension": 48},
        "training": {"epochs": 40, "learning_rate": 0.03},
        "evaluation": {"require_all_splits": True, "require_frozen_test": False},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = runner.invoke(app, ["train", "router", "--config", str(config_path), "--json"])
    assert result.exit_code == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["artifact"]).is_file()
    assert Path(payload["report"]).is_file()
    assert payload["seed"] == 5
    assert payload["metrics"]["trained"]["test"]["cases"] == 2
    assert payload["recommendation"]["promote"] is False
