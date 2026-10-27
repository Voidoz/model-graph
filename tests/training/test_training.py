"""Deterministic training, serialization, split discipline, and artifact validation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from training_support import render_messages, synthetic_dataset

from modelgraph.training.artifact import (
    RouterArtifact,
    artifact_schema,
    validate_router_artifact,
)
from modelgraph.training.config import RouterTrainingConfig
from modelgraph.training.dataset import RouterDataset
from modelgraph.training.errors import RouterTrainingError
from modelgraph.training.features import RouterView
from modelgraph.training.model import train_router
from modelgraph.training.predict import Predictor


def _config(tmp_path: Path) -> RouterTrainingConfig:
    return RouterTrainingConfig(
        dataset="unused",
        artifact=str(tmp_path / "artifact.json"),
        report=str(tmp_path / "report.md"),
        seed=20261025,
        features={"dimension": 64, "bigrams": True},
        training={"epochs": 80, "learning_rate": 0.03},
        evaluation={"require_all_splits": True, "require_frozen_test": True},
    )


def test_training_is_deterministic(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    config = _config(tmp_path)
    first = train_router(dataset, config, write=False)
    second = train_router(dataset, config, write=False)
    assert first.artifact == second.artifact
    assert first.predictor.artifact.sha256() == second.predictor.artifact.sha256()
    assert first.artifact["classes"] == ["coding", "story"]
    assert first.artifact["features"] == {"bigrams": True}
    assert first.artifact["seed"] == 20261025
    assert first.artifact["training"]["train_count"] == 8
    assert validate_router_artifact(first.artifact) == []
    assert first.report["calibration"]["fit_split"] == "validation"
    assert first.report["abstain"]["threshold"] >= 0.0


def test_calibration_and_threshold_use_validation_only(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    config = _config(tmp_path)
    baseline = train_router(dataset, config, write=False)
    changed_examples = [
        replace(
            example,
            oracle_route=("story" if example.oracle_route == "coding" else "coding"),
            target=("story" if example.oracle_route == "coding" else "coding"),
        )
        if example.split == "test"
        else example
        for example in dataset.examples
    ]
    changed = RouterDataset(
        path=None,
        examples=changed_examples,
        summary=dataset.summary,
        dataset_manifest_hash=dataset.dataset_manifest_hash,
    )
    other = train_router(changed, config, write=False)
    assert baseline.artifact["weights"] == other.artifact["weights"]
    assert baseline.artifact["calibration"] == other.artifact["calibration"]
    assert baseline.artifact["abstain"] == other.artifact["abstain"]
    assert baseline.artifact["metrics"]["test"] != other.artifact["metrics"]["test"]
    assert baseline.artifact["calibration"]["fit_count"] == 4
    assert baseline.report["calibration"]["fit_split"] == "validation"
    assert baseline.report["abstain"]["threshold"] == other.report["abstain"]["threshold"]


def test_calibration_uses_validation_correctness_not_a_default_key(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    examples = list(dataset.examples)
    index = next(
        i
        for i, item in enumerate(examples)
        if item.split == "validation" and not item.ambiguous and item.oracle_route == "coding"
    )
    examples[index] = replace(examples[index], oracle_route="story", target="story")
    dataset = RouterDataset(
        path=None,
        examples=examples,
        summary=dataset.summary,
        dataset_manifest_hash=dataset.dataset_manifest_hash,
    )
    result = train_router(dataset, _config(tmp_path), write=False)
    assert result.artifact["calibration"]["fit_count"] == 4
    assert "single-class" not in result.artifact["calibration"]["notes"]
    assert result.artifact["calibration"]["temperature"] >= 1.0


def test_report_reliability_uses_the_same_routed_population_as_metrics(
    tmp_path: Path,
) -> None:
    result = train_router(synthetic_dataset(), _config(tmp_path), write=False)
    for split in ("validation", "test"):
        metrics = result.report["metrics"]["trained"][split]
        reliability = result.report["reliability"][split]
        assert reliability["after"]["n"] == metrics["calibration_cases"]
        assert reliability["after"]["ece"] == pytest.approx(metrics["ece"])
        assert reliability["before"]["n"] == reliability["after"]["n"]


def test_bigram_feature_switch_is_recorded_and_used(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.features.bigrams = False
    result = train_router(synthetic_dataset(), config, write=False)
    assert result.artifact["features"] == {"bigrams": False}
    assert result.predictor.bigrams is False
    loaded = Predictor.load(
        RouterArtifact.from_dict(result.artifact).save(tmp_path / "bigram.json")
    )
    assert loaded.bigrams is False
    for example in synthetic_dataset().split("test"):
        assert (
            loaded.decision(_view(example)).to_json()
            == result.predictor.decision(_view(example)).to_json()
        )


def test_predictor_passes_the_serialized_bigram_switch_to_featurize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modelgraph.training import predict as predict_module

    config = _config(tmp_path)
    config.features.bigrams = False
    result = train_router(synthetic_dataset(), config, write=False)
    loaded = Predictor.load(RouterArtifact.from_dict(result.artifact).save(tmp_path / "spy.json"))
    seen: list[bool | None] = []
    real_featurize = predict_module.featurize

    def spy(view, candidate, **kwargs):
        seen.append(kwargs.get("bigrams"))
        return real_featurize(view, candidate, **kwargs)

    monkeypatch.setattr(predict_module, "featurize", spy)
    loaded.decision(_view(synthetic_dataset().split("test")[0]))
    assert seen
    assert all(value is False for value in seen)


def test_test_only_route_does_not_change_classes_or_weights(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    baseline = train_router(dataset, _config(tmp_path), write=False)
    examples = []
    changed = False
    for example in dataset.examples:
        if (
            not changed
            and example.split == "test"
            and not example.ambiguous
            and example.oracle_route == "coding"
        ):
            examples.append(replace(example, oracle_route="music", target="music"))
            changed = True
        else:
            examples.append(example)
    assert changed
    held_out = RouterDataset(
        path=None,
        examples=examples,
        summary=dataset.summary,
        dataset_manifest_hash=dataset.dataset_manifest_hash,
    )
    result = train_router(held_out, _config(tmp_path), write=False)
    assert result.artifact["classes"] == baseline.artifact["classes"] == ["coding", "story"]
    assert result.artifact["weights"] == baseline.artifact["weights"]
    assert result.artifact["calibration"] == baseline.artifact["calibration"]
    assert result.artifact["abstain"] == baseline.artifact["abstain"]


def test_serialization_round_trip_preserves_predictions(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    result = train_router(dataset, _config(tmp_path), write=False)
    artifact = RouterArtifact.from_dict(result.artifact)
    path = artifact.save(tmp_path / "artifact.json")
    loaded = Predictor.load(path)
    for example in dataset.split("test"):
        original = result.predictor.decision(_view(example))
        restored = loaded.decision(_view(example))
        assert restored.to_json() == original.to_json()
        assert restored.probabilities == pytest.approx(original.probabilities)
    assert loaded.artifact.sha256() == artifact.sha256()


def test_write_artifact_and_report(tmp_path: Path) -> None:
    result = train_router(synthetic_dataset(), _config(tmp_path), write=True)
    assert result.artifact_path is not None and result.artifact_path.is_file()
    assert result.report_path is not None and result.report_path.is_file()
    report_json = result.report_path.with_suffix(".json")
    assert report_json.is_file()
    document = json.loads(report_json.read_text())
    assert document["schema"] == "modelgraph/router-training-report/v0"
    assert document["dataset"]["counts"] == {"train": 8, "validation": 5, "test": 5}
    assert document["calibration"]["fit_split"] == "validation"
    assert "Reliability table" in result.report_path.read_text()


def test_missing_validation_split_is_rejected(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    only_train = RouterDataset(
        path=None,
        examples=[item for item in dataset.examples if item.split == "train"],
        summary=dataset.summary,
        dataset_manifest_hash=dataset.dataset_manifest_hash,
    )
    with pytest.raises(RouterTrainingError, match="validation"):
        train_router(only_train, _config(tmp_path), write=False)


def test_frozen_test_requirement_is_enforced(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    summary = dict(dataset.summary)
    summary["dataset"] = dict(summary["dataset"])
    summary["dataset"]["frozen_test"] = False
    unfrozen = RouterDataset(
        path=None,
        examples=dataset.examples,
        summary=summary,
        dataset_manifest_hash=dataset.dataset_manifest_hash,
    )
    with pytest.raises(RouterTrainingError, match="frozen test"):
        train_router(unfrozen, _config(tmp_path), write=False)


def test_artifact_rejects_unknown_schema_and_bad_weights(tmp_path: Path) -> None:
    dataset = synthetic_dataset()
    artifact = train_router(dataset, _config(tmp_path), write=False).artifact
    broken = json.loads(json.dumps(artifact))
    broken["schema"] = "modelgraph/router-artifact/v99"
    assert validate_router_artifact(broken)
    with pytest.raises(Exception, match="invalid router artifact"):
        RouterArtifact.from_dict(broken)
    short = json.loads(json.dumps(artifact))
    short["weights"]["coding"] = short["weights"]["coding"][:-1]
    assert any("expected" in issue for issue in validate_router_artifact(short))


def _view(example):
    return RouterView(
        request=example.request,
        routes=tuple(example.menu_routes),
        messages=render_messages(example.request),
        prompt_hash=example.prompt_hash,
    )


def test_artifact_schema_is_valid_json_schema() -> None:
    schema = artifact_schema()
    Draft202012Validator.check_schema(schema)
    assert schema["$id"] == "urn:modelgraph:training:router-artifact:v0"
    assert schema["properties"]["schema"]["const"] == "modelgraph/router-artifact/v0"
