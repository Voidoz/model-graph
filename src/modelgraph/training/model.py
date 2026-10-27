"""Deterministic conditional-logit training for the router.

The model is a multinomial conditional logit: every menu candidate gets a score from a sparse
feature vector, and a softmax over the candidates gives the class distribution. It is trained
with full-batch gradient descent or Adam in plain Python. There is no randomness in the default
path (zero initialization, deterministic example order), so the same dataset and config produce
byte-identical weights; the recorded seed is part of the recipe and future training options that
need randomness must use it.

Group discipline: the loader already rejects a group that appears in two splits. Training reads
only ``train``, calibration and threshold selection read only ``validation``, and ``test`` is
evaluated once. The trainer never fits or selects on test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelgraph import __version__
from modelgraph.training.artifact import RouterArtifact, build_artifact
from modelgraph.training.calibration import (
    ThresholdChoice,
    apply_temperature,
    evaluate_predictions,
    fit_temperature,
    incumbent_predictions,
    reliability_table,
    select_abstain_threshold,
)
from modelgraph.training.config import RouterTrainingConfig
from modelgraph.training.dataset import RouterDataset, RouterExample
from modelgraph.training.errors import RouterTrainingError
from modelgraph.training.features import RouterView, feature_dimension, featurize
from modelgraph.training.predict import Predictor
from modelgraph.training.rules import evaluate_rules


@dataclass
class TrainingResult:
    artifact: dict[str, Any]
    artifact_path: Path | None
    report: dict[str, Any]
    report_path: Path | None
    predictor: Predictor
    dataset: RouterDataset
    config: RouterTrainingConfig


@dataclass
class _Encoded:
    example: RouterExample
    candidates: list[str]
    vectors: list[dict[int, float]]
    target_index: int
    target: str


class ConditionalLogit:
    """A sparse conditional-logit classifier with deterministic training."""

    def __init__(
        self,
        classes: Sequence[str],
        *,
        dimension: int,
        seed: int,
    ):
        self.classes = list(classes)
        self.total_dimension = feature_dimension(dimension)
        self.seed = seed
        self.weights: dict[str, list[float]] = {
            name: [0.0] * self.total_dimension for name in self.classes
        }

    def score(self, name: str, vector: dict[int, float]) -> float:
        weights = self.weights[name]
        return sum(weights[index] * value for index, value in vector.items())

    def probabilities(
        self, candidates: Sequence[str], vectors: Sequence[dict[int, float]]
    ) -> dict[str, float]:
        scores = {
            name: self.score(name, vector) for name, vector in zip(candidates, vectors, strict=True)
        }
        maximum = max(scores.values()) if scores else 0.0
        exponents = {name: math.exp(value - maximum) for name, value in scores.items()}
        total = sum(exponents.values())
        if total <= 0:
            uniform = 1.0 / len(scores) if scores else 0.0
            return {name: uniform for name in scores}
        return {name: value / total for name, value in exponents.items()}

    def train(
        self,
        encoded: Sequence[_Encoded],
        *,
        epochs: int,
        learning_rate: float,
        l2: float,
        optimizer: str,
        class_weight: str,
        beta1: float,
        beta2: float,
        epsilon: float,
    ) -> list[float]:
        if not encoded:
            raise RouterTrainingError("training split has no examples")
        weights = [1.0] * len(encoded)
        if class_weight == "balanced":
            counts: dict[str, int] = {}
            for item in encoded:
                counts[item.target] = counts.get(item.target, 0) + 1
            for index, item in enumerate(encoded):
                weights[index] = len(encoded) / (len(counts) * counts[item.target])
            mean_weight = sum(weights) / len(weights)
            if mean_weight > 0:
                weights = [value / mean_weight for value in weights]
        moments = {
            name: ([0.0] * self.total_dimension, [0.0] * self.total_dimension)
            for name in self.classes
        }
        history: list[float] = []
        for epoch in range(1, epochs + 1):
            gradients = {name: [0.0] * self.total_dimension for name in self.classes}
            loss = 0.0
            for index, item in enumerate(encoded):
                probabilities = self.probabilities(item.candidates, item.vectors)
                target = item.target
                loss -= weights[index] * math.log(max(probabilities.get(target, 0.0), 1e-12))
                for candidate, vector in zip(item.candidates, item.vectors, strict=True):
                    coefficient = weights[index] * (
                        probabilities[candidate] - (1.0 if candidate == target else 0.0)
                    )
                    if coefficient == 0.0:
                        continue
                    gradient = gradients[candidate]
                    for bucket, value in vector.items():
                        gradient[bucket] += coefficient * value
            for name in self.classes:
                gradient = gradients[name]
                current = self.weights[name]
                for bucket in range(self.total_dimension):
                    regularized = gradient[bucket] + l2 * current[bucket]
                    gradient[bucket] = regularized
                    loss += 0.5 * l2 * current[bucket] * current[bucket]
            loss /= len(encoded)
            self._step(gradients, moments, epoch, learning_rate, optimizer, beta1, beta2, epsilon)
            history.append(loss)
        return history

    def _step(
        self,
        gradients: dict[str, list[float]],
        moments: dict[str, tuple[list[float], list[float]]],
        step: int,
        learning_rate: float,
        optimizer: str,
        beta1: float,
        beta2: float,
        epsilon: float,
    ) -> None:
        correction1 = 1.0 - beta1**step
        correction2 = 1.0 - beta2**step
        for name in self.classes:
            current = self.weights[name]
            gradient = gradients[name]
            if optimizer == "gd":
                for bucket, value in enumerate(gradient):
                    current[bucket] -= learning_rate * value
                continue
            first, second = moments[name]
            for bucket, value in enumerate(gradient):
                first[bucket] = beta1 * first[bucket] + (1.0 - beta1) * value
                second[bucket] = beta2 * second[bucket] + (1.0 - beta2) * value * value
                m_hat = first[bucket] / correction1
                v_hat = second[bucket] / correction2
                current[bucket] -= learning_rate * m_hat / (math.sqrt(v_hat) + epsilon)


def train_router(
    dataset: RouterDataset,
    config: RouterTrainingConfig,
    *,
    command: str | None = None,
    write: bool = True,
) -> TrainingResult:
    """Train, calibrate, select an abstain threshold, and write the artifact/report."""
    _validate_splits(dataset, config)
    train_examples = dataset.split("train")
    validation_examples = dataset.split("validation")
    test_examples = dataset.split("test")
    classes = _ordered_classes(dataset)
    encoded = _encode(train_examples, classes, config)
    model = ConditionalLogit(classes, dimension=config.features.dimension, seed=config.seed)
    history = model.train(
        encoded,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        l2=config.training.l2,
        optimizer=config.training.optimizer,
        class_weight=config.training.class_weight,
        beta1=config.training.beta1,
        beta2=config.training.beta2,
        epsilon=config.training.epsilon,
    )
    calibration_rows = _raw_rows(model, validation_examples, config)
    calibration_rows = [
        row for row in calibration_rows if row["route"] is not None and not row["ambiguous"]
    ]
    fit = fit_temperature(calibration_rows, bins=config.calibration.bins)
    raw_validation = _raw_rows(model, validation_examples, config)
    raw_test = _raw_rows(model, test_examples, config)
    calibrated_validation = _calibrated_predictions(raw_validation, fit.temperature)
    calibrated_test = _calibrated_predictions(raw_test, fit.temperature)
    cost_matrix = config.cost_matrix()
    threshold_choice = select_abstain_threshold(
        calibrated_validation, cost_matrix, steps=config.abstain.steps
    )
    metrics: dict[str, Any] = {}
    reliability: dict[str, Any] = {}
    for name, raw in (("validation", raw_validation), ("test", raw_test)):
        calibrated = calibrated_validation if name == "validation" else calibrated_test
        metrics[name] = evaluate_predictions(
            calibrated, threshold_choice.threshold, cost_matrix, bins=config.calibration.bins
        )
        metrics[name]["uncalibrated"] = _uncalibrated_summary(raw, fit.temperature)
        reliability[name] = _reliability_report(
            raw,
            fit.temperature,
            config.calibration.bins,
            threshold_choice.threshold,
        )
    rules: dict[str, Any] = {}
    for name, examples in (
        ("validation", validation_examples),
        ("test", test_examples),
    ):
        evaluated = evaluate_rules(examples, config=config.rule_config(), cost_matrix=cost_matrix)
        rules[name] = evaluated["metrics"]
    incumbent = {
        "validation": evaluate_predictions(
            incumbent_predictions(validation_examples),
            0.0,
            cost_matrix,
            bins=config.calibration.bins,
        ),
        "test": evaluate_predictions(
            incumbent_predictions(test_examples),
            0.0,
            cost_matrix,
            bins=config.calibration.bins,
        ),
    }
    dataset_block = {
        "id": dataset.dataset_id,
        "version": dataset.dataset_version,
        "hash": dataset.dataset_manifest_hash or "unknown",
        "split_counts": dataset.counts,
        "group_count": sum(int(value) for value in dataset.summary.get("groups", {}).values()),
        "frozen_test": dataset.frozen_test,
    }
    artifact = build_artifact(
        classes=classes,
        weights=model.weights,
        dimension=config.features.dimension,
        bigrams=config.features.bigrams,
        seed=config.seed,
        calibration={
            "method": "temperature",
            "temperature": fit.temperature,
            "fit_split": "validation",
            "fit_count": fit.count,
            "ece": fit.ece,
            "nll": fit.nll,
            "bins": config.calibration.bins,
            "notes": fit.notes,
        },
        abstain={
            "threshold": threshold_choice.threshold,
            "fit_split": "validation",
            "cost_matrix": cost_matrix.to_dict(),
            "validation": threshold_choice.summary(),
            "grid_steps": config.abstain.steps,
            "notes": config.abstain.notes
            or "threshold minimizes validation cost; test is reported once",
        },
        training={
            "epochs": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "l2": config.training.l2,
            "optimizer": config.training.optimizer,
            "class_weight": config.training.class_weight,
            "train_count": len(encoded),
            "final_loss": history[-1] if history else 0.0,
            "loss_history": history,
        },
        dataset=dataset_block,
        provenance={
            "source": "mg train router",
            "command": command or "mg train router --config <config>",
            "config_hash": config.config_hash(),
            "code_version": __version__,
        },
        metrics=metrics,
    )
    artifact_model = RouterArtifact.from_dict(artifact)
    predictor = Predictor(artifact_model)
    report = _build_report(
        dataset=dataset,
        config=config,
        command=command,
        artifact=artifact,
        artifact_model=artifact_model,
        fit=fit,
        threshold=threshold_choice,
        metrics=metrics,
        reliability=reliability,
        rules=rules,
        incumbent=incumbent,
    )
    artifact_path = None
    report_path = None
    if write:
        artifact_path = artifact_model.save(_resolve(config.artifact))
        if config.report:
            from modelgraph.training.report import write_training_report

            report_path = write_training_report(report, _resolve(config.report))
    return TrainingResult(
        artifact=artifact,
        artifact_path=artifact_path,
        report=report,
        report_path=report_path,
        predictor=predictor,
        dataset=dataset,
        config=config,
    )


def _validate_splits(dataset: RouterDataset, config: RouterTrainingConfig) -> None:
    counts = dataset.counts
    missing = [name for name, count in counts.items() if count == 0]
    if config.evaluation.require_all_splits and missing:
        raise RouterTrainingError(
            "dataset is missing required split(s): "
            + ", ".join(missing)
            + "; export a frozen group split before training"
        )
    if config.evaluation.require_frozen_test and not dataset.frozen_test:
        raise RouterTrainingError(
            "dataset does not declare a frozen test split; refusing to train/test on it"
        )
    if not dataset.split("train"):
        raise RouterTrainingError("dataset has no train examples")
    if not dataset.split("validation"):
        raise RouterTrainingError(
            "dataset has no validation examples; calibration/abstention must be fitted on "
            "validation, not test"
        )


def _ordered_classes(dataset: RouterDataset) -> list[str]:
    # The class inventory comes only from the train split. Deriving it from validation or
    # test labels would leak held-out route names (and a weight slot for them) into the
    # artifact and would make a test-only class appear at scoring time.
    classes = sorted(
        {
            example.oracle_route
            for example in dataset.split("train")
            if example.oracle_route is not None
        }
    )
    if len(classes) < 2:
        raise RouterTrainingError("training needs at least two oracle route classes")
    return classes


def _encode(
    examples: Sequence[RouterExample],
    classes: Sequence[str],
    config: RouterTrainingConfig,
) -> list[_Encoded]:
    encoded: list[_Encoded] = []
    for example in sorted(examples, key=lambda item: item.example_id):
        if example.oracle_route is None:
            continue
        if example.ambiguous and not config.training.include_ambiguous:
            continue
        view = _view(example)
        candidates = [route.id for route in view.routes if route.id in classes]
        target = example.oracle_route
        if target not in candidates:
            raise RouterTrainingError(
                f"example {example.example_id} target {target!r} is not in its "
                f"candidate set {candidates}"
            )
        vectors = [
            featurize(
                view,
                candidate,
                dimension=config.features.dimension,
                bigrams=config.features.bigrams,
                candidate_index=index,
                candidate_count=len(candidates),
            )
            for index, candidate in enumerate(candidates)
        ]
        encoded.append(
            _Encoded(
                example=example,
                candidates=candidates,
                vectors=vectors,
                target_index=candidates.index(target),
                target=target,
            )
        )
    return encoded


def _view(example: RouterExample) -> RouterView:
    return RouterView(
        request=example.request,
        routes=tuple(example.menu_routes),
        menu_text="",
        messages=example.messages,
        prompt_hash=example.prompt_hash,
    )


def _raw_rows(
    model: ConditionalLogit,
    examples: Sequence[RouterExample],
    config: RouterTrainingConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in sorted(examples, key=lambda item: item.example_id):
        view = _view(example)
        candidates = [route.id for route in view.routes if route.id in model.classes]
        vectors = [
            featurize(
                view,
                candidate,
                dimension=config.features.dimension,
                bigrams=config.features.bigrams,
                candidate_index=index,
                candidate_count=len(candidates),
            )
            for index, candidate in enumerate(candidates)
        ]
        probabilities = model.probabilities(candidates, vectors)
        route = max(probabilities, key=lambda name: (probabilities[name], name))
        correct = bool(
            not example.ambiguous
            and example.oracle_route is not None
            and route == example.oracle_route
        )
        rows.append(
            {
                "example_id": example.example_id,
                "probabilities": probabilities,
                "route": route,
                "confidence": probabilities.get(route, 0.0),
                "correct": correct,
                "correct_route": correct,
                "ambiguous": example.ambiguous,
                "oracle_route": example.oracle_route,
                "group": example.group,
            }
        )
    return rows


def _calibrated_predictions(
    rows: Sequence[dict[str, Any]], temperature: float
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        calibrated = apply_temperature(row["probabilities"], temperature)
        route = max(calibrated, key=lambda name: (calibrated[name], name))
        predictions.append(
            {
                "example_id": row["example_id"],
                "route": route,
                "confidence": calibrated.get(route, 0.0),
                "correct_route": bool(row["correct_route"]),
                "ambiguous": bool(row["ambiguous"]),
                "oracle_route": row["oracle_route"],
                "group": row["group"],
            }
        )
    return predictions


def _uncalibrated_summary(rows: Sequence[dict[str, Any]], temperature: float) -> dict[str, Any]:
    routed = [row for row in rows if row["route"] is not None and not row["ambiguous"]]
    confidences = [float(row["confidence"]) for row in routed]
    correctness = [bool(row["correct_route"]) for row in routed]
    table = reliability_table(confidences, correctness)
    return {
        "route_accuracy": (
            sum(1 for row in routed if row["correct_route"]) / len(routed) if routed else None
        ),
        "ece": table["ece"],
        "n": table["n"],
        "temperature": 1.0,
        "calibrated_temperature": temperature,
    }


def _reliability_report(
    rows: Sequence[dict[str, Any]],
    temperature: float,
    bins: int,
    threshold: float,
) -> dict[str, Any]:
    """Reliability of the routed, non-ambiguous, threshold-accepted predictions.

    The calibration metric in ``evaluate_predictions`` uses exactly this population. The
    table must use it too, otherwise the report shows two different ECE values for the
    same split (one before and one after threshold abstention). The "before" table uses
    the same selected cases with their uncalibrated confidence so the two rows compare
    like with like.
    """

    raw_by_id = {str(row["example_id"]): row for row in rows}
    calibrated = _calibrated_predictions(rows, temperature)
    selected = [
        row
        for row in calibrated
        if row["route"] is not None
        and not row["ambiguous"]
        and float(row["confidence"]) >= threshold
    ]
    before_confidences: list[float] = []
    before_correctness: list[bool] = []
    for row in selected:
        raw = raw_by_id[str(row["example_id"])]
        before_confidences.append(float(raw["confidence"]))
        before_correctness.append(bool(raw["correct_route"]))
    after = reliability_table(
        [float(row["confidence"]) for row in selected],
        [bool(row["correct_route"]) for row in selected],
        bins=bins,
    )
    return {
        "temperature": temperature,
        "threshold": threshold,
        "before": reliability_table(before_confidences, before_correctness, bins=bins),
        "after": after,
    }


def _build_report(
    *,
    dataset: RouterDataset,
    config: RouterTrainingConfig,
    command: str | None,
    artifact: dict[str, Any],
    artifact_model: RouterArtifact,
    fit: Any,
    threshold: ThresholdChoice,
    metrics: dict[str, Any],
    reliability: dict[str, Any],
    rules: dict[str, Any],
    incumbent: dict[str, Any],
) -> dict[str, Any]:
    limitations: list[str] = []
    if not dataset.frozen_test:
        limitations.append("The dataset does not declare a frozen test split.")
    if dataset.summary.get("excluded", {}).get("unlabeled"):
        limitations.append(
            f"{dataset.summary['excluded']['unlabeled']} routing example(s) had no positive "
            "label and were excluded."
        )
    if dataset.dataset_id == "unknown":
        limitations.append("No T-0023 dataset manifest was present; provenance is file-level.")
    if len(dataset.examples) < 100:
        limitations.append(
            f"Only {len(dataset.examples)} labeled examples; calibration and threshold "
            "estimates are high-variance."
        )
    contamination = (
        (dataset.manifest or {}).get("spec", {}).get("eval_contamination")
        if dataset.manifest
        else None
    )
    if not contamination or contamination.get("status") != "checked":
        limitations.append("Eval contamination was not checked for this dataset.")
    if "single-class" in fit.notes:
        limitations.append(
            "Validation route predictions were single-class, so temperature scaling correctly "
            "stayed at 1.0; this run does not exercise a non-identity temperature. The unit "
            "tests cover fitting on overconfident rows."
        )
    limitations.append(
        "No end-to-end quality delta was measured here: e2e needs live worker providers and "
        "a real suite; the report compares routing only."
    )
    return {
        "schema": "modelgraph/router-training-report/v0",
        "artifact": {
            "path": config.artifact,
            "schema": artifact["schema"],
            "version": artifact["artifact_version"],
            "sha256": artifact_model.sha256(),
            "classes": artifact["classes"],
        },
        "command": command,
        "config": config.to_dict(),
        "seed": config.seed,
        "dataset": {
            "summary": dataset.summary,
            "manifest_hash": dataset.dataset_manifest_hash,
            "counts": dataset.counts,
            "classes": artifact["classes"],
        },
        "calibration": {
            "method": "temperature",
            "fit_split": "validation",
            "fit_count": fit.count,
            "temperature": fit.temperature,
            "notes": fit.notes,
        },
        "abstain": {
            "threshold": threshold.threshold,
            "cost_matrix": config.cost_matrix().to_dict(),
            "validation": threshold.summary(),
            "grid_steps": config.abstain.steps,
        },
        "metrics": {
            "trained": metrics,
            "rules": rules,
            "incumbent": incumbent,
        },
        "reliability": reliability,
        "limitations": limitations,
        "recommendation": _recommendation(metrics, rules, incumbent),
    }


def _recommendation(
    metrics: dict[str, Any], rules: dict[str, Any], incumbent: dict[str, Any]
) -> dict[str, Any]:
    trained_test = metrics.get("test", {})
    rules_test = rules.get("test", {})
    incumbent_test = incumbent.get("test", {})
    trained_agreement = trained_test.get("oracle_agreement")
    trained_cost = trained_test.get("expected_cost_per_case")
    comparators = [
        value.get("oracle_agreement")
        for value in (rules_test, incumbent_test)
        if isinstance(value.get("oracle_agreement"), (int, float))
    ]
    comparator_costs = [
        value.get("expected_cost_per_case")
        for value in (rules_test, incumbent_test)
        if isinstance(value.get("expected_cost_per_case"), (int, float))
    ]
    best_agreement = max(comparators, default=None)
    best_cost = min(comparator_costs, default=None)
    agreement_improved = (
        trained_agreement is not None
        and best_agreement is not None
        and trained_agreement > best_agreement
    )
    cost_improved = trained_cost is not None and best_cost is not None and trained_cost < best_cost
    if agreement_improved and cost_improved:
        reason = (
            "On this synthetic frozen test split the trained router improves both oracle "
            "agreement and cost per case against the scripted comparators. Promotion is still "
            "not claimed: the incumbent is a stand-in and end-to-end quality was not measured."
        )
    elif cost_improved:
        reason = (
            "On this synthetic frozen test split the trained router lowers cost per case "
            f"({trained_cost:.3f} vs best comparator {best_cost:.3f}) mainly by abstaining on "
            "ambiguous requests, but oracle agreement does not improve "
            f"({trained_agreement!r} vs {best_agreement!r}). The lower cost is real for this "
            "cost matrix and fixture, not evidence of a real e2e quality win; the incumbent is "
            "a scripted stand-in and e2e was not measured."
        )
    else:
        reason = (
            "On this synthetic frozen test split the trained router does not improve oracle "
            "agreement or cost per case over the scripted comparators. Promotion is not "
            "claimed; the incumbent is a stand-in and e2e was not measured."
        )
    return {
        "promote": False,
        "reason": reason,
        "trained_test_oracle_agreement": trained_agreement,
        "best_comparator_test_oracle_agreement": best_agreement,
        "trained_test_cost_per_case": trained_cost,
        "best_comparator_test_cost_per_case": best_cost,
        "agreement_improved_on_this_test_split": agreement_improved,
        "cost_improved_on_this_test_split": cost_improved,
    }


def _resolve(value: str) -> Path:
    return Path(value).expanduser()
