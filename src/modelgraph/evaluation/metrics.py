"""Metric primitives shared by node and routing evaluation.

Everything here is deterministic for a fixed input and seed. Proportions use Wilson
intervals; means and latency percentiles use a seeded percentile bootstrap. ECE and
threshold sweeps are descriptive for the evaluated partition; fitting a calibration map is
deliberately out of scope for v0.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

LOWER_IS_BETTER = frozenset(
    {
        "cost_total_usd",
        "cost_per_task_usd",
        "latency_mean_seconds",
        "latency_p50_seconds",
        "latency_p95_seconds",
        "protocol_error_rate",
        "protocol_errors_per_100_calls",
        "ece",
        "abstain_rate",
        "max_calibration_error",
        "oracle_gap",
        "hop_count_mean",
        "hop_count_p95",
    }
)


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    total = sum(weights)
    if total <= 0 or len(values) != len(weights):
        return None
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Linear-interpolated percentile in ``[0, 1]`` (``0.95`` for p95)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(fraction, 0.0), 1.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> list[float] | None:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float | None] | None = None,
    *,
    resamples: int = 2000,
    seed: int = 1234,
    confidence: float = 0.95,
) -> list[float] | None:
    """Seeded percentile bootstrap CI for a scalar statistic."""
    if not values:
        return None
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    estimator = statistic or mean_or_none
    rng = random.Random(seed)
    n = len(values)
    estimates: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimate = estimator(sample)
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return None
    alpha = (1 - confidence) / 2
    low = percentile(estimates, alpha)
    high = percentile(estimates, 1 - alpha)
    if low is None or high is None:  # pragma: no cover - estimates is non-empty
        return None
    return [low, high]


def latency_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean_seconds": None,
            "p50_seconds": None,
            "p95_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "count": len(values),
        "mean_seconds": mean_or_none(values),
        "p50_seconds": percentile(values, 0.5),
        "p95_seconds": percentile(values, 0.95),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def reliability_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Equal-width reliability bins, expected calibration error, and maximum bin gap."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correctness labels must have the same length")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    total = len(confidences)
    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        members = [
            (confidence, label)
            for confidence, label in zip(confidences, correct, strict=True)
            if (lower <= confidence < upper) or (index == n_bins - 1 and confidence == 1.0)
        ]
        count = len(members)
        mean_confidence = mean_or_none([item[0] for item in members])
        accuracy = mean_or_none([1.0 if item[1] else 0.0 for item in members])
        gap = (
            abs(accuracy - mean_confidence)
            if accuracy is not None and mean_confidence is not None
            else None
        )
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    ece = 0.0 if total else None
    if ece is not None:
        for entry in bins:
            if entry["gap"] is not None:
                ece += entry["count"] / total * entry["gap"]
    gaps = [entry["gap"] for entry in bins if entry["gap"] is not None]
    return {
        "n": total,
        "bins": bins,
        "ece": ece,
        "max_calibration_error": max(gaps) if gaps else None,
    }


def threshold_sweep(
    records: Sequence[dict[str, Any]],
    *,
    steps: int = 20,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Coverage/selective-accuracy curve for confidence thresholds.

    ``records`` entries need ``action`` (``route`` or ``abstain``/``error``), ``confidence``
    (number or ``None``), and ``correct`` (bool). Lowering the threshold raises coverage.
    """
    if thresholds is None:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        thresholds = [index / steps for index in range(steps + 1)]
    total = len(records)
    sweep: list[dict[str, Any]] = []
    for threshold in thresholds:
        covered = [
            record
            for record in records
            if record.get("action") == "route"
            and isinstance(record.get("confidence"), (int, float))
            and float(record["confidence"]) >= threshold
        ]
        correct = sum(1 for record in covered if record.get("correct") is True)
        sweep.append(
            {
                "threshold": round(float(threshold), 6),
                "n_covered": len(covered),
                "n_correct": correct,
                "coverage": safe_div(len(covered), total),
                "selective_accuracy": safe_div(correct, len(covered)),
                "abstain_rate": 1 - (len(covered) / total) if total else None,
            }
        )
    return sweep


def confusion_matrix(
    oracle_labels: Sequence[str],
    predicted_labels: Sequence[str],
    *,
    extra_labels: Sequence[str] = ("abstain", "error"),
) -> dict[str, Any]:
    """Row-normalized-free confusion matrix over child ids plus abstain/error labels."""
    if len(oracle_labels) != len(predicted_labels):
        raise ValueError("oracle and predicted labels must have the same length")
    children = sorted((set(oracle_labels) | set(predicted_labels)) - set(extra_labels))
    labels = children + [label for label in extra_labels]
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for oracle, predicted in zip(oracle_labels, predicted_labels, strict=True):
        if oracle not in index:
            labels.append(oracle)
            index[oracle] = len(labels) - 1
            for matrix_row in matrix:
                matrix_row.append(0)
            matrix.append([0 for _ in labels])
        if predicted not in index:
            labels.append(predicted)
            index[predicted] = len(labels) - 1
            for matrix_row in matrix:
                matrix_row.append(0)
            matrix.append([0 for _ in labels])
        matrix[index[oracle]][index[predicted]] += 1
    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        row_index = index[label]
        support = sum(matrix[row_index])
        predicted_count = sum(matrix[column][row_index] for column in range(len(labels)))
        true_positive = matrix[row_index][row_index]
        precision = safe_div(true_positive, predicted_count)
        recall = safe_div(true_positive, support)
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "support": support,
            "predicted": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    total = sum(sum(row) for row in matrix)
    diagonal = sum(matrix[index[label]][index[label]] for label in labels)
    return {
        "labels": labels,
        "matrix": matrix,
        "total": total,
        "accuracy": safe_div(diagonal, total),
        "per_class": per_class,
    }


def abstain_metrics(
    predicted_abstain: Sequence[bool],
    genuinely_ambiguous: Sequence[bool],
) -> dict[str, Any]:
    if len(predicted_abstain) != len(genuinely_ambiguous):
        raise ValueError("predicted and ambiguous labels must have the same length")
    true_positive = sum(
        1
        for predicted, actual in zip(predicted_abstain, genuinely_ambiguous, strict=True)
        if predicted and actual
    )
    false_positive = sum(
        1
        for predicted, actual in zip(predicted_abstain, genuinely_ambiguous, strict=True)
        if predicted and not actual
    )
    false_negative = sum(
        1
        for predicted, actual in zip(predicted_abstain, genuinely_ambiguous, strict=True)
        if not predicted and actual
    )
    true_negative = sum(
        1
        for predicted, actual in zip(predicted_abstain, genuinely_ambiguous, strict=True)
        if not predicted and not actual
    )
    precision = safe_div(true_positive, true_positive + false_positive)
    recall = safe_div(true_positive, true_positive + false_negative)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    total = len(predicted_abstain)
    return {
        "n": total,
        "ambiguous": true_positive + false_negative,
        "abstained": true_positive + false_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "abstain_rate": safe_div(true_positive + false_positive, total),
    }


def metric_direction(name: str) -> str:
    """Return ``min`` for lower-is-better metrics, otherwise ``max``."""
    return "min" if name in LOWER_IS_BETTER else "max"


def threshold_result(name: str, threshold: float, value: float | None) -> dict[str, Any]:
    direction = metric_direction(name)
    passed: bool | None
    if value is None:
        passed = None
    elif direction == "min":
        passed = value <= threshold
    else:
        passed = value >= threshold
    return {
        "metric": name,
        "threshold": threshold,
        "value": value,
        "direction": direction,
        "passed": passed,
    }
