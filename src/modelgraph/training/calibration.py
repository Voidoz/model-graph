"""Temperature scaling, reliability reporting, and cost-sensitive abstention.

Calibration is fitted on the validation split only. The test split is used once, for the
reported numbers. The abstain threshold is selected on validation predictions using an explicit
cost matrix; the test set never selects a threshold.

The primitives are pure Python and deterministic. A temperature of ``1.0`` means no scaling; a
grid search over a fixed log-spaced grid chooses the value that minimizes validation NLL of the
chosen route's correctness.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from modelgraph.training.errors import RouterTrainingError

DEFAULT_TEMPERATURE_GRID = tuple(0.05 * (400.0 ** (index / 160.0)) for index in range(0, 161))
DEFAULT_BINS = 10


@dataclass(frozen=True)
class CostMatrix:
    """Documented costs, in the same units for every outcome.

    ``unnecessary_abstain`` is the cost of abstaining on a clear request. Set it high for a
    graph with no fallback (abstention becomes a run failure) and lower when a fallback exists.
    """

    correct_route: float = 0.0
    misroute: float = 1.0
    unnecessary_abstain: float = 0.35
    misroute_ambiguous: float = 0.6
    correct_abstain: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RouterTrainingError(f"cost matrix value {name!r} must be a finite number")
            if float(value) < 0:
                raise RouterTrainingError(f"cost matrix value {name!r} must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "correct_route": float(self.correct_route),
            "misroute": float(self.misroute),
            "unnecessary_abstain": float(self.unnecessary_abstain),
            "misroute_ambiguous": float(self.misroute_ambiguous),
            "correct_abstain": float(self.correct_abstain),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CostMatrix:
        known = set(cls.__dataclass_fields__)
        unknown = set(value) - known
        if unknown:
            raise RouterTrainingError(f"unknown cost matrix keys: {sorted(unknown)}")
        return cls(**{key: float(item) for key, item in value.items()})


@dataclass(frozen=True)
class CalibrationFit:
    temperature: float
    nll: float | None
    ece: float | None
    count: int
    notes: str = ""


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    cost_per_case: float
    coverage: float
    abstain_rate: float
    oracle_agreement: float | None
    curve: tuple[dict[str, float], ...] = field(default_factory=tuple)

    def summary(self) -> dict[str, Any]:
        return {
            "cost_per_case": self.cost_per_case,
            "coverage": self.coverage,
            "abstain_rate": self.abstain_rate,
            "oracle_agreement": self.oracle_agreement,
        }


def apply_temperature(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not probabilities:
        return {}
    powered = {
        name: (max(0.0, float(value)) ** (1.0 / temperature))
        for name, value in probabilities.items()
    }
    total = sum(powered.values())
    if total <= 0 or not math.isfinite(total):
        uniform = 1.0 / len(probabilities)
        return {name: uniform for name in probabilities}
    return {name: value / total for name, value in powered.items()}


def fit_temperature(
    rows: Sequence[dict[str, Any]],
    *,
    bins: int = DEFAULT_BINS,
    grid: Sequence[float] = DEFAULT_TEMPERATURE_GRID,
) -> CalibrationFit:
    """Fit one temperature on ``rows`` of ``{probabilities, route, correct}``.

    ``route`` is the raw argmax route. Rows where the model's argmax is the abstain class are
    not route-confidence observations and should be filtered by the caller.
    """
    usable = [
        row
        for row in rows
        if isinstance(row.get("probabilities"), dict) and row.get("route") is not None
    ]
    if not usable:
        return CalibrationFit(1.0, None, None, 0, "no validation route predictions")
    observed = {bool(row.get("correct")) for row in usable}
    if len(observed) < 2:
        return CalibrationFit(
            1.0,
            nll(usable, 1.0),
            reliability_table(_confidences(usable, 1.0), _correct(usable), bins=bins)["ece"],
            len(usable),
            "validation correctness is single-class; temperature fixed at 1.0",
        )
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in grid:
        if temperature <= 0:
            continue
        value = nll(usable, temperature)
        if value is None:
            continue
        if value < best_nll - 1e-12:
            best_nll = value
            best_temperature = temperature
    # Deterministic tie-break: prefer the temperature closest to 1.0 on equal NLL.
    tied = [
        temperature
        for temperature in grid
        if temperature > 0 and abs((nll(usable, temperature) or float("inf")) - best_nll) <= 1e-12
    ]
    if tied:
        best_temperature = min(tied, key=lambda item: (abs(item - 1.0), item))
    confidences = _confidences(usable, best_temperature)
    table = reliability_table(confidences, _correct(usable), bins=bins)
    return CalibrationFit(
        temperature=best_temperature,
        nll=best_nll if math.isfinite(best_nll) else None,
        ece=table["ece"],
        count=len(usable),
        notes="temperature fit on validation route predictions",
    )


def nll(
    rows: Sequence[dict[str, Any]], temperature: float, *, epsilon: float = 1e-12
) -> float | None:
    """Binary NLL of chosen-route confidence after temperature scaling."""
    if not rows:
        return None
    total = 0.0
    count = 0
    for row in rows:
        probabilities = row.get("probabilities")
        route_value = row.get("route")
        if (
            not isinstance(probabilities, dict)
            or not isinstance(route_value, str)
            or route_value not in probabilities
        ):
            continue
        route = route_value
        calibrated = apply_temperature(probabilities, temperature)
        confidence = min(max(calibrated.get(route, 0.0), epsilon), 1.0 - epsilon)
        correct = bool(row.get("correct"))
        total += -math.log(confidence if correct else 1.0 - confidence)
        count += 1
    return total / count if count else None


def brier(confidences: Sequence[float], correct: Sequence[bool]) -> float | None:
    if not confidences:
        return None
    return sum(
        (float(confidence) - (1.0 if truth else 0.0)) ** 2
        for confidence, truth in zip(confidences, correct, strict=True)
    ) / len(confidences)


def reliability_table(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Equal-width reliability bins plus ECE and the maximum bin gap."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correctness labels must have the same length")
    if bins < 1:
        raise ValueError("bins must be at least 1")
    entries: list[dict[str, Any]] = []
    total = len(confidences)
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            (float(confidence), bool(truth))
            for confidence, truth in zip(confidences, correct, strict=True)
            if (lower <= confidence < upper) or (index == bins - 1 and confidence == 1.0)
        ]
        count = len(members)
        mean_confidence = _mean([item[0] for item in members])
        accuracy = _mean([1.0 if item[1] else 0.0 for item in members])
        gap = (
            abs(accuracy - mean_confidence)
            if accuracy is not None and mean_confidence is not None
            else None
        )
        entries.append(
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
        for entry in entries:
            if entry["gap"] is not None:
                ece += entry["count"] / total * entry["gap"]
    gaps = [entry["gap"] for entry in entries if entry["gap"] is not None]
    return {
        "n": total,
        "bins": entries,
        "ece": ece,
        "max_calibration_error": max(gaps) if gaps else None,
    }


def ece(
    confidences: Sequence[float], correct: Sequence[bool], *, bins: int = DEFAULT_BINS
) -> float | None:
    value = reliability_table(confidences, correct, bins=bins).get("ece")
    return float(value) if isinstance(value, (int, float)) else None


def select_abstain_threshold(
    predictions: Sequence[dict[str, Any]],
    cost_matrix: CostMatrix,
    *,
    steps: int = 100,
) -> ThresholdChoice:
    """Choose a confidence threshold on validation predictions only.

    Each prediction must carry ``route`` (raw argmax route or ``None`` when the model's argmax
    is abstain), ``confidence`` (calibrated confidence of that route), ``correct_route``,
    ``ambiguous`` and ``oracle_route``.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not predictions:
        raise RouterTrainingError("cannot select an abstain threshold from zero predictions")
    curve: list[dict[str, float]] = []
    for index in range(steps + 1):
        threshold = index / steps
        metrics = evaluate_predictions(predictions, threshold, cost_matrix)
        curve.append(
            {
                "threshold": threshold,
                "cost_per_case": float(metrics["expected_cost_per_case"]),
                "coverage": float(metrics["coverage"]),
                "abstain_rate": float(metrics["abstain_rate"]),
                "oracle_agreement": float(metrics["oracle_agreement"] or 0.0),
            }
        )
    selected = min(
        curve,
        key=lambda row: (row["cost_per_case"], -row["coverage"], row["threshold"]),
    )
    return ThresholdChoice(
        threshold=float(selected["threshold"]),
        cost_per_case=float(selected["cost_per_case"]),
        coverage=float(selected["coverage"]),
        abstain_rate=float(selected["abstain_rate"]),
        oracle_agreement=float(selected["oracle_agreement"]),
        curve=tuple(curve),
    )


def evaluate_predictions(
    predictions: Sequence[dict[str, Any]],
    threshold: float,
    cost_matrix: CostMatrix,
    *,
    bins: int = DEFAULT_BINS,
) -> dict[str, Any]:
    """Metrics for one threshold: agreement, abstention, calibration, confusion, cost."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    decisions: list[dict[str, Any]] = []
    total_cost = 0.0
    for prediction in predictions:
        route = prediction.get("route")
        confidence = float(prediction.get("confidence") or 0.0)
        ambiguous = bool(prediction.get("ambiguous"))
        correct_route = bool(prediction.get("correct_route"))
        abstain = route is None or confidence < threshold
        if ambiguous:
            if abstain:
                cost = cost_matrix.correct_abstain
                correct = True
            else:
                cost = cost_matrix.misroute_ambiguous
                correct = False
        else:
            if abstain:
                cost = cost_matrix.unnecessary_abstain
                correct = False
            elif correct_route:
                cost = cost_matrix.correct_route
                correct = True
            else:
                cost = cost_matrix.misroute
                correct = False
        total_cost += cost
        decisions.append(
            {
                "prediction": prediction,
                "abstain": abstain,
                "chosen": "abstain" if abstain else route,
                "cost": cost,
                "correct": correct,
                "confidence": confidence,
            }
        )
    n = len(decisions)
    labeled = [row for row in decisions if not row["prediction"].get("ambiguous")]
    ambiguous_rows = [row for row in decisions if row["prediction"].get("ambiguous")]
    routed = [row for row in decisions if not row["abstain"]]
    abstained = [row for row in decisions if row["abstain"]]
    correct_labeled = sum(1 for row in labeled if row["correct"])
    abstain_on_ambiguous = sum(1 for row in ambiguous_rows if row["abstain"])
    routed_labeled = [row for row in routed if not row["prediction"].get("ambiguous")]
    confidences = [row["confidence"] for row in routed_labeled]
    correctness = [bool(row["prediction"].get("correct_route")) for row in routed_labeled]
    table = reliability_table(confidences, correctness, bins=bins)
    confusion: dict[str, dict[str, int]] = {}
    for row in decisions:
        oracle = str(row["prediction"].get("oracle_route") or "abstain")
        chosen = str(row["chosen"])
        confusion.setdefault(oracle, {})
        confusion[oracle][chosen] = confusion[oracle].get(chosen, 0) + 1
    return {
        "cases": n,
        "labeled_cases": len(labeled),
        "ambiguous_cases": len(ambiguous_rows),
        "routed_cases": len(routed),
        "abstained_cases": len(abstained),
        "coverage": (len(routed) / n) if n else None,
        "abstain_rate": (len(abstained) / n) if n else None,
        "oracle_agreement": (correct_labeled / len(labeled)) if labeled else None,
        "selective_accuracy": (
            sum(1 for row in routed_labeled if row["prediction"].get("correct_route"))
            / len(routed_labeled)
            if routed_labeled
            else None
        ),
        "abstain_precision": (abstain_on_ambiguous / len(abstained) if abstained else None),
        "abstain_recall": (abstain_on_ambiguous / len(ambiguous_rows) if ambiguous_rows else None),
        "expected_cost_per_case": total_cost / n if n else None,
        "total_cost": total_cost,
        "ece": table["ece"],
        "max_calibration_error": table["max_calibration_error"],
        "calibration_cases": table["n"],
        "brier": brier(confidences, correctness),
        "confusion": confusion,
        "threshold": threshold,
    }


def incumbent_predictions(examples: Sequence[Any]) -> list[dict[str, Any]]:
    """Turn recorded routing decisions in a dataset into comparison predictions."""
    predictions: list[dict[str, Any]] = []
    for example in examples:
        output = example.output if hasattr(example, "output") else {}
        chosen = output.get("chosen_route")
        decision = output.get("chosen_decision")
        confidence = output.get("confidence")
        ambiguous = bool(example.ambiguous)
        oracle_route = example.oracle_route
        route = chosen if decision == "route" and isinstance(chosen, str) else None
        correct_route = bool(route is not None and route == oracle_route)
        predictions.append(
            {
                "route": route,
                "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0.0,
                "correct_route": correct_route,
                "ambiguous": ambiguous,
                "oracle_route": oracle_route,
            }
        )
    return predictions


def _confidences(rows: Sequence[dict[str, Any]], temperature: float) -> list[float]:
    values: list[float] = []
    for row in rows:
        calibrated = apply_temperature(row["probabilities"], temperature)
        values.append(float(calibrated.get(row["route"], 0.0)))
    return values


def _correct(rows: Sequence[dict[str, Any]]) -> list[bool]:
    return [bool(row.get("correct")) for row in rows]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None
