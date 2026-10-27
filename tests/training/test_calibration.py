"""Calibration, ECE, reliability, and cost-sensitive abstention."""

from __future__ import annotations

import math

import pytest

from modelgraph.training.calibration import (
    CostMatrix,
    apply_temperature,
    ece,
    evaluate_predictions,
    fit_temperature,
    nll,
    reliability_table,
    select_abstain_threshold,
)
from modelgraph.training.errors import RouterTrainingError


def test_reliability_table_and_ece_are_exact_for_known_bins() -> None:
    table = reliability_table([0.1, 0.9, 0.9], [False, True, True], bins=10)
    assert table["n"] == 3
    assert table["bins"][1]["count"] == 1
    assert table["bins"][1]["confidence"] == pytest.approx(0.1)
    assert table["bins"][1]["accuracy"] == pytest.approx(0.0)
    assert table["bins"][9]["count"] == 2
    assert table["bins"][9]["confidence"] == pytest.approx(0.9)
    assert table["bins"][9]["accuracy"] == pytest.approx(1.0)
    assert table["ece"] == pytest.approx((1 / 3) * 0.1 + (2 / 3) * 0.1)
    assert table["max_calibration_error"] == pytest.approx(0.1)
    assert ece([0.1, 0.9, 0.9], [False, True, True]) == pytest.approx(table["ece"])


def test_apply_temperature_normalizes() -> None:
    calibrated = apply_temperature({"a": 0.8, "b": 0.2}, 2.0)
    assert sum(calibrated.values()) == pytest.approx(1.0)
    assert calibrated["a"] > calibrated["b"]
    assert calibrated["a"] < 0.8


def test_temperature_is_fit_to_validation_rows() -> None:
    # Overconfident and wrong: a temperature above 1 should reduce NLL.
    rows = [
        {"probabilities": {"coding": 0.95, "story": 0.05}, "route": "coding", "correct": False},
        {"probabilities": {"coding": 0.9, "story": 0.1}, "route": "coding", "correct": False},
        {"probabilities": {"coding": 0.1, "story": 0.9}, "route": "story", "correct": True},
    ]
    fit = fit_temperature(rows)
    assert fit.count == 3
    assert fit.temperature > 1.0
    assert fit.nll is not None
    assert fit.ece is not None
    assert "validation" in fit.notes


def test_single_class_validation_leaves_temperature_at_one() -> None:
    rows = [
        {"probabilities": {"coding": 0.9, "story": 0.1}, "route": "coding", "correct": True},
        {"probabilities": {"coding": 0.8, "story": 0.2}, "route": "coding", "correct": True},
    ]
    fit = fit_temperature(rows)
    assert fit.temperature == 1.0
    assert "single-class" in fit.notes


def test_cost_matrix_rejects_negative_values() -> None:
    with pytest.raises(RouterTrainingError, match="non-negative"):
        CostMatrix(misroute=-1.0)


def test_abstain_threshold_minimizes_validation_cost() -> None:
    predictions = [
        {
            "route": "coding",
            "confidence": 0.9,
            "correct_route": True,
            "ambiguous": False,
            "oracle_route": "coding",
        },
        {
            "route": "story",
            "confidence": 0.9,
            "correct_route": True,
            "ambiguous": False,
            "oracle_route": "story",
        },
        {
            "route": "coding",
            "confidence": 0.3,
            "correct_route": False,
            "ambiguous": True,
            "oracle_route": "coding",
        },
    ]
    choice = select_abstain_threshold(predictions, CostMatrix(), steps=100)
    # Routing the ambiguous case costs 0.6; abstaining on it costs 0. The chosen threshold is
    # therefore above 0.3 and at or below 0.9.
    assert 0.3 < choice.threshold <= 0.9
    assert choice.cost_per_case == pytest.approx(0.0)
    assert choice.coverage == pytest.approx(2 / 3)
    metrics = evaluate_predictions(predictions, choice.threshold, CostMatrix())
    assert metrics["abstain_recall"] == pytest.approx(1.0)
    assert metrics["abstain_precision"] == pytest.approx(1.0)
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["oracle_agreement"] == pytest.approx(1.0)


def test_unnecessary_abstention_cost_changes_the_threshold() -> None:
    predictions = [
        {
            "route": "coding",
            "confidence": 0.6,
            "correct_route": False,
            "ambiguous": False,
            "oracle_route": "story",
        }
    ]
    cheap = select_abstain_threshold(predictions, CostMatrix(unnecessary_abstain=0.1))
    expensive = select_abstain_threshold(predictions, CostMatrix(unnecessary_abstain=2.0))
    # With cheap abstention the low-confidence misroute is rejected; with expensive
    # abstention routing is cheaper. The threshold therefore moves down, not up.
    assert cheap.threshold > 0.6
    assert expensive.threshold <= 0.6
    assert cheap.cost_per_case < expensive.cost_per_case
    assert not math.isnan(expensive.cost_per_case)


def test_nll_divides_by_valid_rows_only() -> None:
    valid = {
        "probabilities": {"coding": 0.9, "story": 0.1},
        "route": "coding",
        "correct": True,
    }
    invalid = {"probabilities": {}, "route": None, "correct": True}
    assert nll([valid, invalid], 1.0) == pytest.approx(nll([valid], 1.0))
    assert nll([invalid], 1.0) is None
