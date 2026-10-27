"""Inference for a serialized router.

The predictor is intentionally tiny and dependency-free. It scores every menu candidate (plus
the learned abstain candidate, when present), applies temperature scaling, and applies the
artifact's abstain threshold. The output method returns the canonical decision JSON the runtime
parser accepts; no runtime code is forked.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelgraph.training.artifact import RouterArtifact, load_router_artifact
from modelgraph.training.features import (
    ABSTAIN_CLASS,
    RouterView,
    featurize,
    hash_dimension,
)


@dataclass(frozen=True)
class RouterDecision:
    """One parsed-trained-router decision, including diagnostics for reports."""

    action: str
    route: str | None
    confidence: float
    reason: str
    probabilities: dict[str, float]
    threshold: float
    raw_confidence: float

    def to_json(self) -> dict[str, Any]:
        if self.action == "route":
            return {
                "decision": "route",
                "to": self.route,
                "confidence": round(self.confidence, 6),
                "reason": self.reason,
            }
        return {
            "decision": "abstain",
            "confidence": round(self.confidence, 6),
            "reason": self.reason,
        }

    def text(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False)


class Predictor:
    """Score menu candidates and emit a canonical router decision."""

    def __init__(self, artifact: RouterArtifact):
        self.artifact = artifact
        self.classes = artifact.classes
        self.weights = artifact.weights
        self.total_dimension = artifact.feature_dimension
        self.dimension = hash_dimension(self.total_dimension)
        self.temperature = artifact.temperature
        self.threshold = artifact.abstain_threshold
        self.bigrams = artifact.bigrams

    @classmethod
    def load(cls, path: str | Path, *, verify_sha256: str | None = None) -> Predictor:
        return cls(load_router_artifact(path, verify_sha256=verify_sha256))

    def candidates(self, view: RouterView) -> list[str]:
        candidates = view.route_ids()
        if ABSTAIN_CLASS in self.classes:
            candidates.append(ABSTAIN_CLASS)
        return candidates

    def scores(self, view: RouterView) -> dict[str, float]:
        candidates = self.candidates(view)
        result: dict[str, float] = {}
        for index, candidate in enumerate(candidates):
            vector = featurize(
                view,
                candidate,
                dimension=self.dimension,
                bigrams=self.bigrams,
                candidate_index=index,
                candidate_count=len(candidates),
            )
            weights = self.weights.get(candidate)
            if weights is None:
                # A menu route with no training evidence still appears in the menu. Give it a
                # neutral score; the softmax will only select it if the rest is weak.
                result[candidate] = 0.0
                continue
            result[candidate] = sum(weights[index_] * value for index_, value in vector.items())
        return result

    def probabilities(
        self, view: RouterView, *, temperature: float | None = None
    ) -> dict[str, float]:
        scores = self.scores(view)
        if not scores:
            return {}
        scale = self.temperature if temperature is None else temperature
        if scale <= 0:
            raise ValueError("temperature must be positive")
        return _softmax({name: score / scale for name, score in scores.items()})

    def decision(self, view: RouterView) -> RouterDecision:
        probabilities = self.probabilities(view)
        if not probabilities:
            return RouterDecision(
                "abstain",
                None,
                0.0,
                "no scorable route in the menu",
                {},
                self.threshold,
                0.0,
            )
        best = max(probabilities, key=lambda name: (probabilities[name], name))
        best_probability = probabilities[best]
        raw_best = max(self.probabilities(view, temperature=1.0).items(), key=lambda item: item[1])
        raw_confidence = float(raw_best[1])
        if best == ABSTAIN_CLASS:
            return RouterDecision(
                "abstain",
                None,
                best_probability,
                f"trained router abstained (p_abstain={best_probability:.3f})",
                probabilities,
                self.threshold,
                raw_confidence,
            )
        if best_probability < self.threshold:
            return RouterDecision(
                "abstain",
                None,
                best_probability,
                (
                    f"calibrated confidence {best_probability:.3f} is below the abstain "
                    f"threshold {self.threshold:.3f}"
                ),
                probabilities,
                self.threshold,
                raw_confidence,
            )
        return RouterDecision(
            "route",
            best,
            best_probability,
            (
                f"trained router selected {best} (p={best_probability:.3f}, "
                f"threshold={self.threshold:.3f})"
            ),
            probabilities,
            self.threshold,
            raw_confidence,
        )

    def decision_json(self, view: RouterView) -> str:
        return self.decision(view).text()


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    maximum = max(scores.values())
    exponents = {name: math.exp(value - maximum) for name, value in scores.items()}
    total = sum(exponents.values())
    if total <= 0 or not math.isfinite(total):
        uniform = 1.0 / len(scores)
        return {name: uniform for name in scores}
    return {name: value / total for name, value in exponents.items()}
