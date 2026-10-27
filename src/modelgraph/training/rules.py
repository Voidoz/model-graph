"""Deterministic rules/keyword router baseline.

This is the floor the trained classifier must beat. It reads the same rendered menu as the
runtime, scores lexical overlap with each route's id and rendered card, honours optional explicit
keyword lists from the recipe config, and abstains when the signal is weak or ambiguous. It emits
the canonical decision JSON, so it can be compared with the trained adapter through the same
router executor.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from modelgraph.training.calibration import CostMatrix, evaluate_predictions
from modelgraph.training.dataset import RouterExample
from modelgraph.training.features import RouterView, tokenize


@dataclass(frozen=True)
class RuleConfig:
    keywords: dict[str, tuple[str, ...]] = field(default_factory=dict)
    min_score: float = 1.0
    ambiguity_margin: float = 0.12
    confidence_floor: float = 0.2
    confidence_ceiling: float = 0.95

    def __post_init__(self) -> None:
        if self.min_score < 0:
            raise ValueError("min_score must be non-negative")
        if not 0 <= self.ambiguity_margin <= 1:
            raise ValueError("ambiguity_margin must be in [0, 1]")
        if not 0 <= self.confidence_floor <= self.confidence_ceiling <= 1:
            raise ValueError("confidence_floor must be <= confidence_ceiling and both in [0, 1]")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RuleConfig:
        keywords = {
            str(route): tuple(str(item) for item in items)
            for route, items in (value.get("keywords") or {}).items()
        }
        return cls(
            keywords=keywords,
            min_score=float(value.get("min_score", 1.0)),
            ambiguity_margin=float(value.get("ambiguity_margin", 0.12)),
            confidence_floor=float(value.get("confidence_floor", 0.2)),
            confidence_ceiling=float(value.get("confidence_ceiling", 0.95)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": {route: list(words) for route, words in sorted(self.keywords.items())},
            "min_score": self.min_score,
            "ambiguity_margin": self.ambiguity_margin,
            "confidence_floor": self.confidence_floor,
            "confidence_ceiling": self.confidence_ceiling,
        }


@dataclass(frozen=True)
class RuleDecision:
    route: str | None
    confidence: float
    reason: str
    action: str
    scores: dict[str, float]

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


class RuleRouter:
    """A keyword-overlap router with an explicit ambiguity margin."""

    def __init__(self, config: RuleConfig | None = None):
        self.config = config or RuleConfig()

    def scores(self, view: RouterView) -> dict[str, float]:
        request_tokens = set(tokenize(view.request))
        result: dict[str, float] = {}
        for route in view.routes:
            id_tokens = set(tokenize(route.id.replace("-", " ").replace("_", " ")))
            definition_tokens = id_tokens | set(tokenize(route.text))
            score = 0.0
            for token in request_tokens:
                if token in id_tokens:
                    score += 2.5
                if token in definition_tokens:
                    score += 1.0
            explicit = {
                token
                for keyword in self.config.keywords.get(route.id, ())
                for token in tokenize(keyword)
            }
            score += 2.0 * len(request_tokens & explicit)
            result[route.id] = score
        return result

    def decide(self, view: RouterView) -> RuleDecision:
        scores = self.scores(view)
        if not scores:
            return RuleDecision(None, 0.0, "no menu routes", "abstain", {})
        ordered = sorted(
            scores.items(),
            key=lambda item: (-item[1], view.route_ids().index(item[0])),
        )
        top_route, top = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        if top < self.config.min_score:
            return RuleDecision(
                None,
                0.2,
                f"no route keyword reached min_score={self.config.min_score:g}",
                "abstain",
                scores,
            )
        margin = (top - second) / max(top, 1.0)
        confidence = min(
            max(0.5 + 0.45 * margin, self.config.confidence_floor),
            self.config.confidence_ceiling,
        )
        if margin < self.config.ambiguity_margin and second > 0:
            return RuleDecision(
                None,
                min(confidence, 0.45),
                (f"ambiguous keyword signal: {top_route}={top:g} vs {ordered[1][0]}={second:g}"),
                "abstain",
                scores,
            )
        return RuleDecision(
            top_route,
            confidence,
            f"keyword score {top_route}={top:g} (margin={margin:.2f})",
            "route",
            scores,
        )


def evaluate_rules(
    examples: Sequence[RouterExample],
    *,
    config: RuleConfig | None = None,
    cost_matrix: CostMatrix | None = None,
) -> dict[str, Any]:
    router = RuleRouter(config)
    predictions: list[dict[str, Any]] = []
    for example in examples:
        view = RouterView(
            request=example.request,
            routes=tuple(example.menu_routes),
            menu_text="",
            messages=example.messages,
            prompt_hash=example.prompt_hash,
        )
        decision = router.decide(view)
        predictions.append(
            {
                "route": decision.route,
                "confidence": decision.confidence,
                "correct_route": bool(
                    decision.route is not None
                    and example.oracle_route is not None
                    and decision.route == example.oracle_route
                ),
                "ambiguous": example.ambiguous,
                "oracle_route": example.oracle_route,
            }
        )
    return {
        "predictions": predictions,
        "metrics": evaluate_predictions(predictions, 0.0, cost_matrix or CostMatrix()),
        "config": router.config.to_dict(),
    }
