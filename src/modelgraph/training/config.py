"""Reference training config for the local router recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from modelgraph.contracts.loader import load_document
from modelgraph.training.calibration import CostMatrix
from modelgraph.training.errors import RouterTrainingError
from modelgraph.training.rules import RuleConfig
from modelgraph.util import canonical_json, sha256_text

CONFIG_SCHEMA = "modelgraph/router-training-config/v0"


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class FeatureParams(_Config):
    dimension: int = Field(default=512, ge=17, le=4096)
    bigrams: bool = True


class TrainingParams(_Config):
    epochs: int = Field(default=300, ge=1, le=100000)
    learning_rate: float = Field(default=0.03, gt=0)
    l2: float = Field(default=0.001, ge=0)
    optimizer: Literal["adam", "gd"] = "adam"
    class_weight: Literal["none", "balanced"] = "balanced"
    include_ambiguous: bool = False
    beta1: float = Field(default=0.9, ge=0, lt=1)
    beta2: float = Field(default=0.999, ge=0, lt=1)
    epsilon: float = Field(default=1e-8, gt=0)


class CalibrationParams(_Config):
    method: Literal["temperature"] = "temperature"
    bins: int = Field(default=10, ge=1, le=100)
    fit_split: Literal["validation"] = "validation"
    notes: str | None = None


class AbstainParams(_Config):
    fit_split: Literal["validation"] = "validation"
    steps: int = Field(default=100, ge=1, le=10000)
    cost_matrix: dict[str, float] = Field(default_factory=lambda: CostMatrix().to_dict())
    notes: str | None = None

    @field_validator("cost_matrix")
    @classmethod
    def _valid_cost_matrix(cls, value: dict[str, float]) -> dict[str, float]:
        CostMatrix.from_dict(value)
        return value


class EvaluationParams(_Config):
    require_all_splits: bool = True
    require_frozen_test: bool = False


class RuleParams(_Config):
    keywords: dict[str, list[str]] = Field(default_factory=dict)
    min_score: float = Field(default=1.0, ge=0)
    ambiguity_margin: float = Field(default=0.12, ge=0, le=1)
    confidence_floor: float = Field(default=0.2, ge=0, le=1)
    confidence_ceiling: float = Field(default=0.95, ge=0, le=1)

    def to_rule_config(self) -> RuleConfig:
        return RuleConfig(
            keywords={route: tuple(words) for route, words in self.keywords.items()},
            min_score=self.min_score,
            ambiguity_margin=self.ambiguity_margin,
            confidence_floor=self.confidence_floor,
            confidence_ceiling=self.confidence_ceiling,
        )


class RouterTrainingConfig(_Config):
    schema_version: Literal["modelgraph/router-training-config/v0"] = Field(
        default="modelgraph/router-training-config/v0", alias="schema"
    )
    dataset: str
    artifact: str
    report: str | None = None
    seed: int = 20261025
    features: FeatureParams = Field(default_factory=FeatureParams)
    training: TrainingParams = Field(default_factory=TrainingParams)
    calibration: CalibrationParams = Field(default_factory=CalibrationParams)
    abstain: AbstainParams = Field(default_factory=AbstainParams)
    evaluation: EvaluationParams = Field(default_factory=EvaluationParams)
    rules: RuleParams = Field(default_factory=RuleParams)
    notes: str | None = None

    def cost_matrix(self) -> CostMatrix:
        return CostMatrix.from_dict(self.abstain.cost_matrix)

    def rule_config(self) -> RuleConfig:
        return self.rules.to_rule_config()

    def config_hash(self) -> str:
        return sha256_text(
            canonical_json(self.model_dump(mode="json", by_alias=True, exclude_none=True))
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


def load_training_config(path: str | Path) -> RouterTrainingConfig:
    source = Path(path)
    try:
        document = load_document(source)
    except Exception as exc:
        raise RouterTrainingError(f"cannot load training config {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise RouterTrainingError(f"training config {source} must be a mapping")
    try:
        return RouterTrainingConfig.model_validate(document)
    except Exception as exc:
        raise RouterTrainingError(f"invalid training config {source}: {exc}") from exc
