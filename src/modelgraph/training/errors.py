"""Training-specific errors with a stable code."""

from __future__ import annotations

from modelgraph.errors import ModelGraphError


class RouterTrainingError(ModelGraphError):
    """A router dataset, artifact, config, or training run is invalid."""

    code = "MG-7000"


class RouterDataError(RouterTrainingError):
    """The exported dataset cannot be turned into router examples."""

    code = "MG-7001"


class RouterArtifactError(RouterTrainingError):
    """A router artifact is missing, malformed, or fails its integrity check."""

    code = "MG-7002"


class SpecialistTrainingError(RouterTrainingError):
    """A specialist config, dataset, artifact, or training run is invalid."""

    code = "MG-7010"


class SpecialistDataError(SpecialistTrainingError):
    """The T-0023 dataset cannot be used for specialist training."""

    code = "MG-7011"


class SpecialistPromotionError(SpecialistTrainingError):
    """A specialist artifact cannot be promoted without sufficient evidence."""

    code = "MG-7012"


class SpecialistArtifactError(SpecialistTrainingError):
    """A generated specialist artifact or its schema is invalid."""

    code = "MG-7013"
