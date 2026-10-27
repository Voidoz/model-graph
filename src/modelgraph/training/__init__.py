"""Offline router training, calibration, and the trained-router adapter.

The package is deliberately dependency-light: training uses plain Python (no torch,
numpy, or scikit-learn), so the recipe runs on the project's normal development
environment. See ``docs/training.md`` §4 and ``training/README.md``.

Importing this package is cheap; submodules that pull in the exporter or evaluation code are
imported explicitly so the runtime provider can load an artifact without importing the whole
training toolchain.
"""

from modelgraph.training.artifact import (
    ARTIFACT_SCHEMA,
    RouterArtifact,
    load_router_artifact,
    validate_router_artifact,
)
from modelgraph.training.features import (
    FEATURE_VERSION,
    MenuRoute,
    RouterView,
    featurize,
    view_from_example,
    view_from_messages,
)
from modelgraph.training.predict import Predictor, RouterDecision

__all__ = [
    "ARTIFACT_SCHEMA",
    "FEATURE_VERSION",
    "MenuRoute",
    "Predictor",
    "RouterArtifact",
    "RouterDecision",
    "RouterView",
    "featurize",
    "load_router_artifact",
    "validate_router_artifact",
    "view_from_example",
    "view_from_messages",
]
