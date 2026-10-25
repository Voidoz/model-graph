"""Evaluation-specific errors.

They reuse the repository error taxonomy rather than inventing new ranges: suite authoring and
selection problems are validation errors (``MG-1001``); runner failures are runtime errors
(``MG-2000``).
"""

from __future__ import annotations

from modelgraph.errors import ModelGraphError


class SuiteError(ModelGraphError):
    """The suite is missing, malformed, or inconsistent with its level."""

    code = "MG-1001"


class EvalError(ModelGraphError):
    """The evaluation runner cannot complete as requested."""

    code = "MG-2000"
