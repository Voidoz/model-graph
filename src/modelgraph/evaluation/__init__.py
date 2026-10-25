"""Node, routing, and end-to-end evaluation harness.

Offline-first: the deterministic mock provider is the default, and live providers are gated by
``MG_EVAL_LIVE=1`` or ``allow_live=True``. Reports are machine-readable JSON plus a Markdown
summary, both referencing the suite hash and eval run manifest.
"""

from modelgraph.evaluation.checks import JudgeHook, JudgeRequest, JudgeVerdict
from modelgraph.evaluation.comparison import compare_reports, load_report
from modelgraph.evaluation.e2e import E2EConfiguration, run_e2e_suite, score_blinded_pair
from modelgraph.evaluation.errors import EvalError, SuiteError
from modelgraph.evaluation.node import run_node_suite
from modelgraph.evaluation.report import (
    EvalReport,
    build_primary_metric,
    validate_report,
    write_report,
)
from modelgraph.evaluation.routing import run_routing_suite
from modelgraph.evaluation.suite import EvalSuite, load_suite

__all__ = [
    "E2EConfiguration",
    "EvalError",
    "EvalReport",
    "EvalSuite",
    "JudgeHook",
    "JudgeRequest",
    "JudgeVerdict",
    "SuiteError",
    "build_primary_metric",
    "compare_reports",
    "load_report",
    "load_suite",
    "run_e2e_suite",
    "run_node_suite",
    "run_routing_suite",
    "score_blinded_pair",
    "validate_report",
    "write_report",
]
