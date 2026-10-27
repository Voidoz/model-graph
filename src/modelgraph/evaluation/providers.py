"""Provider wiring for evaluation.

Mock and ``trained-router`` runs are offline and deterministic. Any live provider is gated
explicitly so CI never reaches a network by accident. Provider construction is delegated to the
existing ``build_provider`` factory; evaluation only decides when it is allowed.
"""

from __future__ import annotations

import json
import os
from typing import Any

from modelgraph.compiler.ir import CompiledNode
from modelgraph.contracts.manifest import ModelManifest
from modelgraph.evaluation.errors import EvalError
from modelgraph.evaluation.suite import EvalCase, EvalSuite
from modelgraph.providers import build_provider
from modelgraph.providers.base import Health, Provider
from modelgraph.providers.mock import MockProvider, MockScenario

LIVE_ENV = "MG_EVAL_LIVE"
LIVE_VALUES = {"1", "true", "yes", "on"}
LIVE_PROVIDERS = {"auto", "ollama", "openai-compatible", "llamacpp"}
OFFLINE_PROVIDERS = {"mock", "trained-router", "mixed"}


def live_allowed(explicit: bool = False) -> bool:
    return explicit or os.environ.get(LIVE_ENV, "").strip().lower() in LIVE_VALUES


def require_live(provider: str, *, allow_live: bool) -> None:
    if provider in OFFLINE_PROVIDERS:
        return
    if provider not in LIVE_PROVIDERS:
        raise EvalError(f"unknown eval provider {provider!r}")
    if not live_allowed(allow_live):
        raise EvalError(
            f"provider {provider!r} is a live path and is gated; set {LIVE_ENV}=1 or pass "
            "allow_live=True. Offline suites must use the mock provider.",
            details={"provider": provider, "env": LIVE_ENV},
        )


class ProviderPool:
    """One provider per node for live runs, a fresh scripted mock per case for offline runs."""

    def __init__(self, kind: str, *, allow_live: bool) -> None:
        require_live(kind, allow_live=allow_live)
        self.kind = kind
        self._live: dict[str, Provider] = {}
        self._health: dict[str, dict[str, Any]] = {}

    def for_case(self, case: EvalCase, node: CompiledNode) -> Provider:
        if self.kind == "mock":
            script = normalize_mock_script(case.mock)
            scenario = MockScenario(responses={node.id: script} if script else {})
            max_context = node.model.spec.context_window if node.model else 32768
            return MockProvider(scenario, max_context=max_context)
        manifest = node.model
        if manifest is None:
            raise EvalError(f"node {node.id!r} has no model manifest")
        key = manifest.metadata.id
        if key not in self._live:
            provider = _build_live_provider(self.kind, manifest)
            self._live[key] = provider
            health = provider.health()
            self._health[key] = _health_dict(health)
        return self._live[key]

    def health(self) -> dict[str, dict[str, Any]]:
        return dict(self._health)

    def close(self) -> None:
        for provider in self._live.values():
            close = getattr(provider, "close", None)
            if callable(close):
                close()
        self._live.clear()


def normalize_mock_script(script: Any) -> list[str]:
    """Turn a case's ``mock`` value into an ordered list of raw provider responses."""
    if script is None:
        return []
    if isinstance(script, str):
        return [script]
    if isinstance(script, dict):
        return [json.dumps(script, ensure_ascii=False)]
    if isinstance(script, list):
        normalized: list[str] = []
        for item in script:
            if isinstance(item, dict):
                normalized.append(json.dumps(item, ensure_ascii=False))
            else:
                normalized.append(str(item))
        return normalized
    raise EvalError(f"unsupported mock script type: {type(script).__name__}")


def provider_summary(
    kind: str, suite: EvalSuite, pool: ProviderPool | None = None
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "kind": kind,
        "gated_live": kind in LIVE_PROVIDERS,
        "live_gate_env": LIVE_ENV,
        "deterministic_expected": kind in OFFLINE_PROVIDERS,
        "suite_provider": suite.provider,
    }
    if pool is not None and pool.health():
        summary["health"] = pool.health()
    return summary


def build_live_provider(kind: str, manifest: ModelManifest) -> Provider:
    """Build a live provider for one manifest; callers must gate with :func:`require_live`."""
    return _build_live_provider(kind, manifest)


def _build_live_provider(kind: str, manifest: ModelManifest) -> Provider:
    if kind == "auto":
        return build_provider(manifest)
    spec = manifest.spec.model_copy(update={"provider": kind})
    forced = manifest.model_copy(update={"spec": spec})
    return build_provider(forced)


def _health_dict(health: Health) -> dict[str, Any]:
    return {"ok": health.ok, "detail": health.detail}
