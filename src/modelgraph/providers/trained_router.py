"""Provider adapter for a serialized local router artifact.

The provider participates in the normal runtime contract: the router executor renders the exact
menu/request prompt, this adapter scores it, and it returns the same decision JSON an LLM would
return. No graph or executor semantics are forked.

Manifest convention:

```yaml
spec:
  provider: trained-router
  model: runs/training/router-triage/artifact.json   # artifact path
  integrity:
    sha256: <64 hex chars>                            # optional but recommended
```

A relative ``model`` path is resolved against the current directory first, then the project
root, so the same manifest works from the repository root and from a checker invocation.
"""

from __future__ import annotations

from pathlib import Path

from modelgraph.contracts.manifest import ModelManifest
from modelgraph.paths import project_root
from modelgraph.providers.base import (
    Capabilities,
    ChatRequest,
    ChatResponse,
    Health,
    ProviderFailure,
    Usage,
)
from modelgraph.training.errors import RouterArtifactError
from modelgraph.training.features import view_from_messages
from modelgraph.training.predict import Predictor
from modelgraph.util import estimate_tokens


class TrainedRouterProvider:
    """Score the rendered menu with a serialized conditional-logit router."""

    name = "trained-router"

    def __init__(self, manifest: ModelManifest):
        self.manifest = manifest
        self.artifact_path = resolve_artifact_path(manifest)
        expected = manifest.spec.integrity.sha256
        try:
            artifact = Predictor.load(self.artifact_path, verify_sha256=expected)
        except RouterArtifactError as exc:
            code = "MG-3006" if exc.path else "MG-3005"
            raise ProviderFailure(
                f"trained router unavailable: {exc}",
                code=code,
                retryable=False,
                details={"artifact": str(self.artifact_path)},
            ) from exc
        self.predictor = artifact

    def capabilities(self) -> Capabilities:
        return Capabilities(
            chat=True,
            json_mode=True,
            streaming=False,
            seeded=True,
            max_context=self.manifest.spec.context_window,
        )

    def health(self) -> Health:
        return Health(ok=True, detail=f"trained router artifact {self.artifact_path}")

    def chat(self, request: ChatRequest) -> ChatResponse:
        try:
            view = view_from_messages(request.messages)
        except ValueError as exc:
            raise ProviderFailure(
                f"trained router could not parse the rendered prompt: {exc}",
                code="MG-3006",
                retryable=False,
            ) from exc
        decision = self.predictor.decision(view)
        text = decision.text()
        prompt_tokens = estimate_tokens(request.prompt_text())
        output_tokens = estimate_tokens(text)
        return ChatResponse(
            text=text,
            usage=Usage(prompt_tokens, output_tokens, approximated=True),
            finish_reason="stop",
            provider_meta={
                "trained_router": True,
                "artifact_sha256": self.predictor.artifact.sha256(),
                "artifact_version": self.predictor.artifact.document.get("artifact_version"),
                "schema": self.predictor.artifact.document.get("schema"),
                "calibrated": True,
                "temperature": self.predictor.temperature,
                "abstain_threshold": self.predictor.threshold,
                "confidence": decision.confidence,
                "probabilities": decision.probabilities,
            },
        )


def resolve_artifact_path(manifest: ModelManifest) -> Path:
    raw = Path(manifest.spec.model).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, project_root() / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ProviderFailure(
        f"trained router artifact not found: {raw}",
        code="MG-3005",
        retryable=False,
        details={"candidates": [str(item) for item in candidates]},
    )
