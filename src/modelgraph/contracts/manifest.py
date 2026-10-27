"""Typed model manifests."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from modelgraph.contracts.constraints import GraphId, Semver, Sha256Hex


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ProviderKind(StrEnum):
    mock = "mock"
    openai_compatible = "openai-compatible"
    ollama = "ollama"
    llamacpp = "llamacpp"
    custom = "custom"
    trained_router = "trained-router"


class Capability(StrEnum):
    chat = "chat"
    json_mode = "json_mode"
    streaming = "streaming"
    seeded = "seeded"
    grammar = "grammar"


class Pricing(StrEnum):
    per_token = "per_token"
    per_request = "per_request"
    free = "free"
    unknown = "unknown"


class NetworkAccess(StrEnum):
    none = "none"
    loopback = "loopback"
    internet = "internet"


class ManifestMetadata(_Model):
    id: GraphId
    name: str
    version: Semver
    description: str
    labels: dict[str, str] = Field(default_factory=dict)
    authors: list[str] | None = None


class ProviderLimits(_Model):
    default_temperature: float | None = None
    default_max_output_tokens: int | None = None
    timeout_seconds: float | None = None


class Cost(_Model):
    pricing: Pricing
    input_per_million_usd: float = 0.0
    output_per_million_usd: float = 0.0
    per_request_usd: float = 0.0
    notes: str | None = None


class Latency(_Model):
    p50: float | None = None
    p95: float | None = None


class Resources(_Model):
    vram_gb: float | None = None
    ram_gb: float | None = None
    disk_gb: float | None = None
    cpu_only: bool | None = None
    network_access: NetworkAccess | None = None


class License(_Model):
    name: str
    url: str | None = None
    weights_uri: str | None = None
    commercial_use: bool | None = None
    redistribution: bool | None = None
    notes: str | None = None


class Integrity(_Model):
    sha256: Sha256Hex | None = None
    revision: str | None = None
    verified_at: str | None = None


class EvalInfo(_Model):
    suite: str | None = None
    report: str | None = None
    score: float | None = None
    measured_at: str | None = None
    notes: str | None = None


class ModelSpec(_Model):
    provider: ProviderKind
    model: str
    endpoint: str | None = None
    context_window: int
    max_output_tokens: int | None = None
    capabilities: list[Capability] = Field(default_factory=list)
    limits: ProviderLimits = Field(default_factory=ProviderLimits)
    cost: Cost = Field(default_factory=lambda: Cost(pricing=Pricing.free))
    latency_ms: Latency = Field(default_factory=Latency)
    resources: Resources = Field(default_factory=Resources)
    license: License
    integrity: Integrity = Field(default_factory=Integrity)
    eval: EvalInfo | None = None
    notes: str | None = None


class ModelManifest(_Model):
    apiVersion: Literal["modelgraph/v0"]
    kind: Literal["Model"]
    metadata: ManifestMetadata
    spec: ModelSpec

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)
