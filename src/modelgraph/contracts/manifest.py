"""Typed model manifests."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ManifestMetadata(_Model):
    id: str
    name: str
    version: str
    description: str
    labels: dict[str, str] = Field(default_factory=dict)
    authors: list[str] | None = None


class ProviderLimits(_Model):
    default_temperature: float | None = None
    default_max_output_tokens: int | None = None
    timeout_seconds: float | None = None


class Cost(_Model):
    pricing: str = "free"
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
    network_access: str | None = None


class License(_Model):
    name: str
    url: str | None = None
    weights_uri: str | None = None
    commercial_use: bool | None = None
    redistribution: bool | None = None
    notes: str | None = None


class Integrity(_Model):
    sha256: str | None = None
    revision: str | None = None
    verified_at: str | None = None


class EvalInfo(_Model):
    suite: str | None = None
    report: str | None = None
    score: float | None = None
    measured_at: str | None = None
    notes: str | None = None


class ModelSpec(_Model):
    provider: str
    model: str
    endpoint: str | None = None
    context_window: int
    max_output_tokens: int | None = None
    capabilities: list[str] = Field(default_factory=list)
    limits: ProviderLimits = Field(default_factory=ProviderLimits)
    cost: Cost = Field(default_factory=Cost)
    latency_ms: Latency = Field(default_factory=Latency)
    resources: Resources = Field(default_factory=Resources)
    license: License
    integrity: Integrity = Field(default_factory=Integrity)
    eval: EvalInfo | None = None
    notes: str | None = None


class ModelManifest(_Model):
    apiVersion: str
    kind: str
    metadata: ManifestMetadata
    spec: ModelSpec

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)
