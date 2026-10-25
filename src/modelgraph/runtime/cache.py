"""Exact-match response cache.

The cache is deliberately simple and inspectable:

* the key is a SHA-256 over the rendered messages, generation parameters, and the
  identity of everything that can change a response (graph hash, node, renderer
  version, model manifest hash, resolved model revision/version, and the effective
  provider/model);
* one JSON file per key under ``<cache-dir>/<sha256>.json``;
* writes use a temporary file, ``fsync``, and ``os.replace`` so a concurrent reader
  sees either a complete old entry or a complete new one, never a partial file;
* corrupt or partially written entries are ignored (and removed when writes are
  allowed) instead of failing a run;
* entries contain the response text, normalized usage, finish reason, provider
  metadata, creation time, and *cache-key metadata only*. Request messages and
  prompts are never stored, only a hash of the messages.

Privacy choice (T-0016): a response that matches any default secret/PII pattern is
**not cached**. The runtime emits ``cache.skipped`` with the pattern names (never
the matched text) and the next identical request misses. Redacting the cache
metadata alone was rejected because the response body is the sensitive part.

There is no TTL and no semantic matching. Invalidation is by key only, which is why
the key includes the graph hash, model manifest hash, and resolved revision.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from modelgraph.compiler.ir import CompiledGraph, CompiledNode
from modelgraph.providers.base import ChatRequest, ChatResponse, Usage
from modelgraph.security import scan_data
from modelgraph.util import canonical_json, sha256_text
from modelgraph.version import RENDERER_VERSION

CACHE_ENTRY_SCHEMA = "modelgraph/cache-entry/v0"
CACHE_KEY_SCHEMA = "modelgraph/cache-key/v0"

CACHE_MODES: tuple[str, ...] = ("on", "off", "readonly")
CacheMode = Literal["on", "off", "readonly"]


@dataclass(frozen=True)
class CacheKey:
    """A content-derived cache key plus the non-secret metadata used to build it."""

    digest: str
    metadata: dict[str, Any]

    @property
    def value(self) -> str:
        return f"sha256:{self.digest}"

    @property
    def filename(self) -> str:
        return f"{self.digest}.json"


@dataclass(frozen=True)
class CacheEntry:
    """A cached provider response and its creation metadata."""

    text: str
    usage: Usage
    finish_reason: str = "stop"
    provider_meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    key: str = ""
    key_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CACHE_ENTRY_SCHEMA,
            "key": self.key,
            "created_at": self.created_at,
            "key_metadata": dict(self.key_metadata),
            "response": {
                "text": self.text,
                "usage": self.usage.to_dict(),
                "finish_reason": self.finish_reason,
                "provider_meta": dict(self.provider_meta),
            },
        }

    @classmethod
    def from_dict(cls, data: Any, *, expected_key: str | None = None) -> CacheEntry:
        """Parse an entry, raising ``ValueError`` when it is corrupt or partial."""

        if not isinstance(data, dict):
            raise ValueError("cache entry must be a JSON object")
        if data.get("schema") != CACHE_ENTRY_SCHEMA:
            raise ValueError("cache entry schema mismatch")
        key = data.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("cache entry has no key")
        if expected_key is not None and key != expected_key:
            raise ValueError("cache entry key mismatch")
        created_at = data.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("cache entry has no created_at")
        response = data.get("response")
        if not isinstance(response, dict):
            raise ValueError("cache entry has no response")
        text = response.get("text")
        if not isinstance(text, str):
            raise ValueError("cache entry response text is missing")
        usage_data = response.get("usage")
        if not isinstance(usage_data, dict):
            raise ValueError("cache entry usage is missing")
        usage = Usage(
            input_tokens=_non_negative_int(usage_data.get("input_tokens"), "input_tokens"),
            output_tokens=_non_negative_int(usage_data.get("output_tokens"), "output_tokens"),
            approximated=bool(usage_data.get("approximated", False)),
        )
        finish_reason = response.get("finish_reason", "stop")
        if not isinstance(finish_reason, str):
            raise ValueError("cache entry finish_reason must be a string")
        provider_meta = response.get("provider_meta", {})
        if not isinstance(provider_meta, dict):
            raise ValueError("cache entry provider_meta must be an object")
        key_metadata = data.get("key_metadata", {})
        if not isinstance(key_metadata, dict):
            raise ValueError("cache entry key_metadata must be an object")
        return cls(
            text=text,
            usage=usage,
            finish_reason=finish_reason,
            provider_meta=dict(provider_meta),
            created_at=created_at,
            key=key,
            key_metadata=dict(key_metadata),
        )


@dataclass
class CacheStats:
    """Per-run cache accounting; never shared between runs by the runtime."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    skipped: int = 0
    avoided_tokens: int = 0
    avoided_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "skipped": self.skipped,
            "avoided_tokens": self.avoided_tokens,
            "avoided_cost_usd": round(self.avoided_cost_usd, 8),
        }


class ResponseCache:
    """A local, filesystem-backed exact-match response cache.

    ``on`` reads and writes; ``readonly`` reads but never writes (a miss stays a
    miss); ``off`` touches neither reads nor writes. A cache hit consumes zero
    provider tokens/cost, so the executor does not charge the budget for it.
    """

    def __init__(self, directory: str | Path, *, mode: str = "on") -> None:
        normalized = mode.strip().lower()
        if normalized not in CACHE_MODES:
            raise ValueError(f"cache mode must be one of {', '.join(CACHE_MODES)}; got {mode!r}")
        self.directory = Path(directory)
        self.mode = normalized

    @property
    def readable(self) -> bool:
        return self.mode in {"on", "readonly"}

    @property
    def writable(self) -> bool:
        return self.mode == "on"

    def path_for(self, key: CacheKey | str) -> Path:
        filename = key.filename if isinstance(key, CacheKey) else _filename_for(key)
        return self.directory / filename

    def get(self, key: CacheKey) -> CacheEntry | None:
        """Return a valid entry, or ``None`` for a miss/corrupt/partial entry."""

        if not self.readable:
            return None
        path = self.path_for(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CacheEntry.from_dict(data, expected_key=key.value)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            # A partial write from a crashed process, a truncated copy, or an entry
            # whose key/schema does not match. Ignore it; when writes are allowed the
            # next successful response atomically replaces it.
            if self.writable:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
            return None

    def put(self, key: CacheKey, entry: CacheEntry) -> bool:
        """Atomically store ``entry``. Returns ``False`` when the write was skipped."""

        if not self.writable:
            return False
        payload = entry.to_dict()
        payload["key"] = key.value
        payload["key_metadata"] = dict(key.metadata)
        try:
            _write_atomic(self.path_for(key), payload)
        except (OSError, TypeError, ValueError):
            return False
        return True


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"cache entry usage.{field_name} must be an integer")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"cache entry usage.{field_name} must be non-negative")
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"cache entry usage.{field_name} must be an integer") from exc
        if parsed < 0:
            raise ValueError(f"cache entry usage.{field_name} must be non-negative")
        return parsed
    raise ValueError(f"cache entry usage.{field_name} must be an integer")


def _filename_for(key: str) -> str:
    digest = key.split(":", 1)[-1]
    if not digest or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"invalid cache key {key!r}")
    return f"{digest}.json"


def build_cache_key(
    *,
    node: CompiledNode,
    graph: CompiledGraph,
    request: ChatRequest,
    provider: str,
    renderer_version: str = RENDERER_VERSION,
) -> CacheKey:
    """Build the exact-match key.

    The effective ``provider`` adapter name is part of the key: running the same graph
    with ``--provider mock`` and ``--provider ollama`` must not share responses even
    though the model manifest is identical. ``timeout_seconds`` is intentionally not
    part of the key because it does not change the generated content.
    """

    model = node.model
    messages = [message.to_dict() for message in request.messages]
    metadata = {
        "node_id": node.id,
        "graph_hash": graph.graph_hash,
        "renderer_version": renderer_version,
        "model_manifest_hash": node.model_hash,
        "resolved_revision": model.spec.integrity.revision if model is not None else None,
        "model_version": model.metadata.version if model is not None else None,
        "manifest_provider": model.spec.provider.value if model is not None else None,
        "provider": provider,
        "model": request.model,
        "generation": {
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "json_mode": request.json_mode,
            "stop": list(request.stop),
            "seed": request.seed,
        },
        "messages_hash": sha256_text(canonical_json(messages)),
    }
    payload = {
        "schema": CACHE_KEY_SCHEMA,
        **{key: value for key, value in metadata.items() if key != "messages_hash"},
        "messages": messages,
    }
    digest = sha256_text(canonical_json(payload)).split(":", 1)[1]
    return CacheKey(digest=digest, metadata=metadata)


def response_has_default_secret(response: ChatResponse | Any) -> list[str]:
    """Return matching default pattern names for a response, without matched text.

    The default pattern set includes PII patterns as well as credentials, so an e-mail
    or home path also disables caching for that response.
    """

    if isinstance(response, ChatResponse):
        value: Any = {"text": response.text, "provider_meta": response.provider_meta}
    else:
        value = response
    return sorted({finding.pattern for finding in scan_data(value)})


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(descriptor)
    except OSError:  # pragma: no cover - platform dependent
        pass
    finally:
        os.close(descriptor)
