"""Provider contract shared by mock, local, and hosted adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from modelgraph.errors import ProviderError


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    approximated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "approximated": self.approximated,
        }


@dataclass(frozen=True)
class Capabilities:
    chat: bool = True
    json_mode: bool = False
    streaming: bool = False
    seeded: bool = False
    max_context: int = 8192

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat": self.chat,
            "json_mode": self.json_mode,
            "streaming": self.streaming,
            "seeded": self.seeded,
            "max_context": self.max_context,
        }


@dataclass(frozen=True)
class Health:
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[Message]
    temperature: float = 0.2
    max_output_tokens: int = 1024
    json_mode: bool = False
    seed: int | None = None
    timeout_seconds: float = 120.0
    stop: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_id(self) -> str | None:
        value = self.metadata.get("node_id")
        return str(value) if value is not None else None

    def prompt_text(self) -> str:
        return "\n".join(message.content for message in self.messages)


@dataclass(frozen=True)
class ChatResponse:
    text: str
    usage: Usage
    finish_reason: str = "stop"
    provider_meta: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...

    def health(self) -> Health: ...

    def chat(self, request: ChatRequest) -> ChatResponse: ...


class ProviderFailure(ProviderError):
    """A provider call failed; carries a stable code and retryability."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "MG-3000",
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details=details)
        self.code = code
        self.retryable = retryable
