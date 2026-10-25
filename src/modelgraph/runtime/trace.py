"""Trace recording and replay primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelgraph.providers.base import (
    Capabilities,
    ChatRequest,
    ChatResponse,
    Health,
    ProviderFailure,
    Usage,
)
from modelgraph.security import RedactionReport, redact_data
from modelgraph.util import utc_now


@dataclass
class TraceRecorder:
    run_id: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {"ts": utc_now(), "run_id": self.run_id, "event": event, **fields}
        self.events.append(record)
        return record

    def write(self, path: Path, *, redact: bool = True) -> RedactionReport | None:
        """Write the event stream, redacting secret/PII-shaped text by default.

        The in-memory events are left untouched so the producing run can still use them;
        only the persisted JSONL is scrubbed. The returned report contains pattern names,
        counts, field paths, and truncated hashes, never the matched text.
        """

        report: RedactionReport | None = None
        events: list[dict[str, Any]] = self.events
        if redact:
            redacted, report = redact_data(self.events)
            events = redacted
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return report


def read_events(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


class ReplayProvider:
    """Serve recorded provider responses in order, without a network call."""

    name = "replay"

    def __init__(self, events: list[dict[str, Any]]):
        self.events = [
            event
            for event in events
            if event.get("event") in {"provider.response", "provider.error"}
        ]
        self.index = 0

    def capabilities(self) -> Capabilities:
        return Capabilities(json_mode=True, max_context=1_000_000)

    def health(self) -> Health:
        return Health(ok=True, detail="replay provider")

    def chat(self, request: ChatRequest) -> ChatResponse:
        if self.index >= len(self.events):
            raise ProviderFailure(
                "replay has no recorded response for this call",
                code="MG-3006",
                retryable=False,
            )
        event = self.events[self.index]
        self.index += 1
        if event.get("event") == "provider.error":
            raise ProviderFailure(
                str(event.get("message", "recorded provider failure")),
                code=str(event.get("code", "MG-3000")),
                retryable=bool(event.get("retryable", False)),
            )
        usage_data = event.get("usage") or {}
        return ChatResponse(
            text=str(event.get("text", "")),
            usage=Usage(
                input_tokens=int(usage_data.get("input_tokens", 0)),
                output_tokens=int(usage_data.get("output_tokens", 0)),
                approximated=bool(usage_data.get("approximated", False)),
            ),
            finish_reason=str(event.get("finish_reason", "stop")),
            provider_meta={"replayed": True},
        )
