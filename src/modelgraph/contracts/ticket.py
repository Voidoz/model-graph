"""Ticket frontmatter parsing and validation."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from modelgraph.contracts.constraints import TicketDate, TicketId
from modelgraph.contracts.schema import validate_document
from modelgraph.errors import Issue, ModelGraphError

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


class TicketStatus(StrEnum):
    todo = "todo"
    ready = "ready"
    in_progress = "in-progress"
    blocked = "blocked"
    review = "review"
    done = "done"
    dropped = "dropped"


class TicketArea(StrEnum):
    foundations = "foundations"
    compiler = "compiler"
    runtime = "runtime"
    protocol = "protocol"
    providers = "providers"
    evaluation = "evaluation"
    training = "training"
    registry = "registry"
    experience = "experience"
    security = "security"
    observability = "observability"
    docs = "docs"


class TicketSize(StrEnum):
    XS = "XS"
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"


class TicketPriority(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class TicketFrontmatter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: TicketId
    title: str
    status: TicketStatus
    phase: int
    area: TicketArea
    size: TicketSize
    priority: TicketPriority
    depends_on: list[TicketId] = []
    owner: str | None = None
    created: TicketDate
    updated: TicketDate
    labels: list[str] = []
    blocked_reason: str | None = None


def parse_frontmatter(text: str) -> dict[str, Any]:
    match = _FRONTMATTER.match(text)
    if not match:
        raise ModelGraphError("ticket has no YAML frontmatter")
    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        raise ModelGraphError("ticket frontmatter must be a mapping")
    return data


def load_ticket(path: str | Path) -> TicketFrontmatter:
    source = Path(path)
    data = parse_frontmatter(source.read_text(encoding="utf-8"))
    issues: list[Issue] = validate_document(data, "ticket", path=str(source))
    if issues:
        raise ModelGraphError(
            "; ".join(issue.render() for issue in issues),
            path=str(source),
            details={"issues": [issue.__dict__ for issue in issues]},
        )
    return TicketFrontmatter.model_validate(data)
