"""Ticket frontmatter fixture validation."""

from __future__ import annotations

from pathlib import Path

from modelgraph.contracts.ticket import load_ticket


def test_ticket_null_owner_and_blocker_are_accepted() -> None:
    ticket = load_ticket(
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "contracts"
        / "ticket"
        / "valid"
        / "minimal.ticket.md"
    )
    assert ticket.owner is None
    assert ticket.blocked_reason is None
