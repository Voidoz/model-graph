"""Every real ticket validates, and the template is explicitly excluded."""

from __future__ import annotations

from pathlib import Path

from modelgraph.contracts.ticket import load_ticket

ROOT = Path(__file__).resolve().parents[2]
TICKETS = ROOT / ".scratch" / "tickets"


def test_every_ticket_validates() -> None:
    paths = sorted(TICKETS.glob("T-*.md"))
    assert len(paths) >= 30
    for path in paths:
        ticket = load_ticket(path)
        assert path.name.startswith(f"{ticket.id}-")


def test_template_is_explicitly_excluded() -> None:
    template = TICKETS / "_template.md"
    assert template.exists()
    assert template not in sorted(TICKETS.glob("T-*.md"))
    assert not template.name.startswith("T-")


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
