"""String patterns shared by the contract models.

The JSON Schemas in ``schemas/`` are canonical. These aliases mirror the patterns that the
schemas impose so the Pydantic models reject the same shapes; ``tests/contracts/test_parity.py``
compares them and fails on drift.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

GRAPH_ID = r"^[a-z][a-z0-9-]{2,63}$"
NODE_ID = r"^[a-z][a-z0-9-]{1,62}$"
ARTIFACT_NAME = r"^[a-z][a-z0-9_]{0,63}$"
SEMVER = r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?$"
SUBGRAPH_REF = (
    r"^[a-z][a-z0-9-]{2,63}@(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?$"
)
CONTENT_HASH = r"^sha256:[0-9a-f]{64}$"
SHA256_HEX = r"^[0-9a-f]{64}$"
TICKET_ID = r"^T-[0-9]{4}$"
TICKET_DATE = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"

GraphId = Annotated[str, Field(pattern=GRAPH_ID)]
NodeId = Annotated[str, Field(pattern=NODE_ID)]
ArtifactName = Annotated[str, Field(pattern=ARTIFACT_NAME)]
Semver = Annotated[str, Field(pattern=SEMVER)]
SubgraphRef = Annotated[str, Field(pattern=SUBGRAPH_REF)]
ContentHash = Annotated[str, Field(pattern=CONTENT_HASH)]
Sha256Hex = Annotated[str, Field(pattern=SHA256_HEX)]
TicketId = Annotated[str, Field(pattern=TICKET_ID)]
TicketDate = Annotated[str, Field(pattern=TICKET_DATE)]
