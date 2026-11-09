"""The current-value route, ``POST values``, read through the supervised child.

Mounted under ``/api/directory`` by :mod:`.directory_api`. The body carries only
channel names; the route reads, never writes, and opens no control session: the
lifespan's supervisor asks its read-only child, and a child that cannot answer
costs rows that read ``unavailable`` with the reason.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from osprey.interfaces.channel_finder.directory_values_ipc import MAX_NAMES, ValueRow

router = APIRouter()


class ValuesRequest(BaseModel):
    """At most :data:`~.directory_values_ipc.MAX_NAMES` channel names and nothing else."""

    model_config = ConfigDict(extra="forbid")

    names: list[StrictStr] = Field(max_length=MAX_NAMES)


@router.post("/values")
async def values(body: ValuesRequest, request: Request) -> list[dict[str, Any]]:
    """One ``ValueRow`` per name, in request order."""
    rows: list[ValueRow] = await request.app.state.directory_values.read(body.names)
    return [row.to_wire() for row in rows]
