"""The current-value route, ``GET values?name=…``, read through the supervised child.

Mounted under ``/api/directory`` by :mod:`.directory_api`. The query carries only
channel names; the route reads, never writes, and opens no control session: the
lifespan's supervisor asks its read-only child, and a child that cannot answer
costs rows that read ``unavailable`` with the reason. The app's cache middleware
answers every ``/api/`` response ``no-store``, so a Refresh re-reads.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request

from osprey.interfaces.channel_finder.directory_values_ipc import MAX_NAMES, ValueRow

router = APIRouter()

_Names = Annotated[list[str], Query(max_length=MAX_NAMES)]


@router.get("/values")
async def values(request: Request, name: _Names = []) -> list[dict[str, Any]]:  # noqa: B006
    """One ``ValueRow`` per ``name``, in request order; at most ``MAX_NAMES`` of them."""
    rows: list[ValueRow] = await request.app.state.directory_values.read(name)
    return [row.to_wire() for row in rows]
