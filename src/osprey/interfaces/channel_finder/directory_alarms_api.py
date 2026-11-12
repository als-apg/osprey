"""PV directory alarm route: ``alarms``, every channel the alarm feed holds in alarm now.

Mounted under ``/api/directory`` by :mod:`.directory_api`. The route is a GET that
answers :meth:`~osprey.services.pv_directory.facts.Facts.active_alarms` over the
lifespan's ``DirectoryService``: the counts per severity over every channel, and
the active rows. ``limit=0`` answers the counts alone.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request

from osprey.services.pv_directory.facts import ACTIVE_ALARMS_MAX_ROWS, Facts

router = APIRouter()

_Many = Annotated[list[str] | None, Query()]
_Limit = Annotated[int | None, Query(ge=0, le=ACTIVE_ALARMS_MAX_ROWS)]


@router.get("/alarms")
async def alarms(request: Request, severity: _Many = None, limit: _Limit = None) -> dict[str, Any]:
    """The active alarms of the current target: ``{outcome, source, counts, total, rows, ...}``."""
    state = request.app.state
    facts = Facts(state.directory_service, state.config)
    return await facts.active_alarms(severities=tuple(severity or ()), limit=limit)
