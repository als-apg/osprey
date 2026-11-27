"""channel directory alarm route: ``alarms``, every channel the alarm feed holds in alarm now.

Mounted under ``/api/directory`` by :mod:`.directory_api`. The route is a GET that
answers :meth:`~osprey.services.channel_directory.facts.Facts.active_alarms` over the
lifespan's ``DirectoryService``: the counts per severity over every channel, and
the active rows. ``limit=0`` answers the counts alone.

The route answers what the feed holds now. The lifespan starts the feed, so a
page's first read normally finds it loaded; one that arrives while the feed is
still reading its history waits :data:`FIRST_ANSWER_WAIT_S` at most and is then
answered ``unavailable`` with the reason ``alarm feed still loading``. The
page polls, and its next read has the list.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request

from osprey.services.channel_directory.facts import ACTIVE_ALARMS_MAX_ROWS, Facts

router = APIRouter()

#: The most seconds a read waits for a feed that is still loading. Long enough
#: for a feed that loads at once (a replay file, a short topic) to answer the
#: first read, short enough that a page is never held by a long history.
FIRST_ANSWER_WAIT_S = 0.25

_Many = Annotated[list[str] | None, Query()]
_Limit = Annotated[int | None, Query(ge=0, le=ACTIVE_ALARMS_MAX_ROWS)]


@router.get("/alarms")
async def alarms(request: Request, severity: _Many = None, limit: _Limit = None) -> dict[str, Any]:
    """The active alarms of the current target: ``{outcome, source, counts, total, rows, ...}``."""
    state = request.app.state
    facts = Facts(state.directory_service, state.config)
    return await facts.active_alarms(
        severities=tuple(severity or ()), limit=limit, wait=FIRST_ANSWER_WAIT_S
    )
