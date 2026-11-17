"""PV directory archive route: ``archive``, what the current target's archive holds.

Mounted under ``/api/directory`` by :mod:`.directory_api`. The route is a GET that
answers :meth:`~osprey.services.pv_directory.facts.Facts.archive_summary` over the
lifespan's ``DirectoryService``. It reads the archive and its configuration as
rendered; nothing here writes either.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from osprey.services.pv_directory.facts import Facts

router = APIRouter()


@router.get("/archive")
async def archive(request: Request) -> dict[str, Any]:
    """The archive summary of the current target: ``{outcome, source, archived, by_ioc, ...}``."""
    state = request.app.state
    return await Facts(state.directory_service, state.config).archive_summary()
