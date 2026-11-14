"""PV and IOC page routes: ``pv/{name}``, ``ioc/{name}`` and ``logbook/availability``.

Mounted under ``/api/directory`` by :mod:`.directory_api`. The PV and IOC pages
answer the facts of :class:`~osprey.services.pv_directory.facts.Facts` over the
lifespan's ``DirectoryService``. Logbook availability only says whether the
current target has a logbook source; the panel reads the logbook itself.

``{name}`` is one percent-encoded segment, so a name holding ``/`` arrives whole.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from osprey.services.pv_directory._connectors import (
    DIRECTORY_CONNECTORS_AVAILABLE,
    FLOOR_REASON,
    NO_SOURCE_REASON,
    as_payload,
    logbook_source,
)
from osprey.services.pv_directory.facts import Facts

router = APIRouter()


def _facts(request: Request) -> Facts:
    state = request.app.state
    return Facts(state.directory_service, state.config)


@router.get("/pv/{name:path}")
async def pv_page(name: str, request: Request) -> dict[str, Any]:
    """Every configured section about the PV called *name*: ``{name, outcome, sections}``."""
    return await _facts(request).pv_facts(name)


@router.get("/ioc/{name:path}")
async def ioc_page(name: str, request: Request) -> dict[str, Any]:
    """The ``IocFacts`` of the IOC called *name*."""
    return await _facts(request).ioc_facts(name)


@router.get("/logbook/availability")
async def logbook_availability(request: Request) -> dict[str, Any]:
    """Whether the current target has a logbook: ``{available, source, reason?}``.

    The flag is the descriptor's ``ariel``, so the two cannot disagree.
    """
    from osprey.interfaces.channel_finder import directory_api

    if not DIRECTORY_CONNECTORS_AVAILABLE:
        return {"available": False, "source": None, "reason": FLOOR_REASON}
    config = request.app.state.config
    ariel = directory_api.ariel_configured(config)
    target = (await request.app.state.directory_service.count())["source"]["target"]
    source = logbook_source(target, ariel_configured=ariel)
    if source is None:
        return {"available": False, "source": None, "reason": NO_SOURCE_REASON}
    return {"available": True, "source": as_payload(source)}
