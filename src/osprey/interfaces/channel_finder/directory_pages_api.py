"""PV and IOC page routes: ``pv/{name}``, ``ioc/{name}`` and ``logbook/availability``.

Mounted under ``/api/directory`` by :mod:`.directory_api`.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
