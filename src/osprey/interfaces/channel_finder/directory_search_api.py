"""PV directory search routes: ``search``, ``export.csv`` and ``count``.

Mounted under ``/api/directory`` by :mod:`.directory_api`.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
