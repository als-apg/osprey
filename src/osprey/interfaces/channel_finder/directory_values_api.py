"""The current-value route, ``POST values``, read through the supervised child.

Mounted under ``/api/directory`` by :mod:`.directory_api`.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
