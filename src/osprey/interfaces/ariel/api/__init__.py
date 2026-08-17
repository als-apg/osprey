"""ARIEL Web API.

REST API endpoints for search, entry management, and status.
"""

from osprey.interfaces.ariel.api.routes import router
from osprey.interfaces.ariel.api.schemas import (
    EntriesListResponse,
    EntryCreateRequest,
    EntryCreateResponse,
    EntryResponse,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)

__all__ = [
    "router",
    "EntriesListResponse",
    "EntryCreateRequest",
    "EntryCreateResponse",
    "EntryResponse",
    "SearchRequest",
    "SearchResponse",
    "StatusResponse",
]
