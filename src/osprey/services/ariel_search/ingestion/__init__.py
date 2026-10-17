"""ARIEL ingestion adapters.

This module provides facility-specific adapters for logbook ingestion.
"""

from osprey.services.ariel_search.ingestion.adapters import get_adapter
from osprey.services.ariel_search.ingestion.base import FacilityAdapter

__all__ = [
    "FacilityAdapter",
    "get_adapter",
]
