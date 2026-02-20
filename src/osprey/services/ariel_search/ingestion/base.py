"""Base ingestion adapter interface.

This module defines the abstract base class for ARIEL ingestion adapters.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.models import EnhancedLogbookEntry


class BaseAdapter(ABC):
    """Abstract base class for ingestion adapters.

    Each facility implements their own adapter to convert facility-specific
    logbook formats into the ARIEL schema.

    Attributes:
        config: ARIEL configuration
    """

    def __init__(self, config: "ARIELConfig") -> None:
        """Initialize the adapter with configuration.

        Args:
            config: ARIEL configuration
        """
        self.config = config

    @property
    @abstractmethod
    def source_system_name(self) -> str:
        """Return the source system identifier.

        Examples: 'ALS eLog', 'JLab Logbook', 'ORNL Logbook'
        """

    @abstractmethod
    def fetch_entries(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> AsyncIterator["EnhancedLogbookEntry"]:
        """Fetch entries from the source system.

        Args:
            since: Only fetch entries after this timestamp
            until: Only fetch entries before this timestamp
            limit: Maximum number of entries to fetch

        Yields:
            EnhancedLogbookEntry objects with base fields populated.
            Enhancement fields are added later by enhancement modules.

        Raises:
            IngestionError: If connection to source system fails
        """

    async def count_entries(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int | None:
        """Count entries available in the source system.

        This is an optional method - adapters may return None if counting
        is not supported or too expensive.

        Args:
            since: Only count entries after this timestamp
            until: Only count entries before this timestamp

        Returns:
            Total count of entries, or None if not available
        """
        return None
