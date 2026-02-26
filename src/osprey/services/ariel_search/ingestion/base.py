"""Base ingestion adapter interface.

This module defines the abstract base class for ARIEL ingestion adapters.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.models import (
        EnhancedLogbookEntry,
        FacilityEntryCreateRequest,
    )


class FacilityAdapter(ABC):
    """Abstract base class for ingestion adapters.

    Each facility implements their own adapter to convert facility-specific
    logbook formats into the ARIEL schema. Adapters support both reading
    (fetch_entries) and optionally writing (create_entry) to facility logbooks.

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

    @property
    def supports_write(self) -> bool:
        """Whether this adapter supports creating entries in the facility logbook.

        Override in subclasses that implement write support.
        """
        return False

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

    async def create_entry(self, request: "FacilityEntryCreateRequest") -> str:
        """Create an entry in the facility logbook.

        Args:
            request: Entry creation request with subject, details, etc.

        Returns:
            The facility-assigned entry ID.

        Raises:
            NotImplementedError: If this adapter does not support writes.
            IngestionError: If the write fails.
        """
        raise NotImplementedError(
            f"{self.source_system_name} adapter does not support creating entries"
        )

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


# Backwards-compatible alias
BaseAdapter = FacilityAdapter
