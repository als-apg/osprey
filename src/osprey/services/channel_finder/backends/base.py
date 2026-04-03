"""Abstract base class and data models for PV info backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PVRecord:
    """Metadata record for a single process variable."""

    name: str
    record_type: str = ""
    description: str = ""
    host: str = ""
    ioc: str = ""
    units: str = ""
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Paginated search result from a PV info backend."""

    records: list[PVRecord]
    total_count: int
    has_more: bool
    page: int
    page_size: int


class PVInfoBackend(ABC):
    """Abstract interface for PV metadata backends.

    Implementations provide search and metadata retrieval for process
    variables from a live database or API.
    """

    @abstractmethod
    async def search(
        self,
        pattern: str,
        *,
        record_type: str | None = None,
        ioc: str | None = None,
        description_contains: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> SearchResult:
        """Search for PVs matching a glob pattern with optional filters.

        Args:
            pattern: Glob pattern to match PV names (e.g., ``SR:BPM:*``).
            record_type: Filter by EPICS record type (e.g., ``ai``, ``ao``).
            ioc: Filter by IOC name.
            description_contains: Filter by substring in description.
            page: Page number (1-indexed).
            page_size: Results per page (max 200).
        """

    @abstractmethod
    async def get_metadata(self, pv_names: list[str]) -> list[PVRecord]:
        """Get detailed metadata for specific PV names.

        Args:
            pv_names: List of exact PV names (max 100).

        Returns:
            Records for each found PV. Missing PVs are omitted.
        """
