"""
Abstract base class for archiver connectors.

Provides protocol-agnostic interfaces for retrieving historical data
from various archiver systems (EPICS Archiver Appliance, Tango HDB++, LabVIEW, etc.).

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class ArchiverMetadata:
    """Metadata about archived PV."""

    pv_name: str
    is_archived: bool
    archival_start: datetime | None = None
    archival_end: datetime | None = None
    sampling_period: float | None = None
    description: str | None = None


class ArchiverConnector(ABC):
    """
    Abstract base class for archiver connectors.

    Implementations provide interfaces to different archiver systems
    using a unified API that returns pandas DataFrames.

    Example:
        >>> connector = await ConnectorFactory.create_archiver_connector()
        >>> try:
        >>>     df = await connector.get_data(
        >>>         pv_list=['BEAM:CURRENT', 'BEAM:LIFETIME'],
        >>>         start_date=datetime(2024, 1, 1),
        >>>         end_date=datetime(2024, 1, 2)
        >>>     )
        >>>     print(df.head())
        >>> finally:
        >>>     await connector.disconnect()
    """

    @abstractmethod
    async def connect(self, config: dict[str, Any]) -> None:
        """
        Establish connection to archiver.

        Args:
            config: Archiver-specific configuration

        Raises:
            ConnectionError: If connection cannot be established
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to archiver and cleanup resources."""
        pass

    @abstractmethod
    async def get_data(
        self,
        pv_list: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None,
        processing: str = "raw",
    ) -> pd.DataFrame:
        """
        Retrieve historical data for PVs.

        Args:
            pv_list: List of PV names to retrieve
            start_date: Start of time range
            end_date: End of time range
            precision_ms: Time precision in milliseconds (for downsampling)
            timeout: Optional timeout in seconds
            processing: Aggregation applied within each precision_ms bin. One of
                "raw", "mean", "min", "max", "median", "std", "count". Backends
                that aggregate server-side (EPICS) push it down; the rest apply
                it client-side. Anything else raises ValueError.

        Returns:
            A long-format DataFrame with columns ``timestamp`` (datetime64[ns, UTC]),
            ``channel`` (str), and ``value`` — not dtype-constrained: ``float64``
            when every requested channel's samples are numeric, or whatever
            dtype pandas needs to hold the mix (typically ``object``) once any
            channel is non-numeric (e.g. an enum/status PV archived as a
            string). Sorted by channel then timestamp. Channels are never
            placed on a shared index: each channel contributes only its own
            real samples (or, when ``processing`` bins them, only its own real
            per-bin aggregates), so nothing is ever manufactured — no
            forward-fill, no reindex onto a regular grid, no bin for a period
            with no samples. A PV with no data in range contributes no rows.
            An empty result is an empty frame with these columns, ``value``
            defaulting to ``float64``.

        Raises:
            ConnectionError: If archiver cannot be reached
            TimeoutError: If operation times out
            ValueError: If time range, PV names, or processing mode are
                invalid, or if a non-"raw" processing mode is requested for a
                channel whose values are non-numeric
        """
        pass

    @abstractmethod
    async def get_metadata(self, pv_name: str) -> ArchiverMetadata:
        """
        Get archiving metadata for a PV.

        Args:
            pv_name: Name of the process variable

        Returns:
            ArchiverMetadata with archiving information

        Raises:
            ConnectionError: If archiver cannot be reached
            ValueError: If PV name is invalid
        """
        pass

    @abstractmethod
    async def check_availability(self, pv_names: list[str]) -> dict[str, bool]:
        """
        Check which PVs are archived.

        Args:
            pv_names: List of PV names to check

        Returns:
            Dictionary mapping PV name to availability status
        """
        pass
