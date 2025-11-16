"""
Abstract base class for archiver connectors.

Provides protocol-agnostic interfaces for retrieving historical data
from various archiver systems (EPICS Archiver Appliance, Tango HDB++, LabVIEW, etc.).

Related to Issue #18 - Control System Abstraction (Layer 2)
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
        timeout: int | None = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data for PVs.

        Args:
            pv_list: List of PV names to retrieve
            start_date: Start of time range
            end_date: End of time range
            precision_ms: Time precision in milliseconds (for downsampling)
            timeout: Optional timeout in seconds

        Returns:
            DataFrame with datetime index and PV columns
            Each column contains the time series for one PV

        Raises:
            ConnectionError: If archiver cannot be reached
            TimeoutError: If operation times out
            ValueError: If time range or PV names are invalid
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

