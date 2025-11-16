"""
EPICS Archiver Appliance connector using archivertools.

Provides interface to EPICS Archiver Appliance for historical data retrieval.
Refactored from existing archiver integration code.

Related to Issue #18 - Control System Abstraction (Layer 2 - EPICS Archiver Implementation)
"""

import asyncio
from datetime import datetime
from typing import Any

import pandas as pd

from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey.utils.logger import get_logger

logger = get_logger("epics_archiver_connector")


class EPICSArchiverConnector(ArchiverConnector):
    """
    EPICS Archiver Appliance connector using archivertools.

    Provides access to historical PV data from EPICS Archiver Appliance
    via the archivertools Python library.

    Example:
        >>> config = {
        >>>     'url': 'https://archiver.als.lbl.gov:8443',
        >>>     'timeout': 60
        >>> }
        >>> connector = EPICSArchiverConnector()
        >>> await connector.connect(config)
        >>> df = await connector.get_data(
        >>>     pv_list=['BEAM:CURRENT'],
        >>>     start_date=datetime(2024, 1, 1),
        >>>     end_date=datetime(2024, 1, 2)
        >>> )
    """

    def __init__(self):
        self._connected = False
        self._archiver_client = None

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Initialize archiver client.

        Args:
            config: Configuration with keys:
                - url: Archiver URL (required)
                - timeout: Default timeout in seconds (default: 60)

        Raises:
            ImportError: If archivertools is not installed
            ValueError: If URL is not provided
        """
        try:
            from archivertools import ArchiverClient
            self._ArchiverClient = ArchiverClient
        except ImportError as e:
            raise ImportError(
                "archivertools is required for EPICS archiver. "
                "Install with: pip install archivertools"
            ) from e

        archiver_url = config.get('url')
        if not archiver_url:
            raise ValueError("archiver URL is required for EPICS archiver")

        try:
            self._archiver_client = self._ArchiverClient(archiver_url=archiver_url)
        except Exception as e:
            raise ConnectionError(f"ArchiverClient initialization failed: {e}") from e

        self._timeout = config.get('timeout', 60)
        self._connected = True

        logger.debug(f"EPICS Archiver connector initialized: {archiver_url}")

    async def disconnect(self) -> None:
        """Cleanup archiver connection."""
        self._archiver_client = None
        self._connected = False
        logger.debug("EPICS Archiver connector disconnected")

    async def get_data(
        self,
        pv_list: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None
    ) -> pd.DataFrame:
        """
        Retrieve historical data from EPICS archiver.

        Args:
            pv_list: List of PV names to retrieve
            start_date: Start of time range
            end_date: End of time range
            precision_ms: Time precision in milliseconds (for downsampling)
            timeout: Optional timeout in seconds

        Returns:
            DataFrame with datetime index and PV columns

        Raises:
            RuntimeError: If archiver not connected
            TimeoutError: If operation times out
            ConnectionError: If archiver cannot be reached
            ValueError: If data format is unexpected
        """
        timeout = timeout or self._timeout

        if not self._archiver_client:
            raise RuntimeError("Archiver not connected")

        # Validate inputs
        if not isinstance(start_date, datetime):
            raise TypeError(f"start_date must be a datetime object, got {type(start_date)}")
        if not isinstance(end_date, datetime):
            raise TypeError(f"end_date must be a datetime object, got {type(end_date)}")

        def fetch_data():
            """Synchronous data fetch function."""
            return self._archiver_client.match_data(
                pv_list=pv_list,
                precision=precision_ms,
                start=start_date,
                end=end_date,
                verbose=0
            )

        try:
            # Use asyncio.wait_for for timeout, asyncio.to_thread for async execution
            data = await asyncio.wait_for(
                asyncio.to_thread(fetch_data),
                timeout=timeout
            )

            # Ensure datetime index
            if isinstance(data, pd.DataFrame) and hasattr(data, 'index'):
                data.index = pd.to_datetime(data.index)

                logger.debug(
                    f"Retrieved archiver data: {len(data)} points "
                    f"for {len(pv_list)} PVs"
                )
                return data
            else:
                raise ValueError(f"Unexpected data format: {type(data)}")

        except TimeoutError as e:
            raise TimeoutError(f"Archiver request timed out after {timeout}s") from e
        except ConnectionRefusedError as e:
            raise ConnectionError(
                "Cannot connect to the archiver. "
                "Please check connectivity and SSH tunnels (if required)."
            ) from e
        except Exception as e:
            error_msg = str(e).lower()
            if "connection" in error_msg:
                raise ConnectionError(
                    f"Network connectivity issue with archiver: {e}"
                ) from e
            raise

    async def get_metadata(self, pv_name: str) -> ArchiverMetadata:
        """
        Get archiving metadata for a PV.

        Note: archivertools doesn't expose metadata API directly,
        so this returns basic information.

        Args:
            pv_name: Name of the process variable

        Returns:
            ArchiverMetadata with basic archiving information
        """
        # Basic implementation - could be enhanced with direct archiver API calls
        return ArchiverMetadata(
            pv_name=pv_name,
            is_archived=True,  # Assume true if no error
            description=f"EPICS Archived PV: {pv_name}"
        )

    async def check_availability(self, pv_names: list[str]) -> dict[str, bool]:
        """
        Check which PVs are archived.

        Note: Basic implementation that assumes all PVs are archived.
        Could be enhanced with actual archiver API calls.

        Args:
            pv_names: List of PV names to check

        Returns:
            Dictionary mapping PV name to availability status
        """
        # Basic implementation - could be enhanced with archiver API calls
        return dict.fromkeys(pv_names, True)

