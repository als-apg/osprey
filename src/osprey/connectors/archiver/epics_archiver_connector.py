"""
EPICS Archiver Appliance connector using direct HTTP calls.

Provides interface to EPICS Archiver Appliance for historical data retrieval.
Refactored from existing archiver integration code.

"""

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

import pandas as pd

from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey.utils.logger import get_logger

logger = get_logger("epics_archiver_connector")


class EPICSArchiverConnector(ArchiverConnector):
    """
    EPICS Archiver Appliance connector using direct HTTP calls.

    Provides access to historical PV data from EPICS Archiver Appliance
    via direct HTTP requests using Python stdlib.

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
        self._url = None

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Initialize archiver connection.

        Args:
            config: Configuration with keys:
                - url: Archiver URL (required)
                - timeout: Default timeout in seconds (default: 60)

        Raises:
            ValueError: If URL is not provided
        """
        archiver_url = config.get("url")
        if not archiver_url:
            raise ValueError("archiver URL is required for EPICS archiver")

        self._url = archiver_url
        self._timeout = config.get("timeout", 60)
        self._connected = True

        logger.debug(f"EPICS Archiver connector initialized: {archiver_url}")

    async def disconnect(self) -> None:
        """Cleanup archiver connection."""
        self._url = None
        self._connected = False
        logger.debug("EPICS Archiver connector disconnected")

    def _fetch_single_pv(self, pv: str, start_str: str, end_str: str) -> pd.Series:
        """
        Fetch archived data for a single PV via direct HTTP.

        Args:
            pv: PV name (may include processing operators, e.g. mean_60(SR:DCCT))
            start_str: ISO 8601 UTC start time string
            end_str: ISO 8601 UTC end time string

        Returns:
            pd.Series with DatetimeIndex; empty Series if no data

        Raises:
            ValueError: If PV has array-valued samples (waveform PV)
            ConnectionError: If the HTTP request fails
        """
        params = urllib.parse.urlencode(
            {"pv": pv, "from": start_str, "to": end_str, "fetchLatestMetadata": "true"}
        )
        url = f"{self._url}/retrieval/data/getData.json?{params}"
        req = urllib.request.Request(url, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot connect to archiver at {self._url}: {e}") from e

        # Empty response: [] or [{"meta": ..., "data": []}]
        if not payload:
            return pd.Series(dtype=float, name=pv)
        data_points = payload[0].get("data", [])
        if not data_points:
            return pd.Series(dtype=float, name=pv)

        # Check for waveform PV (array-valued val)
        if isinstance(data_points[0].get("val"), list):
            raise ValueError(f"Waveform PVs not supported: {pv}")

        timestamps = pd.to_datetime(
            [dp["secs"] for dp in data_points], unit="s"
        ) + pd.to_timedelta([dp["nanos"] for dp in data_points], unit="ns")
        values = [dp["val"] for dp in data_points]

        return pd.Series(values, index=timestamps, name=pv)

    async def get_data(
        self,
        pv_list: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None,
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

        if not self._connected:
            raise RuntimeError("Archiver not connected")

        # Validate inputs
        if not isinstance(start_date, datetime):
            raise TypeError(f"start_date must be a datetime object, got {type(start_date)}")
        if not isinstance(end_date, datetime):
            raise TypeError(f"end_date must be a datetime object, got {type(end_date)}")

        def fetch_data():
            """Synchronous data fetch function."""
            return self._archiver_client.match_data(
                pv_list=pv_list, precision=precision_ms, start=start_date, end=end_date, verbose=0
            )

        try:
            # Use asyncio.wait_for for timeout, asyncio.to_thread for async execution
            data = await asyncio.wait_for(asyncio.to_thread(fetch_data), timeout=timeout)

            if not isinstance(data, pd.DataFrame):
                raise ValueError(f"Unexpected data format: {type(data)}")

            # archivertools returns DataFrame with columns [secs, nanos, PV1, PV2, ...]
            # Convert secs/nanos to DatetimeIndex and drop those columns
            if "secs" in data.columns and "nanos" in data.columns:
                # Combine seconds and nanoseconds into datetime
                timestamps = pd.to_datetime(data["secs"], unit="s") + pd.to_timedelta(
                    data["nanos"], unit="ns"
                )
                data = data.drop(columns=["secs", "nanos"])
                data.index = timestamps
            elif not isinstance(data.index, pd.DatetimeIndex):
                # Fallback: try to convert existing index to datetime
                data.index = pd.to_datetime(data.index)

            logger.debug(f"Retrieved archiver data: {len(data)} points for {len(pv_list)} PVs")
            return data

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
                raise ConnectionError(f"Network connectivity issue with archiver: {e}") from e
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
            description=f"EPICS Archived PV: {pv_name}",
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
