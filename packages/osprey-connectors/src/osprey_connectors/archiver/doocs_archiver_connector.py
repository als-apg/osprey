"""
DOOCS local history connector using doocs4py.

Provides interface to the DOOCS local histories.

Author: Frank Mayet (DESY, MXL)
Date: 2026-07-01
"""

import asyncio
from datetime import datetime
from operator import itemgetter
from typing import Any

import numpy as np
import pandas as pd

from osprey_connectors.archiver._timerange import (
    aggregate_long_frame,
    resolve_processing,
    utc_window,
)
from osprey_connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey_connectors.logger import get_logger

logger = get_logger("doocs_archiver_connector")


class DOOCSArchiverConnector(ArchiverConnector):
    """
    DOOCS local history connector.

    Provides access to local history data of a given DOOCS property if available.

    A centered moving average can be applied by supplying `avg_window` (seconds).
    It averages over a real time span, so it operates on the archived samples at
    their own irregular timestamps -- no resampling onto a uniform grid.

    Example:
        >>> config = {
        >>>     'avg_window': 20
        >>> }
        >>> connector = DOOCSArchiverConnector()
        >>> await connector.connect(config)
        >>> df = await connector.get_data(
        >>>     pv_list=['FACILITY/DEVICE/LOCATION/PROPERTY'],
        >>>     start_date=datetime(2026, 7, 1),
        >>>     end_date=datetime(2026, 7, 2)
        >>> )
    """

    def __init__(self):
        self._connected = False
        self._avg_window = None
        self._doocs4py = None
        self._timeout = 60

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Configure DOOCS environment and test connection.

        Args:
            config: Configuration with optional keys:
                - avg_window: Centered moving-average window in seconds
                  (default: None, no smoothing)
                - timeout: Default request timeout in seconds (default: 60)

        Raises:
            ImportError: If doocs4py is not installed
        """
        # Import doocs4py here and give clear error if not installed
        try:
            import doocs4py

            self._doocs4py = doocs4py
            logger.debug(
                f"DOOCS archiver connector: doocs4py version {self._doocs4py.__version__} loaded"
            )
        except ImportError:
            raise ImportError("doocs4py is required for the DOOCS connector.") from None

        # Test connection using a doocs4py.names call, listing all FACILITYs
        try:
            facilities = [f[1] for f in self._doocs4py.names("*")]
            logger.debug(
                "DOOCS archiver connector: ENS connection successful."
                f"Available FACILITIEs: {len(facilities)}"
            )
        except Exception:
            raise Exception("DOOCS archiver connector failed to connect to the ENS.") from None

        self._avg_window = config.get("avg_window", None)
        self._timeout = config.get("timeout", 60)

        self._connected = True
        logger.debug("DOOCS archiver connector initialized")

    async def disconnect(self) -> None:
        """Cleanup archiver."""
        self._connected = False
        self._doocs4py = None
        logger.debug("DOOCS archiver connector disconnected")

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
        Retrieve historical data from the DOOCS local histories.

        Args:
            pv_list: List of DOOCS property addresses
            start_date: Start of time range
            end_date: End of time range
            precision_ms: Time precision (affects downsampling)
            timeout: Timeout in seconds. ``None`` falls back to the connector's
                configured default rather than waiting indefinitely.
            processing: Aggregation applied within each precision_ms bin. One of
                "raw", "mean", "min", "max", "median", "std", "count". Applied
                client-side via pandas resampling. Anything else raises ValueError.

        Returns:
            The canonical long frame — see :meth:`ArchiverConnector.get_data`.

        Raises:
            RuntimeError: If archiver not connected, or a DOOCS property's
                history cannot be read
            TypeError: If start_date or end_date are not datetime objects
            TimeoutError: If the request times out
            ValueError: If a non-raw processing mode is requested for a
                channel that carries non-numeric values
        """

        # asyncio.wait_for(timeout=None) would block indefinitely.
        timeout = timeout if timeout is not None else self._timeout

        if not self._connected:
            raise RuntimeError("DOOCS archiver not connected")

        # A naive datetime's .timestamp() resolves against the *host* zone;
        # convert explicitly so the window means the same thing on every box.
        start_utc, end_utc = utc_window(start_date, end_date)
        resolved = resolve_processing(processing, precision_ms)

        def fetch_all() -> dict[str, pd.Series]:
            data = {}
            for add in pv_list:
                hist_data_dict = self._read_history(
                    add,
                    start_utc.timestamp(),
                    end_utc.timestamp(),
                    self._avg_window,
                )
                if hist_data_dict is None:
                    raise RuntimeError(f"DOOCS archiver connector: Cannot read history for {add}")
                timestamps = pd.to_datetime(hist_data_dict.get("time", []), unit="s", utc=True)
                values = hist_data_dict.get("data", [])
                data[add] = pd.Series(values, index=timestamps, name=add)
            return data

        try:
            series_dict = await asyncio.wait_for(asyncio.to_thread(fetch_all), timeout=timeout)

            data = aggregate_long_frame(series_dict, resolved)

            logger.debug(
                f"Retrieved DOOCS archiver data: {len(data)} rows "
                f"across {len(pv_list)} DOOCS properties"
            )
            return data

        except TimeoutError as e:
            raise TimeoutError(f"DOOCS archiver request timed out after {timeout}s") from e

    async def get_metadata(self, pv_name: str) -> ArchiverMetadata:
        """Get archiver metadata."""
        return ArchiverMetadata(
            pv_name=pv_name,
            is_archived=True,
            description=f"DOOCS archived PV: {pv_name}",
        )

    async def check_availability(self, pv_names: list[str]) -> dict[str, bool]:
        """Check availability based on .HIST property name extension."""
        if not self._connected or self._doocs4py is None:
            return dict.fromkeys(pv_names, False)

        avail = {}
        for add in pv_names:
            hist_address = add
            if not hist_address.endswith(".HIST"):
                hist_address = add + ".HIST"
            try:
                if self._doocs4py.names(hist_address):
                    avail[add] = True
                else:
                    avail[add] = False
            except Exception:
                avail[add] = False

        return avail

    def _read_history(
        self,
        address: str,
        start_time: float,
        end_time: float,
        avg_window: float | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Read history data from DOOCS using doocs4py. Timestamps are in UNIX format.

        Parameters
        ----------
        address:
            DOOCS history address. ".HIST" is appended automatically if missing.
        start_time, end_time:
            Time range in UNIX timestamps.
        avg_window:
            Length (in seconds) of a centered moving average over the archived
            samples at their own irregular timestamps.

        Returns
        -------
        A dict with "time" and "data" arrays holding the smoothed series if
        ``avg_window`` asked for one and the raw samples otherwise, or None if
        no data was retrieved. Every timestamp is a real archived one.
        """

        start_ts: int = int(start_time)
        stop_ts: int = int(end_time)

        try:
            if not address.endswith(".HIST"):
                address = address + ".HIST"
            hist_address = self._doocs4py.Address(address)

            current_stop = stop_ts
            all_data = []

            while True:
                ttii = self._doocs4py.types.TTII(
                    start_ts, current_stop, 256, 0
                )  # 256 means Archiver
                result = self._doocs4py.get(hist_address, ttii)

                # Check if the newly fetched chunk is empty to prevent infinite loops
                if not result.value:
                    break

                chunk = result.value
                all_data.extend(chunk)

                oldest_in_chunk = chunk[0][0]

                # Failsafe to break if the timestamp stops advancing
                if current_stop == oldest_in_chunk:
                    break

                current_stop = oldest_in_chunk

                if current_stop <= start_ts:
                    break

            if not all_data:
                return None

            n_entries = len(all_data)
            raw_time = np.fromiter(map(itemgetter(0), all_data), dtype=float, count=n_entries)
            raw_data = np.fromiter(map(itemgetter(3), all_data), dtype=float, count=n_entries)

            # Remove duplicates and ensure monotonically increasing time.
            # np.unique returns sorted unique values, which the routines below require.
            raw_time, unique_indices = np.unique(raw_time, return_index=True)
            raw_data = raw_data[unique_indices]

            # Optional centered moving average over the real samples. The
            # time-based window handles irregular spacing and keeps the real
            # timestamps, returning exactly one value per input sample.
            out_data = (
                pd.Series(raw_data, index=pd.to_datetime(raw_time, unit="s"))
                .rolling(pd.Timedelta(seconds=avg_window), center=True)
                .mean()
                .to_numpy()
                if avg_window is not None and avg_window > 0
                else raw_data
            )

            return {"time": raw_time, "data": out_data}

        except Exception:
            return None
