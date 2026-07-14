"""
DOOCS local history connector using doocs4py.

Provides interface to the DOOCS local histories.

Author: Frank Mayet (DESY, MXL)
Date: 2026-07-01
"""

import asyncio
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey.utils.logger import get_logger

logger = get_logger("doocs_archiver_connector")


class DOOCSArchiverConnector(ArchiverConnector):
    """
    DOOCS local history connector.

    Provides access to local history data of a given DOOCS property if available.

    A moving average can be applied, by supplying `avg_window`. If not None, it forces
    a uniform grid, so the window has a well-defined constant dt to operate on.

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

    async def connect(self, config: dict[str, Any]) -> None:
        """
        Configure DOOCS environment and test connection.

        Args:
            config: No config needed for DOOCS

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

        self._connected = True
        logger.debug("DOOCS archiver connector initialized")

    async def disconnect(self) -> None:
        """Cleanup archiver."""
        self._connected = False
        logger.debug("DOOCS archiver connector disconnected")

    async def get_data(
        self,
        pv_list: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic historical data.

        Args:
            pv_list: List of DOOCS property addresses
            start_date: Start of time range
            end_date: End of time range
            precision_ms: Time precision (affects downsampling)
            timeout: Timeout in seconds

        Returns:
            DataFrame with datetime index and columns for each DOOCS property
        """

        if not self._connected:
            raise RuntimeError("DOOCS archiver not connected")

        # Validate inputs
        if not isinstance(start_date, datetime):
            raise TypeError(f"start_date must be a datetime object, got {type(start_date)}")
        if not isinstance(end_date, datetime):
            raise TypeError(f"end_date must be a datetime object, got {type(end_date)}")

        duration = (end_date - start_date).total_seconds()

        # Limit number of points for performance
        # Use precision_ms to determine sampling
        num_points = min(int(duration / (precision_ms / 1000.0)), 10000)
        num_points = max(num_points, 10)  # At least 10 points

        def fetch_all() -> dict[str, pd.Series]:
            data = {}
            for add in pv_list:
                hist_data_dict = self._read_history(
                    add,
                    start_date.timestamp(),
                    end_date.timestamp(),
                    num_points,
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

            if len(pv_list) == 1:
                data = pd.DataFrame(series_dict)
            else:
                # Align multi-PV series to a common precision_ms grid via ffill
                resolution = f"{max(1, precision_ms)}ms"
                grid = pd.date_range(start=start_date, end=end_date, freq=resolution)
                # Archiver timestamps are always UTC; ensure the grid matches
                if grid.tz is None:
                    grid = grid.tz_localize("UTC")
                aligned = {}
                for pv, series in series_dict.items():
                    if series.empty:
                        aligned[pv] = pd.Series(index=grid, dtype=float, name=pv)
                    else:
                        reindexed = series.reindex(series.index.union(grid)).ffill()
                        aligned[pv] = reindexed.reindex(grid)
                data = pd.DataFrame(aligned)

            logger.debug(
                f"Retrieved DOOCS archiver data: {len(data)} points "
                f"for {len(pv_list)} DOOCS properties"
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
        if not self._connected:
            dict.fromkeys(pv_names, False)

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
        max_points: int | None = None,
        avg_window: float | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Read history data from DOOCS using doocs4py. Timestamps are in UNIX format.

        Parameters
        ----------
        address:
            DOOCS history address. ".HIST" is appended automatically if missing.
        start_time, end_time:
            Time range in UNIX timestamps.
        max_points:
            If given and the number of retrieved samples exceeds it, the data is
            resampled onto a uniform time grid (constant dt) of at most this many
            points using a zero-order hold. If max_points exceeds the number of
            available points, the grid falls back to the full resolution (no
            upsampling).
        avg_window:
            Length (in seconds) of a centered moving average applied to the
            resampled data. Supplying avg_window forces a uniform grid even when
            max_points is None (falling back to full resolution), so the window has
            a well-defined constant dt to operate on.

        Returns
        -------
        A dict with "time" and "data" arrays holding the most processed series
        available (smoothed > reduced > raw), or None if no data was retrieved.
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

            raw_time = np.array([entry[0] for entry in all_data], dtype=float)
            raw_data = np.array([entry[3] for entry in all_data], dtype=float)

            # Remove duplicates and ensure monotonically increasing time.
            # np.unique returns sorted unique values, which the routines below require.
            raw_time, unique_indices = np.unique(raw_time, return_index=True)
            raw_data = raw_data[unique_indices]

            # These will hold the resampled / smoothed series if produced.
            reduced_time = reduced_data = None
            smooth_data = None

            # Build a uniform grid if either a point limit or an averaging window
            # is requested. The grid never exceeds the available point count, so
            # max_points=None (with avg_window set) falls back to full resolution.
            if max_points is not None or (avg_window is not None and avg_window > 0):
                if max_points is None:
                    n_points = raw_time.size
                else:
                    n_points = min(max_points, raw_time.size)

                reduced_time = np.linspace(raw_time[0], raw_time[-1], n_points)

                # Zero-order hold: most recent sample at or before each grid point.
                idx = np.searchsorted(raw_time, reduced_time, side="right") - 1
                idx = np.clip(idx, 0, raw_time.size - 1)
                reduced_data = raw_data[idx]

                # Optional centered moving average over the constant-dt grid.
                # Requires at least 2 grid points to define dt.
                if avg_window is not None and avg_window > 0 and n_points > 1:
                    dt = reduced_time[1] - reduced_time[0]
                    win = max(1, int(round(avg_window / dt)))
                    if win > 1:
                        kernel = np.ones(win) / win
                        smooth_data = np.convolve(reduced_data, kernel, mode="same")
                        # Correct edge underweighting from zero-padding in 'same'.
                        norm = np.convolve(np.ones_like(reduced_data), kernel, mode="same")
                        smooth_data = smooth_data / norm

            # Build metadata describing the request and the retrieved raw data.
            metadata = {
                "raw_count": int(raw_time.size),
                "max_points": max_points,
                "avg_window": avg_window,
                "start_iso": np.datetime64(int(raw_time[0]), "s").astype(str),
                "end_iso": np.datetime64(int(raw_time[-1]), "s").astype(str),
            }

            # Return the most processed series available, along with metadata.
            if smooth_data is not None:
                out_time, out_data = reduced_time, smooth_data
            elif reduced_data is not None:
                out_time, out_data = reduced_time, reduced_data
            else:
                out_time, out_data = raw_time, raw_data

            return {"time": out_time, "data": out_data, "metadata": metadata}

        except Exception:
            return None
