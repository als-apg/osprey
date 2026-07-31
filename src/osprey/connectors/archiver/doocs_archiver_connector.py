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

from osprey.connectors.archiver._timerange import (
    aggregate_series,
    long_frame,
    require_datetime,
    resolve_processing,
    to_utc,
)
from osprey.connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey.utils.logger import get_logger

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
            The canonical long frame — see :meth:`ArchiverConnector.get_data`
            for the full contract (columns, dtypes, ordering, and the rule that
            nothing is ever manufactured).

        Raises:
            RuntimeError: If archiver not connected, or a DOOCS property's
                history cannot be read
            TypeError: If start_date or end_date are not datetime objects
            TimeoutError: If the request times out
            ValueError: If a non-raw processing mode is requested for a
                channel that carries non-numeric values (see
                :func:`~osprey.connectors.archiver._timerange.aggregate_series`)
        """

        # An omitted timeout means "the connector's default", not "wait forever":
        # asyncio.wait_for(timeout=None) blocks indefinitely, so an unresponsive
        # ENS would hang the caller. Matches EPICS and MongoDB.
        timeout = timeout if timeout is not None else self._timeout

        if not self._connected:
            raise RuntimeError("DOOCS archiver not connected")

        require_datetime(start_date, end_date)

        # A naive datetime's .timestamp() resolves against the *host* zone; convert
        # explicitly so the window means the same thing on every deploy box. Deriving
        # duration from the converted bounds (rather than the raw start_date/end_date)
        # also tolerates a mixed naive/aware pair, which would otherwise raise TypeError
        # on subtraction before either bound got a chance to be normalized.
        start_utc = to_utc(start_date)
        end_utc = to_utc(end_date)
        resolved = resolve_processing(processing, precision_ms)

        # max_points=None for every mode, "raw" included: it skips
        # _read_history's own zero-order-hold decimation (a np.linspace grid
        # plus nearest-sample-at-or-before hold) so every real archived sample
        # reaches aggregate_series below. That decimation forces a regular grid
        # and forward-fills onto it -- both explicitly prohibited by this
        # framework's "nothing is manufactured" contract, and for "raw" it
        # meant only 2 of 10,000 returned timestamps were ever real archived
        # ones. A real aggregate mode needs the underlying raw samples to
        # aggregate *over* regardless (finding #1 -- the bug this whole fix
        # pass exists for -- would otherwise starve mean/count/std down to a
        # single value per bin); "raw" now needs the same thing for the same
        # reason: decimate_raw (inside aggregate_series) keeps each bin's own
        # real last sample, at its own real timestamp, which requires the real
        # samples in the first place.
        def fetch_all() -> dict[str, pd.Series]:
            data = {}
            for add in pv_list:
                hist_data_dict = self._read_history(
                    add,
                    start_utc.timestamp(),
                    end_utc.timestamp(),
                    None,
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

            # Each channel is aggregated over its own real samples independently
            # of every other requested channel -- aggregate_series drops any bin
            # with no samples rather than emitting one, so no grid alignment or
            # bin-width floor is needed regardless of how differently two
            # channels are sampled. "raw" decimates via aggregate_series's
            # decimate_raw path: one real sample per precision_ms bin, at its
            # own real timestamp.
            series = {
                pv: aggregate_series(s, precision_ms, resolved) for pv, s in series_dict.items()
            }
            data = long_frame(series)

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
            Length (in seconds) of a centered moving average. The window is a real
            duration, so it applies to whatever series is in hand -- the max_points
            grid if one was requested, otherwise the archived samples at their own
            irregular timestamps. It never introduces a grid of its own, and it
            returns exactly one value per input sample, at that sample's own time.

        Returns
        -------
        A dict with "time" and "data" arrays holding the most processed series
        available (smoothed > reduced > raw), or None if no data was retrieved.
        Timestamps are the real archived ones unless max_points requested a grid.
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

            # Build a uniform grid only when a point limit is requested. The grid
            # used to be built for avg_window too, because the moving average was
            # a fixed-width convolution kernel that needed a constant dt to turn
            # a duration into a sample count. A time-based rolling window needs
            # no such thing, and the grid was never free: its timestamps are
            # np.linspace positions rather than archived ones, and its values are
            # forward-filled onto them by the zero-order hold below -- both
            # prohibited by this framework's "nothing is manufactured" contract.
            # get_data always passes max_points=None, so it no longer reaches
            # this branch at all; an explicit max_points is a caller asking for
            # a fixed-size grid, which is what they get.
            if max_points is not None:
                n_points = min(max_points, raw_time.size)
                reduced_time = np.linspace(raw_time[0], raw_time[-1], n_points)

                # Zero-order hold: most recent sample at or before each grid point.
                idx = np.searchsorted(raw_time, reduced_time, side="right") - 1
                idx = np.clip(idx, 0, raw_time.size - 1)
                reduced_data = raw_data[idx]

            # Optional centered moving average over whichever series we have: the
            # decimated grid when max_points asked for one, the real samples
            # otherwise. The window is a real duration, so irregular sample
            # spacing is fine -- each output value averages the samples that
            # actually fall within avg_window of its own timestamp, and every
            # timestamp returned is one an archived sample really carries.
            #
            # An offset window also shrinks at the edges rather than zero-padding
            # (no renormalization pass needed) and always returns exactly one
            # value per input sample: np.convolve(mode="same") returned
            # max(n_samples, window_width), so an avg_window wider than the
            # queried span used to hand back `data` longer than `time` and blow
            # up in get_data with a length mismatch.
            if avg_window is not None and avg_window > 0:
                src_time = reduced_time if reduced_time is not None else raw_time
                src_data = reduced_data if reduced_data is not None else raw_data
                smooth_data = (
                    pd.Series(src_data, index=pd.to_datetime(src_time, unit="s"))
                    .rolling(pd.Timedelta(seconds=avg_window), center=True)
                    .mean()
                    .to_numpy()
                )

            # Build metadata describing the request and the retrieved raw data.
            metadata = {
                "raw_count": int(raw_time.size),
                "max_points": max_points,
                "avg_window": avg_window,
                "start_iso": np.datetime64(int(raw_time[0]), "s").astype(str),
                "end_iso": np.datetime64(int(raw_time[-1]), "s").astype(str),
            }

            # Return the most processed series available, along with metadata.
            # Smoothing keeps whichever timestamps it operated on -- the real
            # ones unless max_points asked for a grid.
            if smooth_data is not None:
                out_time = reduced_time if reduced_time is not None else raw_time
                out_data = smooth_data
            elif reduced_data is not None:
                out_time, out_data = reduced_time, reduced_data
            else:
                out_time, out_data = raw_time, raw_data

            return {"time": out_time, "data": out_data, "metadata": metadata}

        except Exception:
            return None
