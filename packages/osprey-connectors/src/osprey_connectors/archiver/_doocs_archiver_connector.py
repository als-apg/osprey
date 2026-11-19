"""
DOOCS local history connector using doocs4py.

Provides interface to the DOOCS local histories.

Author: Frank Mayet (DESY, MXL)
Date: 2026-07-01
"""

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Protocol, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import model_validator
from typing_extensions import override

from osprey_connectors.archiver._base import ArchiveCoverage, ChannelArchive
from osprey_connectors.archiver._timerange import (
    ProcessingMode,
    aggregate_long_frame,
    resolve_processing,
    utc_window,
)
from osprey_connectors.config_sections import ConnectorSettings
from osprey_connectors.connection import ConnectionSettings, read_connection_settings
from osprey_connectors.logger import get_logger
from osprey_connectors.service_answers import BackendSpec, Unavailable

logger = get_logger("doocs_archiver_connector")


class _HistoryReply(Protocol):
    """What ``doocs4py.get`` answers for a ``.HIST`` address."""

    @property
    def value(self) -> Sequence[Sequence[float]]:
        """The archived entries, each ``(time, _, _, value)``, oldest first."""
        ...


class _Doocs4pyTypes(Protocol):
    """The ``doocs4py.types`` the connector builds a history request from."""

    def TTII(self, start: float, stop: float, mode: int, flags: int, /) -> object: ...


class _Doocs4py(Protocol):
    """The surface of :mod:`doocs4py` the connector calls; doocs4py ships no types."""

    __version__: str
    types: _Doocs4pyTypes

    def names(self, pattern: str) -> Sequence[Sequence[str]]: ...

    def Address(self, address: str) -> object: ...

    def get(self, address: object, request: object, /) -> _HistoryReply: ...


class _History(TypedDict):
    """One property's history: parallel arrays of UNIX timestamps and values."""

    time: npt.NDArray[np.float64]
    data: npt.NDArray[np.float64]


class DoocsArchiverSettings(ConnectorSettings):
    """The DOOCS archive's block, ``archiver.settings``, read; every key optional.

    Attributes:
        connection: ``timeout_s``; ``auth`` and ``tls`` are refused, because
            doocs4py takes no login and no CA.
        avg_window: The centered moving-average window in seconds; ``None``
            for no smoothing.
    """

    connection: ConnectionSettings
    avg_window: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _read(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        connection = read_connection_settings(
            data,
            where="archiver.settings",
            logins=frozenset(),
            tls=False,
            unsupported_because=(
                "the DOOCS archiver reaches the ENS through doocs4py, which takes no "
                "login and no CA"
            ),
        )
        return {**data, "connection": connection}


class DOOCSArchiverConnector(ChannelArchive):
    """
    DOOCS local history connector.

    Provides access to local history data of a given DOOCS property if available.

    A centered moving average can be applied by supplying `avg_window` (seconds).
    It averages over a real time span, so it operates on the archived samples at
    their own irregular timestamps -- no resampling onto a uniform grid.

    The spec's settings, every key optional:

    - avg_window: Centered moving-average window in seconds (default: None,
      no smoothing)
    - timeout_s: Default request timeout in seconds (default: 60)

    The ENS is not asked when the archive is built; :meth:`probe` asks it.

    Raises:
        ValueError: If the block names ``auth``, ``tls`` or a flat ``timeout``
        ImportError: If doocs4py is not installed

    Example:
        >>> archive = DOOCSArchiverConnector(BackendSpec(
        >>>     kind="archive", type="doocs_archiver", target="live", rehearsal=False,
        >>>     settings={'avg_window': 20},
        >>> ))
        >>> frame = await archive.history(
        >>>     ['FACILITY/DEVICE/LOCATION/PROPERTY'],
        >>>     start_at=datetime(2026, 7, 1, tzinfo=UTC),
        >>>     end_at=datetime(2026, 7, 2, tzinfo=UTC),
        >>> )
    """

    settings_model: ClassVar[type[ConnectorSettings]] = DoocsArchiverSettings
    SETTINGS_KEY: ClassVar[str] = "archiver.settings"

    def __init__(
        self, spec: BackendSpec, *, project_dir: Path | None = None, **seams: object
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        config = self._settings(DoocsArchiverSettings)
        # Import doocs4py here and give clear error if not installed
        try:
            import doocs4py

            library: _Doocs4py = doocs4py
            self._doocs4py = library
            logger.debug(f"DOOCS archiver connector: doocs4py version {library.__version__} loaded")
        except ImportError:
            raise ImportError("doocs4py is required for the DOOCS connector.") from None

        self._avg_window: float | None = config.avg_window
        self._timeout: float = config.connection.timeout_or(60)
        logger.debug("DOOCS archiver connector initialized")

    @override
    async def probe(self, *, timeout_s: float | None = None) -> str | Unavailable:
        """Whether the ENS answers: how many facilities it lists, or why not."""
        bound = timeout_s if timeout_s is not None else self._timeout

        def list_facilities() -> int:
            return len([f[1] for f in self._doocs4py.names("*")])

        try:
            facilities = await asyncio.wait_for(asyncio.to_thread(list_facilities), timeout=bound)
        except TimeoutError:
            return Unavailable(cause="timeout", reason=f"the ENS did not answer within {bound}s")
        except Exception:
            return Unavailable(
                cause="unreachable",
                reason="DOOCS archiver connector failed to connect to the ENS.",
            )
        return f"the ENS lists {facilities} facilities"

    @override
    async def history(
        self,
        addresses: Sequence[str],
        *,
        start_at: datetime,
        end_at: datetime,
        precision_ms: int = 1000,
        processing: ProcessingMode = "raw",
        timeout_s: float | None = None,
    ) -> pd.DataFrame | Unavailable:
        """
        Retrieve historical data from the DOOCS local histories.

        Args:
            addresses: List of DOOCS property addresses
            start_at: Start of the time range, timezone-aware.
            end_at: End of the time range, timezone-aware.
            precision_ms: Time precision (affects downsampling)
            processing: Aggregation applied within each precision_ms bin. One of
                "raw", "mean", "min", "max", "median", "std", "count". Applied
                client-side via pandas resampling. Anything else raises
                ArchiveQueryError.
            timeout_s: Timeout in seconds. ``None`` falls back to the connector's
                configured default rather than waiting indefinitely.

        Returns:
            The canonical long frame — see :meth:`ChannelArchive.history` — or
            ``Unavailable`` when a DOOCS property's history cannot be read, or
            the request times out.

        Raises:
            TypeError: If start_at or end_at is not a datetime
            NaiveDatetimeError: If start_at or end_at is naive
            ArchiveQueryError: If a non-raw processing mode is requested for a
                channel that carries non-numeric values
        """

        # asyncio.wait_for(timeout=None) would block indefinitely.
        bound: float = timeout_s if timeout_s is not None else self._timeout

        start_utc, end_utc = utc_window(start_at, end_at)
        resolved = resolve_processing(processing, precision_ms, start_utc)

        # Quoted: ``pd.Series`` is not subscriptable at runtime.
        def fetch_all() -> "dict[str, pd.Series[float]] | Unavailable":
            data: dict[str, pd.Series[float]] = {}
            for add in addresses:
                hist_data_dict = self._read_history(
                    add,
                    start_utc.timestamp(),
                    end_utc.timestamp(),
                    self._avg_window,
                )
                if hist_data_dict is None:
                    return Unavailable(
                        cause="failed",
                        reason=f"DOOCS archiver connector: Cannot read history for {add}",
                    )
                timestamps = pd.to_datetime(hist_data_dict["time"], unit="s", utc=True)
                data[add] = pd.Series(hist_data_dict["data"], index=timestamps, name=add)
            return data

        try:
            series_dict = await asyncio.wait_for(asyncio.to_thread(fetch_all), timeout=bound)
        except TimeoutError:
            return Unavailable(
                cause="timeout", reason=f"DOOCS archiver request timed out after {bound}s"
            )
        if isinstance(series_dict, Unavailable):
            return series_dict

        data = aggregate_long_frame(series_dict, resolved)

        logger.debug(
            f"Retrieved DOOCS archiver data: {len(data)} rows "
            f"across {len(addresses)} DOOCS properties"
        )
        return data

    @override
    async def coverage(
        self, addresses: Sequence[str], *, timeout_s: float | None = None
    ) -> Mapping[str, ArchiveCoverage] | Unavailable:
        """Whether each channel is archived, from its ``.HIST`` property name extension.

        A lookup that raises answers ``is_archived`` ``None``: the archiver
        could not say. DOOCS states no bounds.
        """
        bound = timeout_s if timeout_s is not None else self._timeout
        doocs4py = self._doocs4py

        def lookup_all() -> dict[str, ArchiveCoverage]:
            covered: dict[str, ArchiveCoverage] = {}
            for add in addresses:
                hist_address = add
                if not hist_address.endswith(".HIST"):
                    hist_address = add + ".HIST"
                try:
                    archived = bool(doocs4py.names(hist_address))
                except Exception as e:
                    covered[add] = ArchiveCoverage(
                        address=add,
                        is_archived=None,
                        unknown_reason=(
                            f"the DOOCS name service could not be asked: {type(e).__name__}"
                        ),
                    )
                else:
                    covered[add] = ArchiveCoverage(address=add, is_archived=archived)
            return covered

        try:
            return await asyncio.wait_for(asyncio.to_thread(lookup_all), timeout=bound)
        except TimeoutError:
            return Unavailable(cause="timeout", reason=f"the ENS did not answer within {bound}s")

    def _read_history(
        self,
        address: str,
        start_time: float,
        end_time: float,
        avg_window: float | None = None,
    ) -> _History | None:
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

        doocs4py = self._doocs4py

        try:
            if not address.endswith(".HIST"):
                address = address + ".HIST"
            hist_address = doocs4py.Address(address)

            current_stop: float = stop_ts
            all_data: list[Sequence[float]] = []

            while True:
                ttii = doocs4py.types.TTII(start_ts, current_stop, 256, 0)  # 256 means Archiver
                result = doocs4py.get(hist_address, ttii)

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
            raw_time = np.fromiter(
                (entry[0] for entry in all_data), dtype=np.float64, count=n_entries
            )
            raw_data = np.fromiter(
                (entry[3] for entry in all_data), dtype=np.float64, count=n_entries
            )

            # Remove duplicates and ensure monotonically increasing time.
            # np.unique returns sorted unique values, which the routines below require.
            raw_time, unique_indices = np.unique(raw_time, return_index=True)
            raw_data = raw_data[unique_indices]

            # Optional centered moving average over the real samples. The
            # time-based window handles irregular spacing and keeps the real
            # timestamps, returning exactly one value per input sample.
            out_data: npt.NDArray[np.float64] = (
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
