"""MYA archiver connector, reached through the myquery HTTP service.

Jefferson Lab archives channel history in MYA and serves it over myquery, whose
Python client is ``jlab_archiver_client``. myquery exposes three endpoints this
connector uses, and the choice between them is what most of the code below is
about:

* ``interval`` -- every archived event for one channel, at its true timestamp.
* ``mystats`` -- server-side statistics per time bin (min/max/mean/stdev/...).
* ``mysampler`` -- one value per grid point, sample-and-hold.

``mysampler`` is deliberately NOT used. It answers on a grid shared by every
requested channel, carrying the value last in effect at each grid point rather
than a sample actually recorded there. That is forward-fill onto a shared index,
which is exactly what :meth:`ArchiverConnector.get_data` forbids: the frame must
carry each channel's own real samples, with no bin for a period that had none.
``interval`` costs more bytes for a wide window and is the honest answer.

Aggregates go to ``mystats`` where it can compute them, the same shape as the
EPICS connector's server-side operators. ``median`` is the exception -- MYA does
not compute one -- so that mode alone falls back to fetching raw events and
binning client-side, the way the DOOCS connector does for every mode.

Example:
    >>> connector = MYAArchiverConnector()
    >>> await connector.connect({'deployment': 'ops'})
    >>> df = await connector.get_data(
    ...     ['IGLK100HVPSkVolts'],
    ...     datetime(2026, 9, 9, tzinfo=timezone.utc),
    ...     datetime(2026, 9, 11, tzinfo=timezone.utc),
    ...     precision_ms=600_000,
    ...     processing='mean',
    ... )
"""

import asyncio
import math
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from osprey_connectors.archiver._timerange import (
    Processing,
    aggregate_long_frame,
    long_frame,
    reject_non_numeric,
    resolve_processing,
    utc_window,
)
from osprey_connectors.archiver.base import ArchiverConnector, ArchiverMetadata
from osprey_connectors.config import get_facility_timezone
from osprey_connectors.logger import get_logger

logger = get_logger("mya_archiver_connector")

#: OSPREY processing mode -> the name mystats gives that statistic. mystats also
#: reports duration/integration/rms/updateCount, which OSPREY has no mode for.
#: ``median`` is absent because MYA does not compute one; see
#: :meth:`MYAArchiverConnector.get_data`.
_MYSTATS_STAT = {
    "mean": "mean",
    "min": "min",
    "max": "max",
    "std": "stdev",
    "count": "eventCount",
}

#: The MYA deployment queried when the config names none. myquery's own client
#: defaults to "history"; "ops" is the live machine, which is what a control
#: assistant asking about recent history almost always wants.
_DEFAULT_DEPLOYMENT = "ops"

_DEFAULT_TIMEOUT_S = 60


class MYAArchiverConnector(ArchiverConnector):
    """Archiver connector for Jefferson Lab's MYA, via the myquery service."""

    def __init__(self) -> None:
        self._connected = False
        # The client library and its config singleton, bound in connect(). Typed
        # Any because jlab_archiver_client ships no stubs and is imported lazily,
        # so there is no symbol to name here on a machine without it.
        self._client: Any = None
        self._config: Any = None
        self._timeout = _DEFAULT_TIMEOUT_S
        self._deployment = _DEFAULT_DEPLOYMENT
        self._timezone: ZoneInfo | None = None

    async def connect(self, config: dict[str, Any]) -> None:
        """Initialize the myquery client.

        Args:
            config: Configuration block, every key optional:

                - ``myquery_server``: myquery host. Default: the client
                  library's own (``epicsweb.jlab.org``).
                - ``protocol``: ``http`` or ``https``. Default: the library's.
                - ``deployment``: MYA deployment to query. Default ``ops``.
                - ``timeout``: Default request timeout in seconds. Default 60.
                - ``timezone``: IANA zone the myquery server's naive timestamps
                  are in. Defaults to the facility timezone, which is right
                  whenever the archiver and the machine share a site.

        Raises:
            ImportError: If ``jlab_archiver_client`` is not installed.
            ConnectionError: If the client cannot be initialized.
        """
        try:
            import jlab_archiver_client as jac
        except ImportError:
            raise ImportError(
                "jlab_archiver_client is required for the MYA archiver connector. "
                "Install with: pip install jlab-archiver-client"
            ) from None

        self._client = jac
        self._config = jac.config.config

        try:
            # Only the settings actually given are applied. Writing None over
            # the library's defaults would leave the client with no server at
            # all, and those defaults are right for every JLab machine.
            overrides = {
                key: config[key] for key in ("myquery_server", "protocol") if config.get(key)
            }
            if overrides:
                self._config.set(**overrides)
            if not self._config.myquery_server:
                raise ValueError("myquery_server is required for the MYA archiver")
        except Exception as e:
            raise ConnectionError(f"MYA archiver client initialization failed: {e}") from e

        self._deployment = config.get("deployment") or _DEFAULT_DEPLOYMENT
        self._timeout = config.get("timeout", _DEFAULT_TIMEOUT_S)
        zone = config.get("timezone")
        self._timezone = ZoneInfo(zone) if zone else get_facility_timezone()
        self._connected = True

        logger.debug(
            "MYA archiver connected: %s://%s (deployment=%s)",
            self._config.protocol,
            self._config.myquery_server,
            self._deployment,
        )

    async def disconnect(self) -> None:
        """Release the client. myquery is stateless HTTP; nothing to close."""
        self._client = None
        self._config = None
        self._connected = False
        logger.debug("MYA archiver disconnected")

    async def get_data(
        self,
        channels: list[str],
        start_date: datetime,
        end_date: datetime,
        precision_ms: int = 1000,
        timeout: int | None = None,
        processing: str = "raw",
    ) -> pd.DataFrame:
        """Retrieve historical data for one or more channels.

        Args:
            channels: Channel addresses (MYA channel names).
            start_date: Start of the time range.
            end_date: End of the time range.
            precision_ms: Bin width in milliseconds; ``<= 0`` means full
                resolution.
            timeout: Optional timeout in seconds.
            processing: Aggregation within each bin. ``mean``/``min``/``max``/
                ``std``/``count`` are computed by mystats server-side;
                ``median`` is binned client-side from raw events because MYA
                computes no median; ``raw`` decimates real events locally.

        Returns:
            The canonical long frame -- ``timestamp``/``channel``/``value``,
            sorted by channel then timestamp.

        Raises:
            RuntimeError: If the archiver is not connected.
            TimeoutError: If the request times out.
            ConnectionError: If myquery cannot be reached.
            ValueError: If the time range or processing mode is invalid, or a
                non-raw mode is asked of a non-numeric channel.
        """
        if not self._connected:
            raise RuntimeError("MYA archiver not connected")

        timeout = timeout or self._timeout
        start_utc, end_utc = utc_window(start_date, end_date)
        resolved = resolve_processing(processing, precision_ms)

        if end_utc <= start_utc:
            raise ValueError(f"end_date must be after start_date (got {start_date} to {end_date})")

        # myquery is timezone-naive and reads every bound in its own server's
        # local zone, so the UTC window is converted rather than relabelled.
        start_local = start_utc.astimezone(self._timezone).replace(tzinfo=None)
        end_local = end_utc.astimezone(self._timezone).replace(tzinfo=None)

        server_side = resolved.mode in _MYSTATS_STAT

        def fetch_all() -> dict[str, pd.Series]:
            if server_side:
                return self._fetch_stats(channels, start_local, end_local, resolved)
            return self._fetch_events(channels, start_local, end_local)

        try:
            series = await asyncio.wait_for(asyncio.to_thread(fetch_all), timeout=timeout)

            if server_side:
                # mystats already binned, so aggregate_series is skipped -- but
                # its non-numeric check must still run: MYA answers an aggregate
                # query on an enum channel with something, and handing that back
                # labelled as a mean would be a lie.
                for s in series.values():
                    reject_non_numeric(s, resolved)
                data = long_frame(series)
            else:
                # "raw" decimates, "median" aggregates; both client-side.
                data = aggregate_long_frame(series, resolved)

            logger.debug(
                "Retrieved MYA archiver data: %d rows across %d channels "
                "(processing=%s, precision_ms=%d)",
                len(data),
                len(channels),
                resolved.mode,
                precision_ms,
            )
            return data

        except TimeoutError as e:
            raise TimeoutError(f"MYA archiver request timed out after {timeout}s") from e
        except ConnectionRefusedError as e:
            raise ConnectionError(
                "Cannot connect to the MYA archiver. Check connectivity and configuration."
            ) from e
        except Exception as e:
            if "connection" in str(e).lower():
                raise ConnectionError(f"Network issue reaching the MYA archiver: {e}") from e
            raise

    def _fetch_events(
        self, channels: list[str], start: datetime, end: datetime
    ) -> dict[str, pd.Series]:
        """Every archived event per channel, from myquery's interval endpoint.

        One query per channel rather than the client's ``run_parallel`` helper:
        that helper forward-fills each channel onto a shared index, and the
        frame must carry each channel's own real samples and nothing else.
        """
        series: dict[str, pd.Series] = {}
        for channel in channels:
            query = self._client.query.IntervalQuery(
                channel=channel, begin=start, end=end, deployment=self._deployment
            )
            interval = self._client.interval.Interval(query)
            interval.run()
            series[channel] = self._localize(interval.data, channel)
        return series

    def _fetch_stats(
        self, channels: list[str], start: datetime, end: datetime, resolved: Processing
    ) -> dict[str, pd.Series]:
        """One server-side aggregate per bin, from myquery's mystats endpoint.

        mystats is told how MANY bins to cut the range into, not how wide each
        one should be, so the requested width becomes a bin count here.
        """
        span_ms = (end - start).total_seconds() * 1000
        num_bins = max(1, math.ceil(span_ms / resolved.precision_ms))

        query = self._client.query.MyStatsQuery(
            pvlist=channels,
            start=start,
            end=end,
            num_bins=num_bins,
            deployment=self._deployment,
        )
        mystats = self._client.mystats.MyStats(query)
        mystats.run()

        frame = mystats.data
        stat = _MYSTATS_STAT[resolved.mode]
        if frame is None or len(frame) == 0:
            return {channel: self._localize(None, channel) for channel in channels}

        # mystats answers with a (timestamp, stat) MultiIndex carrying every
        # statistic it computed; keep the one that was asked for.
        try:
            wide = frame.xs(stat, level="stat")
        except KeyError as e:
            raise ValueError(
                f"MYA returned no {stat!r} statistic for processing={resolved.mode!r}"
            ) from e

        return {
            channel: self._localize(wide[channel] if channel in wide.columns else None, channel)
            for channel in channels
        }

    def _localize(self, data: "pd.Series | None", channel: str) -> pd.Series:
        """Stamp one channel's naive myquery timestamps as UTC-aware.

        A bin that MYA had no samples for comes back as NaN; an empty bin is not
        a sample, so it contributes no row and is dropped here.
        """
        if data is None or len(data) == 0:
            return pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"), name=channel)

        s = data.dropna()
        index = pd.DatetimeIndex(pd.to_datetime(s.index))
        if index.tz is None:
            # tz_localize with the zone object rather than a fixed offset, so a
            # window crossing a DST boundary converts correctly on both sides.
            index = index.tz_localize(self._timezone, ambiguous=True, nonexistent="shift_forward")
        return pd.Series(s.to_numpy(), index=index.tz_convert("UTC"), name=channel)

    async def get_metadata(self, channel: str) -> ArchiverMetadata:
        """Look up archiving metadata for one channel.

        Args:
            channel: Channel address.

        Returns:
            :class:`ArchiverMetadata` for the channel; ``is_archived`` is False
            when MYA knows nothing about it.

        Raises:
            ValueError: If the address matched more than one channel, or a
                channel other than the one asked for -- both mean a SQL
                wildcard slipped into the address.
        """

        def fetch_metadata() -> ArchiverMetadata:
            query = self._client.query.ChannelQuery(pattern=channel, deployment=self._deployment)
            found = self._client.channel.Channel(query)
            found.run()

            matches = found.matches or []
            if not matches:
                return ArchiverMetadata(
                    channel=channel,
                    is_archived=False,
                    description=f"Unknown channel: {channel}",
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Archiver matched more than one channel for {channel!r}: "
                    f"{[m['name'] for m in matches]}. The address looks like a pattern."
                )

            match = matches[0]["name"]
            if match != channel:
                raise ValueError(
                    f"Archiver returned the wrong channel. Requested: {channel}. Received: {match}."
                )
            return ArchiverMetadata(
                channel=match,
                is_archived=True,
                description=f"Archived channel: {match}",
            )

        return await asyncio.wait_for(asyncio.to_thread(fetch_metadata), timeout=self._timeout)

    async def check_availability(self, channels: list[str]) -> dict[str, bool]:
        """Report which channels MYA archives.

        Args:
            channels: Channel addresses to check.

        Returns:
            Mapping of channel address to whether MYA archives it.
        """

        def fetch_availability() -> dict[str, bool]:
            # Sequential on purpose: myquery answers one pattern per request,
            # and a burst of them from a wide channel list is what the server
            # would feel first.
            available = dict.fromkeys(channels, False)
            for name in channels:
                query = self._client.query.ChannelQuery(pattern=name, deployment=self._deployment)
                found = self._client.channel.Channel(query)
                found.run()
                matches = found.matches or []
                available[name] = len(matches) == 1 and matches[0]["name"] == name
            return available

        return await asyncio.wait_for(asyncio.to_thread(fetch_availability), timeout=self._timeout)
