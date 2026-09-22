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

MYA records only significant updates, and a value stays in effect until the next
one, so a channel whose last change predates the window would otherwise answer
with nothing at all -- "no data" for a setpoint whose value is perfectly well
known. The ``interval`` query therefore asks for the prior point, which myquery
documents as "the most recent update prior to the start, to give a value at the
start of the query"; see :meth:`MYAArchiverConnector._clamp_prior`.

Aggregates go to ``mystats`` where it can compute them, the same shape as the
EPICS connector's server-side operators. ``median`` is the exception -- MYA does
not compute one -- so that mode alone falls back to fetching raw events and
binning client-side, the way the DOOCS connector does for every mode.

Timestamps are requested as epoch milliseconds (``unix_timestamps_ms``), which
name an instant outright. myquery's default spelling is a naive local wall clock,
and the autumn fall-back hour repeats every wall-clock time in it, so the same
string names two instants an hour apart with nothing to tell them apart. Epoch
milliseconds have no such hour. This requires jlab-archiver-client 4.0.1; see
:meth:`MYAArchiverConnector._localize`.

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
from datetime import datetime, timedelta
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

#: The client release that honours ``unix_timestamps_ms``. Earlier ones accept
#: the flag, ask myquery for epoch milliseconds, and then parse the integers
#: that come back as *nanoseconds* -- dating every sample to January 1970. The
#: guard in :meth:`MYAArchiverConnector._localize` refuses that silently-wrong
#: data rather than serving it.
_REQUIRED_CLIENT = "4.0.1"


def _like_literal(channel: str) -> str:
    """Quote a channel name so myquery's SQL ``LIKE`` matches it and nothing else.

    The channel endpoint matches SQL patterns, where ``_`` stands for any single
    character -- and underscores are everywhere in accelerator channel names. So
    an unquoted address silently matches its siblings: ``IGLK100HVPSkVolt_``
    matches ``IGLK100HVPSkVolts``, which would make :meth:`get_metadata` reject a
    valid address as a pattern and :meth:`check_availability` call an archived
    channel unavailable.

    Verified against epicsweb: the server honours backslash escapes, and quoting
    a name that needs no quoting returns it unchanged.

    Args:
        channel: The exact channel name to look up.

    Returns:
        The same name with every ``LIKE`` metacharacter backslash-escaped.
    """
    # Backslash first: quoting it after the others would re-quote their escapes.
    for special in ("\\", "%", "_"):
        channel = channel.replace(special, f"\\{special}")
    return channel


class MYAArchiverConnector(ArchiverConnector):
    """Archiver connector for Jefferson Lab's MYA, via the myquery service."""

    def __init__(self) -> None:
        self._connected = False
        # The client library, bound in connect(). Typed Any because
        # jlab_archiver_client ships no stubs and is imported lazily, so there
        # is no symbol to name here on a machine without it.
        self._client: Any = None
        self._urls: dict[str, str] = {}
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
                - ``timezone``: IANA zone the myquery server reads query bounds
                  in. Returned samples carry their own instant and need no zone;
                  this is only how a UTC window is spelled for the server, whose
                  bounds are a naive wall clock. Defaults to the facility
                  timezone, which is right whenever the archiver and the machine
                  share a site.

        Raises:
            ImportError: If ``jlab_archiver_client`` is not installed.
            ConnectionError: If the client cannot be initialized.
        """
        try:
            import jlab_archiver_client as jac
        except ImportError:
            raise ImportError(
                "jlab_archiver_client is required for the MYA archiver connector. "
                f"Install with: pip install 'jlab-archiver-client>={_REQUIRED_CLIENT}'"
            ) from None

        self._client = jac

        try:
            # The library's settings are read, never written. `config.set()`
            # mutates a process-global singleton: a second connector with no
            # overrides would silently inherit this one's server, and two
            # connected instances would clobber each other. Every endpoint URL
            # is built here and handed to the query object per call instead.
            settings = jac.config.config
            server = config.get("myquery_server") or settings.myquery_server
            protocol = config.get("protocol") or settings.protocol
            if not server:
                raise ValueError("myquery_server is required for the MYA archiver")
            base = f"{protocol}://{server}"
            # Each path is read off the singleton directly rather than through
            # `config.snapshot()`, which through 4.0.1 reports `point_path`
            # under the `mystats_path` key -- every aggregate query would go to
            # the wrong endpoint.
            self._urls = {
                "interval": f"{base}{settings.interval_path}",
                "mystats": f"{base}{settings.mystats_path}",
                "channel": f"{base}{settings.channel_path}",
            }
        except Exception as e:
            raise ConnectionError(f"MYA archiver client initialization failed: {e}") from e

        self._deployment = config.get("deployment") or _DEFAULT_DEPLOYMENT
        self._timeout = config.get("timeout") or _DEFAULT_TIMEOUT_S
        zone = config.get("timezone")
        self._timezone = ZoneInfo(zone) if zone else get_facility_timezone()
        self._connected = True

        logger.debug(
            "MYA archiver connected: %s (deployment=%s)",
            self._urls["interval"],
            self._deployment,
        )

    async def disconnect(self) -> None:
        """Release the client. myquery is stateless HTTP; nothing to close."""
        self._client = None
        self._urls = {}
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
            RuntimeError: If the archiver is not connected, or the client
                library is too old to honour ``unix_timestamps_ms``.
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

        server_side = resolved.mode in _MYSTATS_STAT

        def fetch_all() -> dict[str, pd.Series]:
            if server_side:
                return self._fetch_stats(channels, start_utc, end_utc, resolved)
            return self._fetch_events(channels, start_utc, end_utc)

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

    def _server_local(self, moment: datetime) -> datetime:
        """The naive wall clock myquery reads a UTC instant as.

        Query bounds are the one place the server's zone still matters: myquery
        renders ``begin``/``end`` with no offset, so a UTC window has to be
        spelled in the server's own local time. Returned samples carry epoch
        milliseconds and need no such conversion.
        """
        return moment.astimezone(self._timezone).replace(tzinfo=None)

    def _empty(self, channel: str) -> pd.Series:
        """The no-samples answer for one channel: no rows, right index type."""
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"), name=channel)

    def _fetch_events(
        self, channels: list[str], start_utc: datetime, end_utc: datetime
    ) -> dict[str, pd.Series]:
        """Every archived event per channel, from myquery's interval endpoint.

        One query per channel rather than the client's ``run_parallel`` helper:
        that helper forward-fills each channel onto a shared index, and the
        frame must carry each channel's own real samples and nothing else.
        """
        begin = self._server_local(start_utc)
        end = self._server_local(end_utc)
        series: dict[str, pd.Series] = {}
        for channel in channels:
            query = self._client.query.IntervalQuery(
                channel=channel,
                begin=begin,
                end=end,
                deployment=self._deployment,
                unix_timestamps_ms=True,
                prior_point=True,
            )
            interval = self._client.interval.Interval(query, url=self._urls["interval"])
            interval.run()
            series[channel] = self._clamp_prior(self._localize(interval.data, channel), start_utc)
        return series

    def _clamp_prior(self, s: pd.Series, start_utc: datetime) -> pd.Series:
        """Stamp the prior point at the window start whose value it reports.

        ``prior_point`` returns the last update *before* the window, carrying
        its own original timestamp. Left there it would be a sample outside the
        range the caller asked for, and it would land in a bin that is not in
        the window at all. Moved to the window start it says what it is there to
        say -- this was the value in effect when the window opened -- which is
        what MYA's record-on-change model means by the value at that instant.

        A real sample recorded exactly at the window start is the better
        witness, so the prior point gives way to it rather than doubling it.
        """
        if s.empty:
            return s
        start = pd.Timestamp(start_utc)
        if not (s.index < start).any():
            return s
        moved = pd.Series(
            s.to_numpy(), index=s.index.where(s.index >= start, start), name=s.name
        ).sort_index()
        return moved[~moved.index.duplicated(keep="last")]

    def _fetch_stats(
        self,
        channels: list[str],
        start_utc: datetime,
        end_utc: datetime,
        resolved: Processing,
    ) -> dict[str, pd.Series]:
        """One server-side aggregate per bin, from myquery's mystats endpoint.

        mystats is told how MANY bins to cut the range into, not how wide each
        one should be, so a requested width is served by choosing the window the
        count divides evenly. When the width does not divide the window, the
        whole bins are fetched over the part it does divide and the ragged
        remainder is fetched as one more bin -- the partial final bin pandas'
        ``resample`` produces for the same request, which is what every other
        shipped archiver returns here. Truncating instead would drop real data
        inside the window the caller asked for.
        """
        width = resolved.precision_ms
        span_ms = (end_utc - start_utc) // timedelta(milliseconds=1)
        whole = span_ms // width

        if whole == 0:
            # Narrower than a single bin: one partial bin over the whole window.
            return self._mystats(channels, start_utc, end_utc, 1, resolved)

        split_utc = start_utc + timedelta(milliseconds=whole * width)
        series = self._mystats(channels, start_utc, split_utc, whole, resolved)
        if split_utc < end_utc:
            tail = self._mystats(channels, split_utc, end_utc, 1, resolved)
            series = {
                channel: pd.concat([series[channel], tail[channel]]).sort_index()
                for channel in channels
            }
        return series

    def _mystats(
        self,
        channels: list[str],
        start_utc: datetime,
        end_utc: datetime,
        num_bins: int,
        resolved: Processing,
    ) -> dict[str, pd.Series]:
        """One mystats round trip, cut into ``num_bins`` equal bins."""
        query = self._client.query.MyStatsQuery(
            pvlist=channels,
            start=self._server_local(start_utc),
            end=self._server_local(end_utc),
            num_bins=num_bins,
            deployment=self._deployment,
            unix_timestamps_ms=True,
        )
        mystats = self._client.mystats.MyStats(query, url=self._urls["mystats"])
        mystats.run()

        frame = mystats.data
        stat = _MYSTATS_STAT[resolved.mode]
        if frame is None or len(frame) == 0:
            return {channel: self._empty(channel) for channel in channels}

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
        """Read one channel's epoch-millisecond timestamps as UTC instants.

        A bin that MYA had no samples for comes back as NaN; an empty bin is not
        a sample, so it contributes no row and is dropped here.

        Raises:
            RuntimeError: If the index is not integral, which means the client
                library ignored ``unix_timestamps_ms`` and has already parsed
                the timestamps -- as nanoseconds, dating everything to 1970.
        """
        if data is None or len(data) == 0:
            return self._empty(channel)

        s = data.dropna()
        if s.empty:
            return self._empty(channel)

        index = pd.Index(s.index)
        if not pd.api.types.is_integer_dtype(index):
            # A version guard by behaviour rather than by version string: this
            # is the exact symptom, and it needs no packaging dependency to
            # read. The failure it refuses is silent -- every timestamp lands
            # in January 1970 -- which an archiver must never serve.
            raise RuntimeError(
                "MYA returned timestamps that are not epoch milliseconds, so "
                "jlab_archiver_client ignored unix_timestamps_ms. Install "
                f"jlab-archiver-client>={_REQUIRED_CLIENT}: earlier releases read those "
                "integers as nanoseconds and date every sample to 1970."
            )
        return pd.Series(
            s.to_numpy(), index=pd.to_datetime(index, unit="ms", utc=True), name=channel
        )

    def _lookup(self, channel: str) -> list[dict[str, Any]]:
        """Every channel myquery's channel endpoint matches for one address."""
        query = self._client.query.ChannelQuery(
            pattern=_like_literal(channel), deployment=self._deployment
        )
        found = self._client.channel.Channel(query, url=self._urls["channel"])
        found.run()
        return found.matches or []

    async def get_metadata(self, channel: str) -> ArchiverMetadata:
        """Look up archiving metadata for one channel.

        Args:
            channel: Channel address.

        Returns:
            :class:`ArchiverMetadata` for the channel; ``is_archived`` is False
            when MYA knows nothing about it.

        Raises:
            RuntimeError: If the archiver is not connected.
            ValueError: If the address is a SQL pattern rather than a channel
                name, or matched a channel other than the one asked for.
        """
        if not self._connected:
            raise RuntimeError("MYA archiver not connected")
        if "%" in channel:
            # `_` is a wildcard too, but it is also a legitimate character in
            # accelerator channel names, so only `%` gives the caller away.
            raise ValueError(
                f"Archiver address {channel!r} contains '%', a SQL wildcard. The address "
                "looks like a pattern; archiver addresses are exact channel names."
            )

        def fetch_metadata() -> ArchiverMetadata:
            matches = self._lookup(channel)

            if not matches:
                return ArchiverMetadata(
                    channel=channel,
                    is_archived=False,
                    description=f"Unknown channel: {channel}",
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Archiver matched more than one channel for {channel!r}: "
                    f"{[m['name'] for m in matches]}."
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

        Raises:
            RuntimeError: If the archiver is not connected.
            TimeoutError: If any one channel's lookup times out.
        """
        if not self._connected:
            raise RuntimeError("MYA archiver not connected")

        def probe(name: str) -> bool:
            matches = self._lookup(name)
            return len(matches) == 1 and matches[0]["name"] == name

        available = dict.fromkeys(channels, False)
        for name in channels:
            # Sequential on purpose: myquery answers one pattern per request,
            # and a burst of them from a wide channel list is what the server
            # would feel first. Bounded per channel rather than per sweep --
            # one timeout over the whole list would throw away every answer
            # already fetched, and could not be widened for a longer list.
            available[name] = await asyncio.wait_for(
                asyncio.to_thread(probe, name), timeout=self._timeout
            )
        return available
