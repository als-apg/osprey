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
which is exactly what :meth:`ChannelArchive.history` forbids: the frame must
carry each channel's own real samples, with no bin for a period that had none.
``interval`` costs more bytes for a wide window and is the honest answer.

MYA records only significant updates, and a value stays in effect until the next
one, so a channel whose last change predates the window would otherwise answer
with nothing at all -- "no data" for a setpoint whose value is perfectly well
known. The ``interval`` query therefore asks for the prior point, which myquery
documents as "the most recent update prior to the start, to give a value at the
start of the query". It is returned at the instant it was recorded, before the
window, the way the EPICS connector passes through the sample the archiver
appliance returns ahead of ``from``: a real sample at its real time, never a
value relabelled to the window start.

Aggregates go to ``mystats`` where it can compute them, the same shape as the
EPICS connector's server-side operators. ``median`` is the exception -- MYA does
not compute one -- so that mode alone falls back to fetching raw events and
binning client-side, the way the DOOCS connector does for every mode. Both paths
cut their bins from the window start, so a window opening off the midnight
lattice does not answer on two different grids depending on the mode.

Timestamps are requested as epoch milliseconds (``unix_timestamps_ms``), which
name an instant outright. myquery's default spelling is a naive local wall clock,
and the autumn fall-back hour repeats every wall-clock time in it, so the same
string names two instants an hour apart with nothing to tell them apart. Epoch
milliseconds have no such hour. This requires jlab-archiver-client 4.0.1; see
:meth:`MYAArchiverConnector._localize`.

Example:
    >>> archive = MYAArchiverConnector(BackendSpec(
    ...     kind='archive', type='mya_archiver', target='live', rehearsal=False,
    ...     settings={'deployment': 'ops'},
    ... ))
    >>> frame = await archive.history(
    ...     ['IGLK100HVPSkVolts'],
    ...     start_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
    ...     end_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
    ...     precision_ms=600_000,
    ...     processing='mean',
    ... )
"""

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd
from typing_extensions import override

from osprey_connectors._contract_types import ContractAny
from osprey_connectors.archiver._base import ArchiveCoverage, ChannelArchive
from osprey_connectors.archiver._timerange import (
    Processing,
    ProcessingMode,
    SampleSeries,
    aggregate_long_frame,
    long_frame,
    reject_non_numeric,
    resolve_processing,
    utc_window,
)
from osprey_connectors.config import get_facility_timezone
from osprey_connectors.connection import read_connection_settings
from osprey_connectors.errors import ArchiveQueryError, ConnectorUnavailableError
from osprey_connectors.logger import get_logger
from osprey_connectors.service_answers import BackendSpec, Unavailable

logger = get_logger("mya_archiver_connector")

#: OSPREY processing mode -> the name mystats gives that statistic. mystats also
#: reports duration/integration/rms/updateCount, which OSPREY has no mode for.
#: ``median`` is absent because MYA does not compute one; see
#: :meth:`MYAArchiverConnector.history`.
_MYSTATS_STAT: dict[ProcessingMode, str] = {
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


class _ClientTooOldError(RuntimeError):
    """The client library ignored ``unix_timestamps_ms``; private to this module."""


class _NoStatisticError(ValueError):
    """mystats answered without the statistic asked for; private to this module."""


class _MyquerySettings(Protocol):
    """``jlab_archiver_client.config.config``, the library's endpoint settings."""

    myquery_server: str
    protocol: str
    interval_path: str
    mystats_path: str
    channel_path: str


class _ConfigModule(Protocol):
    """``jlab_archiver_client.config``."""

    config: _MyquerySettings


class _QueryModule(Protocol):
    """``jlab_archiver_client.query``: the request objects, one per endpoint."""

    def IntervalQuery(
        self,
        *,
        channel: str,
        begin: datetime,
        end: datetime,
        deployment: str,
        unix_timestamps_ms: bool,
        prior_point: bool,
    ) -> object: ...

    def MyStatsQuery(
        self,
        *,
        pvlist: list[str],
        start: datetime,
        end: datetime,
        num_bins: int,
        deployment: str,
        unix_timestamps_ms: bool,
    ) -> object: ...

    def ChannelQuery(self, *, pattern: str, deployment: str) -> object: ...


class _Interval(Protocol):
    """An ``interval`` request; ``data`` is set by ``run()``."""

    data: "SampleSeries | None"

    def run(self) -> None: ...


class _IntervalModule(Protocol):
    """``jlab_archiver_client.interval``."""

    def Interval(self, query: object, *, url: str) -> _Interval: ...


class _MyStats(Protocol):
    """A ``mystats`` request; ``data`` is set by ``run()``."""

    data: "pd.DataFrame | None"

    def run(self) -> None: ...


class _MyStatsModule(Protocol):
    """``jlab_archiver_client.mystats``."""

    def MyStats(self, query: object, *, url: str) -> _MyStats: ...


class _Channel(Protocol):
    """A ``channel`` lookup; ``matches`` is set by ``run()``."""

    matches: list[Mapping[str, object]] | None

    def run(self) -> None: ...


class _ChannelModule(Protocol):
    """``jlab_archiver_client.channel``."""

    def Channel(self, query: object, *, url: str) -> _Channel: ...


class _JlabArchiverClient(Protocol):
    """The surface of ``jlab_archiver_client`` the connector calls; it ships no types."""

    config: _ConfigModule
    query: _QueryModule
    interval: _IntervalModule
    mystats: _MyStatsModule
    channel: _ChannelModule


def _like_literal(address: str) -> str:
    """Quote a channel name so myquery's SQL ``LIKE`` matches it and nothing else.

    The channel endpoint matches SQL patterns, where ``_`` stands for any single
    character -- and underscores are everywhere in accelerator channel names. So
    an unquoted address silently matches its siblings: ``IGLK100HVPSkVolt_``
    matches ``IGLK100HVPSkVolts``, which would make :meth:`coverage` call an
    archived channel ambiguous, or not archived.

    Verified against epicsweb: the server honours backslash escapes, and quoting
    a name that needs no quoting returns it unchanged.

    Args:
        address: The exact channel name to look up.

    Returns:
        The same name with every ``LIKE`` metacharacter backslash-escaped.
    """
    # Backslash first: quoting it after the others would re-quote their escapes.
    for special in ("\\", "%", "_"):
        address = address.replace(special, f"\\{special}")
    return address


class MYAArchiverConnector(ChannelArchive):
    """Archiver connector for Jefferson Lab's MYA, via the myquery service.

    The spec's settings, every key optional:

    - ``myquery_server``: myquery host. Default: the client library's own
      (``epicsweb.jlab.org``).
    - ``protocol``: ``http`` or ``https``. Default: the library's.
    - ``deployment``: MYA deployment to query. Default ``ops``.
    - ``timeout_s``: Default request timeout in seconds. Default 60.
    - ``auth`` and ``tls`` are refused: the client library sends every request
      itself, with no login and no per-connection CA. A site CA reaches it
      through the process trust store, which ``images.site_ca`` extends.
    - ``timezone``: IANA zone the myquery server reads query bounds in.
      Returned samples carry their own instant and need no zone; this is only
      how a UTC window is spelled for the server, whose bounds are a naive
      wall clock. Defaults to the facility timezone, which is right whenever
      the archiver and the machine share a site.

    Raises:
        ValueError: If the block names ``auth``, ``tls`` or a flat ``timeout``.
        ImportError: If ``jlab_archiver_client`` is not installed.
        ConnectorUnavailableError: If the client cannot be initialized.
    """

    def __init__(
        self, spec: BackendSpec, *, project_dir: Path | None = None, **seams: object
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        # The settings block is the archive's extension contract; this backend reads its keys unchecked.
        config: Mapping[str, ContractAny] = spec.settings
        connection = read_connection_settings(
            config,
            where="archiver.settings",
            logins=frozenset(),
            tls=False,
            unsupported_because=(
                "the MYA client (jlab-archiver-client) sends every request itself, "
                "with no login and no per-connection CA"
            ),
        )
        try:
            import jlab_archiver_client
        except ImportError:
            raise ImportError(
                "jlab_archiver_client is required for the MYA archiver connector. "
                f"Install with: pip install 'jlab-archiver-client>={_REQUIRED_CLIENT}'"
            ) from None

        jac: _JlabArchiverClient = jlab_archiver_client
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
            raise ConnectorUnavailableError(
                f"MYA archiver client initialization failed: {e}"
            ) from e

        self._deployment: str = config.get("deployment") or _DEFAULT_DEPLOYMENT
        self._timeout: float = connection.timeout_or(_DEFAULT_TIMEOUT_S)
        zone = config.get("timezone")
        self._timezone: ZoneInfo = ZoneInfo(zone) if zone else get_facility_timezone()

        logger.debug(
            "MYA archiver connected: %s (deployment=%s)",
            self._urls["interval"],
            self._deployment,
        )

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
        """Retrieve historical data for one or more channels.

        Args:
            addresses: Channel addresses (MYA channel names).
            start_at: Start of the time range, timezone-aware.
            end_at: End of the time range, timezone-aware.
            precision_ms: Bin width in milliseconds; ``<= 0`` means full
                resolution.
            processing: Aggregation within each bin. ``mean``/``min``/``max``/
                ``std``/``count`` are computed by mystats server-side;
                ``median`` is binned client-side from raw events because MYA
                computes no median; ``raw`` decimates real events locally.
            timeout_s: Optional timeout in seconds.

        Returns:
            The canonical long frame -- ``timestamp``/``address``/``value``,
            sorted by address then timestamp -- or ``Unavailable`` when myquery
            gives no answer, or the client library is too old to honour
            ``unix_timestamps_ms`` (``misconfigured``).

        Raises:
            ArchiveQueryError: If the time range or processing mode is invalid,
                or a non-raw mode is asked of a non-numeric channel.
        """
        if timeout_s is None:
            timeout_s = self._timeout
        start_utc, end_utc = utc_window(start_at, end_at)
        resolved = resolve_processing(processing, precision_ms, start_utc)

        if end_utc <= start_utc:
            raise ArchiveQueryError(
                f"end_at must be after start_at (got {start_at} to {end_at})", parameter="end_at"
            )

        server_side = resolved.mode in _MYSTATS_STAT
        channels = list(addresses)

        def fetch_all() -> dict[str, SampleSeries]:
            if server_side:
                return self._fetch_stats(channels, start_utc, end_utc, resolved)
            return self._fetch_events(channels, start_utc, end_utc)

        try:
            series = await asyncio.wait_for(asyncio.to_thread(fetch_all), timeout=timeout_s)
        except TimeoutError:
            return Unavailable(
                cause="timeout", reason=f"MYA archiver request timed out after {timeout_s}s"
            )
        except ConnectionRefusedError:
            return Unavailable(
                cause="unreachable",
                reason="Cannot connect to the MYA archiver. Check connectivity and configuration.",
            )
        except _ClientTooOldError as e:
            return Unavailable(cause="misconfigured", reason=str(e))
        except _NoStatisticError as e:
            return Unavailable(cause="bad_reply", reason=str(e))
        except Exception as e:
            logger.warning("MYA archiver read failed", exc_info=True)
            if "connection" in str(e).lower():
                return Unavailable(
                    cause="unreachable", reason=f"Network issue reaching the MYA archiver: {e}"
                )
            return Unavailable(cause="failed", reason=f"the MYA archiver read failed: {e}")

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

    def _library(self) -> _JlabArchiverClient:
        """The client library the archive was built with."""
        return self._client

    def _server_local(self, moment: datetime) -> datetime:
        """The naive wall clock myquery reads a UTC instant as.

        Query bounds are the one place the server's zone still matters: myquery
        renders ``begin``/``end`` with no offset, so a UTC window has to be
        spelled in the server's own local time. Returned samples carry epoch
        milliseconds and need no such conversion.
        """
        return moment.astimezone(self._timezone).replace(tzinfo=None)

    def _empty(self, address: str) -> SampleSeries:
        """The no-samples answer for one channel: no rows, right index type."""
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"), name=address)

    def _fetch_events(
        self, addresses: list[str], start_utc: datetime, end_utc: datetime
    ) -> dict[str, SampleSeries]:
        """Every archived event per channel, from myquery's interval endpoint.

        The prior point comes back with the rest, at its own timestamp before
        the window; see the module docstring.

        One query per channel rather than the client's ``run_parallel`` helper:
        that helper forward-fills each channel onto a shared index, and the
        frame must carry each channel's own real samples and nothing else.
        """
        begin = self._server_local(start_utc)
        end = self._server_local(end_utc)
        client = self._library()
        series: dict[str, SampleSeries] = {}
        for channel in addresses:
            query = client.query.IntervalQuery(
                channel=channel,
                begin=begin,
                end=end,
                deployment=self._deployment,
                unix_timestamps_ms=True,
                prior_point=True,
            )
            interval = client.interval.Interval(query, url=self._urls["interval"])
            interval.run()
            series[channel] = self._localize(interval.data, channel)
        return series

    def _fetch_stats(
        self,
        addresses: list[str],
        start_utc: datetime,
        end_utc: datetime,
        resolved: Processing,
    ) -> dict[str, SampleSeries]:
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
            return self._mystats(addresses, start_utc, end_utc, 1, resolved)

        split_utc = start_utc + timedelta(milliseconds=whole * width)
        series = self._mystats(addresses, start_utc, split_utc, whole, resolved)
        if split_utc < end_utc:
            tail = self._mystats(addresses, split_utc, end_utc, 1, resolved)
            series = {
                channel: pd.concat([series[channel], tail[channel]]).sort_index()
                for channel in addresses
            }
        return series

    def _mystats(
        self,
        addresses: list[str],
        start_utc: datetime,
        end_utc: datetime,
        num_bins: int,
        resolved: Processing,
    ) -> dict[str, SampleSeries]:
        """One mystats round trip, cut into ``num_bins`` equal bins."""
        client = self._library()
        query = client.query.MyStatsQuery(
            pvlist=addresses,
            start=self._server_local(start_utc),
            end=self._server_local(end_utc),
            num_bins=num_bins,
            deployment=self._deployment,
            unix_timestamps_ms=True,
        )
        mystats = client.mystats.MyStats(query, url=self._urls["mystats"])
        mystats.run()

        frame = mystats.data
        stat = _MYSTATS_STAT[resolved.mode]
        if frame is None or len(frame) == 0:
            return {channel: self._empty(channel) for channel in addresses}

        # mystats answers with a (timestamp, stat) MultiIndex carrying every
        # statistic it computed; keep the one that was asked for.
        try:
            wide = frame.xs(stat, level="stat")
        except KeyError as e:
            raise _NoStatisticError(
                f"MYA returned no {stat!r} statistic for processing={resolved.mode!r}"
            ) from e

        return {
            channel: self._localize(wide[channel] if channel in wide.columns else None, channel)
            for channel in addresses
        }

    def _localize(self, data: "SampleSeries | None", address: str) -> SampleSeries:
        """Read one channel's epoch-millisecond timestamps as UTC instants.

        A bin that MYA had no samples for comes back as NaN; an empty bin is not
        a sample, so it contributes no row and is dropped here.

        Raises:
            _ClientTooOldError: If the index is not integral, which means the
                client library ignored ``unix_timestamps_ms`` and has already
                parsed the timestamps -- as nanoseconds, dating everything to 1970.
        """
        if data is None or len(data) == 0:
            return self._empty(address)

        s = data.dropna()
        if s.empty:
            return self._empty(address)

        index = pd.Index(s.index)
        if not pd.api.types.is_integer_dtype(index):
            # A version guard by behaviour rather than by version string: this
            # is the exact symptom, and it needs no packaging dependency to
            # read. The failure it refuses is silent -- every timestamp lands
            # in January 1970 -- which an archiver must never serve.
            raise _ClientTooOldError(
                "MYA returned timestamps that are not epoch milliseconds, so "
                "jlab_archiver_client ignored unix_timestamps_ms. Install "
                f"jlab-archiver-client>={_REQUIRED_CLIENT}: earlier releases read those "
                "integers as nanoseconds and date every sample to 1970."
            )
        return pd.Series(
            s.to_numpy(), index=pd.to_datetime(index, unit="ms", utc=True), name=address
        )

    def _lookup(self, address: str) -> list[Mapping[str, object]]:
        """Every channel myquery's channel endpoint matches for one address."""
        client = self._library()
        query = client.query.ChannelQuery(
            pattern=_like_literal(address), deployment=self._deployment
        )
        found = client.channel.Channel(query, url=self._urls["channel"])
        found.run()
        return found.matches or []

    def _matched(self, address: str) -> ArchiveCoverage:
        """One channel's coverage, from myquery's channel endpoint. Runs in a worker thread."""
        matches = self._lookup(address)
        if not matches:
            return ArchiveCoverage(address=address, is_archived=False)
        if len(matches) > 1:
            return ArchiveCoverage(
                address=address,
                is_archived=None,
                unknown_reason=(
                    f"Archiver matched more than one channel for {address!r}: "
                    f"{[m['name'] for m in matches]}."
                ),
            )
        match = matches[0]["name"]
        if match != address:
            return ArchiveCoverage(
                address=address,
                is_archived=None,
                unknown_reason=(
                    f"Archiver returned the wrong channel. Requested: {address}. Received: {match}."
                ),
            )
        return ArchiveCoverage(address=address, is_archived=True)

    @override
    async def coverage(
        self, addresses: Sequence[str], *, timeout_s: float | None = None
    ) -> Mapping[str, ArchiveCoverage] | Unavailable:
        """Report which channels MYA archives.

        MYA states no bounds, so a channel's coverage says only whether it is
        archived; a lookup that matched another channel, or more than one,
        cannot say, and names what it matched.

        Args:
            addresses: Channel addresses to check.
            timeout_s: The seconds each channel's lookup may spend; ``None`` is
                the configured default.

        Returns:
            One :class:`ArchiveCoverage` per address, or ``Unavailable`` when
            a lookup gets no answer.

        Raises:
            ArchiveQueryError: If an address is a SQL pattern rather than a
                channel name.
        """
        for address in addresses:
            if "%" in address:
                # `_` is a wildcard too, but it is also a legitimate character in
                # accelerator channel names, so only `%` gives the caller away.
                raise ArchiveQueryError(
                    f"Archiver address {address!r} contains '%', a SQL wildcard. The address "
                    "looks like a pattern; archiver addresses are exact channel names.",
                    parameter="addresses",
                    address=address,
                )
        bound = timeout_s if timeout_s is not None else self._timeout
        covered: dict[str, ArchiveCoverage] = {}
        for address in addresses:
            # Sequential on purpose: myquery answers one pattern per request,
            # and a burst of them from a wide channel list is what the server
            # would feel first. Bounded per channel rather than per sweep --
            # one timeout over the whole list would throw away every answer
            # already fetched, and could not be widened for a longer list.
            try:
                covered[address] = await asyncio.wait_for(
                    asyncio.to_thread(self._matched, address), timeout=bound
                )
            except TimeoutError:
                return Unavailable(
                    cause="timeout", reason=f"MYA channel lookup timed out after {bound}s"
                )
            except Exception as e:
                logger.warning("MYA channel lookup failed", exc_info=True)
                return Unavailable(
                    cause="failed", reason=f"the MYA channel lookup failed: {type(e).__name__}"
                )
        return covered
