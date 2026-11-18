"""Shared time-range and processing helpers for archiver connectors.

Private to :mod:`osprey_connectors.archiver`. Refuses naive query bounds and
normalizes aware ones to UTC, resolves processing modes for server-side (EPICS
operator) and client-side (pandas aggregation) backends, and assembles
per-channel series into the canonical long frame without ever manufacturing a
sample.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, TypeAlias, TypeGuard, get_args

import pandas as pd
from pydantic import InstanceOf

from osprey_connectors._contract_types import ContractAny
from osprey_connectors.errors import ArchiveQueryError, require_aware
from osprey_connectors.model import FrozenModel

#: A canonical processing mode: the aggregation ChannelArchive.history
#: applies within each bin, or raw for none.
ProcessingMode = Literal["raw", "mean", "min", "max", "median", "std", "count"]

#: Every :data:`ProcessingMode`, raw first.
PROCESSING_MODES: tuple[ProcessingMode, ...] = get_args(ProcessingMode)

#: One channel's samples, indexed by timestamp. The element is the archived
#: channel value the ``value`` column of ``ChannelArchive.history`` carries,
#: which that contract leaves open: a float, an enum string or a waveform's
#: ``list[float]``. pandas' stubs bound a Series element to scalar dtypes, which
#: a waveform trace is not, so the open value is named rather than guessed. A
#: string, because ``pd.Series`` is not subscriptable at runtime.
SampleSeries: TypeAlias = "pd.Series[ContractAny]"

# The canonical long-format columns every archiver connector returns.
LONG_COLUMNS = ("timestamp", "address", "value")

# EPICS Archiver Appliance operator names, keyed by our canonical mode. "raw"
# maps to lastSample so a binned raw query keeps its long-standing behavior.
_EPICS_OPERATORS: dict[ProcessingMode, str] = {
    "raw": "lastSample",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "median": "median",
    "std": "std",
    "count": "count",
}


def _is_processing_mode(value: str) -> TypeGuard[ProcessingMode]:
    """Whether ``value`` names one of :data:`PROCESSING_MODES`."""
    return value in PROCESSING_MODES


def utc_window(start_at: object, end_at: object) -> tuple[datetime, datetime]:
    """Validate a caller's query bounds and return them as UTC instants.

    Both bounds must be timezone-aware: a naive bound names no instant, and
    the package does not guess the zone it was meant in.

    Args:
        start_at: The caller's start bound, unvalidated.
        end_at: The caller's end bound, unvalidated.

    Returns:
        ``(start_utc, end_utc)``, both timezone-aware in UTC.

    Raises:
        TypeError: If either bound is not a ``datetime``, naming which one.
        NaiveDatetimeError: If either bound is naive, naming which one.
    """
    start = require_aware(start_at, name="start_at")
    end = require_aware(end_at, name="end_at")
    return start.astimezone(UTC), end.astimezone(UTC)


class Processing(FrozenModel):
    """A processing mode rendered for both backend families.

    Attributes:
        mode: The canonical mode name, one of :data:`PROCESSING_MODES`.
        precision_ms: The bin width this mode was resolved against; ``<= 0``
            means full resolution.
        epics_operator: Archiver Appliance operator prefix to wrap the PV name
            with (e.g. ``"mean_60"``), or ``None`` when no server-side binning
            applies — either full resolution was requested, or the bin width is
            not a whole number of seconds, which the operator syntax cannot
            express. ``precision_ms`` tells those two apart.
        origin: The instant the bin lattice is anchored on — the query window's
            start, in UTC. Every mode and every channel in one request bins on
            this one grid.
    """

    mode: ProcessingMode
    precision_ms: int
    epics_operator: str | None
    origin: InstanceOf[pd.Timestamp]


def resolve_processing(processing: str, precision_ms: int, window_start: datetime) -> Processing:
    """Validate a processing mode and render it for both backend families.

    Args:
        processing: One of :data:`PROCESSING_MODES`.
        precision_ms: Bin width in milliseconds; ``<= 0`` means full resolution.
        window_start: The query window's start. Bins are anchored here, so the
            same window and width always cut the same lattice — see
            :attr:`Processing.origin`. Timezone-aware, as :func:`utc_window`
            returns it.

    Returns:
        The resolved :class:`Processing`.

    Raises:
        ArchiveQueryError: If ``processing`` is not a known mode, or if a non-raw
            mode is requested with a non-positive ``precision_ms``.
        NaiveDatetimeError: If ``window_start`` is naive.
    """
    require_aware(window_start, name="window_start")
    if not _is_processing_mode(processing):
        raise ArchiveQueryError(
            f"Unknown processing mode {processing!r}. Valid modes: {', '.join(PROCESSING_MODES)}",
            parameter="processing",
        )
    if processing != "raw" and precision_ms <= 0:
        raise ArchiveQueryError(
            f"processing={processing!r} requires precision_ms > 0 (got {precision_ms}); "
            "an aggregate needs a bin width.",
            parameter="precision_ms",
        )
    # The appliance's operator syntax takes whole seconds, so a width that is
    # not a multiple of 1000 ms has no faithful operator: return None rather
    # than flooring to a width the caller did not ask for.
    if precision_ms > 0 and precision_ms % 1000 == 0:
        operator = f"{_EPICS_OPERATORS[processing]}_{precision_ms // 1000}"
    else:
        operator = None
    origin = pd.Timestamp(window_start).tz_convert("UTC")
    return Processing(
        mode=processing,
        precision_ms=precision_ms,
        epics_operator=operator,
        origin=origin,
    )


def long_frame(series: dict[str, SampleSeries]) -> pd.DataFrame:
    """Build the canonical long frame from per-channel series.

    Each channel contributes exactly its own real samples, with no shared index
    and no fill; an empty series contributes no rows. ``value`` is never
    coerced — the input series' dtypes flow through ``concat`` promotion.

    Args:
        series: Mapping of channel address to its sample series, each with a
            UTC-aware ``DatetimeIndex`` and values of any dtype.

    Returns:
        A frame with :data:`LONG_COLUMNS`, sorted by address then timestamp. An
        empty mapping (or a mapping of only empty series) yields the empty frame
        with the same columns, ``value`` defaulting to ``float64``.
    """
    live = {channel: s for channel, s in series.items() if not s.empty}
    if not live:
        # Built explicitly, not via concat: `pd.concat({})` raises
        # ValueError("No objects to concatenate").
        return pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                "address": pd.Series(dtype=str),
                "value": pd.Series(dtype="float64"),
            }
        )
    # Series in, Series out; with the element type unknown the stubs cannot
    # tell the Series overload of `concat` from the DataFrame one.
    stacked: SampleSeries = pd.concat(live, names=["address", "timestamp"])
    frame = stacked.rename("value").reset_index()
    # Real samples can arrive at any datetime64 resolution; the contract is ns.
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns, UTC]")
    frame = frame.sort_values(["address", "timestamp"], ignore_index=True)
    return frame[list(LONG_COLUMNS)]


def _datetime_index(s: SampleSeries) -> pd.DatetimeIndex:
    """The ``DatetimeIndex`` of one channel's samples.

    Raises:
        TypeError: If ``s`` is not indexed by timestamps, naming the channel.
    """
    index = s.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            f"channel {s.name!r} must be indexed by timestamps, got {type(index).__name__}"
        )
    return index


def decimate_raw(s: SampleSeries, precision_ms: int, origin: pd.Timestamp) -> SampleSeries:
    """Keep the last real sample in each ``precision_ms`` bin, at its true timestamp.

    Do NOT replace with ``s.resample(...).agg("last")`` — that relabels the kept
    sample at the bin's leading edge, fabricating a timestamp nothing was ever
    recorded at.

    Args:
        s: One channel's real samples, time-sorted (ascending), any dtype — no
            numeric check, so an enum/status channel round-trips.
        precision_ms: Bin width in milliseconds; ``<= 0`` means full resolution.
        origin: The instant the lattice is anchored on, from
            :attr:`Processing.origin`. Samples before it — an archiver's prior
            point, say — fall into bins stepping backwards from it.

    Returns:
        The subsequence of ``s`` at each bin's last real sample, in original
        order. ``s`` unchanged when empty, when ``precision_ms <= 0``, or when
        it is already sparser than one sample per bin.

    Raises:
        TypeError: If a non-empty ``s`` to be binned is not indexed by
            timestamps.
    """
    if precision_ms <= 0 or s.empty:
        return s
    # Anchored on the query window, the same instant `aggregate_series` hands
    # `resample`: raw and the aggregate modes must bin the same window on the
    # same grid, and epoch-anchored `floor` would diverge from both for any
    # width that does not divide a day.
    width = pd.Timedelta(milliseconds=precision_ms)
    index = _datetime_index(s)
    bins = origin + (index - origin) // width * width
    return s[~bins.duplicated(keep="last")]


def reject_non_numeric(s: SampleSeries, resolved: Processing) -> None:
    """Enforce the contract that only ``raw`` is valid for a non-numeric channel.

    Backends that aggregate server-side never go through :func:`aggregate_series`,
    so they must call this on each returned channel themselves.

    Args:
        s: One channel's samples, ``s.name`` set to the channel so the error can
            name it. An empty series is always accepted.
        resolved: The resolved processing mode. ``"raw"`` is always accepted.

    Raises:
        ArchiveQueryError: If a non-raw mode was requested for a non-empty
            channel whose values are non-numeric, naming the channel.
    """
    if resolved.mode == "raw" or s.empty or pd.api.types.is_numeric_dtype(s):
        return
    channel = repr(s.name) if s.name is not None else "<unnamed channel>"
    raise ArchiveQueryError(
        f"Cannot apply processing={resolved.mode!r} to channel {channel}: its values are "
        "non-numeric (enum/status channels only support processing='raw')",
        parameter="processing",
        address=None if s.name is None else str(s.name),
    )


def aggregate_series(s: SampleSeries, resolved: Processing) -> SampleSeries:
    """Bin one channel's real samples, dropping periods that contained no samples.

    Empty bins are dropped rather than emitted as NaN, so a sparse channel
    queried at a fine width returns fewer rows than it has samples, never more.
    An empty ``s`` is returned unchanged for every mode.

    Args:
        s: One channel's real samples, UTC-aware ``DatetimeIndex``, with
            ``s.name`` set to the channel.
        resolved: The resolved processing mode; ``resolved.precision_ms`` is the
            bin width.

    Returns:
        For ``"raw"``: whatever :func:`decimate_raw` returns. For every other
        mode: one value per non-empty bin, indexed at ns resolution.

    Raises:
        ArchiveQueryError: If a non-raw mode is applied to a non-empty channel
            holding non-numeric values — see :func:`reject_non_numeric`.
        TypeError: If a non-empty ``s`` is not indexed by timestamps.
    """
    # "raw" must never reach the resampler: `resample(...).agg("last")` relabels
    # the kept sample at the bin's leading edge, fabricating a timestamp nothing
    # was ever recorded at.
    if resolved.mode == "raw":
        return decimate_raw(s, resolved.precision_ms, resolved.origin)
    if s.empty:
        return s
    reject_non_numeric(s, resolved)
    if s.index.dtype != "datetime64[ns, UTC]":
        # Resampling a datetime64[s, UTC] index at a sub-second width raises
        # ZeroDivisionError (pandas 3.0).
        s = s.set_axis(_datetime_index(s).as_unit("ns"))
    # origin= pins the lattice to the query window. Without it `resample`
    # defaults to origin="start_day", midnight of *this channel's* first
    # sample, so two channels in one request whose data starts on different
    # days would answer the same window on different grids.
    resampler = s.resample(f"{resolved.precision_ms}ms", origin=resolved.origin)
    counts = resampler.count()
    aggregated = resampler.agg(resolved.mode)
    return aggregated[counts > 0]


def aggregate_long_frame(series: dict[str, SampleSeries], resolved: Processing) -> pd.DataFrame:
    """Bin every channel client-side, then assemble the canonical long frame.

    Backends that bin server-side (EPICS) do not use it — they call
    :func:`long_frame` on the already-binned series after :func:`reject_non_numeric`.

    Args:
        series: Mapping of channel name to its raw sample series, each with a
            UTC-aware ``DatetimeIndex``.
        resolved: The resolved processing mode, applied to every channel
            independently.

    Returns:
        The canonical long frame — see :func:`long_frame`.

    Raises:
        ArchiveQueryError: If a non-raw mode is applied to a non-empty channel
            holding non-numeric values — see :func:`reject_non_numeric`.
    """
    return long_frame({channel: aggregate_series(s, resolved) for channel, s in series.items()})
