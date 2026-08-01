"""Shared time-range and processing helpers for archiver connectors.

Private to :mod:`osprey.connectors.archiver`. Collects three rules every archiver
backend needs and that were previously duplicated per connector (or missing): how
a caller's datetime becomes a UTC instant on the wire, how a processing mode
renders for a server-side (EPICS operator) or client-side (pandas aggregation)
backend, and how per-channel series are assembled into the canonical long frame
without ever manufacturing a sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from osprey.utils.config import get_facility_timezone

PROCESSING_MODES = ("raw", "mean", "min", "max", "median", "std", "count")

# The canonical long-format contract every archiver connector returns. Sorted by
# channel then timestamp; an empty result is an empty frame with these columns
# and dtypes.
LONG_COLUMNS = ("timestamp", "channel", "value")

# EPICS Archiver Appliance operator names, keyed by our canonical mode. "raw"
# maps to lastSample so a binned raw query keeps its long-standing behavior.
_EPICS_OPERATORS = {
    "raw": "lastSample",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "median": "median",
    "std": "std",
    "count": "count",
}

# pandas resample aggregation names, keyed by our canonical mode. "raw" has no
# entry on purpose: it is decimated by decimate_raw, and the aggregation that
# looks equivalent, `resample(...).agg("last")`, is not — it relabels the kept
# sample at the bin's leading edge. Rendering "last" here would put that
# substitution one attribute access away.
_PANDAS_AGGS = {
    "mean": "mean",
    "min": "min",
    "max": "max",
    "median": "median",
    "std": "std",
    "count": "count",
}


def require_datetime(start_date: object, end_date: object) -> None:
    """Raise ``TypeError`` unless both query bounds are ``datetime`` objects.

    Without it :func:`to_utc` fails on a ``str`` with an ``AttributeError``
    naming ``tzinfo`` — an error that says nothing about which argument was
    wrong.

    Args:
        start_date: The caller's start bound, unvalidated.
        end_date: The caller's end bound, unvalidated.

    Raises:
        TypeError: If either bound is not a ``datetime``, naming which one.
    """
    for name, value in (("start_date", start_date), ("end_date", end_date)):
        if not isinstance(value, datetime):
            raise TypeError(f"{name} must be a datetime object, got {type(value)}")


def to_utc(dt: datetime) -> datetime:
    """Return ``dt`` as a timezone-aware UTC datetime.

    An aware datetime is converted. A naive one carries no zone, so it is read
    as facility-local — matching how the rest of the framework reads operator
    wall-clock times — and then converted. ``get_facility_timezone()`` degrades
    to UTC when ``system.timezone`` is unset, so an unconfigured deployment
    still treats naive input as UTC.

    Args:
        dt: The datetime to normalize.

    Returns:
        The same instant, timezone-aware in UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=get_facility_timezone())
    return dt.astimezone(UTC)


@dataclass(frozen=True)
class Processing:
    """A processing mode rendered for both backend families.

    Attributes:
        mode: The canonical mode name, one of :data:`PROCESSING_MODES`.
        precision_ms: The bin width this mode was resolved against; ``<= 0``
            means full resolution. Carried here so the width a backend resolved
            with is necessarily the width it bins with.
        epics_operator: Archiver Appliance operator prefix to wrap the PV name
            with (e.g. ``"mean_60"``), or ``None`` when no server-side binning
            applies — either full resolution was requested, or the bin width is
            not a whole number of seconds, which the operator syntax cannot
            express. ``precision_ms`` tells those two apart; see
            :meth:`EPICSArchiverConnector.get_data`, which turns the second into
            a ``ValueError``.
        pandas_agg: The pandas resample aggregation name for client-side
            backends, or ``None`` for ``"raw"`` — which is decimated by
            :func:`decimate_raw`, not resampled. Read it only from the non-raw
            branch that :func:`aggregate_series` already takes.
    """

    mode: str
    precision_ms: int
    epics_operator: str | None
    pandas_agg: str | None


def resolve_processing(processing: str, precision_ms: int) -> Processing:
    """Validate a processing mode and render it for both backend families.

    Args:
        processing: One of :data:`PROCESSING_MODES`.
        precision_ms: Bin width in milliseconds; ``<= 0`` means full resolution.

    Returns:
        The resolved :class:`Processing`.

    Raises:
        ValueError: If ``processing`` is not a known mode, or if a non-raw mode
            is requested with a non-positive ``precision_ms`` — silently
            returning raw samples for an aggregate request is the failure this
            helper exists to prevent.
    """
    if processing not in PROCESSING_MODES:
        raise ValueError(
            f"Unknown processing mode {processing!r}. Valid modes: {', '.join(PROCESSING_MODES)}"
        )
    if processing != "raw" and precision_ms <= 0:
        raise ValueError(
            f"processing={processing!r} requires precision_ms > 0 (got {precision_ms}); "
            "an aggregate needs a bin width."
        )
    # The appliance's operator syntax takes a whole number of seconds, so a
    # width that is not a multiple of 1000 ms has no faithful operator. Return
    # None rather than flooring: `max(1, precision_ms // 1000)` silently served
    # both a 500 ms and a 1500 ms request at 1 s. This is the one place the rule
    # lives — EPICS turns the None into a ValueError, and the client-side
    # backends bin at the exact width themselves.
    if precision_ms > 0 and precision_ms % 1000 == 0:
        operator = f"{_EPICS_OPERATORS[processing]}_{precision_ms // 1000}"
    else:
        operator = None
    return Processing(
        mode=processing,
        precision_ms=precision_ms,
        epics_operator=operator,
        pandas_agg=_PANDAS_AGGS.get(processing),
    )


def long_frame(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Build the canonical long frame from per-channel series.

    Each channel contributes exactly its own real samples, with no shared index
    and no fill — see :meth:`ArchiverConnector.get_data` for the contract. A
    channel whose series is empty contributes no rows and does not appear at
    all, rather than being padded with NaNs to keep a column.

    ``value`` is not dtype-constrained and nothing here coerces it: the input
    series' own dtype flows through, so an all-numeric mapping stays ``float64``
    by ordinary ``concat`` promotion while a mapping containing an enum/status
    channel takes whatever dtype holds the mix (typically ``object``).

    Args:
        series: Mapping of channel name to its sample series, each with a
            UTC-aware ``DatetimeIndex`` and values of any dtype.

    Returns:
        A frame with :data:`LONG_COLUMNS`, sorted by channel then timestamp. An
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
                "channel": pd.Series(dtype=str),
                "value": pd.Series(dtype="float64"),
            }
        )
    # Concatenating the mapping keys each channel's samples under its own name,
    # producing a (channel, timestamp) MultiIndex that reset_index turns into
    # the two long-format label columns -- no per-channel frame construction.
    frame = pd.concat(live, names=["channel", "timestamp"]).rename("value").reset_index()
    # Real samples can arrive at any datetime64 resolution; the contract is ns.
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns, UTC]")
    frame = frame.sort_values(["channel", "timestamp"], ignore_index=True)
    return frame[list(LONG_COLUMNS)]


def decimate_raw(s: pd.Series, precision_ms: int) -> pd.Series:
    """Keep the last real sample in each ``precision_ms`` bin, at its true timestamp.

    "raw" processing's own binning: it computes no derived value, so every row
    it returns is a real, previously-recorded sample carrying its own real
    timestamp. Do NOT replace it with
    ``s.resample(f"{precision_ms}ms").agg("last")`` — that returns the same
    *value* but relabels it at the bin's leading edge, fabricating a timestamp
    nothing was ever recorded at.

    Args:
        s: One channel's real samples, time-sorted (ascending), any dtype — no
            numeric check, so an enum/status channel round-trips.
        precision_ms: Bin width in milliseconds; ``<= 0`` means full resolution.

    Returns:
        The subsequence of ``s`` at each bin's last real sample, in original
        order. ``s`` unchanged when empty, when ``precision_ms <= 0``, or when
        it is already sparser than one sample per bin.
    """
    if precision_ms <= 0 or s.empty:
        return s
    # Anchor the lattice the way `aggregate_series`' `resample` does, not the
    # way `DatetimeIndex.floor` does. `floor` is epoch-anchored; `resample`
    # defaults to origin="start_day", i.e. midnight of the first sample's day.
    # The two agree for every width that divides a day and diverge otherwise
    # (measured: at 7000 ms they sit up to 6 s apart, date-dependent), which
    # left `processing="raw"` and `processing="mean"` binning the same window on
    # two different grids. Reproducing resample's anchor here rather than moving
    # `aggregate_series` onto `floor` keeps existing aggregate output unchanged.
    width = pd.Timedelta(milliseconds=precision_ms)
    origin = s.index.min().normalize()
    bins = origin + (s.index - origin) // width * width
    return s[~bins.duplicated(keep="last")]


def reject_non_numeric(s: pd.Series, resolved: Processing) -> None:
    """Enforce the contract that only ``raw`` is valid for a non-numeric channel.

    Aggregating (mean/min/max/median/std/count) over enum or status values is
    undefined, so every backend must raise rather than return something that
    looks like an aggregate. Client-side backends get this free through
    :func:`aggregate_series`; a backend that pushes the aggregation down to its
    server (EPICS) never calls that, so it must call this on each returned
    channel instead — otherwise ``mean_60(SR:MODE)`` comes back with its raw
    ``CW``/``STANDBY`` values labelled as means.

    Args:
        s: One channel's samples, ``s.name`` set to the channel so the error can
            name it. An empty series is always accepted — a channel that matched
            no samples must not fail the query for every other one.
        resolved: The resolved processing mode. ``"raw"`` is always accepted.

    Raises:
        ValueError: If a non-raw mode was requested for a non-empty channel
            whose values are non-numeric, naming the channel.
    """
    if resolved.mode == "raw" or s.empty or pd.api.types.is_numeric_dtype(s):
        return
    channel = repr(s.name) if s.name is not None else "<unnamed channel>"
    raise ValueError(
        f"Cannot apply processing={resolved.mode!r} to channel {channel}: its values are "
        "non-numeric (enum/status channels only support processing='raw')"
    )


def aggregate_series(s: pd.Series, resolved: Processing) -> pd.Series:
    """Bin one channel's real samples, dropping periods that contained no samples.

    Dropping empty bins rather than emitting them as NaN is the single reason no
    bin-width floor is needed anywhere: ``resample`` can never upsample onto a
    fixed grid, so a sparse channel queried at a fine width returns *fewer* rows
    than it has samples, never more. An empty ``s`` is returned unchanged for
    every mode — there is nothing to aggregate or to reject, and one empty
    channel must not fail the query for every other one.

    Args:
        s: One channel's real samples, UTC-aware ``DatetimeIndex``, named
            (``s.name`` set to the channel) so a non-numeric aggregation request
            can name the offending channel.
        resolved: The resolved processing mode. ``resolved.precision_ms`` is the
            bin width — it travels with the mode rather than alongside it, so a
            caller cannot resolve at one width and bin at another.
            ``resolved.pandas_agg`` names the per-bin aggregation, and is
            ``None`` for ``"raw"``, which is decimated by :func:`decimate_raw`
            instead of resampled.

    Returns:
        For ``"raw"``: whatever :func:`decimate_raw` returns. For every other
        mode: one value per *non-empty* bin, aggregated with
        ``resolved.pandas_agg``, indexed at ns resolution.

    Raises:
        ValueError: If a non-raw mode is applied to a non-empty channel holding
            non-numeric values — see :func:`reject_non_numeric`.
    """
    if resolved.mode == "raw":
        return decimate_raw(s, resolved.precision_ms)
    if s.empty:
        return s
    reject_non_numeric(s, resolved)
    if s.index.dtype != "datetime64[ns, UTC]":
        # Load-bearing, not defensive dtype churn. A source that builds its
        # index with unit="s" yields datetime64[s, UTC], and resampling that at
        # "500ms" raises ZeroDivisionError: integer modulo by zero (pandas 3.0).
        s = s.set_axis(s.index.as_unit("ns"))
    resampler = s.resample(f"{resolved.precision_ms}ms")
    counts = resampler.count()
    aggregated = resampler.agg(resolved.pandas_agg)
    return aggregated[counts > 0]
