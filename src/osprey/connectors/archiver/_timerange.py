"""Shared time-range and processing helpers for archiver connectors.

Private to :mod:`osprey.connectors.archiver`. Normalizes query bounds to UTC,
resolves processing modes for server-side (EPICS operator) and client-side
(pandas aggregation) backends, and assembles per-channel series into the
canonical long frame without ever manufacturing a sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from osprey.utils.config import localize_facility

PROCESSING_MODES = ("raw", "mean", "min", "max", "median", "std", "count")

# The canonical long-format columns every archiver connector returns.
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


def require_datetime(start_date: object, end_date: object) -> None:
    """Raise ``TypeError`` unless both query bounds are ``datetime`` objects.

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

    An aware datetime is converted. A naive one means facility wall-clock and is
    read via :func:`~osprey.utils.config.localize_facility`, which degrades to
    UTC when ``system.timezone`` is unset.

    Args:
        dt: The datetime to normalize.

    Returns:
        The same instant, timezone-aware in UTC.
    """
    return localize_facility(dt).astimezone(UTC)


def utc_window(start_date: datetime, end_date: datetime) -> tuple[datetime, datetime]:
    """Validate a caller's query bounds and return them as UTC instants.

    Args:
        start_date: The caller's start bound, unvalidated.
        end_date: The caller's end bound, unvalidated.

    Returns:
        ``(start_utc, end_utc)``, both timezone-aware in UTC.

    Raises:
        TypeError: If either bound is not a ``datetime``, naming which one.
    """
    require_datetime(start_date, end_date)
    return to_utc(start_date), to_utc(end_date)


@dataclass(frozen=True)
class Processing:
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
    """

    mode: str
    precision_ms: int
    epics_operator: str | None


def resolve_processing(processing: str, precision_ms: int) -> Processing:
    """Validate a processing mode and render it for both backend families.

    Args:
        processing: One of :data:`PROCESSING_MODES`.
        precision_ms: Bin width in milliseconds; ``<= 0`` means full resolution.

    Returns:
        The resolved :class:`Processing`.

    Raises:
        ValueError: If ``processing`` is not a known mode, or if a non-raw mode
            is requested with a non-positive ``precision_ms``.
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
    # The appliance's operator syntax takes whole seconds, so a width that is
    # not a multiple of 1000 ms has no faithful operator: return None rather
    # than flooring to a width the caller did not ask for.
    if precision_ms > 0 and precision_ms % 1000 == 0:
        operator = f"{_EPICS_OPERATORS[processing]}_{precision_ms // 1000}"
    else:
        operator = None
    return Processing(mode=processing, precision_ms=precision_ms, epics_operator=operator)


def long_frame(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Build the canonical long frame from per-channel series.

    Each channel contributes exactly its own real samples, with no shared index
    and no fill; an empty series contributes no rows. ``value`` is never
    coerced — the input series' dtypes flow through ``concat`` promotion.

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
    frame = pd.concat(live, names=["channel", "timestamp"]).rename("value").reset_index()
    # Real samples can arrive at any datetime64 resolution; the contract is ns.
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns, UTC]")
    frame = frame.sort_values(["channel", "timestamp"], ignore_index=True)
    return frame[list(LONG_COLUMNS)]


def decimate_raw(s: pd.Series, precision_ms: int) -> pd.Series:
    """Keep the last real sample in each ``precision_ms`` bin, at its true timestamp.

    Do NOT replace with ``s.resample(...).agg("last")`` — that relabels the kept
    sample at the bin's leading edge, fabricating a timestamp nothing was ever
    recorded at.

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
    # Anchor the lattice the way `resample` does (origin="start_day", midnight
    # of the first sample's day), not epoch-anchored `floor`: the two diverge
    # for widths that don't divide a day, and raw and aggregate modes must bin
    # the same window on the same grid.
    width = pd.Timedelta(milliseconds=precision_ms)
    origin = s.index.min().normalize()
    bins = origin + (s.index - origin) // width * width
    return s[~bins.duplicated(keep="last")]


def reject_non_numeric(s: pd.Series, resolved: Processing) -> None:
    """Enforce the contract that only ``raw`` is valid for a non-numeric channel.

    Backends that aggregate server-side never go through :func:`aggregate_series`,
    so they must call this on each returned channel themselves.

    Args:
        s: One channel's samples, ``s.name`` set to the channel so the error can
            name it. An empty series is always accepted.
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
        ValueError: If a non-raw mode is applied to a non-empty channel holding
            non-numeric values — see :func:`reject_non_numeric`.
    """
    # "raw" must never reach the resampler: `resample(...).agg("last")` relabels
    # the kept sample at the bin's leading edge, fabricating a timestamp nothing
    # was ever recorded at.
    if resolved.mode == "raw":
        return decimate_raw(s, resolved.precision_ms)
    if s.empty:
        return s
    reject_non_numeric(s, resolved)
    if s.index.dtype != "datetime64[ns, UTC]":
        # Resampling a datetime64[s, UTC] index at a sub-second width raises
        # ZeroDivisionError (pandas 3.0).
        s = s.set_axis(s.index.as_unit("ns"))
    resampler = s.resample(f"{resolved.precision_ms}ms")
    counts = resampler.count()
    aggregated = resampler.agg(resolved.mode)
    return aggregated[counts > 0]


def aggregate_long_frame(series: dict[str, pd.Series], resolved: Processing) -> pd.DataFrame:
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
        ValueError: If a non-raw mode is applied to a non-empty channel holding
            non-numeric values — see :func:`reject_non_numeric`.
    """
    return long_frame({channel: aggregate_series(s, resolved) for channel, s in series.items()})
