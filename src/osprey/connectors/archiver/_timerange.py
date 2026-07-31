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

# pandas resample aggregation names, keyed by our canonical mode. "raw" is kept
# here for symmetry (every mode renders a Processing.pandas_agg, "raw" included
# — see test_each_mode_renders_for_both_backends) even though aggregate_series
# never resamples with it: "raw" is decimated by decimate_raw instead, which
# keeps a bin's last *real* sample rather than relabeling it at the bin edge the
# way `resample(...).agg("last")` would.
_PANDAS_AGGS = {
    "raw": "last",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "median": "median",
    "std": "std",
    "count": "count",
}


def to_utc(dt: datetime) -> datetime:
    """Return ``dt`` as a timezone-aware UTC datetime.

    An aware datetime is converted. A naive one carries no zone, so it is read as
    facility-local — matching how the rest of the framework reads operator
    wall-clock times — and then converted. ``get_facility_timezone()`` degrades to
    UTC when ``system.timezone`` is unset, so an unconfigured deployment sees
    naive input treated as UTC exactly as before.

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
        epics_operator: Archiver Appliance operator prefix to wrap the PV name
            with (e.g. ``"mean_60"``), or ``None`` when no server-side binning
            applies.
        pandas_agg: The pandas resample aggregation name for client-side backends.
    """

    mode: str
    epics_operator: str | None
    pandas_agg: str


def resolve_processing(processing: str, precision_ms: int) -> Processing:
    """Validate a processing mode and render it for both backend families.

    Args:
        processing: One of :data:`PROCESSING_MODES`.
        precision_ms: Bin width in milliseconds; ``<= 0`` means full resolution.

    Returns:
        The resolved :class:`Processing`.

    Raises:
        ValueError: If ``processing`` is not a known mode, or if a non-raw mode is
            requested with a non-positive ``precision_ms`` — an aggregate with no
            bin width is meaningless, and silently returning raw samples instead
            is the failure mode this helper exists to prevent.
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
    if precision_ms > 0:
        n_secs = max(1, precision_ms // 1000)
        operator = f"{_EPICS_OPERATORS[processing]}_{n_secs}"
    else:
        operator = None
    return Processing(
        mode=processing,
        epics_operator=operator,
        pandas_agg=_PANDAS_AGGS[processing],
    )


def long_frame(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Build the canonical long frame from per-channel series.

    Nothing is manufactured: the frame contains exactly the real samples each
    input series carries, with no shared index and no fill. A channel whose
    series is empty contributes no rows and does not appear in the output at
    all — it is not padded with NaNs to preserve a "one column per channel"
    shape the way the old wide format required.

    ``value`` is not dtype-constrained. A channel may carry non-numeric
    samples (e.g. an enum/status PV archived as a string), and no channel is
    coerced or dropped for being non-numeric — this function performs no
    dtype coercion at all, so the input series' own dtype flows straight
    through. When every input series is numeric, ordinary ``pandas.concat``
    type promotion keeps the combined ``value`` column ``float64``; once any
    channel is non-numeric, the column takes on whatever dtype pandas needs to
    hold the mix (typically ``object`` once a non-numeric channel is combined
    with a numeric one).

    Args:
        series: Mapping of channel name to its sample series. Each series must
            have a UTC-aware ``DatetimeIndex``. Values may be any dtype the
            source produced — numeric, string, or otherwise.

    Returns:
        A frame with :data:`LONG_COLUMNS`, sorted by channel then timestamp. An
        empty mapping (or a mapping of only empty series) yields the empty frame
        with the same columns; ``value`` defaults to ``float64`` in that case,
        since there is no data to infer a dtype from.
    """
    frames = [
        pd.DataFrame(
            {
                "timestamp": s.index,
                "channel": channel,
                "value": s.to_numpy(),
            }
        )
        for channel, s in series.items()
        if not s.empty
    ]
    if not frames:
        return pd.DataFrame(
            {
                "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                "channel": pd.Series(dtype=str),
                "value": pd.Series(dtype="float64"),
            }
        )
    frame = pd.concat(frames, ignore_index=True)
    # Real samples can arrive at any datetime64 resolution; the contract is ns.
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns, UTC]")
    frame = frame.sort_values(["channel", "timestamp"], ignore_index=True)
    return frame[list(LONG_COLUMNS)]


def decimate_raw(s: pd.Series, precision_ms: int) -> pd.Series:
    """Keep the last real sample in each ``precision_ms`` bin, with its true timestamp.

    This is "raw" processing's own binning. Unlike an aggregate mode it never
    computes a derived value, so the row kept is a real, previously-recorded
    sample at its own real timestamp — never a bin-edge timestamp. Contrast with
    ``s.resample(f"{precision_ms}ms").agg("last")``, which would relabel that
    same sample at the *bin's* start time instead, fabricating a timestamp
    nothing was ever recorded at — exactly what this framework's "nothing is
    manufactured" contract forbids. Dtype-agnostic (no numeric check), so an
    enum/status channel still round-trips under ``processing="raw"``. A channel
    already sparser than ``precision_ms`` is left bit-identical: with at most one
    real sample per bin, there is nothing to drop.

    Args:
        s: One channel's real samples, time-sorted (ascending), any dtype.
        precision_ms: Bin width in milliseconds; ``<= 0`` means full resolution.

    Returns:
        The subsequence of ``s`` at each bin's last real sample, in original
        order. ``s`` unchanged when it is empty or ``precision_ms <= 0``.
    """
    if precision_ms <= 0 or s.empty:
        return s
    bins = s.index.floor(f"{precision_ms}ms")
    return s[~pd.Series(bins).duplicated(keep="last").to_numpy()]


def aggregate_series(s: pd.Series, precision_ms: int, resolved: Processing) -> pd.Series:
    """Bin one channel's real samples, dropping periods that contained no samples.

    This is the single reason no bin-width floor is needed anywhere: a bin with
    no samples is dropped rather than emitted as NaN, so ``resample`` can never
    upsample onto a fixed grid. A sparse channel queried at a fine
    ``precision_ms`` returns *fewer* rows than it has samples, never more. An
    empty ``s`` is returned unchanged for every mode, "raw" included — there is
    nothing to aggregate or to reject as non-numeric, and a channel that
    matched zero samples must not fail a query for every *other* requested
    channel.

    Args:
        s: One channel's real samples, UTC-aware ``DatetimeIndex``. The index may
            carry any datetime64 resolution (e.g. a source that builds its index
            with ``unit="s"`` yields ``datetime64[s, UTC]``); it is normalized to
            ns internally so ``resample`` can bin at any ``precision_ms``,
            including sub-second or non-whole-second widths that a coarser
            resolution cannot represent. Named (``s.name`` set to the channel)
            so a non-numeric aggregation request can name the offending channel.
        precision_ms: Bin width in milliseconds. For ``"raw"``, ``<= 0`` means
            full resolution; otherwise it is the width ``decimate_raw`` keeps
            one real sample per bin from.
        resolved: The resolved processing mode; ``resolved.pandas_agg`` names the
            aggregation to apply per bin (unused for ``"raw"`` — see
            :func:`decimate_raw`).

    Returns:
        For ``"raw"``: ``s`` decimated by :func:`decimate_raw` — one real sample
        (its own real timestamp, unchanged value) per ``precision_ms`` bin, or
        every real sample when ``precision_ms <= 0``. For every other mode: one
        value per non-empty ``precision_ms`` bin, aggregated with
        ``resolved.pandas_agg``, indexed at ns resolution. An empty ``s`` is
        returned unchanged regardless of mode.

    Raises:
        ValueError: If ``resolved.mode`` is not ``"raw"``, ``s`` is non-empty,
            and ``s`` holds non-numeric values (e.g. an enum/status channel
            archived as a string) — aggregating (mean/min/max/median/std/count)
            over non-numeric samples is undefined. ``raw`` is always valid for
            any channel, numeric or not.
    """
    if resolved.mode == "raw":
        return decimate_raw(s, precision_ms)
    if s.empty:
        return s
    if not pd.api.types.is_numeric_dtype(s):
        channel = repr(s.name) if s.name is not None else "<unnamed channel>"
        raise ValueError(
            f"Cannot apply processing={resolved.mode!r} to channel {channel}: its values are "
            "non-numeric (enum/status channels only support processing='raw')"
        )
    if s.index.dtype != "datetime64[ns, UTC]":
        s = s.set_axis(s.index.as_unit("ns"))
    resampler = s.resample(f"{precision_ms}ms")
    counts = resampler.count()
    aggregated = resampler.agg(resolved.pandas_agg)
    return aggregated[counts > 0]
