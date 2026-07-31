"""Shared time-range and processing helpers for archiver connectors.

Private to :mod:`osprey.connectors.archiver`. Collects three rules every archiver
backend needs and that were previously duplicated per connector (or missing): how
a caller's datetime becomes a UTC instant on the wire, how a processing mode
renders for a server-side (EPICS operator) or client-side (pandas aggregation)
backend, and how per-PV series are aligned onto one common grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from osprey.utils.config import get_facility_timezone

PROCESSING_MODES = ("raw", "mean", "min", "max", "median", "std", "count")

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

# pandas resample aggregation names, keyed by our canonical mode.
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


def align_to_grid(
    series: dict[str, pd.Series],
    start: datetime,
    end: datetime,
    precision_ms: int,
) -> pd.DataFrame:
    """Align per-PV series onto one common ``precision_ms`` grid.

    Each series is forward-filled onto the grid so every column shares an index.
    A series with no samples becomes an all-NaN float column rather than being
    dropped, so the frame always carries one column per requested PV.

    Args:
        series: Mapping of PV name to its sample series.
        start: Grid start; naive values are read as UTC.
        end: Grid end.
        precision_ms: Grid resolution in milliseconds (floored at 1).

    Returns:
        A DataFrame indexed by the grid, one column per entry in ``series``.
    """
    resolution = f"{max(1, precision_ms)}ms"
    grid = pd.date_range(start=start, end=end, freq=resolution)
    # Archiver timestamps are UTC; a naive grid must be labeled to match.
    if grid.tz is None:
        grid = grid.tz_localize("UTC")
    aligned = {}
    for pv, s in series.items():
        if s.empty:
            aligned[pv] = pd.Series(index=grid, dtype=float, name=pv)
        else:
            aligned[pv] = s.reindex(s.index.union(grid)).ffill().reindex(grid)
    return pd.DataFrame(aligned)
