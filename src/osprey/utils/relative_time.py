"""Declarative relative timestamps, resolved against an anchor at load time.

Demo and seed data (simulation scenario bundles, the standalone ARIEL demo
logbook) express timestamps *relative* to "now" — ``{days_ago, time}`` — rather
than as absolute dates. They resolve to concrete datetimes at the moment data
enters the system (``osprey sim apply`` for scenario bundles; ``osprey ariel
ingest``/``quickstart`` for the generic adapter), so the data always lands at a
recent, deterministic wall-clock position without ever mutating the source file
at build time.

This is the single shared primitive — kept in :mod:`osprey.utils` so neither
the simulation engine nor the ARIEL ingestion layer depends on the other.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from datetime import time as dtime


@dataclass(frozen=True)
class RelativeTimestamp:
    """A timestamp expressed relative to an apply-/ingest-time anchor.

    ``days_ago`` is a whole-day shift back from the anchor date; ``time`` pins
    the resolved time-of-day (tz-naive local), so hardcoded prose clock times in
    an entry body stay consistent with the resolved timestamp.
    """

    days_ago: int
    time: dtime


def resolve_relative_timestamp(spec: RelativeTimestamp, now: datetime) -> datetime:
    """Resolve a relative timestamp against an anchor ``now``.

    Shifts ``now`` back ``spec.days_ago`` whole calendar days and pins the
    resolved time-of-day to ``spec.time``. The shift is computed on the date
    component, so it is DST-safe (it counts calendar days, not 24-hour spans)
    and the resolved wall-clock time is exactly ``spec.time`` regardless of any
    DST transition between the two dates. The result carries ``now``'s tzinfo
    (naive in, naive out).

    Args:
        spec: The relative timestamp (whole-day offset plus time-of-day).
        now: The anchor instant. For scenario bundles this is the shared
            apply-time T0 (so narrative and telemetry resolve against one clock);
            for generic ingestion it is the ingest-time "now".

    Returns:
        The resolved timestamp, with the same tzinfo as ``now``.
    """
    target_date = (now - timedelta(days=spec.days_ago)).date()
    return datetime.combine(target_date, spec.time, tzinfo=now.tzinfo)
