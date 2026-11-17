"""Declarative relative timestamps, resolved against an anchor at load time.

Demo and seed data (simulation scenario bundles and their narratives, seed
files the generic ARIEL adapter reads) express timestamps *relative* to "now" —
``{days_ago, time}`` — rather than as absolute dates. They resolve to concrete
datetimes at the moment data enters the system (``osprey sim apply`` and the
deploy-time seed for scenario narratives; ``osprey ariel ingest``/``quickstart``
for the generic adapter), so the data always lands at a
recent, deterministic wall-clock position without ever mutating the source file
at build time.

This is the single shared primitive, living in :mod:`osprey_connectors` so
neither the simulation engine nor the ARIEL ingestion layer depends on the
other.
"""

from datetime import datetime, timedelta
from datetime import time as dtime
from typing import Annotated, TypeAlias

from pydantic import BeforeValidator, Field
from pydantic_core import PydanticCustomError

from osprey_connectors.model import FrozenModel

__all__ = [
    "RelativeTimestamp",
    "TimeOfDay",
    "resolve_relative_timestamp",
]


def _local_time(value: object) -> object:
    """An ``HH:MM[:SS]`` text as the time of day it states; a time with an offset is refused."""
    if isinstance(value, str):
        try:
            value = dtime.fromisoformat(value)
        except ValueError:
            raise PydanticCustomError(
                "time_of_day", "Input should be an HH:MM[:SS] time of day"
            ) from None
    if isinstance(value, dtime) and value.tzinfo is not None:
        raise PydanticCustomError(
            "time_offset", "Input is local time and must not carry a timezone offset"
        )
    return value


#: A wall-clock time of day without a timezone offset. A string is read as
#: ``HH:MM[:SS]`` (``08:00`` is ``08:00:00``); a time that states an offset is
#: refused, because the facility timezone places it. JSON mode writes it as
#: :meth:`datetime.time.isoformat` does, ``HH:MM:SS``.
TimeOfDay: TypeAlias = Annotated[dtime, BeforeValidator(_local_time)]


class RelativeTimestamp(FrozenModel):
    """A timestamp expressed relative to an apply-/ingest-time anchor.

    ``days_ago`` is a whole-day shift back from the anchor date, 0 or more and
    never a ``bool``; ``time`` pins the resolved time-of-day (local, without an
    offset), so hardcoded prose clock times in an entry body stay consistent
    with the resolved timestamp. Read from a file it is ``{days_ago, time}``
    with ``time`` an ``HH:MM[:SS]`` text, and it writes ``time`` back as
    ``HH:MM:SS``.
    """

    days_ago: int = Field(ge=0)
    time: TimeOfDay


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
