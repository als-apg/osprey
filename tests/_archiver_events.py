"""Archiver events and top-off records stated as a test states them, read as the view reads them.

A test writes an event as the facility file and the simulator view state it, a
mapping; the series and the archive rewrite take the view's records. These
validate one into the other, and leave a record as it is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import TypeAdapter

from osprey_connectors.simulation.records import ArchiverEvent, TopoffRecord

#: The reader of one event, as the view reads it.
EVENT: TypeAdapter[ArchiverEvent] = TypeAdapter(ArchiverEvent)


def event(raw: Mapping[str, object] | ArchiverEvent) -> ArchiverEvent:
    """One event as the view's record."""
    return EVENT.validate_python(raw)


def events(raws: Iterable[Mapping[str, object] | ArchiverEvent]) -> list[ArchiverEvent]:
    """A script of events as the view's records."""
    return [event(raw) for raw in raws]


def topoff(raw: Mapping[str, object] | TopoffRecord | None) -> TopoffRecord | None:
    """A ``topoff`` seed record as the view's record, or ``None``."""
    if raw is None or isinstance(raw, TopoffRecord):
        return raw
    return TopoffRecord.model_validate(raw)
