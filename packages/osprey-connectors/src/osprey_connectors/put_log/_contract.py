"""What a put log answers, and in what shape.

The shared answers and the rules every backend follows are in
:mod:`osprey_connectors.service_answers`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Self

from pydantic import AwareDatetime, model_validator

from osprey_connectors.model import FrozenModel
from osprey_connectors.service_answers import (
    LogPage,
    Unavailable,
    _Backend,
    _require_one_time,
)


class PutLogEntry(FrozenModel):
    """One logged write, as the log recorded it.

    ``address`` is what was written: an EPICS PV, a Tango attribute, a DOOCS
    property. ``field`` is an EPICS field put's ``.HIHI``, else ``None``.
    ``put_at`` is when the write happened. A log can record a stamp that cannot
    be placed on the clock, such as an IOC's local time with no zone: then
    ``put_at`` is ``None`` and ``recorded_stamp`` holds the stamp's text, and no
    zone is guessed for it. ``recorded_stamp`` is ``None`` whenever ``put_at`` is set.
    """

    put_at: AwareDatetime | None = None
    recorded_stamp: str | None = None
    address: str
    field: str | None = None
    old: object = None
    new: object = None
    user: str | None = None
    client: str | None = None

    @model_validator(mode="after")
    def _one_time(self) -> Self:
        _require_one_time(self.put_at, self.recorded_stamp)
        return self


class PutLog(_Backend, ABC):
    """Read a log of channel writes."""

    @abstractmethod
    async def history(
        self,
        address: str,
        *,
        start_at: datetime | None = None,
        limit: int,
        timeout_s: float,
    ) -> LogPage[PutLogEntry] | Unavailable:
        """The logged writes of *address*, ``PutLogEntry`` rows, field puts included, newest first.

        With *start_at*, an aware instant, only writes at or after it, and those
        whose time cannot be placed; a naive one raises
        :class:`~osprey_connectors.errors.NaiveDatetimeError`.
        """

    @abstractmethod
    async def probe(self, *, timeout_s: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""
