"""What a put log answers, and in what shape.

The shared answers and the rules every backend follows are in
:mod:`osprey_connectors.service_answers`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from osprey_connectors.model import FrozenModel
from osprey_connectors.service_answers import HistoryPage, Unavailable, _Backend


class PutRow(FrozenModel):
    """One logged write, as the log recorded it.

    ``address`` is what was written: an EPICS PV, a Tango attribute, a DOOCS
    property. ``field`` is an EPICS field put's ``.HIHI``, else ``None``.
    """

    at: str | None = None
    address: str
    field: str | None = None
    old: object = None
    new: object = None
    user: str | None = None
    client: str | None = None


class PutLog(_Backend, ABC):
    """Read a log of channel writes."""

    @abstractmethod
    async def history(
        self, address: str, since: datetime | None, limit: int, *, timeout_s: float
    ) -> HistoryPage[PutRow]:
        """The logged writes of *address*, ``PutRow`` rows, field puts included, newest first."""

    @abstractmethod
    async def probe(self, *, timeout_s: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""
