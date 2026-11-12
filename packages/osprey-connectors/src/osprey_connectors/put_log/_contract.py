"""What a put log answers, and in what shape.

The shared answers and the rules every backend follows are in
:mod:`osprey_connectors.service_answers`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from osprey_connectors.service_answers import HistoryPage, Unavailable, _Backend


@dataclass(frozen=True, kw_only=True)
class PutRow:
    """One logged write, as the log recorded it.

    ``channel`` is what was written: an EPICS PV, a Tango attribute, a DOOCS
    property. ``field`` is an EPICS field put's ``.HIHI``, else ``None``.
    """

    at: str | None = None
    channel: str | None = None
    field: str | None = None
    old: Any = None
    new: Any = None
    user: str | None = None
    client: str | None = None


class PutLog(_Backend, ABC):
    """Read a log of channel writes."""

    @abstractmethod
    async def history(
        self, channel: str, since: datetime | None, limit: int, deadline: float
    ) -> HistoryPage:
        """The logged writes of *channel*, ``PutRow`` rows, field puts included, newest first."""

    @abstractmethod
    async def probe(self, deadline: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""
