"""What an alarm log answers, and in what shape.

The shared answers and the rules every backend follows are in
:mod:`osprey_connectors.service_answers`; the configuration record is the
alarm tree's, from :mod:`osprey_connectors.alarm_state`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from osprey_connectors.alarm_state import AlarmConfigRecord
from osprey_connectors.service_answers import HistoryPage, Unavailable, _Backend


@dataclass(frozen=True, kw_only=True)
class AlarmHistoryRow:
    """One logged alarm event, as the logger recorded it."""

    time: str | None = None
    severity: str | None = None
    status: str | None = None
    message: str | None = None
    user: str | None = None
    kind: Literal["state", "config"] = "state"


class AlarmLog(_Backend, ABC):
    """Read an alarm logger."""

    @abstractmethod
    async def history(
        self, channel: str, since: datetime | None, limit: int, deadline: float
    ) -> HistoryPage[AlarmHistoryRow]:
        """The logged alarm events of *channel*, ``AlarmHistoryRow`` rows, newest first."""

    @abstractmethod
    async def config_for(self, channel: str, deadline: float) -> AlarmConfigRecord | Unavailable:
        """The last logged alarm configuration of *channel*."""

    @abstractmethod
    async def probe(self, deadline: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""
