"""What an alarm log answers, and in what shape.

The shared answers and the rules every backend follows are in
:mod:`osprey_connectors.service_answers`; the configuration record is the
alarm tree's, from :mod:`osprey_connectors.alarm_state`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime

from osprey_connectors.alarm_state import AlarmConfiguration
from osprey_connectors.alarm_state._contract import _Severity
from osprey_connectors.model import FrozenModel
from osprey_connectors.service_answers import LogPage, Unavailable, _Backend


class AlarmLogEntry(FrozenModel):
    """One logged alarm event, as the logger recorded it.

    ``event_at`` is when the logger says it happened, ``None`` when the row gave
    no time it could be placed by. ``severity`` is the logged one, its ``_ACK``
    folded into ``acknowledged`` as :class:`~osprey_connectors.alarm_state.AlarmSeverity`
    ``.of_recorded`` reads it.
    """

    event_at: AwareDatetime | None = None
    severity: _Severity = None
    acknowledged: bool = False
    status: str | None = None
    message: str | None = None
    user: str | None = None
    kind: Literal["state", "config"] = "state"


class AlarmLog(_Backend, ABC):
    """Read an alarm logger."""

    @abstractmethod
    async def history(
        self,
        address: str,
        *,
        start_at: datetime | None = None,
        limit: int,
        timeout_s: float,
    ) -> LogPage[AlarmLogEntry] | Unavailable:
        """The logged alarm events of *address*, ``AlarmLogEntry`` rows, newest first.

        With *start_at*, an aware instant, only events at or after it, and those
        whose time cannot be placed; a naive one raises
        :class:`~osprey_connectors.errors.NaiveDatetimeError`.
        """

    @abstractmethod
    async def config_for(
        self, address: str, *, timeout_s: float
    ) -> AlarmConfiguration | Unavailable:
        """The last logged alarm configuration of *address*."""

    @abstractmethod
    async def probe(self, *, timeout_s: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""
