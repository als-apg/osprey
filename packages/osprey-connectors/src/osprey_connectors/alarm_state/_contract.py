"""What an alarm-state feed answers, and in what shape.

The shared answers and the rules every backend follows are in
:mod:`osprey_connectors.service_answers`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from osprey_connectors.service_answers import Unavailable, _Backend


@dataclass(frozen=True, kw_only=True)
class AlarmStateRecord:
    """What one PV's alarm is now. The backend alone sets ``stale``; others pass it on."""

    severity: str | None = None
    status: str | None = None
    latched: bool = False
    acknowledged: bool = False
    message: str | None = None
    time: str | None = None
    stale: bool = False


@dataclass(frozen=True, kw_only=True)
class AlarmConfigRecord:
    """One PV's place in the alarm tree."""

    in_tree: Literal["yes", "no", "unknown"]
    path: str | None = None
    guidance: Any = None
    displays: Any = None
    enabled: bool | None = None
    latching: bool | None = None


class AlarmState(_Backend, ABC):
    """Follow a live alarm feed.

    :meth:`start` begins whatever feeds the snapshot and :meth:`aclose` ends it;
    both are idempotent. Until the feed has caught up, reads answer
    ``Unavailable("alarm feed still loading")``, and once caught up with no
    message at all, ``Unavailable("no message from the feed yet")``: never a
    clear state and never ``in_tree: "no"``. A read of a started feed waits for
    catch-up up to its deadline and only then answers ``alarm feed still
    loading``; a feed that already knows its answer, unreachable or unbuildable,
    answers at once.
    """

    @abstractmethod
    async def start(self) -> None:
        """Begin feeding the snapshot."""

    @abstractmethod
    async def aclose(self) -> None:
        """Stop feeding the snapshot."""

    @abstractmethod
    async def state_for(self, pv: str, deadline: float) -> AlarmStateRecord | Unavailable | None:
        """The alarm state of *pv*, waiting up to *deadline* seconds for catch-up.

        ``None`` is a caught-up feed that holds none for it.
        """

    @abstractmethod
    async def config_for(self, pv: str, deadline: float) -> AlarmConfigRecord | Unavailable:
        """The live alarm configuration of *pv*, waiting up to *deadline* seconds for catch-up."""

    @abstractmethod
    async def probe(self, deadline: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""
