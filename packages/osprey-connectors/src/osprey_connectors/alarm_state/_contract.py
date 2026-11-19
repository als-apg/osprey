"""What an alarm-state feed answers, and in what shape.

The shared answers and the rules every backend follows are in
:mod:`osprey_connectors.service_answers`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BeforeValidator,
    SerializerFunctionWrapHandler,
    model_serializer,
)
from typing_extensions import override

from osprey_connectors.model import FrozenModel
from osprey_connectors.service_answers import NotListed, Unavailable, _Backend

#: What an alarm service appends to a severity an operator has acknowledged.
_ACK_SUFFIX = "_ACK"


class AlarmSeverity(StrEnum):
    """How bad an alarm is, as the alarm services spell it; members are in order, worst last."""

    OK = "OK"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    INVALID = "INVALID"
    UNDEFINED = "UNDEFINED"

    @property
    def rank(self) -> int:
        """The severity's place in the order, higher is worse: ``OK`` is 0."""
        return _RANKS[self]

    @classmethod
    def of_recorded(cls, word: object) -> tuple[AlarmSeverity | None, bool]:
        """``(severity, acknowledged)`` of the severity word an alarm service recorded.

        A word ending in ``_ACK`` is that severity, acknowledged. ``None`` stays
        ``None``; any other value is read as its text, and a word that is no
        severity reads as ``UNDEFINED``, so it is never taken for a recovery.
        """
        if word is None:
            return None, False
        recorded = word if isinstance(word, str) else str(word)
        bare = recorded.removesuffix(_ACK_SUFFIX)
        try:
            severity = cls(bare)
        except ValueError:
            severity = cls.UNDEFINED
        return severity, bare != recorded


_RANKS = {severity: rank for rank, severity in enumerate(AlarmSeverity)}


def _severity_member(value: object) -> object:
    """A severity field's value as its :class:`AlarmSeverity` member, from the member or its word.

    Any other value is passed on, for the field to refuse.
    """
    if isinstance(value, str) and not isinstance(value, AlarmSeverity):
        try:
            return AlarmSeverity(value)
        except ValueError:
            return value
    return value


#: A record's severity: a member, or its word, which the strict model reads as the member.
_Severity = Annotated[AlarmSeverity | None, BeforeValidator(_severity_member)]


class AlarmCondition(FrozenModel):
    """What one channel's alarm is now. The backend alone sets ``stale``; others pass it on.

    ``severity`` is the alarm server's, its ``_ACK`` folded into ``acknowledged``.
    ``event_at`` is when the alarm server says the state began, ``None`` when it
    gave no time it could be placed by.
    """

    severity: _Severity = None
    status: str | None = None
    latched: bool = False
    acknowledged: bool = False
    message: str | None = None
    event_at: AwareDatetime | None = None
    stale: bool = False


class AlarmEntry(FrozenModel):
    """One channel the feed holds a state for.

    ``address`` is the channel address; ``path`` is the alarm-tree place the state
    was reported under, ``None`` for a backend with no tree. Tango alarm devices
    and DOOCS alarm servers fill ``path`` where they keep a hierarchy.
    """

    address: str
    path: str | None
    condition: AlarmCondition


class AlarmStates(FrozenModel):
    """Every channel's alarm state now, and the counts per severity.

    ``counts`` is derived from the rows' severities, ``OK`` included and a ``None``
    severity not counted, so it cannot disagree with them. ``last_contact_at`` is
    when, in UTC, the backend last heard from the feed (a message or an offset
    check), ``None`` when it never has; ``stale`` is the backend's flag. ``counts``
    is derived, never passed, and is part of the record's dump.
    """

    rows: tuple[AlarmEntry, ...]
    stale: bool
    last_contact_at: AwareDatetime | None

    @property
    def counts(self) -> Mapping[str, int]:
        """The rows per severity, by severity name."""
        counts: dict[str, int] = {}
        for row in self.rows:
            severity = row.condition.severity
            if severity is not None:
                counts[severity.value] = counts.get(severity.value, 0) + 1
        return dict(sorted(counts.items()))

    @model_serializer(mode="wrap")
    def _dump_with_counts(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        dumped: dict[str, object] = handler(self)
        return {**dumped, "counts": self.counts}


class AlarmConfiguration(FrozenModel):
    """One channel's place in the alarm tree."""

    in_tree: Literal["yes", "no", "unknown"]
    path: str | None = None
    guidance: object = None
    displays: object = None
    enabled: bool | None = None
    latching: bool | None = None


class AlarmState(_Backend, ABC):
    """Follow a live alarm feed.

    :meth:`start` begins whatever feeds the snapshot and :meth:`aclose` ends it;
    both are idempotent. Until the feed has caught up, reads answer
    ``Unavailable(cause="not_ready", reason="alarm feed still loading")``, and once
    caught up with no message at all, ``Unavailable(cause="unreachable",
    reason="no message from the feed yet")``: never a
    clear state and never ``in_tree: "no"``. A read of a started feed waits for
    catch-up up to its deadline and only then answers ``alarm feed still
    loading``; a feed that already knows its answer, unreachable or unbuildable,
    answers at once.
    """

    @override
    @abstractmethod
    async def start(self) -> None:
        """Begin feeding the snapshot."""

    @override
    @abstractmethod
    async def aclose(self) -> None:
        """Stop feeding the snapshot."""

    @abstractmethod
    async def state_for(
        self, address: str, *, timeout_s: float
    ) -> AlarmCondition | NotListed | Unavailable:
        """The alarm state of *address*, waiting up to *timeout_s* seconds for catch-up.

        :class:`NotListed` is a caught-up feed that holds none for it.
        """

    @abstractmethod
    async def states(self, *, timeout_s: float) -> AlarmStates | Unavailable:
        """Every channel's alarm state, waiting up to *timeout_s* seconds for catch-up.

        Before catch-up it answers as :meth:`state_for` does, never an empty list.
        """

    @abstractmethod
    async def config_for(
        self, address: str, *, timeout_s: float
    ) -> AlarmConfiguration | Unavailable:
        """The live alarm configuration of *address*, waiting up to *timeout_s* seconds for catch-up."""

    @abstractmethod
    async def probe(self, *, timeout_s: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""
