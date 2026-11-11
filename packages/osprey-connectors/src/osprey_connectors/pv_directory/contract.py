"""What every directory-kind backend answers, and in what shape.

A backend is read-only and never raises: a service that gives no answer is a
value, :class:`Unavailable`, or a page whose outcome says so. Every method is
async and takes ``deadline``, the seconds this call may spend; the caller owns
the overall budget and hands the same figure to calls it runs concurrently.

Backends are built as ``Backend(spec, *, project_dir=None, **seams)``. The
constructor does no I/O and there is no connect step, so building one answers
nothing about the service; ``probe`` does. Out-of-tree classes follow the same
signature. A seam the class does not take is ignored, because the factory
passes the same keywords to every backend of a kind.

The records stay plain so one conversion makes their JSON: a list, a mapping or
a string stays what the service answered, as the service recorded it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "ALARM_STALE_WINDOW_S",
    "AlarmConfigRecord",
    "AlarmHistoryRow",
    "AlarmLog",
    "AlarmState",
    "AlarmStateRecord",
    "BackendSpec",
    "DirectoryPage",
    "DirectoryRecord",
    "DirectoryRow",
    "HistoryPage",
    "Outcome",
    "PVDirectory",
    "PutLog",
    "PutRow",
    "SearchQuery",
    "Source",
    "Unavailable",
]

#: Seconds after which a Kafka alarm feed that has been silent reads as stale.
ALARM_STALE_WINDOW_S: float = 60.0

_Origin = Literal["curated", "directory", "unknown"]


class Outcome(StrEnum):
    """How a lookup ended. ``NOT_LISTED`` is only an exact-name lookup's answer."""

    FOUND = "found"
    INACTIVE = "inactive"
    NOT_LISTED = "not_listed"
    NO_MATCH = "no_match"
    UNAVAILABLE = "unavailable"


def _require_reason(outcome: Outcome, reason: str | None) -> None:
    if outcome is Outcome.UNAVAILABLE and not reason:
        raise ValueError("an unavailable outcome carries its reason")


@dataclass(frozen=True)
class Unavailable:
    """The service gave no answer, and why, in words a reader can act on."""

    reason: str


@dataclass(frozen=True, kw_only=True)
class Source:
    """Which machine and service an answer came from. ``rehearsal`` is true on ``standin``."""

    target: Literal["va", "live", "standin"]
    service: str
    label: str
    rehearsal: bool


@dataclass(frozen=True, kw_only=True)
class BackendSpec:
    """One backend to build; ``pv_directory_targets`` makes these from the config."""

    kind: Literal["pv_directory", "alarm_log", "alarm_state", "put_log"]
    type: str
    target: str
    settings: Mapping[str, Any]
    rehearsal: bool


@dataclass(frozen=True, kw_only=True)
class DirectoryRow:
    """One PV. Backends leave ``origin`` unknown: only the service can compute it."""

    name: str
    origin: _Origin = "unknown"
    status: Literal["Active", "Inactive"] | None = None
    description: str | None = None
    record_type: str | None = None
    ioc: str | None = None
    host: str | None = None
    last_announced: str | None = None


@dataclass(frozen=True, kw_only=True)
class DirectoryRecord:
    """One PV in full. Every property is optional and a string, as measured."""

    row: DirectoryRow
    properties: Mapping[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class SearchQuery:
    """One search, built only by the service; callers pass these fields as keywords.

    Everything goes to the server except ``origin``, which the service computes
    over the fetched set. A query keys the service's cache, so it hashes: the
    repeatable filters become tuples and ``properties`` stays out of the hash.
    """

    words: str = ""
    name: str | None = None
    properties: Mapping[str, str] = field(default_factory=dict, hash=False)
    status: str | None = None
    ioc: tuple[str, ...] = ()
    record_type: tuple[str, ...] = ()
    origin: tuple[str, ...] = ()
    page: int = 0
    page_size: int = 50

    def __post_init__(self) -> None:
        for name in ("ioc", "record_type", "origin"):
            object.__setattr__(self, name, _filter_values(getattr(self, name)))


def _filter_values(values: str | Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return (values,) if isinstance(values, str) else tuple(values)


@dataclass(frozen=True, kw_only=True)
class DirectoryPage:
    """One page of a search.

    ``searched`` is the search in plain words and ``request`` the literal request
    sent. ``partial`` lists ``{part, reason}`` for what missed the deadline;
    ``facets`` maps ``origin``, ``ioc`` and ``record_type`` to ``{value, count}``
    lists, and is ``None`` with ``facets_reason`` when it could not be computed.
    """

    outcome: Outcome
    searched: str
    request: str
    source: Source
    total: int | None = None
    rows: list[DirectoryRow] = field(default_factory=list)
    truncated: bool = False
    partial: list[Mapping[str, str]] = field(default_factory=list)
    facets: Mapping[str, list[Mapping[str, Any]]] | None = None
    facets_reason: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_reason(self.outcome, self.reason)


@dataclass(frozen=True, kw_only=True)
class AlarmHistoryRow:
    """One logged alarm event, as the logger recorded it."""

    time: str | None = None
    severity: str | None = None
    status: str | None = None
    message: str | None = None
    user: str | None = None
    kind: Literal["state", "config"] = "state"


@dataclass(frozen=True, kw_only=True)
class PutRow:
    """One logged write, as the log recorded it; ``field`` is a field put's ``.HIHI``."""

    at: str | None = None
    pv: str | None = None
    field: str | None = None
    old: Any = None
    new: Any = None
    user: str | None = None
    client: str | None = None


@dataclass(frozen=True, kw_only=True)
class HistoryPage:
    """Logged rows of one PV. No rows means nothing was logged, not that nothing happened."""

    outcome: Outcome
    source: Source
    rows: Sequence[AlarmHistoryRow | PutRow] = field(default_factory=list)
    truncated: bool = False
    retention_known: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_reason(self.outcome, self.reason)


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
    """One PV's place in the alarm tree.

    ``label`` is ``"live configuration"`` from the alarm-state feed and ``"last
    logged configuration"`` from the alarm logger.
    """

    in_tree: Literal["yes", "no", "unknown"]
    path: str | None = None
    guidance: Any = None
    displays: Any = None
    enabled: bool | None = None
    latching: bool | None = None
    label: str


class _Backend:
    """The construction and the source every backend kind shares."""

    def __init__(self, spec: BackendSpec, *, project_dir: Path | None = None, **seams: Any) -> None:
        self.spec = spec
        self.project_dir = project_dir

    @property
    def source(self) -> Source:
        return Source(
            target=self.spec.target,  # type: ignore[arg-type]
            service=self.spec.type,
            label=self._display_name(),
            rehearsal=self.spec.rehearsal,
        )

    def _display_name(self) -> str:
        return self.spec.type


class PVDirectory(_Backend, ABC):
    """Search the control system's own directory of PVs."""

    #: The display name, a class attribute so that reading it needs no I/O.
    label: str = "All PVs"

    def _display_name(self) -> str:
        return self.label

    @abstractmethod
    async def search(self, query: SearchQuery, deadline: float) -> DirectoryPage:
        """One page of the rows *query* matches; no rows is ``NO_MATCH``, not ``NOT_LISTED``."""

    @abstractmethod
    async def count(self, query: SearchQuery, deadline: float) -> int | Unavailable:
        """How many rows *query* matches over the whole directory."""

    @abstractmethod
    async def describe(
        self, name: str, deadline: float
    ) -> DirectoryRecord | Literal[Outcome.NOT_LISTED] | Unavailable:
        """The record of the PV called exactly *name*, or ``NOT_LISTED``."""

    @abstractmethod
    async def property_names(self, deadline: float) -> list[str] | Unavailable:
        """The property names the directory holds."""

    @abstractmethod
    async def property_values(
        self, name: str, prefix: str | None, deadline: float
    ) -> list[str] | Unavailable:
        """The values of property *name*, those starting with *prefix* if given."""

    @abstractmethod
    async def probe(self, deadline: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""


class AlarmLog(_Backend, ABC):
    """Read an alarm logger."""

    @abstractmethod
    async def history(
        self, pv: str, since: datetime | None, limit: int, deadline: float
    ) -> HistoryPage:
        """The logged alarm events of *pv*, ``AlarmHistoryRow`` rows, newest first."""

    @abstractmethod
    async def config_for(self, pv: str, deadline: float) -> AlarmConfigRecord | Unavailable:
        """The last logged alarm configuration of *pv*."""

    @abstractmethod
    async def probe(self, deadline: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""


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


class PutLog(_Backend, ABC):
    """Read a log of channel writes."""

    @abstractmethod
    async def history(
        self, pv: str, since: datetime | None, limit: int, deadline: float
    ) -> HistoryPage:
        """The logged writes of *pv*, ``PutRow`` rows, field puts included, newest first."""

    @abstractmethod
    async def probe(self, deadline: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""
