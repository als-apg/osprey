"""What every read-only service backend answers, whatever its kind.

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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Generic, Literal, TypeVar

from osprey_connectors.types import ControlTarget

#: The row record a :class:`HistoryPage` holds.
_RowT = TypeVar("_RowT")

__all__ = [
    "BackendSpec",
    "HistoryPage",
    "Outcome",
    "Source",
    "Unavailable",
]


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


def _text(recorded: object) -> str | None:
    """A recorded value a record holds as a string; any other is not that field's."""
    return recorded if isinstance(recorded, str) else None


def _flag(recorded: object) -> bool | None:
    """A recorded value a record holds as a flag; any other is not that field's."""
    return recorded if isinstance(recorded, bool) else None


@dataclass(frozen=True)
class Unavailable:
    """The service gave no answer, and why, in words a reader can act on."""

    reason: str


@dataclass(frozen=True, kw_only=True)
class Source:
    """Which kind of service, on which machine, an answer came from.

    ``service`` is the backend's type name; ``rehearsal`` is true on ``standin``.
    """

    kind: Literal["directory", "alarm_state", "alarm_log", "put_log", "logbook", "archive"]
    target: ControlTarget
    service: str
    rehearsal: bool


@dataclass(frozen=True, kw_only=True)
class BackendSpec:
    """One backend to build; ``service_targets`` makes these from the config."""

    kind: Literal["directory", "alarm_log", "alarm_state", "put_log"]
    type: str
    target: ControlTarget
    settings: Mapping[str, object]
    rehearsal: bool


@dataclass(frozen=True, kw_only=True)
class HistoryPage(Generic[_RowT]):
    """Logged rows of one PV. No rows means nothing was logged, not that nothing happened.

    ``rows`` are the records of the kind that answered.
    """

    outcome: Outcome
    source: Source
    rows: Sequence[_RowT] = field(default_factory=list)
    truncated: bool = False
    retention_known: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_reason(self.outcome, self.reason)


class _Backend:
    """The construction and the source every backend kind shares."""

    def __init__(
        self, spec: BackendSpec, *, project_dir: Path | None = None, **seams: object
    ) -> None:
        self.spec = spec
        self.project_dir = project_dir

    @property
    def source(self) -> Source:
        return Source(
            kind=self.spec.kind,
            target=self.spec.target,
            service=self.spec.type,
            rehearsal=self.spec.rehearsal,
        )
