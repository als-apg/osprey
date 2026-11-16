"""What every read-only service backend answers, whatever its kind.

A backend is read-only and never raises: a service that gives no answer is a
value, :class:`Unavailable`, or a page whose outcome says so. Every method is
async and takes the keyword ``timeout_s``, the seconds this call may spend; the caller owns
the overall budget and hands the same figure to calls it runs concurrently.

Backends are built as ``Backend(spec, *, project_dir=None, **seams)``. The
constructor does no I/O and there is no connect step, so building one answers
nothing about the service; ``probe`` does. Out-of-tree classes follow the same
signature. A seam the class does not take is ignored, because the factory
passes the same keywords to every backend of a kind.

The records are frozen pydantic models, validated when built, and stay plain so
one conversion makes their JSON: a list, a mapping or a string stays what the
service answered, as the service recorded it. An instant is the exception: it is
read once, where the answer arrives, into a timezone-aware ``datetime`` named
``<event>_at``. A stamp that names no zone, or that cannot be read, places nothing
on the clock, so its field is ``None`` and no zone is guessed for it; a record
whose service is known to write such stamps keeps the text in a
``recorded_stamp`` field beside it. An instant a caller hands a backend (the
start of a history's window) must be aware too: a naive one is the caller's
error and raises :class:`~osprey_connectors.errors.NaiveDatetimeError`, the one
exception to "never raises".
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Generic, Literal, Self, TypeVar

from pydantic import (
    ValidationError,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    model_validator,
)

from osprey_connectors.model import FrozenModel, FrozenTuple, ReadOnlyMapping
from osprey_connectors.types import ControlTarget

#: The row record a :class:`HistoryPage` holds.
_RowT = TypeVar("_RowT")
#: The type a :data:`_Recorded` field holds.
_RecordedT = TypeVar("_RecordedT")

__all__ = [
    "BackendSpec",
    "HistoryPage",
    "Outcome",
    "ServiceKind",
    "Source",
    "SourceKind",
    "Unavailable",
]

#: A service kind this package builds a backend for, as ``BackendSpec.kind``
#: spells it.
ServiceKind = Literal["directory", "alarm_log", "alarm_state", "put_log"]

#: Every kind of service an answer can come from: a :data:`ServiceKind`, or one
#: the framework answers from itself (the logbook, the archive).
SourceKind = Literal[ServiceKind, "logbook", "archive"]


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


def _require_one_time(placed: datetime | None, recorded_stamp: str | None) -> None:
    """A record holds an instant or the stamp that could not be placed, never both."""
    if placed is not None and recorded_stamp is not None:
        raise ValueError("a placed instant keeps no recorded_stamp; that is for an unplaced one")


def _text(recorded: object) -> str | None:
    """A recorded value a record holds as a string; any other is not that field's."""
    return recorded if isinstance(recorded, str) else None


def _flag(recorded: object) -> bool | None:
    """A recorded value a record holds as a flag; any other is not that field's."""
    return recorded if isinstance(recorded, bool) else None


def _instant(recorded: object) -> datetime | None:
    """A recorded stamp as the aware instant it names, or ``None`` when it names none.

    An aware ``datetime`` is that instant; a string is read as ISO-8601, ``Z`` and
    ``+00:00`` alike. A string with no zone, one that is not ISO-8601 and any other
    value name no instant, and none is guessed for them.
    """
    if isinstance(recorded, str):
        try:
            recorded = datetime.fromisoformat(recorded.strip())
        except ValueError:
            return None
    if not isinstance(recorded, datetime) or recorded.utcoffset() is None:
        return None
    return recorded


def _none_unless_valid(recorded: object, handler: ValidatorFunctionWrapHandler) -> object:
    """*recorded* validated as its field's type, or ``None`` when it is not of it."""
    try:
        return handler(recorded)
    except ValidationError:
        return None


#: A field of a service's answer read as :func:`_text` and :func:`_flag` read one:
#: a value of another type is not that field's, so it reads as ``None``, never
#: refusing the answer it is in.
_Recorded = Annotated[_RecordedT | None, WrapValidator(_none_unless_valid)]


class Unavailable(FrozenModel):
    """The service gave no answer, and why, in words a reader can act on."""

    reason: str


class Source(FrozenModel):
    """Which kind of service, on which machine, an answer came from.

    ``service`` is the backend's type name; ``rehearsal`` is true on ``standin``.
    """

    kind: SourceKind
    target: ControlTarget
    service: str
    rehearsal: bool


class BackendSpec(FrozenModel):
    """One backend to build; ``service_targets`` makes these from the config."""

    kind: ServiceKind
    type: str
    target: ControlTarget
    settings: ReadOnlyMapping[str, object]
    rehearsal: bool


class HistoryPage(FrozenModel, Generic[_RowT]):
    """Logged rows of one PV. No rows means nothing was logged, not that nothing happened.

    ``rows`` are the records of the kind that answered.
    """

    outcome: Outcome
    source: Source
    rows: FrozenTuple[_RowT] = ()
    truncated: bool = False
    retention_known: bool = False
    reason: str | None = None

    @model_validator(mode="after")
    def _unavailable_says_why(self) -> Self:
        _require_reason(self.outcome, self.reason)
        return self


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
