"""What every read-only service backend answers, whatever its kind.

A backend is read-only and never raises: every method answers its value or
:class:`Unavailable`, the one shape of "no answer, and why", and an exact lookup
that finds nothing answers :class:`NotListed`. Every method is
async and takes the keyword ``timeout_s``, the seconds this call may spend; the caller owns
the overall budget and hands the same figure to calls it runs concurrently.

Backends are built as ``Backend(spec, *, project_dir=None, **seams)``. The
constructor reads ``spec.settings`` into the class's ``settings_model`` (so a
block its type refuses raises there, in the backend's words), does no I/O, and
there is no connect step, so building one answers nothing about the service;
``probe`` does. Out-of-tree classes follow the same signature. A seam the class
does not take is ignored, because the factory passes the same keywords to every
backend of a kind.

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

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, ClassVar, Generic, Literal, TypeVar

from pydantic import (
    Field,
    ValidationError,
    ValidatorFunctionWrapHandler,
    WrapValidator,
    model_validator,
)

from osprey_connectors.config_sections import ConnectorSettings
from osprey_connectors.connector_types import ControlTarget
from osprey_connectors.model import FrozenModel, FrozenTuple, parse_instant

#: The row record a :class:`Page` holds.
_RowT = TypeVar("_RowT")
#: The type a :data:`_Recorded` field holds.
_RecordedT = TypeVar("_RecordedT")

__all__ = [
    "BackendSpec",
    "LogPage",
    "NotListed",
    "Page",
    "PageOutcome",
    "ServiceKind",
    "Source",
    "SourceKind",
    "Unavailable",
    "UnavailableCause",
]

#: A service kind this package builds a backend for, as ``BackendSpec.kind``
#: spells it.
ServiceKind = Literal["directory", "alarm_log", "alarm_state", "put_log", "archive"]

#: Every kind of service an answer can come from: a :data:`ServiceKind`, or the
#: logbook, which the framework answers from itself.
SourceKind = Literal[ServiceKind, "logbook"]

#: Why a service gave no answer, as a caller branches on it:
#:
#: - ``unreachable``: the service could not be reached, or was never heard from.
#: - ``timeout``: it did not answer within the call's budget.
#: - ``bad_reply``: it answered, but not with an answer: an error status, an
#:   oversized or malformed body.
#: - ``refused``: the answer was refused before it was asked or read: a request
#:   the backend may not send, or an access proxy's login page.
#: - ``misconfigured``: the deployment's block cannot be used: a bad url, missing
#:   settings, a missing client library.
#: - ``not_ready``: the service is still coming up, or the session moved under
#:   the call; asking again later can answer.
#: - ``unsupported``: the service cannot answer this question at all.
#: - ``failed``: the backend raised, or failed in a way none of the above names.
UnavailableCause = Literal[
    "unreachable",
    "timeout",
    "bad_reply",
    "refused",
    "misconfigured",
    "not_ready",
    "unsupported",
    "failed",
]


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

    An aware ``datetime`` is that instant; a string is read, surrounding
    whitespace aside, by :func:`~osprey_connectors.model.parse_instant`, so
    ``Z`` and ``+00:00`` alike, in UTC. A string with no zone, one that is not
    ISO-8601 and any other value name no instant, and none is guessed for them.
    """
    if isinstance(recorded, str):
        try:
            return parse_instant(recorded.strip())
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
    """The service gave no answer: ``cause`` for a caller to branch on, ``reason`` for a reader.

    ``reason`` is the words a reader acts on; no caller compares it.
    """

    cause: UnavailableCause
    reason: str


class NotListed(FrozenModel):
    """An exact lookup's answer when the service holds nothing at *address*."""

    address: str


class Source(FrozenModel):
    """Which kind of service, on which machine, an answer came from.

    ``service`` is the backend's type name; ``rehearsal`` is true on ``standin``.
    """

    kind: SourceKind
    target: ControlTarget
    service: str
    rehearsal: bool


class BackendSpec(FrozenModel):
    """One backend to build; ``service_targets`` makes these from the config.

    ``settings`` is the backend's block. A mapping given here is read into the
    base :class:`~osprey_connectors.config_sections.ConnectorSettings`, which
    keeps every key as written and refuses nothing; the backend reads it into
    its own ``settings_model`` when it is built.
    """

    kind: ServiceKind
    type: str
    target: ControlTarget
    settings: ConnectorSettings = Field(default_factory=ConnectorSettings)
    rehearsal: bool

    @model_validator(mode="before")
    @classmethod
    def _settings_as_written(cls, data: object) -> object:
        """Settings given as a mapping keep every key, as the base settings model reads it."""
        if not isinstance(data, Mapping) or not isinstance(data.get("settings"), Mapping):
            return data
        return {**data, "settings": ConnectorSettings.model_validate(dict(data["settings"]))}


#: How a page of rows ended: rows matched, or none did.
PageOutcome = Literal["found", "no_match"]


class Page(FrozenModel, Generic[_RowT]):
    """One page of the rows a service answered for a search or a channel's history.

    ``rows`` are the records of the kind that answered; ``truncated`` says more
    rows match than the page holds. ``note`` says, in a reader's words, why a
    page holds no rows. A service that gave no answer answers
    :class:`Unavailable` instead of a page.
    """

    outcome: PageOutcome
    source: Source
    rows: FrozenTuple[_RowT] = ()
    truncated: bool = False
    note: str | None = None


class LogPage(Page[_RowT], Generic[_RowT]):
    """Logged rows of one channel. No rows means nothing was logged, not that nothing happened.

    ``retention_known`` says whether the log states how far back it keeps rows.
    """

    retention_known: bool = False


#: A backend's settings model.
_SettingsT = TypeVar("_SettingsT", bound=ConnectorSettings)


class _Backend:
    """The construction, the lifecycle and the source every backend kind shares.

    The constructor keeps the spec as it was handed and reads its settings
    into :attr:`settings_model` as :attr:`settings`, naming :attr:`SETTINGS_KEY`
    in a refusal. :meth:`start` and :meth:`aclose` do nothing here; a kind
    whose backends hold something open between calls overrides them.
    """

    #: The model this backend's block is read into. A backend that declares
    #: none reads its keys from the base model's
    #: :attr:`~pydantic.BaseModel.model_extra`.
    settings_model: ClassVar[type[ConnectorSettings]] = ConnectorSettings
    #: The dotted key of the block, named in a refusal of it.
    SETTINGS_KEY: ClassVar[str] = "settings"

    def __init__(
        self, spec: BackendSpec, *, project_dir: Path | None = None, **seams: object
    ) -> None:
        self.spec = spec
        #: ``spec.settings`` read into :attr:`settings_model`.
        self.settings: ConnectorSettings = self.settings_model.read(
            spec.settings, key=self.SETTINGS_KEY
        )
        self.project_dir = project_dir

    def _settings(self, model: type[_SettingsT]) -> _SettingsT:
        """:attr:`settings` as *model*, the backend's :attr:`settings_model`.

        Raises:
            TypeError: The class names another model: a subclass overrode
                :attr:`settings_model` with one that is not *model*'s.
        """
        settings = self.settings
        if not isinstance(settings, model):
            raise TypeError(f"{model.__name__} expected, got {type(settings).__name__}")
        return settings

    async def start(self) -> None:
        """Begin whatever the backend keeps open between calls; nothing by default."""

    async def aclose(self) -> None:
        """End what :meth:`start` began; nothing by default. Idempotent."""

    @property
    def source(self) -> Source:
        return Source(
            kind=self.spec.kind,
            target=self.spec.target,
            service=self.spec.type,
            rehearsal=self.spec.rehearsal,
        )
