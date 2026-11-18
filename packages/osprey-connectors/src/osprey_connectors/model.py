"""The package's data model: the bases every record at an interface is built on.

A record that crosses one of the package's interfaces (what a connector
returns, what the write path reads and keeps, a document read from disk, an
IPC frame) is a pydantic model built on one of the two bases here, so every
such record is validated the same way, at the moment it is built:

* **strict:** a field takes a value of its own type, never one that merely
  converts to it (``"1"`` is not an ``int``, ``1`` is not a ``str``). A field
  whose source really does carry another type (an ISO timestamp in a YAML
  document, say) says so on the field, with ``Field(strict=False)`` or a
  validator, rather than loosening the whole model;
* **closed:** a name that is not a field is refused, on construction and on
  validation, so a misspelt key in a document is an error rather than a value
  silently dropped;
* **by name:** a field with an alias is also accepted under its own name, so
  :func:`evolve` can rebuild any model from its fields.

:class:`FrozenModel` is a record that does not change once built, and every
record is one unless its docstring says why it is not. :class:`MutableModel` is
that exception: a record whose fields may be reassigned, each assignment
validated like the construction was. A record a caller is handed is never
one; a builder that needs a changed record makes a new one, with the
constructor or :func:`evolve`.

:func:`evolve` is how a caller, in the package or outside it, copies one of
these records with some fields changed.

A collection a frozen record holds is frozen too, or the record could change
under a reader who was handed it. A sequence is a :data:`FrozenTuple`, a set a
``frozenset``, and a mapping a :data:`ReadOnlyMapping`. Both aliases take the
mutable form on input — a list, a dict — because that is what JSON and the
connector-host IPC decode to, and the record holds the frozen form from then
on.

An instant a record holds is an :data:`Instant`: a timezone-aware
``datetime``, held in UTC, read from an ISO-8601 string that states its offset
(:func:`parse_instant`) and written with an explicit ``+00:00``. A naive
``datetime`` is refused, never given a zone.

A reader of a state file answers in one shape: one file read is a
:class:`ReadResult` (how the read ended, in the :data:`ReadStatus` words, what
it read and where), and a directory listed is a :class:`Listing` (what was
read, and the paths that could not be).

The module imports only pydantic, typing_extensions and the standard library,
and nothing from the package.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Generic, Literal, NoReturn, Self, TypeAlias, TypeVar

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    SerializerFunctionWrapHandler,
    WrapSerializer,
    model_validator,
)
from typing_extensions import override

__all__ = [
    "FrozenModel",
    "FrozenTuple",
    "Instant",
    "Listing",
    "Login",
    "MutableModel",
    "ReadOnlyMapping",
    "ReadResult",
    "ReadStatus",
    "empty_mapping",
    "evolve",
    "parse_instant",
]

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")


class _FrozenDict(dict[_K, _V]):
    """A ``dict`` that refuses every change once built: the mapping a frozen record holds.

    It is a ``dict``, so it compares equal to one with the same items, passes an
    ``isinstance(value, dict)`` check and is written by :func:`json.dumps`; and
    every method that would change it raises :class:`TypeError`, as a
    ``tuple`` does. A copy (:mod:`copy`, :mod:`pickle`) is a new one with the
    same items.
    """

    def _refuse(self, *_args: object, **_kwargs: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} does not support item assignment")

    __setitem__ = _refuse
    __delitem__ = _refuse
    clear = _refuse
    pop = _refuse
    popitem = _refuse
    setdefault = _refuse
    update = _refuse
    __ior__ = _refuse

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[dict[_K, _V]]]:
        return type(self), (dict(self),)


def _as_tuple(value: object) -> object:
    """A list as the tuple a field holds; any other value as it is, for the field to judge."""
    return tuple(value) if isinstance(value, list) else value


def _frozen(value: Mapping[_K, _V]) -> Mapping[_K, _V]:
    """A validated mapping, held as a read-only ``dict``."""
    return _FrozenDict(value)


def _as_dict(value: Mapping[_K, _V], handler: SerializerFunctionWrapHandler) -> object:
    """A read-only mapping, written as the plain dict it holds."""
    return handler(dict(value))


#: A sequence field of a frozen record: a ``tuple``, which a list given to it
#: becomes. The record dumps it as a list in JSON mode, as for any tuple.
FrozenTuple: TypeAlias = Annotated[tuple[_T, ...], BeforeValidator(_as_tuple)]

#: A mapping field of a frozen record: any mapping on input (a dict as JSON
#: decodes one), held as a ``dict`` that refuses every change (an assignment,
#: ``update``, ``pop`` and the rest raise :class:`TypeError`, as on a tuple), and
#: dumped as a plain ``dict``. Being a ``dict``, it compares equal to one, passes
#: ``isinstance(value, dict)`` and is written by :func:`json.dumps` unchanged.
#: Read-only one level deep: a mapping value nested in it is frozen only when its
#: own type says so (``ReadOnlyMapping[str, ReadOnlyMapping[str, int]]``).
ReadOnlyMapping: TypeAlias = Annotated[
    Mapping[_K, _V], AfterValidator(_frozen), WrapSerializer(_as_dict)
]


def parse_instant(text: str) -> datetime:
    """*text*, an ISO-8601 instant that states its UTC offset, as an aware ``datetime`` in UTC.

    ``Z`` and a numeric offset (``+00:00``, ``-07:00``) are both read. A stamp
    that states no offset is refused rather than given a zone: a reader in
    another zone would place it somewhere else. Every reader of a stored
    instant parses through this, and the stdlib hook twins restate it, ``Z``
    rewritten first because Python 3.9's :meth:`datetime.fromisoformat` does
    not read it.

    Raises:
        ValueError: *text* is not an ISO-8601 instant with an offset.
    """
    stamp = text[:-1] + "+00:00" if text.endswith("Z") else text
    moment = datetime.fromisoformat(stamp)
    if moment.utcoffset() is None:
        raise ValueError(f"{text!r} states no UTC offset")
    try:
        return moment.astimezone(UTC)
    except OverflowError:
        raise ValueError(f"{text!r} is outside the years a datetime holds in UTC") from None


def _instant_from_text(value: object) -> object:
    """A stored stamp as the instant it states; any other value as it is, for the field to judge."""
    return parse_instant(value) if isinstance(value, str) else value


def _in_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _stored_instant(value: datetime) -> str:
    """An instant as a file stores it: ISO-8601 with an explicit ``+00:00``, never ``Z``."""
    return value.astimezone(UTC).isoformat()


#: An instant field: a timezone-aware ``datetime`` (a naive one is refused), held
#: in UTC. A string is read through :func:`parse_instant`, so a record read from
#: a file takes the ISO-8601 stamp the file stores; any other value, a number
#: among them, is refused. JSON mode writes it as ISO-8601 with ``+00:00``
#: (``2026-11-16T12:00:00.250000+00:00``), which Python 3.9's
#: :meth:`datetime.fromisoformat` reads.
Instant: TypeAlias = Annotated[
    AwareDatetime,
    BeforeValidator(_instant_from_text),
    AfterValidator(_in_utc),
    PlainSerializer(_stored_instant, when_used="json"),
]


def _login_name(value: str) -> str:
    """*value* must be a name: not blank, no space around it. It is never stripped."""
    if not value.strip() or value != value.strip():
        raise ValueError("is not a name")
    return value


#: A login, wherever a record names one: a grant, a request, an approval, a
#: stop, a connection block's password login. Not blank and no space around
#: it. Validated, never normalised: ``" bob"`` is refused, not taken for
#: ``"bob"``.
Login: TypeAlias = Annotated[str, AfterValidator(_login_name)]


def empty_mapping() -> Mapping[_K, _V]:
    """An empty read-only mapping: the default of a :data:`ReadOnlyMapping` field."""
    return _FrozenDict()


class FrozenModel(BaseModel):
    """A record that does not change once built: strict, closed, frozen."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        validate_by_name=True,
    )


class MutableModel(BaseModel):
    """A record whose fields may be reassigned, each assignment validated.

    Only for a record that states why it is not a :class:`FrozenModel`, such as
    a file entry edited in place so that a rewrite keeps exactly the keys it
    read (``model_dump(exclude_unset=True)``).
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        validate_by_name=True,
    )


_Model = TypeVar("_Model", bound=BaseModel)


def evolve(model: _Model, /, **changes: object) -> _Model:
    """Return a copy of ``model`` with ``changes``, validated like a new one.

    The model's replacement for :func:`dataclasses.replace`. It is not
    ``model.model_copy(update=changes)``, which skips validation and would let
    a copy hold what construction refuses; this one runs the model's
    validators, field and model alike, on the result.
    """
    return type(model).model_validate({**dict(model), **changes})


#: How one read of a state file ended: a usable value, no file at the path, or
#: a file (or a root) that states nothing usable.
ReadStatus: TypeAlias = Literal["ok", "missing", "unreadable"]


class ReadResult(FrozenModel, Generic[_T]):
    """One read of one state file: how it ended, what it read, and where.

    Attributes:
        status: ``ok`` comes with the value; ``missing`` and ``unreadable``
            with none.
        value: What the file states, or ``None``.
        path: The file read, or ``None`` when no path resolved.
    """

    status: ReadStatus
    value: _T | None
    path: Path | None

    @model_validator(mode="after")
    def _value_only_when_read(self) -> Self:
        if (self.value is None) == (self.status == "ok"):
            raise ValueError("a value comes with status ok, and only with it")
        return self


class Listing(FrozenModel, Generic[_T]):
    """One listing of state files: what was read, and the paths that could not be.

    Attributes:
        items: Every file read, in the listing's order.
        unreadable: Every path there that could not be read.
    """

    items: FrozenTuple[_T]
    unreadable: FrozenTuple[Path]
