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

The module imports only pydantic, typing_extensions and the standard library,
and nothing from the package.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, NoReturn, Self, TypeAlias, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    SerializerFunctionWrapHandler,
    WrapSerializer,
)
from typing_extensions import override

__all__ = [
    "FrozenModel",
    "FrozenTuple",
    "MutableModel",
    "ReadOnlyMapping",
    "empty_mapping",
    "evolve",
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
