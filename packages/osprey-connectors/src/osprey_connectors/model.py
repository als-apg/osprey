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

The module imports only pydantic and the standard library, and nothing from
the package.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ConfigDict

__all__ = ["FrozenModel", "MutableModel", "evolve"]


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
