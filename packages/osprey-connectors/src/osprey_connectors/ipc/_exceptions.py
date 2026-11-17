"""Typed-exception contract for the connector-host IPC channel.

When the connector-host child fails, the parent must see the *same* exception
it would have seen had the control-system call run in-process: an operator
reading an error envelope should not be able to tell that a process boundary
was crossed. A traceback string cannot do that — the parent's error rendering
reads structured attributes (``.attempted_value``, ``.reason``, the min/max/step
bounds), and those only survive if the exception is rebuilt from its fields by
calling the real constructor on the far side.

This module is that contract, and nothing else: a registry mapping a
``class_tag`` to an exception class, the field list each class carries, and the
two functions that turn a live exception into a frame body and back:
:func:`encode_exception`, the one producer of a body, and
:func:`decode_exception`, the one consumer. It has no codec, no pipes and no
process management in it, so both halves of the boundary can import it
cheaply and be tested against it without spawning anything.

The exception frame body
------------------------
:class:`ExceptionFrameBody`, three fields and no more: ``class_tag`` names the
exception class, ``message`` is the rendered text, and ``fields`` holds the
values of the structured attributes enumerated below, by attribute name. An
error frame and a batch read's error node both carry one, flat, and nothing
else; decoding maps each field back onto a constructor keyword and calls the
real class.

The registry is derived from the vocabulary
-------------------------------------------
:data:`EXCEPTION_SPECS` holds the two builtins ``ConnectionError`` and
``TimeoutError`` and their package kin
:class:`~osprey_connectors.errors.ConnectRefusedError` (a ``ConnectionError``
outside every root), each rebuilt from its message, and one spec per class in
``osprey_connectors.errors.__all__`` under a contract root
(:class:`~osprey_connectors.errors.ChannelError`,
:class:`~osprey_connectors.errors.ConnectorUnavailableError`,
:class:`~osprey_connectors.errors.ConnectorInternalError`). It is computed at
import, never listed by hand, so every public failure crosses as itself with
no entry here to keep in step. A :class:`~osprey_connectors.errors.ChannelError` spec
carries ``address`` first, by construction; what else a class carries
is its wire detail, held here and nowhere in the class itself, because the
vocabulary says what a failure is and this module says what of it crosses. A
:class:`~osprey_connectors.errors.ChildLinkError` is never in the registry:
the link to a child raises it in the parent, so a frame can never claim one.

A spec names *attributes*, because the attribute is what the parent's
envelope rendering reads: ``ChannelLimitsViolationError`` is constructed with
``value=`` but stores it as ``.attempted_value``, so the spec spells it
``attempted_value`` and maps that name onto the ``value`` keyword at
reconstruction time. A :class:`~osprey_connectors.errors.ConnectorUnavailableError`
carries its text alone: its ``recovery`` is not a wire field, because recovery
is decided by the parent's supervisor, never by the child. Values are read off
a live exception with :func:`getattr` per attribute name, never a ``__dict__`` dump,
so an attribute the class happens to grow is ignored rather than silently
becoming part of the wire format, and an attribute that is absent encodes as
``None`` instead of raising.

Failing closed
--------------
Two unknowns are possible and neither may end in a silent drop. An exception
whose class is a subclass of a registered class crosses as the nearest such
class in its MRO, with that class's attributes and its own text: a connector's
own subclass of a vocabulary class crosses as that vocabulary class, with its
channel. Any other
class encodes as a
:class:`~osprey_connectors.errors.ConnectorInternalError` carrying the original
``repr`` in its message and the original class name as its ``class_tag``; a
``class_tag`` outside the registry, or a field set the real constructor
refuses, decodes as the same class naming the tag and the original message. It
is never a :class:`ConnectionError`: an exception outside the contract is a
connector bug, not a reachability answer. The parent always raises something,
and that something always says what the child actually reported.

Both sides read this file
-------------------------
The connector-host child and the proxy connector in the parent are the two
consumers of this spec. Neither may widen or reshape it unilaterally: adding a
class, a field, or a name mapping is a change to both halves of the boundary at
once, and to the parity tests that read the registry.
"""

from __future__ import annotations

from pydantic import Field

from osprey_connectors import errors
from osprey_connectors.errors import (
    ChannelError,
    ChannelLimitsViolationError,
    ChannelNotFoundError,
    ChannelUnreachableError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
    ChildLinkError,
    ConnectorInternalError,
    ConnectorUnavailableError,
    ConnectRefusedError,
)
from osprey_connectors.model import FrozenModel, ReadOnlyMapping, empty_mapping

__all__ = [
    "EXCEPTION_SPECS",
    "ExceptionFrameBody",
    "ExceptionSpec",
    "decode_exception",
    "encode_exception",
]

#: The roots of the failure vocabulary; every public class under one crosses typed.
_CONTRACT_ROOTS: tuple[type[BaseException], ...] = (
    ChannelError,
    ConnectorUnavailableError,
    ConnectorInternalError,
)


class ExceptionFrameBody(FrozenModel):
    """The exception frame body: the three keys described above.

    ``fields`` holds attribute values by attribute name, in the order of the
    class's spec.
    """

    class_tag: str
    message: str
    fields: ReadOnlyMapping[str, object] = Field(default_factory=empty_mapping)


class ExceptionSpec(FrozenModel):
    """How one exception class crosses the boundary.

    ``attributes`` are the attribute names whose values a frame body's
    ``fields`` carry, in the order the class documents them. ``ctor_kwargs``
    renames an attribute to the constructor keyword that actually accepts it,
    for the one class where the two differ. ``message_kwarg`` names a
    constructor keyword fed from the frame's ``message``, for a class whose
    text is not derivable from its attributes. An empty ``attributes`` marks a
    class rebuilt from its message alone.
    """

    cls: type[BaseException]
    attributes: tuple[str, ...] = ()
    ctor_kwargs: ReadOnlyMapping[str, str] = Field(default_factory=empty_mapping)
    message_kwarg: str | None = None


class _Detail(FrozenModel):
    """What of one vocabulary class crosses beyond its channel."""

    attributes: tuple[str, ...] = ()
    ctor_kwargs: ReadOnlyMapping[str, str] = Field(default_factory=empty_mapping)
    message_kwarg: str | None = None


#: The wire detail of each vocabulary class. A ``ChannelError`` subclass's
#: ``address`` is not listed: the registry puts it first for every one.
_WIRE_DETAIL: dict[type[BaseException], _Detail] = {
    ChannelError: _Detail(message_kwarg="message"),
    ChannelLimitsViolationError: _Detail(
        attributes=(
            "attempted_value",
            "violation_type",
            "violation_reason",
            "min_value",
            "max_value",
            "max_step",
            "current_value",
        ),
        ctor_kwargs={"attempted_value": "value"},
    ),
    ChannelWriteBlockedError: _Detail(
        attributes=("reason", "verdict_reason", "violation"), message_kwarg="message"
    ),
    ChannelWriteFailedError: _Detail(
        attributes=("outcome", "value_written", "observed_value"), message_kwarg="message"
    ),
    ChannelNotFoundError: _Detail(),
    ChannelUnreachableError: _Detail(message_kwarg="message"),
    ConnectorUnavailableError: _Detail(),
    ConnectorInternalError: _Detail(attributes=("class_tag",), message_kwarg="message"),
}


def _vocabulary() -> list[type[BaseException]]:
    """Every class ``osprey_connectors.errors`` publishes under a contract root.

    Never a :class:`~osprey_connectors.errors.ChildLinkError`, whichever root
    it also lies under: only the parent raises one.
    """
    published = (getattr(errors, name) for name in errors.__all__)
    return [
        item
        for item in published
        if isinstance(item, type)
        and issubclass(item, _CONTRACT_ROOTS)
        and not issubclass(item, ChildLinkError)
    ]


def _spec(cls: type[BaseException], detail: _Detail) -> ExceptionSpec:
    channel = ("address",) if issubclass(cls, ChannelError) else ()
    return ExceptionSpec(
        cls=cls,
        attributes=(*channel, *detail.attributes),
        ctor_kwargs=dict(detail.ctor_kwargs),
        message_kwarg=detail.message_kwarg,
    )


#: The registry: the two builtins, their package kin and the vocabulary. Everything else fails
#: closed to :class:`~osprey_connectors.errors.ConnectorInternalError`; a
#: vocabulary class with no wire detail is left out, and the parity test names it.
EXCEPTION_SPECS: dict[str, ExceptionSpec] = {
    "ConnectionError": ExceptionSpec(cls=ConnectionError),
    "TimeoutError": ExceptionSpec(cls=TimeoutError),
    "ConnectRefusedError": ExceptionSpec(cls=ConnectRefusedError),
    **{cls.__name__: _spec(cls, _WIRE_DETAIL[cls]) for cls in _vocabulary() if cls in _WIRE_DETAIL},
}


def _registered_tag(exc: BaseException) -> str | None:
    """The registry tag of ``exc``'s nearest registered class, or ``None``."""
    by_class = {spec.cls: tag for tag, spec in EXCEPTION_SPECS.items()}
    return next((by_class[cls] for cls in type(exc).__mro__ if cls in by_class), None)


def encode_exception(exc: BaseException) -> ExceptionFrameBody:
    """Render ``exc`` as an exception frame body: the one way a failure is put on the wire.

    An exception class outside the registry is normalised here, at the child's
    edge. A subclass of a registered class encodes as that class, with that
    class's attributes and its own text. Anything else encodes as a
    ``ConnectorInternalError`` whose message carries the original ``repr`` and
    whose ``class_tag`` field names the original class — the parent learns
    what actually happened without this module claiming to reconstruct a class
    it does not know.

    Only the spec's attributes are read, and an attribute the instance does not
    carry yields ``None``, so an exception raised by an older or newer build of
    the child still encodes.
    """
    class_tag = _registered_tag(exc)
    if class_tag is None:
        return ExceptionFrameBody(
            class_tag=ConnectorInternalError.__name__,
            message=f"connector host reported an unrecognised error: {exc!r}",
            fields={"class_tag": type(exc).__name__},
        )
    attributes = EXCEPTION_SPECS[class_tag].attributes
    return ExceptionFrameBody(
        class_tag=class_tag,
        message=str(exc),
        fields={name: getattr(exc, name, None) for name in attributes},
    )


def decode_exception(body: ExceptionFrameBody) -> BaseException:
    """Rebuild the real exception *body* describes, failing closed to ``ConnectorInternalError``.

    A tag outside the registry, or a field set the real constructor refuses,
    still produces an exception the parent can raise — one that names the tag
    and repeats the child's message, so nothing is lost by the fallback.
    """
    spec = EXCEPTION_SPECS.get(body.class_tag)
    if spec is None:
        return ConnectorInternalError(
            f"connector host reported an unrecognised error type {body.class_tag!r}: "
            f"{body.message}",
            class_tag=body.class_tag,
        )
    if not spec.attributes:
        return spec.cls(body.message)

    kwargs = {spec.ctor_kwargs.get(name, name): value for name, value in body.fields.items()}
    if spec.message_kwarg is not None:
        kwargs[spec.message_kwarg] = body.message
    try:
        return spec.cls(**kwargs)
    except (TypeError, ValueError) as exc:  # malformed field set — still never dropped
        return ConnectorInternalError(
            f"connector host reported {body.class_tag!r} with unusable fields ({exc}): "
            f"{body.message}",
            class_tag=body.class_tag,
        )
