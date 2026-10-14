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
two functions that turn a live exception into a frame body and back. It has no
codec, no pipes and no process management in it, so both halves of the boundary
can import it cheaply and be tested against it without spawning anything.

The exception frame body
------------------------
Three keys, and no more::

    {"class_tag": str, "message": str, "fields": dict[str, Any]}

``class_tag`` names the exception class, ``message`` is the rendered text, and
``fields`` holds the structured attributes enumerated below. Decoding maps each
field back onto a constructor keyword and calls the real class.

The field lists are the protocol spec
-------------------------------------
:data:`CHANNEL_LIMITS_VIOLATION_FIELDS` and
:data:`CHANNEL_WRITE_BLOCKED_FIELDS` enumerate, by *attribute* name, exactly
what crosses the boundary for the two osprey-defined classes. They are
attribute names rather than constructor keywords because the attribute is what
the parent's envelope rendering reads: ``ChannelLimitsViolationError`` is
constructed with ``value=`` but stores it as ``.attempted_value``, so the spec
spells it ``attempted_value`` and this module maps that name onto the ``value``
keyword at reconstruction time (see :data:`EXCEPTION_SPECS`). Fields are read
off a live exception with :func:`getattr` per spec name — never a ``__dict__``
dump — so an attribute the class happens to grow is ignored rather than
silently becoming part of the wire format, and an attribute that is absent
encodes as ``None`` instead of raising.

Failing closed
--------------
Two unknowns are possible and neither may end in a silent drop. An exception
class outside the registry encodes as a :class:`ConnectionError` carrying the
original ``repr`` in its message; a ``class_tag`` outside the registry decodes
as a :class:`ConnectionError` naming the tag and the original message. The
parent always raises something, and that something always says what the child
actually reported.

Both sides read this file
-------------------------
The connector-host child and the proxy connector in the parent are the two
consumers of this spec. Neither may widen or reshape it unilaterally: adding a
class, a field, or a name mapping is a change to both halves of the boundary at
once, and to the tests that pin the field lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError

__all__ = [
    "CHANNEL_LIMITS_VIOLATION_FIELDS",
    "CHANNEL_WRITE_BLOCKED_FIELDS",
    "EXCEPTION_SPECS",
    "ExceptionSpec",
    "decode_exception",
    "encode_exception",
    "exception_fields",
    "reconstruct_exception",
]

#: Everything ``ChannelLimitsViolationError`` carries across the boundary, by
#: attribute name. ``attempted_value`` is the attribute; the constructor spells
#: the same thing ``value``, and :data:`EXCEPTION_SPECS` holds that mapping.
CHANNEL_LIMITS_VIOLATION_FIELDS: tuple[str, ...] = (
    "channel_address",
    "attempted_value",
    "violation_type",
    "violation_reason",
    "min_value",
    "max_value",
    "max_step",
    "current_value",
)

#: Everything ``ChannelWriteBlockedError`` carries across the boundary. Its text
#: is not listed here: the frame's own ``message`` feeds the constructor's
#: ``message`` keyword, so a custom refusal message survives verbatim rather
#: than being re-rendered from the reason code.
CHANNEL_WRITE_BLOCKED_FIELDS: tuple[str, ...] = ("channel_address", "reason")


@dataclass(frozen=True)
class ExceptionSpec:
    """How one exception class crosses the boundary.

    ``fields`` are attribute names, in the order the class documents them.
    ``ctor_kwargs`` renames a spec field to the constructor keyword that
    actually accepts it, for the one class where the two differ.
    ``message_kwarg`` names a constructor keyword fed from the frame's
    ``message``, for a class whose text is not derivable from its fields. An
    empty ``fields`` marks a builtin rebuilt from its message alone.
    """

    cls: type[BaseException]
    fields: tuple[str, ...] = ()
    ctor_kwargs: dict[str, str] = field(default_factory=dict)
    message_kwarg: str | None = None


#: The registry. Exactly four classes cross this boundary typed; everything else
#: fails closed to :class:`ConnectionError`.
EXCEPTION_SPECS: dict[str, ExceptionSpec] = {
    "ConnectionError": ExceptionSpec(ConnectionError),
    "TimeoutError": ExceptionSpec(TimeoutError),
    "ChannelWriteBlockedError": ExceptionSpec(
        ChannelWriteBlockedError,
        fields=CHANNEL_WRITE_BLOCKED_FIELDS,
        message_kwarg="message",
    ),
    "ChannelLimitsViolationError": ExceptionSpec(
        ChannelLimitsViolationError,
        fields=CHANNEL_LIMITS_VIOLATION_FIELDS,
        ctor_kwargs={"attempted_value": "value"},
    ),
}


def exception_fields(exc: BaseException) -> dict[str, Any]:
    """Read the spec's fields off a live exception.

    Only the enumerated names are read, and a name the instance does not carry
    yields ``None``, so an exception raised by an older or newer build of the
    child still encodes. An exception outside the registry has no fields.
    """
    spec = EXCEPTION_SPECS.get(type(exc).__name__)
    if spec is None:
        return {}
    return {name: getattr(exc, name, None) for name in spec.fields}


def reconstruct_exception(class_tag: str, message: str, fields: dict[str, Any]) -> BaseException:
    """Rebuild the real exception, failing closed to :class:`ConnectionError`.

    A tag outside the registry, or a field set the real constructor refuses,
    still produces an exception the parent can raise — one that names the tag
    and repeats the child's message, so nothing is lost by the fallback.
    """
    spec = EXCEPTION_SPECS.get(class_tag)
    if spec is None:
        return ConnectionError(
            f"connector host reported an unrecognised error type {class_tag!r}: {message}"
        )
    if not spec.fields:
        return spec.cls(message)

    kwargs = {spec.ctor_kwargs.get(name, name): value for name, value in fields.items()}
    if spec.message_kwarg is not None:
        kwargs[spec.message_kwarg] = message
    try:
        return spec.cls(**kwargs)
    except TypeError as exc:  # malformed field set — still never dropped
        return ConnectionError(
            f"connector host reported {class_tag!r} with unusable fields ({exc}): {message}"
        )


def encode_exception(exc: BaseException) -> dict[str, Any]:
    """Render ``exc`` as an exception frame body.

    An exception class outside the registry is normalised here, at the child's
    edge: it encodes as a :class:`ConnectionError` whose message carries the
    original ``repr`` — the parent learns what actually happened without this
    module claiming to reconstruct a class it does not know.
    """
    class_tag = type(exc).__name__
    if class_tag not in EXCEPTION_SPECS:
        return {
            "class_tag": "ConnectionError",
            "message": f"connector host reported an unrecognised error: {exc!r}",
            "fields": {},
        }
    return {"class_tag": class_tag, "message": str(exc), "fields": exception_fields(exc)}


def decode_exception(frame: dict[str, Any]) -> BaseException:
    """Rebuild the exception an exception frame body describes."""
    return reconstruct_exception(
        str(frame.get("class_tag", "")),
        str(frame.get("message", "")),
        dict(frame.get("fields") or {}),
    )
