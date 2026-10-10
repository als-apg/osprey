"""pyAML exceptions for OSPREY connector outcomes, and the mapping onto them.

pyAML tools catch :class:`PyAMLException`, so every connector outcome that a
tool should see as a pyAML failure is re-raised as one of three subclasses:

* :class:`OspreyWriteRefused` - nothing was written: a limits check, a write
  gate, a validation step or the control system refused it, or the control
  target moved under the call. Treat it as a stop, not a retry.
* :class:`OspreyWriteFailed` - the write was attempted and not confirmed. A
  device list also raises it when one channel of a multi-channel batch is
  refused after the connector sent the others, naming those as sent.
* :class:`OspreyReadFailed` - one or more channels produced no value.

The two write exceptions are also the :mod:`osprey.runtime.journal` exceptions of
the same name, so one ``except`` catches a connector refusal mapped here and the
refusal :func:`osprey.runtime.journal.guarded_write` raises outside a journaled
guarded run.

Each carries a short ``reason`` (one line, suitable for a restore report's
``refused`` entries) and the connector exception as ``__cause__``.
"""

from __future__ import annotations

from collections.abc import Sequence

from pyaml.common.exception import PyAMLException

from osprey.errors import (
    ChannelLimitsViolationError,
    ChannelReadFailedError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
)
from osprey.runtime import ControlTargetChangedError
from osprey.runtime import journal as runtime_journal

__all__ = [
    "OspreyReadFailed",
    "OspreyWriteFailed",
    "OspreyWriteRefused",
    "map_read_error",
    "map_write_error",
]

_VIOLATION_PREFIX = "Violation:"


class OspreyWriteRefused(runtime_journal.OspreyWriteRefused, PyAMLException):
    """A write was refused and no value of the call was written.

    Attributes:
        reason: One-line reason taken from the connector's refusal.
        channel_address: The refused channel, when the refusal names one.
    """


class OspreyWriteFailed(runtime_journal.OspreyWriteFailed, PyAMLException):
    """A write was attempted but its outcome was not confirmed.

    Attributes:
        reason: One-line reason taken from the connector's failure.
        channel_address: The failed channel, when the failure names one.
    """


class OspreyReadFailed(PyAMLException):
    """One or more channel reads produced no value.

    Attributes:
        reason: One-line reason taken from the connector's failure.
        addresses: Every address that failed, in the order asked.
    """

    def __init__(self, reason: str, addresses: Sequence[str] = ()) -> None:
        self.reason = reason
        self.addresses = list(addresses)
        named = ", ".join(self.addresses) if self.addresses else "unknown channel"
        super().__init__(f"OSPREY read failed for {named}: {reason}")


def _first_line(text: str) -> str:
    """Return the first non-blank line of ``text``, stripped (``""`` if none)."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _banner_violation(text: str) -> str | None:
    """Return the ``Violation:`` line's text from a limits banner, if present."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(_VIOLATION_PREFIX):
            return stripped[len(_VIOLATION_PREFIX) :].strip() or None
    return None


def _write_reason(exc: BaseException) -> str:
    """Build the one-line reason for a write-side exception.

    A refusal raised from a ``ChannelLimitsViolationError`` takes that error's
    ``violation_reason``; one without that cause falls back to the ``Violation:``
    line of the limits banner in its message.
    """
    if isinstance(exc, ChannelLimitsViolationError):
        return exc.violation_reason
    if isinstance(exc, ChannelWriteBlockedError | ChannelWriteFailedError):
        text = str(exc)
        cause = exc.__cause__
        violation = (
            _first_line(cause.violation_reason)
            if isinstance(cause, ChannelLimitsViolationError)
            else _banner_violation(text)
        )
        detail = violation or _first_line(text)
        if not detail:
            return exc.reason
        return detail if exc.reason in detail else f"{exc.reason}: {detail}"
    return _first_line(str(exc)) or type(exc).__name__


def map_write_error(
    exc: BaseException, address: str | None = None
) -> OspreyWriteRefused | OspreyWriteFailed:
    """Map a connector write exception onto a pyAML exception.

    Both limits-refusal shapes (a direct ``ChannelLimitsViolationError`` and a
    ``ChannelWriteBlockedError`` with reason ``LIMITS`` carrying the banner),
    any other ``ChannelWriteBlockedError`` and a ``ControlTargetChangedError``
    (including ``SwitchInProgressError``) become :class:`OspreyWriteRefused`.
    ``ChannelWriteFailedError`` and any other exception raised by the write
    become :class:`OspreyWriteFailed`: the write was attempted and nothing
    confirms it landed.

    Args:
        exc: The exception the runtime write raised.
        address: The channel written, used when ``exc`` does not name one.

    Returns:
        The mapped exception, with ``exc`` set as its ``__cause__``; raise it
        with ``raise map_write_error(exc) from exc``.
    """
    channel = getattr(exc, "channel_address", None) or address
    reason = _write_reason(exc)
    mapped: OspreyWriteRefused | OspreyWriteFailed
    refusals = (ChannelLimitsViolationError, ChannelWriteBlockedError, ControlTargetChangedError)
    if isinstance(exc, refusals):
        mapped = OspreyWriteRefused(reason, channel)
    else:
        mapped = OspreyWriteFailed(reason, channel)
    mapped.__cause__ = exc
    return mapped


def map_read_error(exc: BaseException, addresses: Sequence[str] = ()) -> OspreyReadFailed:
    """Map a connector read exception onto :class:`OspreyReadFailed`.

    Args:
        exc: The exception the runtime read raised.
        addresses: The addresses asked for; used when ``exc`` is not a
            ``ChannelReadFailedError`` (which names its own failed addresses).

    Returns:
        The mapped exception naming every failed address, with ``exc`` set as
        its ``__cause__``.
    """
    failed = list(exc.addresses if isinstance(exc, ChannelReadFailedError) else addresses)
    reason = _first_line(str(exc)) or type(exc).__name__
    mapped = OspreyReadFailed(reason, failed)
    mapped.__cause__ = exc
    return mapped
