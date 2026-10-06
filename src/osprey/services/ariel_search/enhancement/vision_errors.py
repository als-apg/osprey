"""How a picture module's failed model call is handled: the one classification.

:func:`classify_vision_error` sorts an exception raised by a vision call (a
caption call, a picture embedding call) into the three ways a picture module
can react:

==================  ==========================================================
Kind                Meaning for the module
==================  ==========================================================
``unavailable``     The service cannot be used; ``run_entry`` returns
                    ``unavailable(reason)`` with ``reason`` from
                    :func:`~osprey.services.ariel_search.enhancement.availability.unavailable_reason`.
``transient``       Worth retrying later (timeout, rate limit, server error,
                    anything unrecognised); the entry stops.
``deterministic``   The same picture fails the same way again (a bad request
                    for that picture, an empty reply, a degenerate vector);
                    stored as that picture's failure once the gate allows it.
==================  ==========================================================

There is no availability mapping here: every availability decision is
:func:`~osprey.services.ariel_search.enhancement.availability.unavailable_reason`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from osprey.services.ariel_search.enhancement import availability
from osprey.services.ariel_search.enhancement.availability import (
    exception_chain,
    library_exception_classes,
)
from osprey.services.ariel_search.enhancement.base import ImageEntryOutcome

if TYPE_CHECKING:
    from osprey.services.ariel_search.enhancement.base import PictureGate

__all__ = [
    "EmptyReplyError",
    "VisionErrorKind",
    "classify_vision_error",
    "error_signature",
    "failed_call_outcome",
    "short_error",
]

VisionErrorKind = Literal["unavailable", "transient", "deterministic"]

#: HTTP statuses that say the request for this picture is wrong, not the service.
_DETERMINISTIC_STATUSES = frozenset({400, 413, 415, 422})


class EmptyReplyError(Exception):
    """The model answered with nothing usable (only whitespace or a thinking block)."""


def _status(exc: BaseException) -> int | None:
    """The HTTP status an exception carries (``httpx``/``requests`` response, LiteLLM field)."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def classify_vision_error(exc: BaseException) -> VisionErrorKind:
    """Classify the exception of a failed vision call.

    Args:
        exc: What the call raised.

    Returns:
        ``unavailable`` when :func:`availability.unavailable_reason` gives a
        reason; ``deterministic`` for an :class:`EmptyReplyError`, a
        :class:`~osprey.models.providers.base.DegenerateVectorError`, a LiteLLM
        ``BadRequestError``/``UnprocessableEntityError`` or an HTTP 400, 413,
        415 or 422; ``transient`` for everything else (timeouts, HTTP 429 and
        5xx, and anything unrecognised).
    """
    from osprey.models.providers.base import DegenerateVectorError

    if availability.unavailable_reason(exc) is not None:
        return "unavailable"
    transient = (TimeoutError,) + library_exception_classes(
        "litellm",
        "Timeout",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "BadGatewayError",
    )
    transient += library_exception_classes("httpx", "TimeoutException") + library_exception_classes(
        "requests", "Timeout"
    )
    deterministic = library_exception_classes(
        "litellm", "BadRequestError", "UnprocessableEntityError"
    )
    for item in exception_chain(exc):
        if isinstance(item, (EmptyReplyError, DegenerateVectorError)):
            return "deterministic"
        if isinstance(item, transient):
            return "transient"
        if deterministic and isinstance(item, deterministic):
            return "deterministic"
        status = _status(item)
        if status is not None:
            if status in _DETERMINISTIC_STATUSES:
                return "deterministic"
            return "transient"
    return "transient"


def error_signature(exc: BaseException) -> str:
    """A stable signature of a failure: the same input failing the same way gives the same one.

    Built from the exception class and its HTTP status, never its message,
    which may carry per-request detail.

    Args:
        exc: The exception.

    Returns:
        ``<ClassName>`` or ``<ClassName>:<status>``.
    """
    status = _status(exc)
    name = type(exc).__name__
    return name if status is None else f"{name}:{status}"


def short_error(exc: BaseException) -> str:
    """``<ClassName>: <message>``, the message cut to 300 characters."""
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def failed_call_outcome(exc: BaseException, gate: PictureGate) -> ImageEntryOutcome | None:
    """How a picture module's failed vision call ends the entry, per :func:`classify_vision_error`.

    Args:
        exc: What the call raised.
        gate: The pass's per-picture admission control.

    Returns:
        ``unavailable`` with the availability reason, ``transient_error`` with
        :func:`short_error`, or ``partial`` when the gate does not yet allow a
        deterministic failure to be stored; ``None`` when it does, and the
        module stores the failure as the picture's.
    """
    kind = classify_vision_error(exc)
    if kind == "unavailable":
        return ImageEntryOutcome.unavailable(availability.unavailable_reason(exc) or "unreachable")
    if kind == "transient":
        return ImageEntryOutcome.transient_error(short_error(exc))
    if not gate.deterministic(error_signature(exc)):
        return ImageEntryOutcome.partial()
    return None
