"""Every error the Jev decision client raises.

The tree is closed: no Jev failure escapes as anything but a ``DecisionError``
subclass, and no other module defines one. Build-time refusals (catalog caps,
threshold-file problems) are not Jev errors and raise ``ConfigurationError``.

``error_from_response`` maps one HTTP response to an error. It is pure: it
logs nothing and keeps nothing of the body beyond the short human-readable
message the service chose to return. Raw error bodies can echo the request
state (a 422 repeats its ``input``), so they are never logged, stored or
chained onto the raised exception.

Failures that have no response are built by the client directly, with
``status=None``:

========================================  ===========================  ======================
failure                                   class                        ``error_type``
========================================  ===========================  ======================
an httpx timeout                          ``DecisionUnavailableError``  ``timeout``
any other httpx transport error           ``DecisionUnavailableError``  ``transport``
deadline passed before a chunk or retry   ``DecisionUnavailableError``  ``deadline_exceeded``
state plus longest question too large     ``DecisionRequestError``      ``state_too_large``
no usable API key                         ``DecisionAuthError``         ``no_key``
========================================  ===========================  ======================
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osprey.models.jev.models import JevCall

__all__ = [
    "DecisionAuthError",
    "DecisionError",
    "DecisionRequestError",
    "DecisionUnavailableError",
    "IncompleteAnswersError",
    "error_from_response",
]

REQUEST_ID_HEADER = "x-typesafe-request-id"

_AUTH_STATUSES = frozenset({401, 403})
_REQUEST_STATUSES = frozenset({400, 404, 413, 422})
_UNAVAILABLE_STATUSES = frozenset({402, 408, 429, 524, 529})

_MAX_MESSAGE_CHARS = 500


class DecisionError(Exception):
    """Base of every Jev failure.

    ``status`` is the HTTP status (or the OpenRouter ``error.code`` of a 200
    carrying an error object), ``None`` when no response arrived.
    ``completed_calls`` holds the calls that finished before the failure, so a
    caller can still account for their usage.
    """

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        status: int | None = None,
        request_id: str | None = None,
        completed_calls: list[JevCall] | None = None,
    ) -> None:
        """Record the message and the request's error type, status and request id."""
        super().__init__(message)
        self.message = message
        self.status = status
        self.error_type = error_type
        self.request_id = request_id
        self.completed_calls: list[JevCall] = list(completed_calls or [])

    def __repr__(self) -> str:
        """Show the message with the error type, status and request id."""
        return (
            f"{type(self).__name__}({self.message!r}, error_type={self.error_type!r}, "
            f"status={self.status!r}, request_id={self.request_id!r})"
        )


class DecisionAuthError(DecisionError):
    """The key is missing, invalid or not allowed to make the call."""


class DecisionRequestError(DecisionError):
    """The request (or the reply to it) is unusable; retrying it unchanged will not help."""


class DecisionUnavailableError(DecisionError):
    """The service could not answer now: overload, outage, timeout or no balance."""


class IncompleteAnswersError(DecisionRequestError):
    """The reply omitted answers for some of the questions asked."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "incomplete_answers",
        status: int | None = None,
        request_id: str | None = None,
        completed_calls: list[JevCall] | None = None,
    ) -> None:
        """Record a reply that omitted answers, typed ``incomplete_answers`` by default."""
        super().__init__(
            message,
            error_type=error_type,
            status=status,
            request_id=request_id,
            completed_calls=completed_calls,
        )


def _class_for_status(status: int | None) -> type[DecisionError]:
    if status in _AUTH_STATUSES:
        return DecisionAuthError
    if status in _REQUEST_STATUSES:
        return DecisionRequestError
    if status in _UNAVAILABLE_STATUSES:
        return DecisionUnavailableError
    if status is not None and 400 <= status < 500:
        return DecisionRequestError
    return DecisionUnavailableError


def _decode(body: bytes | str | None) -> Any:
    """Parse a JSON body; ``None`` when it is absent or not JSON."""
    if body is None:
        return None
    try:
        text = body.decode("utf-8") if isinstance(body, bytes | bytearray) else body
        return json.loads(text) if text.strip() else None
    except (UnicodeDecodeError, ValueError):
        return None


def _header(headers: Mapping[str, str] | None, name: str) -> str | None:
    if not headers:
        return None
    value = headers.get(name)
    if value is None:
        wanted = name.lower()
        for key, candidate in headers.items():
            if isinstance(key, str) and key.lower() == wanted:
                value = candidate
                break
    return value if isinstance(value, str) and value else None


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_MESSAGE_CHARS else text[: _MAX_MESSAGE_CHARS - 3] + "..."


def _as_status(code: Any) -> int | None:
    if isinstance(code, bool):
        return None
    if isinstance(code, int):
        return code
    if isinstance(code, str) and code.strip().isdigit():
        return int(code.strip())
    return None


def _detail_message(detail: Mapping[str, Any]) -> str | None:
    for key in ("message", "detail", "msg"):
        value = detail.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _list_detail_message(items: list[Any]) -> str:
    """Join the ``msg`` and ``loc`` of each validation item; ``input`` is never read."""
    parts: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        msg = item.get("msg")
        if not isinstance(msg, str) or not msg:
            continue
        loc = item.get("loc")
        if isinstance(loc, list) and all(isinstance(p, str | int) for p in loc):
            parts.append(f"{'.'.join(str(p) for p in loc)}: {msg}")
        else:
            parts.append(msg)
    return "; ".join(parts) or "request validation failed"


def error_from_response(
    status: int,
    headers: Mapping[str, str] | None,
    body: bytes | str | None,
) -> DecisionError:
    """Map one HTTP response to the ``DecisionError`` it stands for.

    ``body`` is the raw response body. Only the service's own message fields
    reach the error text; ``input`` and ``metadata`` are never read. A 200 is
    passed here only when it cannot be used: either it carries an OpenRouter
    ``error`` object, or it is undecodable or invalid (``malformed_response``).
    """
    payload = _decode(body)
    request_id = _header(headers, REQUEST_ID_HEADER)
    if request_id is None and isinstance(payload, Mapping):
        body_id = payload.get("id")
        if isinstance(body_id, str) and body_id:
            request_id = body_id

    error_obj = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(error_obj, Mapping):
        code = _as_status(error_obj.get("code"))
        raw_code = error_obj.get("code")
        mapped_status = code if status == 200 else status
        message = error_obj.get("message")
        text = message if isinstance(message, str) and message else "OpenRouter returned an error"
        cls = _class_for_status(code if status == 200 else status)
        return cls(
            _clip(text),
            error_type=f"openrouter_{code if code is not None else raw_code}",
            status=mapped_status,
            request_id=request_id,
        )

    if 200 <= status < 300:
        return DecisionRequestError(
            "Jev returned a response that could not be read",
            error_type="malformed_response",
            status=status,
            request_id=request_id,
        )

    cls = _class_for_status(status)
    fallback_type = f"http_{status}"
    fallback_text = f"Jev request failed with HTTP {status}"
    detail = payload.get("detail") if isinstance(payload, Mapping) else None

    if isinstance(detail, Mapping):
        error_type = detail.get("error_type")
        text = _detail_message(detail) or fallback_text
        return cls(
            _clip(text),
            error_type=error_type if isinstance(error_type, str) and error_type else fallback_type,
            status=status,
            request_id=request_id,
        )
    if isinstance(detail, list):
        return cls(
            _clip(_list_detail_message(detail)),
            error_type="validation_error",
            status=status,
            request_id=request_id,
        )
    if isinstance(detail, str) and detail:
        return cls(
            _clip(detail),
            error_type="api_usage_error",
            status=status,
            request_id=request_id,
        )
    return cls(fallback_text, error_type=fallback_type, status=status, request_id=request_id)
