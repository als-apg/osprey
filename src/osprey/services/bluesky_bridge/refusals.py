"""The one constructor every bridge refusal is made by.

A refusal body is ``{"code", "detail", "retryable", **extra}`` under the HTTP
status its code's row in :data:`~.queue_backend.REFUSAL_REASONS` declares.
Panels, the JS queue client and the MCP tools branch on ``code`` and on
``retryable``, so both come from the one table: a code the table does not
declare fails here, where the refusal is made, and never reaches the wire.

Its own module because the route modules that refuse (``draft``,
``validation``, ``app``, ``queue``) cannot all import ``queue`` — it imports
them — and ``queue_backend`` holds the table without serving HTTP.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from .queue_backend import REFUSAL_REASONS

__all__ = ["refusal"]

_BODY_KEYS = frozenset({"code", "detail", "retryable"})


def refusal(code: str, detail: str, /, **extra: Any) -> HTTPException:
    """Build the refusal for ``code``, answered as its row declares.

    Args:
        code: A key of ``REFUSAL_REASONS``.
        detail: The sentence that tells the reader what to do.
        **extra: Facts beside the three body keys (the plan, an item uid, a
            capability record); none may be ``code``, ``detail`` or
            ``retryable``.

    Returns:
        The exception to raise, with the row's status and retryability.

    Raises:
        ValueError: ``code`` has no row, or ``extra`` names a body key.
    """
    record = REFUSAL_REASONS.get(code)
    if record is None:
        raise ValueError(f"{code!r} is not a declared refusal code")
    clash = _BODY_KEYS & extra.keys()
    if clash:
        raise ValueError(f"a refusal extra may not set {sorted(clash)}")
    return HTTPException(
        status_code=record.status,
        detail={"code": code, "detail": detail, "retryable": record.retryable, **extra},
    )
