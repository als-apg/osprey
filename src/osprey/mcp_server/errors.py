"""Structured error envelope for MCP tool responses."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import NoReturn

from fastmcp.exceptions import ToolError
from mcp.types import CallToolResult


def make_error(
    error_type: str,
    error_message: str,
    suggestions: list[str] | None = None,
    details: dict | list | None = None,
) -> NoReturn:
    """Raise a ``fastmcp.ToolError`` carrying the cross-team standard envelope.

    fastmcp converts a raised ``ToolError`` into a ``CallToolResult`` with
    ``isError=True`` and the exception message verbatim as a ``TextContent``
    block. A ``CallToolResult`` returned from a tool does *not* come out with
    ``isError`` set when fastmcp wraps the tool's structured output, so raising
    is the only path that produces a clean error response on the wire.

    Args:
        error_type: Machine-readable error category (e.g. "limits_violation").
        error_message: Human-readable summary of the error.
        suggestions: Actionable next steps for the operator/agent.
        details: Structured data about the error (e.g. violation specifics,
                 channel limits). Included only if provided.
    """
    envelope: dict = {
        "error": True,
        "error_type": error_type,
        "error_message": error_message,
        "suggestions": suggestions or [],
    }
    if details is not None:
        envelope["details"] = details
    raise ToolError(json.dumps(envelope))


#: The key under which a tool names what it was asked for and did not answer:
#: a map from each such item to the source's own text, or ``None`` when the
#: source gave none. A result always carries it (``{}`` when everything
#: answered); a ``nothing_answered`` refusal carries it under ``details``.
MISSING_KEY = "missing"

#: The error word for a request whose every item was tried and none answered.
NOTHING_ANSWERED = "nothing_answered"


def refuse_nothing_answered(
    tool: str,
    noun: str,
    missing: Mapping[str, str | None],
    details: dict | None = None,
) -> NoReturn:
    """Raise the ``nothing_answered`` envelope for a request nothing answered.

    Args:
        tool: The tool name the message opens with.
        noun: What the request asked for, in the plural (``"channels"``).
        missing: Each item asked for, mapped to the source's own text or ``None``.
        details: Further structured facts; ``missing`` is merged into them.
    """
    make_error(
        NOTHING_ANSWERED,
        f"{tool}: none of the {len(missing)} requested {noun} answered",
        [f"Name each entry of details.{MISSING_KEY} and its text to the operator."],
        details={**(details or {}), MISSING_KEY: dict(missing)},
    )


class NothingAnsweredError(Exception):
    """A tool asked for several things answered none of them.

    Raised inside a tool body that an error handler renders. ``cause`` is the
    one error every item failed with, when they all failed alike, so the
    handler can answer with that error's own envelope; it is ``None`` when the
    items failed for different reasons or for none the handler renders typed.
    ``missing`` maps each item to the source's own text, and the handler merges
    it into the envelope's ``details``. ``errors`` holds every item's error, so
    the handler can run the recovery each class of failure calls for even when
    no one cause names the envelope.
    """

    def __init__(
        self,
        cause: BaseException | None,
        *,
        missing: Mapping[str, str | None] | None = None,
        errors: Sequence[BaseException] = (),
    ):
        self.cause = cause
        self.missing = dict(missing) if missing is not None else None
        self.errors = tuple(errors)
        super().__init__(str(cause) if cause is not None else "nothing answered")


def extract_error_envelope(result: CallToolResult) -> dict | None:
    """Pull the structured envelope out of a wire-form error CallToolResult.

    Returns ``None`` when the result is not an error or its content cannot be
    parsed as the envelope dict. Useful for tests and hooks that need to
    inspect the structured payload behind ``isError=True``.
    """
    if not isinstance(result, CallToolResult):
        return None
    if result.isError is not True:
        return None
    for block in result.content or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and parsed.get("error") is True:
            return parsed
    return None
