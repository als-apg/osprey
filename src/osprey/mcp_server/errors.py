"""Structured error envelope for MCP tool responses."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
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

#: The key under which an error names what the server did about the failure:
#: a list of ``{"action": <word>, "text": <sentence>}``, present only when
#: something was done.
RECOVERY_KEY = "recovery"

#: The key under which a write names what happened to each item it was asked
#: for: its rows, one per channel, each carrying its ``outcome`` word. A write
#: has no ``missing`` key; its rows are its map.
RESULTS_KEY = "results"

#: The error word for a request nothing answered: every item was tried, or
#: refused, and none answered.
NOTHING_ANSWERED = "nothing_answered"

#: The key each tool that answers part of a request states its map under.
MAP_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "channel_read": MISSING_KEY,
        "channel_write": RESULTS_KEY,
        "entries_by_ids": MISSING_KEY,
        "model_response_matrix": MISSING_KEY,
        "pyaml_measure": MISSING_KEY,
    }
)


def answer_map(
    tool: str,
    *,
    missing: Mapping[str, str | None] | None = None,
    results: Sequence[Mapping] | None = None,
) -> dict:
    """Return ``{key: map}`` for ``tool``, under the key :data:`MAP_KEYS` declares.

    ``missing`` is the map of a tool declared under :data:`MISSING_KEY`, and
    ``results`` the rows of one declared under :data:`RESULTS_KEY`; the map
    not given is empty.

    Raises:
        KeyError: ``tool`` declares no map key.
        ValueError: The map given is not the one ``tool`` declares.
    """
    key = MAP_KEYS[tool]
    if key == RESULTS_KEY:
        if missing is not None:
            raise ValueError(f"{tool} states its map under {key!r} only")
        return {key: [dict(row) for row in results or ()]}
    if results is not None:
        raise ValueError(f"{tool} states its map under {key!r} only")
    return {key: dict(missing or {})}


def refuse_nothing_answered(
    tool: str,
    noun: str,
    missing: Mapping[str, str | None] | None = None,
    details: dict | None = None,
    *,
    recovery: Sequence[Mapping[str, str]] = (),
    results: Sequence[Mapping] | None = None,
) -> NoReturn:
    """Raise the ``nothing_answered`` envelope for a request nothing answered.

    Args:
        tool: The tool name the message opens with.
        noun: What the request asked for, in the plural (``"channels"``).
        missing: Each item asked for, mapped to the source's own text or ``None``.
        details: Further structured facts; the map is merged into them.
        recovery: What the server did about the failures, each an ``action``
            word and its ``text``; listed under ``details.recovery`` and the
            sentences appended to the message.
        results: The rows of a tool whose map is its rows, in place of ``missing``.
    """
    stated = answer_map(tool, missing=missing, results=results)
    ((key, answers),) = stated.items()
    message = f"{tool}: none of the {len(answers)} requested {noun} answered"
    merged = {**(details or {}), **stated}
    if recovery:
        message += ". " + " ".join(r["text"] for r in recovery)
        merged[RECOVERY_KEY] = [dict(r) for r in recovery]
    make_error(
        NOTHING_ANSWERED,
        message,
        [f"Name each entry of details.{key} and its text to the operator."],
        details=merged,
    )


class NothingAnsweredError(Exception):
    """A tool asked for several things answered none of them.

    Raised inside a tool body that an error handler renders. ``cause`` is the
    one error every item failed with, when they all failed alike, so the
    handler can answer with that error's own envelope; it is ``None`` when the
    items failed for different reasons or for none the handler renders typed.
    ``missing`` maps each item to the source's own text, or ``results`` holds
    the rows of a tool whose map is its rows; never both. The handler merges
    the one given into the envelope's ``details``. ``errors`` holds every
    item's error, so the handler can report the recovery records the
    supervisor attached to each even when no one cause names the envelope.
    """

    def __init__(
        self,
        cause: BaseException | None,
        *,
        missing: Mapping[str, str | None] | None = None,
        results: Sequence[Mapping] | None = None,
        errors: Sequence[BaseException] = (),
    ):
        if missing is not None and results is not None:
            raise ValueError("a request states one map: missing or results, not both")
        self.cause = cause
        self.missing = dict(missing) if missing is not None else None
        self.results = [dict(row) for row in results] if results is not None else None
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
