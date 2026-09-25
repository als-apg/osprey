"""The one reader of the two tool-call attribution headers.

A queue item enqueued by the agent carries the conversation id and the
``tool_use_id`` of the tool call that queued it, so a queue row, a history
entry and the run's start document join to that call's audit record. The MCP
server sends both as request headers (:data:`CONVERSATION_HEADER`,
:data:`TOOL_USE_HEADER`); the bridge reads them here.

Attribution, not a credential: neither header gates anything, and a forged
value mislabels one queue row that the ledger's own record of the call
contradicts. So a value outside the accepted shape costs the attribution and
never the add. The rule mirrors :mod:`osprey.utils.owner_header`: each refusal
logs one warning naming the *shape* that was refused and never the value,
because the value is caller-chosen text; an absent header logs nothing.

This module imports only the standard library.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: The conversation id of the tool call that sent the request.
CONVERSATION_HEADER: str = "X-Osprey-Conversation"

#: The ``tool_use_id`` of the tool call that sent the request.
TOOL_USE_HEADER: str = "X-Osprey-Tool-Use-Id"

#: The longest accepted value.
MAX_ATTRIBUTION_LENGTH: int = 128

_ATTRIBUTION_RE = re.compile(rf"\A[A-Za-z0-9._:-]{{1,{MAX_ATTRIBUTION_LENGTH}}}\Z")


def _accepted(header: str, value: str | None) -> str | None:
    """*value* when it has the accepted shape; ``None`` otherwise."""
    if value is None:
        return None
    if not isinstance(value, str):
        shape = "not a string"
    elif not value:
        shape = "empty"
    elif len(value) > MAX_ATTRIBUTION_LENGTH:
        shape = f"longer than the {MAX_ATTRIBUTION_LENGTH}-character limit"
    elif not _ATTRIBUTION_RE.match(value):
        shape = "outside the accepted charset [A-Za-z0-9._:-]"
    else:
        return value
    logger.warning(
        "Ignoring %s: the value is %s, so this request carries no call attribution. "
        "The value itself is not logged.",
        header,
        shape,
    )
    return None


def attribution_from_headers(
    conversation: str | None, tool_use_id: str | None
) -> tuple[str | None, str | None]:
    """``(conversation_id, tool_use_id)`` from the raw header values.

    Each value is kept verbatim when it matches ``[A-Za-z0-9._:-]{1,128}``,
    and is ``None`` otherwise — absent silently, malformed with one warning.
    """
    return (
        _accepted(CONVERSATION_HEADER, conversation),
        _accepted(TOOL_USE_HEADER, tool_use_id),
    )
