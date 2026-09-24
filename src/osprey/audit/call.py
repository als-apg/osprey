"""The tool call a record is about: its id, its conversation, and what it noted.

The agent harness sends an id with every MCP ``tools/call`` (the ``_meta`` key
:data:`TOOL_USE_ID_META_KEY`, verified in harness build 2.1.267 and later) and
stamps the conversation id on every stdio MCP child it spawns
(``CLAUDE_CODE_SESSION_ID``). The MCP audit middleware reads both once per call
and opens a :func:`call_scope`; every audit record filed inside that scope —
the middleware's own, an inner recorder's through
:func:`osprey.audit.writer.record` — then names the call it belongs to without
a call-site edit. That id is the key that joins a default ledger line to the
full ``tool_call`` record and to the harness's own telemetry.

**The carrier is a mutable holder set before the tool runs.** :func:`call_scope`
sets a fresh :class:`CallFacts` object on a :class:`~contextvars.ContextVar`.
A copied context (``asyncio.create_task``, ``anyio.to_thread.run_sync``, the
worker thread a server runs a synchronous tool on) copies the *reference*, so
it shares the same object: a fact noted there is seen by the middleware when
the call returns. A context started fresh (``loop.run_in_executor``, a bare
``threading.Thread``) carries no scope and notes nothing.

That is the opposite trade from :mod:`osprey.audit.dedup`, whose marker is a
*value* replaced on the context variable and therefore deliberately not seen
across a copy. The difference is right for each: facts are additive
annotations about one call, so a fact noted on a child task is still a fact
about that call; a decision marker silences the layer outside it, so one that
leaked across tasks could suppress a record for a different call.

Like :mod:`osprey.audit.envelope`, this module is a stdlib-only leaf: the
writer, the middleware, the tools and the services import it, and none of them
may inherit an import cycle by doing so.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TOOL_USE_ID_META_KEY",
    "CallFacts",
    "call_scope",
    "current_call",
    "current_tool_use_id",
    "harness_session_id",
    "valid_tool_use_id",
]

#: The ``_meta`` key the agent harness sends the tool-use id under on every
#: MCP ``tools/call`` (verified in harness build 2.1.267).
TOOL_USE_ID_META_KEY = "claudecode/toolUseId"

#: The shape a tool-use id is accepted in. The id names a stamp file, so it
#: must be a safe path component; anything else is dropped rather than kept.
_TOOL_USE_ID = re.compile(r"\A[A-Za-z0-9_-]{1,128}\Z")

#: The conversation id Osprey forces at launch (web terminal, dispatch worker).
_OSPREY_CONVERSATION_ENV = "OSPREY_TELEMETRY_SESSION_ID"

#: The conversation id the agent harness stamps on every stdio MCP child.
_HARNESS_CONVERSATION_ENV = "CLAUDE_CODE_SESSION_ID"


def valid_tool_use_id(value: Any) -> str | None:
    """*value* when it is a well-formed tool-use id, else ``None``.

    A non-string, an empty string, a character outside ``[A-Za-z0-9_-]`` or a
    value over 128 characters is not an id this module will carry.
    """
    if isinstance(value, str) and _TOOL_USE_ID.match(value):
        return value
    return None


def harness_session_id() -> str | None:
    """The conversation id this process runs under, or ``None``.

    ``OSPREY_TELEMETRY_SESSION_ID`` first (the id Osprey forces, which is the
    one the telemetry store is tagged with), then ``CLAUDE_CODE_SESSION_ID``
    (the one the harness stamps on its MCP children).
    """
    return (
        os.environ.get(_OSPREY_CONVERSATION_ENV)
        or os.environ.get(_HARNESS_CONVERSATION_ENV)
        or None
    )


@dataclass
class CallFacts:
    """One tool call's identifiers and the facts noted about it while it ran.

    Mutable on purpose: see the module docstring. ``facts`` is filled by the
    tool (through :func:`note`) and read by the middleware after the call.

    :param tool_use_id: The harness's id for this call, or ``None``.
    :param session_id: The conversation id, or ``None``.
    :param facts: What the tool noted about the call.
    """

    tool_use_id: str | None
    session_id: str | None
    facts: dict[str, Any] = field(default_factory=dict)


_CURRENT: ContextVar[CallFacts | None] = ContextVar("osprey_audit_current_call", default=None)


@contextlib.contextmanager
def call_scope(tool_use_id: str | None, session_id: str | None) -> Iterator[CallFacts]:
    """Bind one tool call's identifiers to this block.

    Entering sets a fresh :class:`CallFacts`, so nothing from an earlier call
    carries in; leaving restores what was there, on the way out of an
    exception too.
    """
    call = CallFacts(tool_use_id=tool_use_id, session_id=session_id)
    token = _CURRENT.set(call)
    try:
        yield call
    finally:
        _CURRENT.reset(token)


def current_call() -> CallFacts | None:
    """The tool call in scope, or ``None`` outside any :func:`call_scope`."""
    return _CURRENT.get()


def current_tool_use_id() -> str | None:
    """The tool-use id of the call in scope, or ``None``."""
    call = _CURRENT.get()
    return call.tool_use_id if call is not None else None
