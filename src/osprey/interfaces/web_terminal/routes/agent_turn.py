"""Whether the expert view's Claude Code process is mid-turn.

``POST /api/agent-turn`` is how the terminal's own child process tells the
server what it is doing. The Claude Code hooks that report it fire on
``UserPromptSubmit`` (busy), on ``Stop`` and ``StopFailure`` (idle), and on
``SessionStart`` (idle, plus whichever transcript the session just opened), so
the server learns of a turn's edges from the process that runs it rather than
by guessing from terminal output.

The payload shape is a fixed interface contract shared with the hook::

    request:  {"session_id": str, "pool_key": str, "state": "busy"|"idle",
               "surface": str, "ts": float, "source"?: str}
    response: {"ok": true, "recorded": bool}

Two identities appear in it and they are not the same thing. ``pool_key`` is
the session key the browser, the PTY pool and the audit ledger all use — it
never changes. ``session_id`` is Claude Code's own session id, which names the
transcript file and *does* change: a ``/clear`` starts a fresh transcript while
the key stays put. So a report both marks the key busy or idle and records
where the key's conversation currently lives, through
:mod:`osprey.interfaces.web_terminal.transcript_map`.

Both identifiers must be canonical session UUIDs. One becomes a key in a JSON
store on disk and the other is persisted and later spliced into a resume argv
and a transcript filename, so the grammar is closed rather than length-bounded
— the same rule, through the same predicate, that the posture surface applies
to this class of identifier.

Only the expert surface is recorded. The simple view's chat pool knows its own
turn boundaries from the SDK client and needs no hook, and a report arriving
with any other surface — including none, when the environment variable that
names it is unset — is accepted and dropped rather than refused: a hook cannot
act on a 4xx, and a refusal would put an error in the operator's terminal for
something the server deliberately ignores.

Like the panel and agent-activity routes, this endpoint rides the loopback
baseline plus the panel token (``PANEL_TIER_ROUTES`` in
:mod:`osprey.interfaces.web_auth`) — the same credential the in-process
companions already carry.

The recorded entry lives on ``app.state.turn_state``, keyed by session key;
the store itself, its reader and the reset a PTY spawn or teardown performs
are :mod:`osprey.interfaces.web_terminal.turn_state`.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from osprey.interfaces.web_terminal import transcript_map
from osprey.interfaces.web_terminal.session_key import is_posture_key
from osprey.interfaces.web_terminal.turn_state import record_turn_state

logger = logging.getLogger(__name__)

router = APIRouter()

#: Length bound on the two identifiers and the source label. Session ids are
#: UUIDs and hook event names are short, so this is generous by an order of
#: magnitude while keeping an unbounded string out of ``app.state``.
_MAX_ID_LEN = 256

#: The one surface whose turns are recorded here.
EXPERT_SURFACE = "expert"


class AgentTurnRequest(BaseModel):
    """Body of ``POST /api/agent-turn``.

    ``session_id`` and ``state`` are the report; the rest is context the hook
    reads out of its environment. That context is defaulted rather than
    required because an unset variable must leave the report ignorable, not
    malformed — a 422 would reach the operator's terminal as hook noise.
    """

    session_id: str = Field(max_length=_MAX_ID_LEN)
    state: Literal["busy", "idle"]
    pool_key: str = Field(default="", max_length=_MAX_ID_LEN)
    surface: str = Field(default="", max_length=_MAX_ID_LEN)
    ts: float | None = None
    source: str | None = Field(default=None, max_length=_MAX_ID_LEN)


@router.post("/api/agent-turn")
async def post_agent_turn(body: AgentTurnRequest, request: Request):
    """Record a turn edge reported by the expert view's Claude Code process.

    Reports for any other surface are accepted and dropped — see the module
    docstring for why that is not a 4xx. ``recorded`` in the response says
    which of the two happened, so a hook author can tell a working install
    from a silently ignored one without reading the server log.

    The transcript id travels with every recorded report, not only with the
    ones that move it: writing it unconditionally means a session that returns
    to its original transcript clears the stale mapping instead of keeping a
    pointer to a conversation it has left. The map's own write is a no-op when
    nothing changed, so the per-turn traffic costs no disk writes.

    An id that is not a canonical session UUID is dropped the same way a
    foreign surface is — silently, with ``recorded`` false — so a malformed
    identifier can never become a store key, a persisted transcript pointer,
    or a fragment of a later resume argv.
    """
    if body.surface != EXPERT_SURFACE:
        return {"ok": True, "recorded": False}

    if not is_posture_key(body.session_id) or (body.pool_key and not is_posture_key(body.pool_key)):
        logger.debug("Ignoring agent-turn report whose identifiers are not session UUIDs")
        return {"ok": True, "recorded": False}

    app = request.app
    key = body.pool_key or body.session_id
    # The hook's clock is the server's clock, so a timestamp ahead of now is a
    # broken or hostile report rather than a fast one. Later readers let a
    # newer stored timestamp outrank other evidence of what a session is
    # doing, and a far-future value would hold that authority forever.
    now = time.time()
    record_turn_state(
        app,
        key,
        state=body.state,
        ts=min(body.ts, now) if body.ts is not None else now,
        transcript_id=body.session_id,
    )
    transcript_map.set(app, key, body.session_id)
    return {"ok": True, "recorded": True}
