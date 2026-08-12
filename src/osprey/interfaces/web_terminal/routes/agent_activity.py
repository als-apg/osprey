"""Agent activity broadcast route.

``POST /api/agent-activity`` lets the agent-side tooling report which surface
it is currently acting on (a panel, a channel, a run, or an artifact).  The
route validates the payload and fans it out to every connected browser via the
``FileEventBroadcaster`` on ``app.state.broadcaster`` (the same SSE stream that
carries file and panel events, ``GET /api/files/events``), so the frontend can
highlight the target of the agent's attention in real time.

The payload shape is a fixed interface contract shared with the frontend::

    request:   {"tool": str, "target": {"kind": "panel"|"channel"|"run"
                                                |"artifact"|"config"|"ui",
                                        "panel"?: str, "detail"?: str}}
    broadcast: {"type": "agent_activity", "tool": ..., "target": {...}, "ts": ...}

The server adds ``type`` and ``ts``; optional target fields are omitted from
the broadcast when absent.  Like the panel routes, this endpoint relies on the
loopback baseline for access control — no additional auth.

Every accepted event is also appended to a bounded in-memory history ring on
``app.state.agent_activity_ring`` before it is broadcast, so a browser that
connects (or reconnects) after the fact can still see what the agent has been
doing.  SSE clients see no difference — the ring is a pure side channel.

``GET /api/agent-activity/recent?limit=N`` reads that ring back, newest first::

    response: {"events": [{"type": "agent_activity", "tool": ...,
                           "target": {...}, "ts": ...}, ...]}

The events are the broadcast frames verbatim, so a browser can feed them
through the same handler it uses for the SSE stream.
"""

from __future__ import annotations

import time
from itertools import islice
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()

# Length bounds: these strings are broadcast verbatim to every connected
# browser, so cap them well above any legitimate value (tool/panel names,
# channel lists) while keeping a runaway payload out of the SSE stream.
_MAX_NAME_LEN = 256
_MAX_DETAIL_LEN = 1024

#: Size of the ``app.state.agent_activity_ring`` history buffer.  A browser
#: replays at most this many recent events on connect; the oldest fall off.
ACTIVITY_RING_MAX = 50


class AgentActivityTarget(BaseModel):
    """The surface the agent is acting on."""

    kind: Literal["panel", "channel", "run", "artifact", "config", "ui"]
    panel: str | None = Field(default=None, max_length=_MAX_NAME_LEN)
    detail: str | None = Field(default=None, max_length=_MAX_DETAIL_LEN)


class AgentActivityRequest(BaseModel):
    """Body of ``POST /api/agent-activity``."""

    tool: str = Field(max_length=_MAX_NAME_LEN)
    target: AgentActivityTarget


def record_activity(request: Request, tool: str, target: dict) -> dict:
    """Stamp an agent-activity frame, append it to the history ring, return it.

    The single place the frame shape is written, so every producer — this
    module's POST route and the panel routes, which mirror agent-origin panel
    commands that never pass through it — puts the same thing in the ring that
    ``GET /api/agent-activity/recent`` serves and the SSE stream carries.

    Appending is not broadcasting: callers that also broadcast pass the
    returned frame on, and callers recording history only just drop it.

    Args:
        request: Incoming FastAPI request carrying ``app.state``.
        tool: Name of the tool (or synthetic verb) the frame reports.
        target: Already-serialised target, optional keys omitted.

    Returns:
        The frame, whether or not the app carries a ring.
    """
    event = {"type": "agent_activity", "tool": tool, "target": target, "ts": time.time()}
    # Apps that mount these routers standalone (tests, embedders) need not
    # carry a ring; history is then simply unavailable, and the caller's own
    # work (a broadcast, or nothing) still runs.
    ring = getattr(request.app.state, "agent_activity_ring", None)
    if ring is not None:
        ring.append(event)
    return event


@router.post("/api/agent-activity")
async def post_agent_activity(body: AgentActivityRequest, request: Request):
    """Broadcast an agent-activity event to all connected browsers via SSE.

    Malformed bodies and unknown target kinds are rejected with 422 by the
    Pydantic model before this handler runs — nothing is broadcast for them.
    The accepted event is recorded in the history ring first, so an event is
    never broadcast without also being in the history a late browser reads.
    """
    event = record_activity(request, body.tool, body.target.model_dump(exclude_none=True))
    request.app.state.broadcaster.broadcast(event)
    return {"ok": True}


@router.get("/api/agent-activity/recent")
async def get_recent_agent_activity(request: Request, limit: int = ACTIVITY_RING_MAX):
    """Return the most recent agent-activity events, newest first.

    A browser that opens mid-session, or reconnects after its SSE stream
    dropped, reads this to rebuild the recent history it never received live.
    Each event is the broadcast frame verbatim, including the server ``ts``.

    ``limit`` is clamped into ``0..ACTIVITY_RING_MAX`` rather than rejected, so
    a caller asking for more than the ring can hold gets everything it has.
    Apps that mount this router without a ring report an empty history.
    """
    ring = getattr(request.app.state, "agent_activity_ring", None)
    if ring is None:
        return {"events": []}
    limit = max(0, min(limit, ACTIVITY_RING_MAX))
    return {"events": list(islice(reversed(ring), limit))}
