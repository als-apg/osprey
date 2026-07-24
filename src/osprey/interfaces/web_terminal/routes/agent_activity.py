"""Agent activity broadcast route.

``POST /api/agent-activity`` lets the agent-side tooling report which surface
it is currently acting on (a panel, a channel, a run, or an artifact).  The
route validates the payload and fans it out to every connected browser via the
``FileEventBroadcaster`` on ``app.state.broadcaster`` (the same SSE stream that
carries file and panel events, ``GET /api/files/events``), so the frontend can
highlight the target of the agent's attention in real time.

The payload shape is a fixed interface contract shared with the frontend::

    request:   {"tool": str, "target": {"kind": "panel"|"channel"|"run"|"artifact",
                                        "panel"?: str, "detail"?: str}}
    broadcast: {"type": "agent_activity", "tool": ..., "target": {...}, "ts": ...}

The server adds ``type`` and ``ts``; optional target fields are omitted from
the broadcast when absent.  Like the panel routes, this endpoint relies on the
loopback baseline for access control — no additional auth.
"""

from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()

# Length bounds: these strings are broadcast verbatim to every connected
# browser, so cap them well above any legitimate value (tool/panel names,
# channel lists) while keeping a runaway payload out of the SSE stream.
_MAX_NAME_LEN = 256
_MAX_DETAIL_LEN = 1024


class AgentActivityTarget(BaseModel):
    """The surface the agent is acting on."""

    kind: Literal["panel", "channel", "run", "artifact"]
    panel: str | None = Field(default=None, max_length=_MAX_NAME_LEN)
    detail: str | None = Field(default=None, max_length=_MAX_DETAIL_LEN)


class AgentActivityRequest(BaseModel):
    """Body of ``POST /api/agent-activity``."""

    tool: str = Field(max_length=_MAX_NAME_LEN)
    target: AgentActivityTarget


@router.post("/api/agent-activity")
async def post_agent_activity(body: AgentActivityRequest, request: Request):
    """Broadcast an agent-activity event to all connected browsers via SSE.

    Malformed bodies and unknown target kinds are rejected with 422 by the
    Pydantic model before this handler runs — nothing is broadcast for them.
    """
    request.app.state.broadcaster.broadcast(
        {
            "type": "agent_activity",
            "tool": body.tool,
            "target": body.target.model_dump(exclude_none=True),
            "ts": time.time(),
        }
    )
    return {"ok": True}
