"""Hand a session key to the Simple view.

``POST /api/session/{key}/handoff``
Body: ``{"to": "simple", "interrupt": false}``

This is the Simple view's half of the one door every surface acquire walks
through. The Expert view acquires its surface on the ``/ws/terminal``
handshake, where a refusal is a close code; the Simple view has no long-lived
socket to refuse on, so it asks here first and only then starts talking to
``POST /api/chat``. Nothing is prompted: the call returns once the key's
process is the Simple one, resuming whatever conversation the key is on.

The refusals :func:`~osprey.interfaces.web_terminal.session_handoff.acquire_surface`
raises are the whole HTTP contract of this route, and every one of them is
answered with a ``detail.error`` slug the client branches on rather than a
sentence it would have to match:

===========================================  ======  =============================
Condition                                    Status  ``detail.error``
===========================================  ======  =============================
another connection or view holds the key     409     ``session_attached_elsewhere``
a terminal holds it and reports no turns     409     ``handoff_needs_interrupt``
a newer request with interrupt took the key  409     ``handoff_superseded``
the chat was torn down as it started         409     ``chat_terminated``
every chat is busy and the pool is full      429     ``chat_capacity``
the previous agent survived its kill         503     ``outgoing_still_running``
the hand-off's premise stopped holding       503     ``outgoing_vanished``,
                                                     ``spawn_not_pooled``
===========================================  ======  =============================

``handoff_needs_interrupt`` is the one the operator can act on: the terminal
holding the key has no turn-state hook installed, so nothing can tell when its
turn ends, and the client offers *Stop and switch now* — the same POST with
``interrupt: true``, which cuts the running turn short instead of waiting for
it. That same POST, made while an earlier one for the key is still waiting,
takes the key over from the earlier one: the waiting request is answered
``handoff_superseded`` and the interrupting one proceeds, so the operator can
end a wait from the transitional state without the abandoned request standing
in the way.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from osprey.interfaces.web_terminal.chat_session_pool import ChatSessionTerminatedError
from osprey.interfaces.web_terminal.operator_session import (
    CLAUDE_SDK_AVAILABLE,
    POSTURE_SOURCE_LIVE,
    OperatorRegistry,
    OperatorSession,
    build_operator_child_env,
)
from osprey.interfaces.web_terminal.session_handoff import (
    SURFACE_SIMPLE,
    ChannelClosed,
    ChannelToken,
    HandoffError,
    HandoffRefused,
    SpawnRequest,
    acquire_surface,
)
from osprey.interfaces.web_terminal.session_key import is_posture_key

logger = logging.getLogger(__name__)

router = APIRouter()


class HandoffRequest(BaseModel):
    """Body of ``POST /api/session/{key}/handoff``.

    ``to`` is a ``Literal`` rather than a plain string because this route
    speaks for exactly one surface. The Expert view takes the key on its
    ``/ws/terminal`` handshake — it needs the socket the refusal closes — so
    ``{"to": "expert"}`` names a hand-off no HTTP request can carry out, and a
    422 saying which field is wrong is a better answer than a 200 that did
    something else.

    ``interrupt`` asks for a running turn on the outgoing surface to be cut
    short rather than waited for. It defaults to False: the ordinary flip
    waits, and cutting a turn short is a gesture the operator makes after
    being told the key is held (``handoff_needs_interrupt``).
    """

    to: Literal["simple"]
    interrupt: bool = False


@router.post("/api/session/{key}/handoff")
async def hand_off_session(key: str, body: HandoffRequest, request: Request) -> Response:
    """Take session *key* for the Simple view and return once its chat is live.

    The chat is resumed onto the key's current transcript, or started fresh
    under the key when it has none — the acquire decides which from the
    transcript map and the transcripts on disk, and the spawn callback below
    only carries out that decision.

    No turn is taken here. The chat this leaves pooled is idle and free of the
    other surface, and ``POST /api/chat`` mints the per-turn guard when the
    operator actually says something.

    Returns:
        200 ``{"state": "simple", "session_id": <key>}``, or 204 with no body
        when the caller disconnected while the hand-off was waiting.

    Raises:
        HTTPException: 400 for a key outside the session-UUID grammar, 503
            when the Agent SDK is not installed, and the refusals listed in
            the module docstring.
    """
    if not is_posture_key(key):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_session_id",
                "message": "session_id must be a Claude session UUID.",
            },
        )
    if not CLAUDE_SDK_AVAILABLE:
        # Before the acquire, not after: a hand-off that tears the terminal
        # down and then finds it has nothing to start would leave the key with
        # no live process at all.
        raise HTTPException(
            status_code=503,
            detail={
                "error": "sdk_unavailable",
                "message": "Claude Agent SDK is not available.",
            },
        )

    cwd: str = request.app.state.project_cwd
    registry: OperatorRegistry = request.app.state.operator_registry

    async def spawn(req: SpawnRequest) -> OperatorSession:
        """Start the Simple view's chat under the key and hand back the pooled session.

        The environment is built the way ``routes/chat.py`` builds it, with one
        simplification this route earns: the key has already been held to the
        posture surface's grammar above, so a store can always answer for it
        and ``live`` is the honest ``posture_source`` here. The builder form is
        load-bearing for the same reason it is there — the pool calls it inside
        the lock hold that registers the creation, so a concurrent posture flip
        lands wholly before or wholly after this child's environment is read.
        """
        session, _ = await registry.get_or_create_chat_session(
            req.key,
            cwd,
            lambda: build_operator_child_env(
                cwd,
                session_key=req.key,
                app=request.app,
                posture_source=POSTURE_SOURCE_LIVE,
            ),
            resume_id=req.resume_id,
        )
        return session

    try:
        # Never under `asyncio.timeout`/`wait_for`/`TaskGroup`: ChannelClosed is
        # a synthetic cancellation raised from inside the call, and their
        # uncancel accounting would turn it into a TimeoutError or swallow it.
        result = await acquire_surface(
            request.app,
            key,
            SURFACE_SIMPLE,
            ChannelToken(request.is_disconnected),
            interrupt=body.interrupt,
            spawn=spawn,
        )
    except ChannelClosed:
        # The operator navigated away or flipped back while the outgoing turn
        # was still finishing. There is nobody to answer, and the hand-off was
        # abandoned before anything was torn down or started.
        logger.info("Hand-off of session %s abandoned; the caller disconnected", key)
        return Response(status_code=204)
    except HandoffRefused as refused:
        raise HTTPException(
            status_code=refused.status,
            detail={"error": refused.error, "message": str(refused)},
        ) from None
    except HandoffError as failed:
        raise HTTPException(
            status_code=503,
            detail={"error": failed.error, "message": str(failed)},
        ) from None
    except ChatSessionTerminatedError:
        # The chat was torn down while this very hand-off was starting it — a
        # posture flip or an explicit DELETE. Retrying respawns it under
        # whatever the store now holds, which is the whole remedy.
        raise HTTPException(
            status_code=409,
            detail={
                "error": "chat_terminated",
                "message": "This chat was terminated while it was starting; try again.",
            },
        ) from None

    logger.info(
        "Session %s is now on the simple surface (%s)",
        key,
        "started" if result.spawned else "already running",
    )
    return JSONResponse(content={"state": SURFACE_SIMPLE, "session_id": key})
