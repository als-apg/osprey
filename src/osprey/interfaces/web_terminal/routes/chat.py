"""REST chat endpoint — send a prompt, get a response via SSE or buffered JSON.

POST /api/chat?stream={true|false}
Body: {"prompt": "...", "chat_id": "..."}

The Simple-mode chat is keyed on a caller-supplied ``chat_id`` and backed by a
registry-managed :class:`OperatorSession` (the ``ChatSessionPool``). Each POST
mints a per-session turn guard, then consumes one
:meth:`OperatorSession.run_turn` — the turn machine (queue consumption,
silence deadline, heartbeats, terminal detection, quiesce/release) lives
there; the two branches here only map its output to SSE frames or a buffered
JSON payload.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from osprey.interfaces.web_terminal.chat_session_pool import ChatCapacityError
from osprey.interfaces.web_terminal.operator_session import (
    CLAUDE_SDK_AVAILABLE,
    OperatorSession,
    TurnInProgressError,
    TurnSilenceTimeout,
    build_clean_env,
    is_terminal_event,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Default turn timeout (SDK silence) when app.state.chat_turn_timeout_s is unset.
_DEFAULT_TURN_TIMEOUT_S = 600
# Cadence of `: heartbeat` SSE comment frames emitted while awaiting SDK events.
HEARTBEAT_INTERVAL_S = 15


class ChatRequest(BaseModel):
    prompt: str
    chat_id: str


def _strip_for_chat(event: dict[str, Any]) -> dict[str, Any]:
    """Return a wire-hygiene copy of *event* for the chat API client.

    The chat client renders a lightweight conversation view and never needs the
    heavy or sensitive payloads the SDK produces. This filter drops them while
    leaving the event's identity (``type``) and light metadata intact:

    - ``tool_use``   drop ``input`` (arguments); keep ``tool_name`` etc.
    - ``tool_result`` drop ``content`` (the result body); keep ``is_error`` etc.
    - ``thinking``   drop ``content``; a bare ``{"type": "thinking"}`` marker survives.
    - ``result``     reduce to exactly ``{"type", "is_error"}`` — cost/duration/
                     turn counts never reach the chat client.
    - every other type (``text``, ``system``, ``error``, ``session_reset``, …)
      passes through unchanged.

    The input event is never mutated; a new dict is returned whenever a change
    applies (and the same object may be returned for pass-through types).
    """
    etype = event.get("type")

    if etype == "tool_use":
        return {k: v for k, v in event.items() if k != "input"}
    if etype == "tool_result":
        return {k: v for k, v in event.items() if k != "content"}
    if etype == "thinking":
        return {k: v for k, v in event.items() if k != "content"}
    if etype == "result":
        return {"type": "result", "is_error": event.get("is_error")}

    return event


def _turn_timeout_s(request: Request) -> float:
    """Resolve the per-turn SDK-silence timeout from app.state (with default)."""
    return getattr(request.app.state, "chat_turn_timeout_s", _DEFAULT_TURN_TIMEOUT_S)


async def _acquire_chat_turn(request: Request, chat_id: str) -> tuple[OperatorSession, int, bool]:
    """Get-or-create the chat session and mint its per-turn guard.

    Returns ``(session, token, was_reused)``. Maps registry/guard failures to the
    HTTP contract: :class:`ChatCapacityError` -> 429, :class:`TurnInProgressError`
    -> 409, both with a machine-readable body.
    """
    cwd: str = request.app.state.project_cwd
    registry = request.app.state.operator_registry

    try:
        session, was_reused = await registry.get_or_create_chat_session(
            chat_id, cwd, build_clean_env(cwd)
        )
    except ChatCapacityError:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "chat_capacity",
                "message": "All chat sessions are busy; try again shortly.",
            },
        ) from None

    try:
        token = session.acquire_turn()
    except TurnInProgressError:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "turn_in_progress",
                "message": "A turn is already in flight for this chat.",
            },
        ) from None

    return session, token, was_reused


@router.post("/api/chat")
async def chat(request: Request, body: ChatRequest, stream: bool = True):
    """Send a prompt to Claude and receive the response.

    Query params:
        stream  If true (default), return an SSE event stream.
                If false, buffer the full response and return JSON.
    """
    if not CLAUDE_SDK_AVAILABLE:
        raise HTTPException(status_code=503, detail="Claude Agent SDK is not available")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must not be empty")

    session, token, was_reused = await _acquire_chat_turn(request, body.chat_id)
    turn_timeout_s = _turn_timeout_s(request)

    if not stream:
        return await _buffered_response(session, token, prompt, was_reused, turn_timeout_s)

    return StreamingResponse(
        _stream_events(session, token, prompt, was_reused, turn_timeout_s),
        media_type="text/event-stream",
    )


async def _stream_events(
    session: OperatorSession,
    token: int,
    prompt: str,
    was_reused: bool,
    turn_timeout_s: float,
):
    """SSE consumer of :meth:`OperatorSession.run_turn` for one turn.

    The turn machine — queue consumption, silence deadline, heartbeat cadence,
    terminal detection, and the quiesce/release discipline — lives in
    ``run_turn``. This wrapper only maps its output to the SSE wire: heartbeat
    markers become comment frames, events are stripped via
    :func:`_strip_for_chat`, a silence timeout becomes an error frame. The
    ``finally`` safety net covers the one exit ``run_turn`` cannot: the outer
    generator being closed before ``run_turn`` ever started (both calls are
    synchronous, so the pair cannot be interleaved by another request).
    """
    try:
        if not was_reused:
            yield _sse(_strip_for_chat({"type": "session_reset"}))

        async for event in session.run_turn(
            prompt, token, timeout_s=turn_timeout_s, heartbeat_s=HEARTBEAT_INTERVAL_S
        ):
            if event.get("type") == "heartbeat":
                yield ": heartbeat\n\n"
                continue
            yield _sse(_strip_for_chat(event))
    except TurnSilenceTimeout:
        yield _sse(_strip_for_chat(_timeout_event()))
    except asyncio.CancelledError:
        # Non-terminal: client disconnect / re-delivered cancellation. run_turn
        # already quiesced + released; re-raise so Starlette sees the cancel.
        raise
    except Exception as exc:
        logger.exception("Chat stream error")
        yield _sse(
            _strip_for_chat(
                {"type": "error", "message": str(exc), "error_type": type(exc).__name__}
            )
        )
    finally:
        if session.release_turn(token):
            session.spawn_quiesce()


def _timeout_event() -> dict[str, Any]:
    """The error event emitted when the SDK goes silent past the turn timeout."""
    return {
        "type": "error",
        "message": "Timeout waiting for response",
        "error_type": "TimeoutError",
    }


async def _buffered_response(
    session: OperatorSession,
    token: int,
    prompt: str,
    was_reused: bool,
    turn_timeout_s: float,
) -> JSONResponse:
    """Run one turn to completion and return a single reduced JSON response.

    Consumes the same :meth:`OperatorSession.run_turn` machine as the SSE
    branch — heartbeat markers are skipped instead of forwarded, and a silence
    timeout maps to 504 instead of an error frame. A ``session_reset`` marker is
    prepended to ``events`` on a fresh session and ``_strip_for_chat`` applies
    to every event. The top-level payload is reduced to exactly
    ``{text, events, is_error, error?}`` — cost/duration/turn counts never
    reach the chat client.
    """
    events: list[dict[str, Any]] = []
    text_parts: list[str] = []
    is_error = False
    error_msg: str | None = None
    status_code = 200

    if not was_reused:
        events.append(_strip_for_chat({"type": "session_reset"}))

    try:
        async for event in session.run_turn(
            prompt, token, timeout_s=turn_timeout_s, heartbeat_s=HEARTBEAT_INTERVAL_S
        ):
            etype = event.get("type")
            if etype == "heartbeat":
                continue

            events.append(_strip_for_chat(event))
            if etype == "text":
                text_parts.append(event.get("content", ""))
            elif etype == "result":
                is_error = bool(event.get("is_error"))
            elif etype == "error" and is_terminal_event(event):
                is_error, error_msg, status_code = True, event.get("message", "Unknown error"), 500
    except TurnSilenceTimeout:
        is_error, error_msg, status_code = True, "Timeout waiting for response", 504
    except asyncio.CancelledError:
        # Abandoned request — run_turn already quiesced + released; re-raise so
        # the server sees the cancellation.
        raise
    finally:
        # Safety net for the never-started case only (see _stream_events).
        if session.release_turn(token):
            session.spawn_quiesce()

    payload: dict[str, Any] = {
        "text": "".join(text_parts),
        "events": events,
        "is_error": is_error,
    }
    if error_msg is not None:
        payload["error"] = error_msg
    return JSONResponse(content=payload, status_code=status_code)


@router.post("/api/chat/{chat_id}/interrupt", status_code=204)
async def interrupt_chat(chat_id: str, request: Request) -> Response:
    """Signal-only interrupt of an in-flight streaming turn.

    Sends ``client.interrupt()`` iff a turn is currently in flight for
    ``chat_id`` — nothing else. It never drains the reader and never releases
    the turn guard; the handler running that turn owns quiesce and release. A
    204 is returned whether or not there was anything to interrupt (idempotent).
    """
    registry = request.app.state.operator_registry
    session = registry.get_chat_session(chat_id)
    if session is not None and session.in_flight:
        try:
            await session.interrupt()
        except Exception:
            logger.debug("Interrupt signal failed for chat %s", chat_id, exc_info=True)
    return Response(status_code=204)


@router.delete("/api/chat/{chat_id}", status_code=204)
async def delete_chat(chat_id: str, request: Request) -> Response:
    """Tear down a chat session (idempotent).

    Delegates to :meth:`OperatorRegistry.terminate_chat_session`, which is
    busy-safe: an idle session is torn down directly; a busy one is
    interrupt-signalled, its stored quiesce/handler awaited within a bound, then
    hard-torn-down regardless. An unknown ``chat_id`` is a no-op. Always 204.
    """
    registry = request.app.state.operator_registry
    await registry.terminate_chat_session(chat_id)
    return Response(status_code=204)


def _sse(event: dict[str, Any]) -> str:
    """Format an event dict as an SSE data line."""
    return f"data: {json.dumps(event)}\n\n"
