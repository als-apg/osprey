"""REST chat endpoint — send a prompt, get a response via SSE or buffered JSON.

POST /api/chat?stream={true|false}
Body: {"prompt": "...", "chat_id": "..."}
A prompt that opens with a slash command the chat does not list is refused
with 400 ``command_not_available``, before any turn opens.

POST /api/chat/{chat_id}/permission
Body: {"request_id": "...", "allow": bool, "answers"?: {...}, "message"?: "..."}
Answers a ``permission_request`` frame the in-flight turn is waiting on.

GET /api/chat/{chat_id}/events
A live SSE feed of every frame the chat's agent emits, opening with a ``hello``
snapshot; how reloaded and second tabs follow a running turn.

GET /api/chat/commands
Answers {"commands": [{"name", "description", "argument_hint", "kind"}, ...]},
the slash commands the chat agent accepts in this project.

The operator chat is keyed on a caller-supplied ``chat_id`` and backed by a
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
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from osprey.agent_runner import HAS_SDK, AgentRunError
from osprey.audit.envelope import POSTURE_SOURCE_PROCESS
from osprey.interfaces.web_terminal.chat_session_pool import ChatSessionTerminatedError
from osprey.interfaces.web_terminal.operator_session import (
    POSTURE_SOURCE_LIVE,
    OperatorSession,
    TurnInProgressError,
    TurnSilenceTimeout,
    build_operator_child_env,
    is_terminal_event,
)
from osprey.interfaces.web_terminal.routes.chat_attachments import mark_sent
from osprey.interfaces.web_terminal.session_handoff import (
    SURFACE_CHAT,
    ChannelClosed,
    ChannelToken,
    HandoffError,
    HandoffRefused,
    SpawnRequest,
    acquire_surface,
)
from osprey.interfaces.web_terminal.session_key import is_posture_key
from osprey.interfaces.web_terminal.slash_commands import (
    list_chat_slash_commands,
    unavailable_command,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Default turn timeout (SDK silence) when app.state.chat_turn_timeout_s is unset.
_DEFAULT_TURN_TIMEOUT_S = 600
# Cadence of `: heartbeat` SSE comment frames emitted while awaiting SDK events.
HEARTBEAT_INTERVAL_S = 15


class ChatRequest(BaseModel):
    """Body of ``POST /api/chat``.

    ``chat_id`` is deliberately unconstrained. It is the caller's own pool key
    — any client that wants a private conversation picks one and keeps using
    it — and narrowing it here would break every embedder that keys chats on
    something of its own while buying nothing: the id never leaves the pool's
    in-memory dict on this path.

    The closed bare-UUID grammar lives on the *posture* surface instead
    (:mod:`osprey.interfaces.web_terminal.session_key`), because that is
    where a caller-supplied key becomes a store key on disk that decides a
    child's execution mode. The practical consequence is worth stating: the
    shipped client mints its chat id with ``crypto.randomUUID()``
    (``static/js/chat.js``), so its chats are addressable by the posture route,
    while a chat id of some other shape still chats perfectly well and simply
    cannot have a posture set on it. Such a child is spawned — and audited —
    as ``posture_source=process`` rather than ``live``, because ``live`` means
    "a store keeps answering for this key" and for an unaddressable id no
    store ever will (see :func:`_acquire_chat_turn`).
    """

    prompt: str
    chat_id: str


class PermissionAnswer(BaseModel):
    """Body of ``POST /api/chat/{chat_id}/permission``.

    ``answers`` (question text -> chosen label, several labels comma-joined)
    answers an ``AskUserQuestion``; for a tool approval it is omitted.
    ``for_session`` allows the tool for the rest of the session, where the
    request offered that (``can_allow_session``).
    """

    request_id: str
    allow: bool
    answers: dict[str, str] | None = None
    message: str | None = None
    for_session: bool = False


#: Longest string value of a tool input the chat client is sent. Inputs are
#: shown as a one-line summary and an expandable detail, never edited.
_INPUT_VALUE_CHARS = 2000
#: Deepest nesting of a tool input the chat client is sent. A container nested
#: deeper is replaced by a marker, so an input built to be bottomless cannot end
#: the stream that carries it.
_INPUT_DEPTH = 8


def clip_input(value: Any, depth: int = 0) -> Any:
    """Return *value* with every string inside it cut to ``_INPUT_VALUE_CHARS``
    and every container below ``_INPUT_DEPTH`` replaced by a marker.

    The live stream and the replay (``routes/session.py``) clip through this one
    function, so a reloaded chat shows the inputs the live chat did.
    """
    if isinstance(value, str):
        return value if len(value) <= _INPUT_VALUE_CHARS else value[:_INPUT_VALUE_CHARS] + "…"
    if isinstance(value, dict):
        if depth >= _INPUT_DEPTH:
            return "{…}"
        return {k: clip_input(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        if depth >= _INPUT_DEPTH:
            return "[…]"
        return [clip_input(v, depth + 1) for v in value]
    return value


def _strip_for_chat(event: dict[str, Any]) -> dict[str, Any]:
    """Return a wire-hygiene copy of *event* for the chat API client.

    The chat renders tool chips, thinking traces and approval cards, so the
    payloads those need pass through, bounded:

    - ``tool_use`` / ``permission_request``  ``input`` with long strings clipped.
    - ``tool_result``  only the clipped ``preview`` the stream already carries.
    - ``result``  reduce to ``{"type", "is_error"}`` plus ``duration_ms`` and
                  ``seq`` when the frame carries them — cost and turn counts
                  never reach the chat client.
    - ``context``  reduce to the window's occupancy: ``used_tokens``,
                   ``window_tokens`` and ``percent``.
    - every other type (``text``, ``thinking``, ``system``, ``error``,
      ``session_reset``, …) passes through unchanged.

    The input event is never mutated; a new dict is returned whenever a change
    applies (and the same object may be returned for pass-through types).
    """
    etype = event.get("type")

    if etype in ("tool_use", "permission_request") and "input" in event:
        return {**event, "input": clip_input(event["input"])}
    if etype == "tool_result":
        return {k: v for k, v in event.items() if k != "content"}
    if etype == "result":
        return {
            "type": "result",
            "is_error": event.get("is_error"),
            **{k: event[k] for k in ("duration_ms", "seq") if k in event},
        }
    if etype == "context":
        return {
            "type": "context",
            "used_tokens": event.get("used_tokens"),
            "window_tokens": event.get("window_tokens"),
            "percent": event.get("percent"),
            "seq": event.get("seq"),
        }

    return event


def _turn_timeout_s(request: Request) -> float:
    """Resolve the per-turn SDK-silence timeout from app.state (with default)."""
    return getattr(request.app.state, "chat_turn_timeout_s", _DEFAULT_TURN_TIMEOUT_S)


async def _acquire_chat_turn(request: Request, chat_id: str) -> tuple[OperatorSession, int, bool]:
    """Take the session key for the chat, then mint this turn's guard.

    Every turn goes through :func:`acquire_surface`, because the key may be
    held by the terminal and only the hand-off door can stand
    that process down before this one starts. A key the chat already holds
    passes straight through: the pooled child is handed back and the spawn
    callback below is never called, so the pool's own environment comparison
    no longer runs per turn. It runs on the one path that still creates a
    child — this key holding nothing live.

    Returns ``(session, token, fresh_conversation)``. ``fresh_conversation``
    is True only when the child that will answer this prompt was started with
    no transcript to resume: the conversation begins here and the client is
    told so with a ``session_reset`` marker. A resumed child continues a
    conversation the operator has already seen — in the other surface, or in this
    one before a restart — so it carries no marker.

    Maps every failure to the HTTP contract with a machine-readable ``error``
    in the body: 409 when another surface or connection is consuming the key,
    when a terminal turn can only be ended by an interrupt
    (``handoff_needs_interrupt`` — the client offers to stop it and ask
    again), when the chat was torn down as it started
    (:class:`ChatSessionTerminatedError`), or when a turn is already in flight
    (:class:`TurnInProgressError`); 429 when the chat pool is full; 503 when
    the hand-off's premise vanished under it. :class:`ChannelClosed` is left
    to the caller — the client that asked for this turn is already gone.
    """
    cwd: str = request.app.state.project_cwd
    registry = request.app.state.operator_registry

    async def spawn(spawn_request: SpawnRequest) -> OperatorSession:
        """Start the chat child the hand-off door asked for, under its key."""
        session, _created = await registry.get_or_create_chat_session(
            spawn_request.key,
            cwd,
            # A BUILDER, not a mapping — and that is load-bearing, not style.
            # The chat pool is keyed on chat_id, so that is this surface's
            # session identity and the key its runtime posture hangs on.
            #
            # The source is "live" only for a chat_id the posture surface can
            # actually address. chat_id is deliberately unconstrained here
            # (see ChatRequest), while the posture routes refuse anything
            # outside the closed bare-UUID grammar with a 400 — so for an
            # embedder-chosen key like "user-42-chat-3" no store will ever
            # answer, and stamping "live" would tell an auditor that a runtime
            # toggle governed a process no toggle can reach. "process" is the
            # honest word for that child: nothing established its posture but
            # the environment it inherited. The shipped client mints its id
            # with crypto.randomUUID(), so it always takes the "live" arm.
            #
            # Reading the store HERE and passing the result down would leave
            # the read and the pool's registration of this creation separated
            # by everything on the way in. Nothing today suspends in that gap,
            # but any awaited work added to this handler (an audit emitter, an
            # identity lookup) would let a concurrent POST /api/terminal/posture
            # land inside it: the flip writes the store, terminates a pool that
            # is still empty, answers 200 — and this creation then registers a
            # child carrying the posture the operator just left. The pool calls
            # the builder inside the lock hold that registers the pending
            # creation, so the flip must land wholly before it (fresh env) or
            # wholly after (superseded creation, 409). See
            # ``ChatSessionPool.get_or_create``.
            lambda: build_operator_child_env(
                cwd,
                session_key=spawn_request.key,
                app=request.app,
                posture_source=(
                    POSTURE_SOURCE_LIVE
                    if is_posture_key(spawn_request.key)
                    else POSTURE_SOURCE_PROCESS
                ),
            ),
            resume_id=spawn_request.resume_id,
        )
        return cast("OperatorSession", session)

    try:
        result = await acquire_surface(
            request.app,
            chat_id,
            SURFACE_CHAT,
            ChannelToken(request.is_disconnected),
            spawn=spawn,
        )
    except HandoffRefused as refused:
        # 409 for a contested key, 429 for a full pool, 503 for an outgoing
        # process that outlived its kill. The slug is what the client branches
        # on; the message is for the log and the developer console.
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
        # The chat was torn down while this very session was starting — a
        # posture flip or an explicit DELETE. Retrying is the whole remedy and
        # it respawns under whatever the store now holds, so say so rather than
        # letting a deliberate teardown surface as a 500.
        raise HTTPException(
            status_code=409,
            detail={
                "error": "chat_terminated",
                "message": (
                    "This chat was terminated while it was starting; send the prompt again."
                ),
            },
        ) from None

    session = cast("OperatorSession", result.session)
    try:
        token = session.acquire_turn()
    except TurnInProgressError:
        # The backstop behind the hand-off's own wait: a turn that started
        # between that wait ending and this guard being taken.
        raise HTTPException(
            status_code=409,
            detail={
                "error": "turn_in_progress",
                "message": "A turn is already in flight for this chat.",
            },
        ) from None

    return session, token, result.spawned and result.resume_id is None


@router.post("/api/chat")
async def chat(request: Request, body: ChatRequest, stream: bool = True):
    """Send a prompt to Claude and receive the response.

    Query params:
        stream  If true (default), return an SSE event stream.
                If false, buffer the full response and return JSON.
    """
    if not HAS_SDK:
        raise HTTPException(status_code=503, detail="Claude Agent SDK is not available")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt must not be empty")

    # A command the chat does not list (``/login``, ``/model``, ``/config``) the
    # CLI would run itself. Refused here, before the key is taken or a turn
    # opens, so no client reaches it. The name goes back for the notice to say.
    # The project's command files are read off the loop.
    command = await asyncio.to_thread(
        unavailable_command, prompt, Path(request.app.state.project_cwd)
    )
    if command is not None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "command_not_available",
                "command": command,
                "message": f"/{command} is not available in this chat.",
            },
        )

    try:
        session, token, fresh_conversation = await _acquire_chat_turn(request, body.chat_id)
    except ChannelClosed:
        # The client hung up while the session key was being handed over.
        # Nothing is left holding anything; there is simply nobody to answer.
        return Response(status_code=204)
    # The prompt is accepted: the files it carries are the conversation's now.
    mark_sent(request.app, body.chat_id, prompt)

    turn_timeout_s = _turn_timeout_s(request)

    if not stream:
        return await _buffered_response(session, token, prompt, fresh_conversation, turn_timeout_s)

    return StreamingResponse(
        _stream_events(session, token, prompt, fresh_conversation, turn_timeout_s),
        media_type="text/event-stream",
    )


async def _stream_events(
    session: OperatorSession,
    token: int,
    prompt: str,
    fresh_conversation: bool,
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
        if fresh_conversation:
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
        # Non-terminal: client disconnect / re-delivered cancellation. The turn
        # runs on as an orphan (see run_turn); re-raise so Starlette sees it.
        raise
    except Exception as exc:
        logger.exception("Chat stream error")
        error_type = exc.error_type if isinstance(exc, AgentRunError) else type(exc).__name__
        yield _sse(
            _strip_for_chat({"type": "error", "message": str(exc), "error_type": error_type})
        )
    finally:
        session.end_consumer(token)


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
    fresh_conversation: bool,
    turn_timeout_s: float,
) -> JSONResponse:
    """Run one turn to completion and return a single reduced JSON response.

    Consumes the same :meth:`OperatorSession.run_turn` machine as the SSE
    branch — heartbeat markers are skipped instead of forwarded, and a silence
    timeout maps to 504 instead of an error frame. A ``session_reset`` marker is
    prepended to ``events`` when the conversation starts with this turn, and
    ``_strip_for_chat`` applies to every event. The top-level payload is reduced to exactly
    ``{text, events, is_error, error?}`` — cost/duration/turn counts never
    reach the chat client.
    """
    events: list[dict[str, Any]] = []
    text_parts: list[str] = []
    is_error = False
    error_msg: str | None = None
    status_code = 200

    if fresh_conversation:
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
        # Abandoned request — the turn runs on as an orphan (see run_turn);
        # re-raise so the server sees the cancellation.
        raise
    finally:
        # Safety net for the never-started case only (see _stream_events).
        session.end_consumer(token)

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
    """Signal-only interrupt of whatever the chat's agent is doing.

    A turn in flight, or a response the CLI started on its own — either can be
    stopped from any tab. It never releases the turn guard; the turn's owner
    (its consumer, or the pump for a turn whose tab went away) does that when
    the interrupt's result arrives. A 204 is returned whether or not there was
    anything to interrupt (idempotent).
    """
    registry = request.app.state.operator_registry
    session = registry.get_chat_session(chat_id)
    if session is not None:
        try:
            await session.interrupt()
        except Exception:
            logger.debug("Interrupt signal failed for chat %s", chat_id, exc_info=True)
    return Response(status_code=204)


@router.post("/api/chat/{chat_id}/permission", status_code=204)
async def answer_chat_permission(
    chat_id: str, body: PermissionAnswer, request: Request
) -> Response:
    """Answer a ``permission_request`` the chat's in-flight turn is waiting on.

    404 ``no_pending_request`` when nothing waits under ``request_id`` — the
    turn was stopped, or the request was already answered from another tab.
    """
    registry = request.app.state.operator_registry
    session = registry.get_chat_session(chat_id)
    if session is None or not await session.resolve_permission(
        body.request_id,
        allow=body.allow,
        answers=body.answers,
        message=body.message,
        for_session=body.for_session,
    ):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_pending_request",
                "message": "No request is waiting under that id.",
            },
        )
    return Response(status_code=204)


@router.get("/api/chat/{chat_id}/events")
async def chat_events(chat_id: str, request: Request) -> Response:
    """A live feed of everything the chat's agent does, for any tab bound to it.

    Opens with a ``hello`` snapshot — whether a turn is running, its prompt,
    and the permission requests waiting — then carries every frame the agent
    emits, whichever tab asked: a reloaded tab picks a running turn back up,
    a second tab follows it, and a response the CLI starts on its own is seen.
    Frames carry ``seq``, so a tab that also holds the turn's own stream draws
    each frame once. 204 with no body while no agent runs for the chat yet: a fresh
    chat has none until its first prompt starts one. That is a normal state, not an
    error; the client retries slowly until a prompt starts one.
    """
    registry = request.app.state.operator_registry
    session = registry.get_chat_session(chat_id)
    if session is None or not session.is_active:
        return Response(status_code=204)
    feed, hello = session.subscribe()

    async def stream():
        try:
            pending = [_strip_for_chat(frame) for frame in hello["pending"]]
            context = _strip_for_chat(hello["context"]) if hello["context"] else None
            yield _sse({**hello, "pending": pending, "context": context})
            while session.feed_open(feed):
                try:
                    frame = await asyncio.wait_for(feed.get(), timeout=HEARTBEAT_INTERVAL_S)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield _sse(_strip_for_chat(frame))
            # The agent is gone. Its last frames, the one that says so among
            # them, can still be queued: send them before the feed ends.
            if not session.is_active:
                while not feed.empty():
                    yield _sse(_strip_for_chat(feed.get_nowait()))
        finally:
            session.unsubscribe(feed)

    return StreamingResponse(stream(), media_type="text/event-stream")


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


@router.get("/api/chat/commands")
def list_chat_commands(request: Request) -> dict[str, Any]:
    """The slash commands the chat agent accepts in this project.

    Read from the directory the chat agent is started in, so every name listed
    is one the agent expands. The CLI's own built-ins (``/clear``, ``/compact``)
    are listed beside the project's commands.
    """
    cwd: str = request.app.state.project_cwd
    return {"commands": [asdict(c) for c in list_chat_slash_commands(Path(cwd))]}


def _sse(event: dict[str, Any]) -> str:
    """Format an event dict as an SSE data line."""
    return f"data: {json.dumps(event)}\n\n"
