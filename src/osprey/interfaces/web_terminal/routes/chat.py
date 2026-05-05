"""REST chat endpoint — send a prompt, get a response via SSE or buffered JSON.

POST /api/chat?stream={true|false}
Body: {"prompt": "..."}

Creates an ephemeral OperatorSession per request (no registry tracking),
streams events from the SDK, and tears down the session on completion or
client disconnect.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from osprey.interfaces.web_terminal.operator_session import (
    CLAUDE_SDK_AVAILABLE,
    OperatorSession,
    build_clean_env,
)

logger = logging.getLogger(__name__)

router = APIRouter()

EVENT_TIMEOUT_S = 300  # Max seconds to wait for a single event from the SDK


class ChatRequest(BaseModel):
    prompt: str


def _is_terminal(event: dict[str, Any]) -> bool:
    """Return True if *event* signals end-of-stream."""
    etype = event.get("type")
    if etype == "result":
        return True
    if etype == "error" and event.get("error_type") != "AssistantMessageError":
        return True
    return False


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

    cwd: str = request.app.state.project_cwd

    if stream:
        return StreamingResponse(
            _stream_events(cwd, prompt),
            media_type="text/event-stream",
        )

    # Buffered mode — collect all events, return JSON.
    return await _buffered_response(cwd, prompt)


async def _stream_events(cwd: str, prompt: str):
    """Async generator that yields SSE-formatted events."""
    session = OperatorSession(cwd=cwd, env=build_clean_env(cwd))
    try:
        await session.start()
        await session.send_prompt(prompt)

        while True:
            try:
                event = await asyncio.wait_for(session._queue.get(), timeout=EVENT_TIMEOUT_S)
            except TimeoutError:
                yield _sse(
                    {
                        "type": "error",
                        "message": "Timeout waiting for response",
                        "error_type": "TimeoutError",
                    }
                )
                return

            if event.get("type") == "keepalive":
                continue

            yield _sse(event)

            if _is_terminal(event):
                return
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.exception("Chat stream error")
        yield _sse({"type": "error", "message": str(exc), "error_type": type(exc).__name__})
    finally:
        await session.stop()


async def _buffered_response(cwd: str, prompt: str) -> JSONResponse:
    """Run the prompt to completion and return a single JSON response."""
    session = OperatorSession(cwd=cwd, env=build_clean_env(cwd))
    try:
        await session.start()
        await session.send_prompt(prompt)

        events: list[dict[str, Any]] = []
        text_parts: list[str] = []
        result_meta: dict[str, Any] = {"is_error": False}

        while True:
            try:
                event = await asyncio.wait_for(session._queue.get(), timeout=EVENT_TIMEOUT_S)
            except TimeoutError:
                return JSONResponse(
                    content={
                        "text": "".join(text_parts),
                        "events": events,
                        "total_cost_usd": result_meta.get("total_cost_usd"),
                        "duration_ms": result_meta.get("duration_ms"),
                        "num_turns": result_meta.get("num_turns"),
                        "is_error": True,
                        "error": "Timeout waiting for response",
                    },
                    status_code=504,
                )

            if event.get("type") == "keepalive":
                continue

            events.append(event)

            if event["type"] == "text":
                text_parts.append(event["content"])
            elif event["type"] == "result":
                result_meta = event
            elif event["type"] == "error" and event.get("error_type") != "AssistantMessageError":
                return JSONResponse(
                    content={
                        "text": "".join(text_parts),
                        "events": events,
                        "total_cost_usd": None,
                        "duration_ms": None,
                        "num_turns": None,
                        "is_error": True,
                        "error": event.get("message", "Unknown error"),
                    },
                    status_code=500,
                )

            if _is_terminal(event):
                break

        return JSONResponse(
            content={
                "text": "".join(text_parts),
                "events": events,
                "total_cost_usd": result_meta.get("total_cost_usd"),
                "duration_ms": result_meta.get("duration_ms"),
                "num_turns": result_meta.get("num_turns"),
                "is_error": result_meta.get("is_error", False),
            }
        )
    finally:
        await session.stop()


def _sse(event: dict[str, Any]) -> str:
    """Format an event dict as an SSE data line."""
    return f"data: {json.dumps(event)}\n\n"
