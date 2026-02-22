"""Channel Finder AI search — thin wrapper around OperatorSession.

Provides session-based AI search for the Channel Finder web interface.
Reuses OperatorSession and OperatorRegistry from the web terminal for
Claude Agent SDK conversation management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from osprey.interfaces.web_terminal.operator_session import (
    OperatorRegistry,
    OperatorSession,
    build_clean_env,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level registry (initialized during app lifespan)
_registry: OperatorRegistry | None = None


def init_search_registry() -> OperatorRegistry:
    """Initialize the module-level search session registry.

    Returns:
        The newly created OperatorRegistry.
    """
    global _registry
    _registry = OperatorRegistry()
    return _registry


async def cleanup_search_registry() -> None:
    """Clean up all active search sessions and reset the registry."""
    global _registry
    if _registry is not None:
        await _registry.cleanup_all()
        _registry = None


def _get_registry() -> OperatorRegistry:
    """Get the search session registry, raising if not initialized."""
    if _registry is None:
        raise RuntimeError("Search session registry not initialized")
    return _registry


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SearchQueryRequest(BaseModel):
    """Request body for a search query."""

    text: str


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@router.post("/api/search/session")
async def create_search_session(request: Request):
    """Create a new AI search session.

    Spawns an OperatorSession backed by the Claude Agent SDK, configured
    with the project directory from app state.

    Returns:
        JSON with the new session_id.
    """
    registry = _get_registry()
    session_id = uuid.uuid4().hex
    cwd = getattr(request.app.state, "project_cwd", None)
    if not cwd:
        raise HTTPException(status_code=500, detail="Project directory not configured")

    try:
        env = build_clean_env(project_cwd=cwd)
        await registry.create_session(session_id, cwd=cwd, env=env)
    except Exception as exc:
        logger.error("Failed to create search session: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create search session: {exc}",
        ) from exc

    return {"session_id": session_id}


@router.post("/api/search/{session_id}/query")
async def search_query(session_id: str, body: SearchQueryRequest):
    """Send a query to an existing search session.

    Args:
        session_id: The search session identifier.
        body: Request body containing the search text.

    Returns:
        JSON confirming the query was submitted for streaming.
    """
    registry = _get_registry()
    session = registry.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Empty query text")

    try:
        await session.send_prompt(text)
    except Exception as exc:
        logger.error("Failed to send query to session %s: %s", session_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to send query: {exc}") from exc

    return {"status": "streaming"}


@router.delete("/api/search/{session_id}")
async def delete_search_session(session_id: str):
    """Terminate a search session.

    Args:
        session_id: The search session identifier to terminate.
    """
    registry = _get_registry()
    session = registry.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    await registry.terminate_session(session_id)
    return {"status": "terminated", "session_id": session_id}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/search/{session_id}")
async def search_ws(websocket: WebSocket, session_id: str):
    """WebSocket bridge for AI search streaming.

    Protocol:
    - Client -> Server JSON: {"type": "prompt", "text": "..."}
    - Client -> Server JSON: {"type": "cancel"}
    - Server -> Client JSON: structured events (text, thinking, tool_use, etc.)
    """
    await websocket.accept()

    registry = _get_registry()
    cwd = getattr(websocket.app.state, "project_cwd", None)

    # Get or create session
    session: OperatorSession | None = registry.get_session(session_id)

    if session is None:
        if not cwd:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Project directory not configured",
                    "error_type": "ConfigError",
                }
            )
            await websocket.close()
            return

        try:
            env = build_clean_env(project_cwd=cwd)
            session = await registry.create_session(session_id, cwd=cwd, env=env)
        except Exception as exc:
            logger.error("Failed to create search session via WS: %s", exc)
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Failed to start search session: {exc}",
                        "error_type": type(exc).__name__,
                    }
                )
            except Exception:
                pass
            await websocket.close()
            return

    forward_task: asyncio.Task | None = None

    async def forward_events():
        """Drain the session queue and send events to the WebSocket."""
        try:
            while True:
                event = await session._queue.get()
                if event.get("type") == "keepalive":
                    continue
                await websocket.send_json(event)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    forward_task = asyncio.create_task(forward_events())

    try:
        # Notify client that search session is ready
        await websocket.send_json({"type": "system", "subtype": "init"})

        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")
            if msg_type == "prompt":
                text = msg.get("text", "").strip()
                if text:
                    await session.send_prompt(text)
            elif msg_type == "cancel":
                await session.cancel()

    except WebSocketDisconnect:
        pass
    finally:
        if forward_task is not None:
            forward_task.cancel()
            try:
                await forward_task
            except asyncio.CancelledError:
                pass
        # Only clean up the session if we created it in this WS handler
        # to avoid killing sessions that are managed externally
        if session is not None:
            await registry.terminate_session_if_owner(session_id, session)
