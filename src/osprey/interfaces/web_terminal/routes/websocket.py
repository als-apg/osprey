"""WebSocket routes for terminal PTY and operator (Agent SDK) sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from osprey.interfaces.web_terminal.operator_session import build_clean_env
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

logger = logging.getLogger(__name__)

router = APIRouter()

_UUID_RE = re.compile(r"^[a-f0-9-]{36}$")


async def _run_output_loop(
    session,
    websocket: WebSocket,
    stop_event: asyncio.Event,
) -> None:
    """Forward PTY bytes to the WebSocket until stopped or process exits."""
    try:
        async for data in session.read_output():
            if stop_event.is_set():
                return
            await websocket.send_bytes(data)
    except Exception:
        pass
    finally:
        if not stop_event.is_set():
            code = session.exit_code
            try:
                await websocket.send_text(json.dumps({"type": "exit", "code": code}))
            except Exception:
                pass


async def _discover_and_notify(
    snapshot: set[str],
    discovery: SessionDiscovery,
    registry,
    current_key: str,
    websocket: WebSocket,
) -> str | None:
    """Discover a newly-created Claude session UUID and notify the client.

    Returns the discovered UUID (or None). Also rekeys the registry entry.
    """
    loop = asyncio.get_event_loop()
    new_id = await loop.run_in_executor(None, discovery.discover_new_session, snapshot)
    if new_id:
        registry.rekey_session(current_key, new_id)
        try:
            await websocket.send_text(json.dumps({"type": "session_info", "session_id": new_id}))
        except Exception:
            pass
    return new_id


def _build_extra_env(
    websocket: WebSocket,
    claude_session_id: str | None,
) -> dict[str, str]:
    """Build the extra environment dict for PTY sessions."""
    extra_env: dict[str, str] = {}
    if claude_session_id:
        extra_env["OSPREY_SESSION_ID"] = claude_session_id
    hooks_env = getattr(websocket.app.state, "hooks_env", {})
    if hooks_env:
        extra_env.update(hooks_env)
    return extra_env


@router.websocket("/ws/terminal")
async def terminal_ws(websocket: WebSocket):
    """WebSocket bridge for terminal I/O with session pool support.

    Protocol:
    - Client -> Server text frames: raw terminal input (keystrokes)
    - Client -> Server JSON: {"type": "resize", "cols": N, "rows": N}
    - Client -> Server JSON: {"type": "switch_session", "session_id": UUID}
    - Server -> Client binary frames: raw PTY output
    - Server -> Client JSON: {"type": "exit", "code": N}
    - Server -> Client JSON: {"type": "session_switched", "session_id": UUID}
    - Server -> Client JSON: {"type": "session_info", "session_id": UUID}
    - Server -> Client JSON: {"type": "error", "message": str}
    """
    await websocket.accept()

    registry = websocket.app.state.pty_registry
    base_shell_command = websocket.app.state.shell_command
    discovery = SessionDiscovery(websocket.app.state.project_cwd)

    # Parse session params from query string
    req_session_id = websocket.query_params.get("session_id")
    mode = websocket.query_params.get("mode", "new")

    # Build the command and determine the initial session key
    if mode == "resume" and req_session_id:
        command: str | list[str] = [base_shell_command, "--resume", req_session_id]
        claude_session_id: str | None = req_session_id
    else:
        command = base_shell_command
        claude_session_id = None

    # Use claude_session_id as pool key for resumes, temp key for new sessions
    current_key = claude_session_id or f"terminal-{uuid.uuid4().hex[:8]}"

    # Wait for the client's initial resize message before spawning the PTY.
    initial_cols, initial_rows = 80, 24
    try:
        first = await asyncio.wait_for(websocket.receive(), timeout=5.0)
        if "text" in first:
            try:
                msg = json.loads(first["text"])
                if msg.get("type") == "resize":
                    initial_cols = msg["cols"]
                    initial_rows = msg["rows"]
            except (json.JSONDecodeError, KeyError):
                pass
    except TimeoutError:
        logger.warning("No initial resize from client within 5s, using defaults")

    # For new sessions, snapshot existing session files before spawning
    snapshot: set[str] | None = None
    if claude_session_id is None:
        snapshot = discovery.snapshot_session_ids()

    extra_env = _build_extra_env(websocket, claude_session_id)

    session, was_reused = registry.get_or_create_session(
        current_key,
        command,
        rows=initial_rows,
        cols=initial_cols,
        extra_env=extra_env if extra_env else None,
    )
    registry.attach_session(current_key)

    # For new sessions, discover the Claude-generated UUID asynchronously
    if snapshot is not None:

        async def _do_initial_discover():
            nonlocal current_key
            found = await _discover_and_notify(
                snapshot, discovery, registry, current_key, websocket
            )
            if found:
                current_key = found

        asyncio.create_task(_do_initial_discover())

    # Start output forwarding
    stop_event = asyncio.Event()
    output_task = asyncio.create_task(_run_output_loop(session, websocket, stop_event))

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                text = message["text"]
                try:
                    msg = json.loads(text)
                except (json.JSONDecodeError, KeyError):
                    msg = None

                if isinstance(msg, dict):
                    msg_type = msg.get("type")

                    if msg_type == "resize":
                        logger.debug("PTY resize: %dx%d", msg["cols"], msg["rows"])
                        session.resize(msg["rows"], msg["cols"])
                        continue

                    if msg_type == "switch_session":
                        target_id = msg.get("session_id", "")
                        if not _UUID_RE.match(target_id):
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": "Invalid session ID format",
                                    }
                                )
                            )
                            continue

                        if target_id == current_key:
                            # Already on this session — no-op
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "session_switched",
                                        "session_id": target_id,
                                    }
                                )
                            )
                            continue

                        try:
                            # 1. Stop current output loop
                            stop_event.set()
                            output_task.cancel()
                            try:
                                await output_task
                            except asyncio.CancelledError:
                                pass

                            # 2. Detach current session (stays alive in pool)
                            registry.detach_session(current_key)

                            # 3. Build command for target
                            target_cmd: str | list[str] = [
                                base_shell_command,
                                "--resume",
                                target_id,
                            ]
                            target_env = _build_extra_env(websocket, target_id)

                            # 4. Get or create target session
                            session, was_reused = registry.get_or_create_session(
                                target_id,
                                target_cmd,
                                rows=initial_rows,
                                cols=initial_cols,
                                extra_env=target_env if target_env else None,
                            )
                            registry.attach_session(target_id)

                            # 5. Notify client
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "session_switched",
                                        "session_id": target_id,
                                    }
                                )
                            )

                            # 6. Start new output loop
                            stop_event = asyncio.Event()
                            output_task = asyncio.create_task(
                                _run_output_loop(session, websocket, stop_event)
                            )

                            # 7. Update tracking
                            current_key = target_id

                            logger.info(
                                "Session switched to %s (reused=%s)",
                                target_id,
                                was_reused,
                            )
                        except Exception:
                            logger.exception("Session switch failed")
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": "Session switch failed",
                                    }
                                )
                            )
                        continue

                # Not a recognized JSON control message — treat as terminal input
                session.write_input(text.encode("utf-8"))

            elif "bytes" in message:
                session.write_input(message["bytes"])

    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        stop_event.set()
        output_task.cancel()
        # Detach instead of terminate — keep session alive in the pool.
        # Only terminate if the process has already died.
        registry.detach_session(current_key)
        if not session.is_alive:
            registry.terminate_session(current_key)


@router.websocket("/ws/operator")
async def operator_ws(websocket: WebSocket):
    """WebSocket bridge for operator-mode (Claude Agent SDK).

    Protocol:
    - Client -> Server JSON: {"type": "prompt", "text": "..."}
    - Client -> Server JSON: {"type": "cancel"}
    - Server -> Client JSON: structured events (text, thinking, tool_use, etc.)
    """
    await websocket.accept()

    registry = websocket.app.state.operator_registry
    cwd = websocket.app.state.project_cwd
    operator_key = f"operator-{uuid.uuid4().hex[:8]}"
    session = None
    forward_task = None

    try:
        env = build_clean_env(project_cwd=cwd)
        session = await registry.create_session(operator_key, cwd=cwd, env=env)
    except Exception as exc:
        logger.error("Failed to create operator session: %s", exc)
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Failed to start operator session: {exc}",
                    "error_type": type(exc).__name__,
                }
            )
        except Exception:
            pass
        await websocket.close()
        return

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
        # Notify client that operator session is ready
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
        if session is not None:
            await registry.terminate_session_if_owner(operator_key, session)
