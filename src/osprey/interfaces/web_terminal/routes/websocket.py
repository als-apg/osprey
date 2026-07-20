"""WebSocket routes for terminal PTY and operator (Agent SDK) sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from osprey.interfaces.web_terminal.operator_session import build_clean_env
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

logger = logging.getLogger(__name__)

router = APIRouter()

_UUID_RE = re.compile(r"^[a-f0-9-]{36}$")


def _read_effort_level(config_path: Path | None) -> str | None:
    """Read claude_code.effort from config.yml."""
    if not config_path or not Path(config_path).exists():
        return None
    try:
        config = yaml.safe_load(Path(config_path).read_text()) or {}
        return config.get("claude_code", {}).get("effort")
    except Exception:
        return None


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
    timeout: float = 15.0,
) -> str | None:
    """Discover a newly-created Claude session UUID and notify the client.

    Returns the discovered UUID (or None). Also rekeys the registry entry.
    """
    loop = asyncio.get_event_loop()
    new_id = await loop.run_in_executor(None, discovery.discover_new_session, snapshot, timeout)
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
    telemetry_session_id: str | None = None,
) -> dict[str, str]:
    """Build the extra environment dict for PTY sessions.

    ``telemetry_session_id`` is the session UUID this terminal's ``claude`` is
    forced onto (via ``--session-id``); it is handed to the workspace
    provenance_locator tool so a filed issue can point back to this session's
    telemetry. Kept separate from ``claude_session_id`` — which drives
    ``OSPREY_SESSION_ID`` (artifact-store relocation) and stays unset for new
    sessions — because the telemetry locator must NOT relocate the artifact store.
    """
    extra_env: dict[str, str] = {}
    if claude_session_id:
        extra_env["OSPREY_SESSION_ID"] = claude_session_id
    if telemetry_session_id:
        extra_env["OSPREY_TELEMETRY_SESSION_ID"] = telemetry_session_id
        extra_env["OSPREY_TELEMETRY_SESSION_START"] = datetime.now(UTC).isoformat()
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

    effort = _read_effort_level(websocket.app.state.config_path)

    # Build the command and determine the initial session key.
    # base_shell_command is list[str] (set by app.lifespan), so unpack with
    # [*base, ...] — nesting would break PtySession's exec (issue #218).
    if mode == "resume" and req_session_id:
        command: list[str] = [*base_shell_command, "--resume", req_session_id]
        claude_session_id: str | None = req_session_id
        telemetry_session_id: str = req_session_id
    else:
        # Force a known session UUID so the workspace provenance_locator tool can
        # hand it back (via OSPREY_TELEMETRY_SESSION_ID, injected below) and it
        # matches the value the OTEL emitter tags records with as session.id — a
        # filed issue's provenance pointer then resolves. Leave claude_session_id
        # None so the new-session snapshot/discovery path is unchanged: discovery
        # simply finds the id we forced. (Not claude_session_id, which would set
        # OSPREY_SESSION_ID and relocate the artifact store.)
        telemetry_session_id = str(uuid.uuid4())
        command = [*base_shell_command, "--session-id", telemetry_session_id]
        claude_session_id = None

    if effort:
        command.extend(["--effort", effort])

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

    # For resumes, snapshot too — a stale/absent --resume-id can make the CLI
    # silently start a fresh session instead of resuming, and this is how we
    # tell the two apart once the PTY is up.
    resume_snapshot: set[str] | None = None
    if mode == "resume" and req_session_id:
        resume_snapshot = discovery.snapshot_session_ids()

    extra_env = _build_extra_env(websocket, claude_session_id, telemetry_session_id)

    session, was_reused = registry.get_or_create_session(
        current_key,
        command,
        rows=initial_rows,
        cols=initial_cols,
        extra_env=extra_env if extra_env else None,
        cwd=websocket.app.state.project_cwd,
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

    # For resumes, confirm the actually-attached session id so the client can
    # tell a live resume from a silently-fresh PTY on a stale --resume-id.
    # Two cases are trusted immediately, with no discovery needed: a reused
    # warm session (same live PTY) and a cold resume whose session file was
    # already on disk before we spawned (the id was genuinely valid). Only a
    # request for an id with no file on disk — stale/absent — is ambiguous:
    # the CLI may fall back to creating a fresh session, so that case is
    # confirmed via the same discovery mechanism (and window) used above —
    # every entry into this branch is racing the CLI's own startup to write
    # a new session file, so it needs the same generous timeout as the
    # new-session path, not a shortened one.
    if resume_snapshot is not None:
        if was_reused or req_session_id in resume_snapshot:
            try:
                await websocket.send_text(
                    json.dumps({"type": "session_info", "session_id": current_key})
                )
            except Exception:
                pass
        else:

            async def _confirm_resume():
                nonlocal current_key
                found = await _discover_and_notify(
                    resume_snapshot, discovery, registry, current_key, websocket
                )
                if found:
                    current_key = found
                else:
                    try:
                        await websocket.send_text(
                            json.dumps({"type": "session_info", "session_id": current_key})
                        )
                    except Exception:
                        pass

            asyncio.create_task(_confirm_resume())

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

                            # 3. Build command for target — unpack base_shell_command
                            #    (list[str]) so a pinned ["npx", "-y", "..."] prefix
                            #    flattens into target_cmd rather than nesting.
                            target_cmd: list[str] = [
                                *base_shell_command,
                                "--resume",
                                target_id,
                            ]
                            if effort:
                                target_cmd.extend(["--effort", effort])
                            target_env = _build_extra_env(websocket, target_id)

                            # 4. Get or create target session
                            session, was_reused = registry.get_or_create_session(
                                target_id,
                                target_cmd,
                                rows=initial_rows,
                                cols=initial_cols,
                                extra_env=target_env if target_env else None,
                                cwd=websocket.app.state.project_cwd,
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


@router.post("/api/terminal/logout")
async def logout_terminal(request: Request):
    """Terminate the user's warm PTY (and operator) session(s) on logout.

    Each Web Terminal container serves a single user (the multi-user
    topology puts one container behind each ``/u/<user>/`` path), so — like
    ``/api/terminal/restart`` — there is no per-caller session to pick out;
    the whole pool is this user's. Unlike restart, which the client
    immediately reconnects to (respawning a fresh PTY under the same
    flow), logout must not leave anything resumable behind: this empties
    both pools — the PTY registry and the operator-mode (Agent SDK)
    registry, the latter a live agent with tool access and therefore the
    more sensitive of the two — via their existing ``cleanup_all``
    primitives, mirroring ``restart_terminal`` (routes/panels.py), so the
    next visitor at a shared browser inherits no live session of either
    kind (closes the M2 warm-session-inheritance hazard). The client
    clears its stored session id and navigates to the landing page
    afterward — it does not reconnect.
    """
    pty_registry = request.app.state.pty_registry
    operator_registry = request.app.state.operator_registry

    # Terminate all PTY sessions (single-user model)
    pty_registry.cleanup_all()
    logger.info("PTY session(s) terminated for logout")

    # Terminate all operator sessions if active
    try:
        await operator_registry.cleanup_all()
    except Exception:
        pass  # May not have active operator sessions

    return {"status": "ok", "message": "Logged out — terminal session terminated"}


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
