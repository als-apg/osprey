"""HTTP, WebSocket, and SSE routes for the OSPREY Web Terminal."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from osprey.interfaces.web_terminal.claude_code_files import ClaudeCodeFileService
from osprey.interfaces.web_terminal.claude_memory_service import (
    ClaudeMemoryService,
    MemoryFileExistsError,
    MemoryFileNotFoundError,
    MemoryValidationError,
)
from osprey.interfaces.web_terminal.operator_session import build_clean_env
from osprey.interfaces.web_terminal.prompt_gallery_service import PromptGalleryService
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    from osprey import __version__

    session_id = getattr(request.app.state, "server_session_id", None)
    return {
        "status": "healthy",
        "service": "web_terminal",
        "session_id": session_id,
        "version": __version__,
    }


@router.get("/api/artifact-server")
async def artifact_server_config(request: Request):
    """Return the artifact gallery server URL for iframe embedding."""
    url = getattr(request.app.state, "artifact_server_url", "http://127.0.0.1:8086")
    return {"url": url}


@router.get("/api/type-registry")
async def get_type_registry():
    """Return the OSPREY type registry for tool/category color mapping."""
    from osprey.mcp_server.type_registry import registry_to_api_dict

    return registry_to_api_dict()


@router.get("/api/ariel-server")
async def ariel_server_config(request: Request):
    """Return the ARIEL logbook server URL for iframe embedding."""
    url = getattr(request.app.state, "ariel_server_url", None)
    return {"url": url, "available": url is not None}


@router.get("/api/tuning-server")
async def tuning_server_config(request: Request):
    """Return the tuning panel server URL for iframe embedding."""
    url = getattr(request.app.state, "tuning_server_url", None)
    return {"url": url, "available": url is not None}


@router.get("/api/channel-finder-server")
async def channel_finder_server_config(request: Request):
    """Return the Channel Finder server URL for iframe embedding."""
    url = getattr(request.app.state, "channel_finder_server_url", None)
    return {"url": url, "available": url is not None}


@router.get("/api/wiki-url")
async def wiki_url(request: Request):
    """Return the external wiki URL for the header link button."""
    try:
        from osprey.mcp_server.common import load_osprey_config

        config = load_osprey_config()
        confluence = config.get("confluence", {})
        base_url = confluence.get("url")
        if base_url:
            default_page = confluence.get("default_page", "")
            full_url = (
                f"{base_url.rstrip('/')}/{default_page.lstrip('/')}" if default_page else base_url
            )
            return {"url": full_url, "available": True}
    except Exception:
        pass
    return {"url": None, "available": False}


@router.get("/api/cui-server")
async def cui_server_config(request: Request):
    """Return the CUI server URL, live health status, and auth token."""
    import urllib.request as _urllib

    url = getattr(request.app.state, "cui_server_url", None)
    healthy = False
    auth_token = None
    if url:
        try:
            req = _urllib.Request(f"{url}/health", method="GET")
            with _urllib.urlopen(req, timeout=1) as resp:
                healthy = resp.status == 200
        except Exception:
            pass
        # Fetch the auth token from CUI's config so the iframe can auto-authenticate
        if healthy:
            try:
                req = _urllib.Request(f"{url}/api/config", method="GET")
                with _urllib.urlopen(req, timeout=1) as resp:
                    import json as _json

                    data = _json.loads(resp.read())
                    auth_token = data.get("authToken")
            except Exception:
                pass
    return {"url": url, "available": healthy, "authToken": auth_token}


@router.get("/api/agentsview-server")
async def agentsview_server_config(request: Request):
    """Return the agentsview session analytics server URL for iframe embedding."""
    url = getattr(request.app.state, "agentsview_server_url", None)
    return {"url": url, "available": url is not None}


class PanelFocusRequest(BaseModel):
    panel: str
    url: str | None = None


@router.get("/api/panel-focus")
async def get_panel_focus(request: Request):
    """Return the currently active panel."""
    active = getattr(request.app.state, "active_panel", None)
    return {"active_panel": active}


@router.post("/api/panel-focus")
async def set_panel_focus(body: PanelFocusRequest, request: Request):
    """Set the active panel and broadcast a focus event via SSE."""
    known = {
        "artifacts",
        "ariel",
        "tuning",
        "channel-finder",
        "session",
        "session-analytics",
    }
    if body.panel not in known:
        raise HTTPException(status_code=422, detail=f"Unknown panel: {body.panel}")
    request.app.state.active_panel = body.panel
    event: dict = {"type": "panel_focus", "panel": body.panel}
    if body.url:
        event["url"] = body.url
    request.app.state.broadcaster.broadcast(event)
    return {"status": "ok", "active_panel": body.panel}


@router.get("/api/sessions")
async def list_sessions(request: Request):
    """Return Claude Code sessions registered for this project."""
    discovery = SessionDiscovery(request.app.state.project_cwd)
    sessions = discovery.list_sessions()
    return {"sessions": [asdict(s) for s in sessions]}


# ---- Session Diagnostics Endpoints ---- #


@router.get("/api/session-server")
async def session_server_config(request: Request):
    """Return the session diagnostics page URL for iframe embedding."""
    base = str(request.base_url).rstrip("/")
    return {"url": f"{base}/static/session.html"}


def _read_session_events(request: Request, reader) -> list[dict]:
    """Read session events, scoped to a session_id if provided."""
    session_id = request.query_params.get("session_id")
    if session_id:
        return reader.read_session_by_id(session_id)
    return reader.read_current_session()


@router.get("/api/session-agents")
async def session_agents(request: Request):
    """Return agent hierarchy and tool call breakdown for the current session."""
    try:
        from osprey.mcp_server.workspace.tools.session_log import _build_agent_summary
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        reader = TranscriptReader(request.app.state.project_cwd)
        events = _read_session_events(request, reader)
        agents = _build_agent_summary(events)

        # Group tool_call events by agent_id
        tool_calls_by_agent: dict[str, list[dict]] = {}
        for ev in events:
            if ev.get("type") != "tool_call":
                continue
            aid = ev.get("agent_id") or "main"
            tool_calls_by_agent.setdefault(aid, []).append(ev)

        return {
            "agents": agents,
            "tool_calls_by_agent": tool_calls_by_agent,
            "total_events": len(events),
        }
    except Exception:
        logger.debug("session-agents: no transcript available", exc_info=True)
        return {"agents": [], "tool_calls_by_agent": {}, "total_events": 0}


@router.get("/api/session-agent-timeline")
async def session_agent_timeline(request: Request):
    """Return the full internal timeline of a subagent."""
    agent_id = request.query_params.get("agent_id", "")
    if not agent_id:
        return {"timeline": [], "error": "agent_id required"}
    try:
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        reader = TranscriptReader(request.app.state.project_cwd)
        session_id = request.query_params.get("session_id")
        timeline = reader.read_agent_timeline(agent_id, session_id=session_id)
        return {"agent_id": agent_id, "timeline": timeline, "count": len(timeline)}
    except Exception:
        logger.debug("session-agent-timeline: failed", exc_info=True)
        return {"agent_id": agent_id, "timeline": [], "count": 0}


@router.get("/api/session-log")
async def session_log(request: Request):
    """Return filtered tool call events from the current session."""
    try:
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        reader = TranscriptReader(request.app.state.project_cwd)
        events = _read_session_events(request, reader)

        # Parse query filters
        agent_filter = request.query_params.get("agent")
        agent_id_filter = request.query_params.get("agent_id")
        tool_filter = request.query_params.get("tool")
        errors_only = request.query_params.get("errors_only", "").lower() == "true"
        since = request.query_params.get("since")
        before = request.query_params.get("before")
        last_n = min(int(request.query_params.get("last_n", "100")), 500)

        # Filter to tool_call events
        filtered = [e for e in events if e.get("type") == "tool_call"]

        if agent_filter:
            filtered = [e for e in filtered if e.get("agent_type") == agent_filter]
        if agent_id_filter:
            filtered = [e for e in filtered if (e.get("agent_id") or "main") == agent_id_filter]
        if tool_filter:
            filtered = [
                e
                for e in filtered
                if tool_filter.lower() in (e.get("tool_name") or e.get("tool") or "").lower()
            ]
        if errors_only:
            filtered = [e for e in filtered if e.get("is_error")]
        if since:
            filtered = [e for e in filtered if e.get("timestamp", "") >= since]
        if before:
            filtered = [e for e in filtered if e.get("timestamp", "") < before]

        total = len(filtered)
        filtered = filtered[-last_n:]

        return {"events": filtered, "total_events": total, "showing": len(filtered)}
    except Exception:
        logger.debug("session-log: no transcript available", exc_info=True)
        return {"events": [], "total_events": 0, "showing": 0}


@router.get("/api/session-summary")
async def session_summary(request: Request):
    """Return artifact inventory with channel extraction for the current session."""
    try:
        from osprey.mcp_server.artifact_store import ArtifactStore
        from osprey.mcp_server.workspace.tools.session_summary import _extract_channels

        store = ArtifactStore(request.app.state.workspace_dir)
        session_id = request.query_params.get("session_id")
        entries = store.list_entries(
            session_filter=session_id if session_id else None,
        )

        total_bytes = 0
        categories: dict[str, int] = {}
        tools_used: set[str] = set()
        artifact_types: dict[str, int] = {}
        entry_dicts = []

        for entry in entries:
            total_bytes += entry.size_bytes
            cat = entry.category or "uncategorized"
            categories[cat] = categories.get(cat, 0) + 1
            if entry.tool_source:
                tools_used.add(entry.tool_source)
            artifact_types[entry.artifact_type] = artifact_types.get(entry.artifact_type, 0) + 1
            d = entry.to_dict()
            d["channels"] = _extract_channels(entry)
            entry_dicts.append(d)

        return {
            "entries": entry_dicts,
            "totals": {
                "entry_count": len(entries),
                "total_bytes": total_bytes,
                "categories": categories,
                "tools_used": sorted(tools_used),
                "artifact_types": artifact_types,
            },
        }
    except Exception:
        logger.debug("session-summary: no artifacts available", exc_info=True)
        return {
            "entries": [],
            "totals": {
                "entry_count": 0,
                "total_bytes": 0,
                "categories": {},
                "tools_used": [],
                "artifact_types": {},
            },
        }


@router.get("/api/session-chat")
async def session_chat(request: Request):
    """Return conversation turns from the current session transcript."""
    try:
        from osprey.mcp_server.workspace.transcript_reader import TranscriptReader

        reader = TranscriptReader(request.app.state.project_cwd)
        session_id = request.query_params.get("session_id")
        if session_id:
            turns = reader.read_chat_history_by_id(session_id)
        else:
            turns = reader.read_current_chat_history()
        return {"turns": turns, "count": len(turns)}
    except Exception:
        logger.debug("session-chat: no transcript available", exc_info=True)
        return {"turns": [], "count": 0}


_UUID_RE = re.compile(r"^[a-f0-9-]{36}$")


def _resolve_workspace(request: Request) -> Path:
    """Resolve workspace dir, optionally scoped to a session.

    Reads ``?session_id=`` query param. Returns the session-scoped
    subdirectory if valid, otherwise the base workspace dir.
    """
    workspace_base: Path = request.app.state.workspace_dir
    session_id = request.query_params.get("session_id")
    if session_id and _UUID_RE.match(session_id):
        return workspace_base / "sessions" / session_id
    return workspace_base


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
    - Client → Server text frames: raw terminal input (keystrokes)
    - Client → Server JSON: {"type": "resize", "cols": N, "rows": N}
    - Client → Server JSON: {"type": "switch_session", "session_id": UUID}
    - Server → Client binary frames: raw PTY output
    - Server → Client JSON: {"type": "exit", "code": N}
    - Server → Client JSON: {"type": "session_switched", "session_id": UUID}
    - Server → Client JSON: {"type": "session_info", "session_id": UUID}
    - Server → Client JSON: {"type": "error", "message": str}
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

                if msg is not None:
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


@router.get("/api/files/tree")
async def file_tree(request: Request):
    """Return the workspace directory tree as JSON."""
    workspace_dir: Path = _resolve_workspace(request)

    if not workspace_dir.exists():
        return {"name": workspace_dir.name, "type": "directory", "children": []}

    def build_tree(directory: Path, depth: int = 0) -> dict:
        node = {
            "name": directory.name,
            "path": str(directory.relative_to(workspace_dir)),
            "type": "directory",
        }
        if depth > 10:
            return node

        children = []
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            entries = []

        for entry in entries:
            # Skip hidden/ignored
            if entry.name.startswith(".") or entry.name in (
                "__pycache__",
                "_notebook_cache",
                "node_modules",
            ):
                continue

            if entry.is_dir():
                children.append(build_tree(entry, depth + 1))
            else:
                children.append(
                    {
                        "name": entry.name,
                        "path": str(entry.relative_to(workspace_dir)),
                        "type": "file",
                        "size": entry.stat().st_size,
                    }
                )
        node["children"] = children
        return node

    return build_tree(workspace_dir)


@router.get("/api/files/content/{filepath:path}")
async def file_content(filepath: str, request: Request):
    """Return file content with path traversal protection."""
    workspace_dir: Path = _resolve_workspace(request)
    resolved = (workspace_dir / filepath).resolve()

    if not resolved.is_relative_to(workspace_dir.resolve()):
        raise HTTPException(status_code=403, detail="Path traversal blocked")

    if not resolved.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not resolved.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    # Limit file size to 1MB for preview
    size = resolved.stat().st_size
    if size > 1_048_576:
        raise HTTPException(status_code=413, detail="File too large for preview (>1MB)")

    # Detect binary files
    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=415, detail="Binary file — preview not supported") from None

    return {
        "path": filepath,
        "content": content,
        "size": size,
        "extension": resolved.suffix,
    }


@router.get("/api/files/events")
async def file_events(request: Request):
    """SSE endpoint for real-time file change events."""
    broadcaster = request.app.state.broadcaster
    q = broadcaster.subscribe()

    async def stream():
        try:
            while True:
                data = await q.get()
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            broadcaster.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---- Agent Settings Endpoints ---- #

# Config sections relevant to the agent (exclude internal/infra sections)
_AGENT_CONFIG_SECTIONS = [
    "control_system",
    "approval",
    "channel_finder",
    "ariel",
    "python_execution",
    "artifact_server",
    "workspace",
    "screen_capture",
]


@router.get("/api/config")
async def get_config(request: Request):
    """Return agent-relevant config sections as structured JSON + raw YAML."""
    config_path: Path | None = request.app.state.config_path
    if not config_path or not config_path.exists():
        raise HTTPException(status_code=404, detail="No config.yml found")

    raw = config_path.read_text(encoding="utf-8")
    try:
        full_config = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=500, detail=f"Invalid YAML: {e}") from e

    # Extract only agent-relevant sections
    sections = {}
    for key in _AGENT_CONFIG_SECTIONS:
        if key in full_config:
            sections[key] = full_config[key]

    return {
        "sections": sections,
        "raw": raw,
        "path": str(config_path),
    }


class ConfigUpdate(BaseModel):
    raw: str


@router.put("/api/config")
async def put_config(body: ConfigUpdate, request: Request):
    """Validate YAML, back up config.yml, and write updated config."""
    config_path: Path | None = request.app.state.config_path
    if not config_path:
        raise HTTPException(status_code=404, detail="No config.yml found")

    # Validate YAML parses cleanly
    try:
        yaml.safe_load(body.raw)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {e}") from e

    # Backup existing config
    if config_path.exists():
        backup_path = config_path.with_suffix(".yml.bak")
        shutil.copy2(config_path, backup_path)
        logger.info("Config backed up to %s", backup_path)

    # Write new config (fsync to ensure data is on disk before restart)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(body.raw)
        f.flush()
        os.fsync(f.fileno())
    logger.info("Config updated at %s", config_path)

    return {"status": "ok", "requires_restart": True}


class ConfigPatch(BaseModel):
    updates: dict[str, object]


@router.patch("/api/config")
async def patch_config(body: ConfigPatch, request: Request):
    """Apply structured field updates to config.yml, preserving comments.

    Accepts dot-notation keys (e.g. ``"control_system.writes_enabled": true``).
    Uses ruamel.yaml round-trip mode so comments, ordering, and formatting
    in the YAML file are retained.
    """
    from osprey.utils.yaml_config import config_update_fields

    config_path: Path | None = request.app.state.config_path
    if not config_path or not config_path.exists():
        raise HTTPException(status_code=404, detail="No config.yml found")

    if not body.updates:
        raise HTTPException(status_code=422, detail="No updates provided")

    # Backup before mutation
    backup_path = config_path.with_suffix(".yml.bak")
    shutil.copy2(config_path, backup_path)

    try:
        config_update_fields(config_path, body.updates)
    except Exception as e:
        # Restore backup on failure
        shutil.copy2(backup_path, config_path)
        logger.error("Config patch failed, restored backup: %s", e)
        raise HTTPException(status_code=500, detail=f"Config update failed: {e}") from e

    logger.info("Config patched (%d fields) at %s", len(body.updates), config_path)
    return {"status": "ok", "requires_restart": True, "fields_updated": len(body.updates)}


@router.post("/api/terminal/restart")
async def restart_terminal(request: Request):
    """Terminate the current PTY session (and operator session if active).

    The existing WebSocket reconnection logic automatically respawns
    a fresh PTY with the updated config.
    """
    pty_registry = request.app.state.pty_registry
    operator_registry = request.app.state.operator_registry

    # Terminate all PTY sessions (single-user model)
    pty_registry.cleanup_all()
    logger.info("All PTY sessions terminated for restart")

    # Terminate all operator sessions if active
    try:
        await operator_registry.cleanup_all()
    except Exception:
        pass  # May not have active operator sessions

    return {"status": "ok", "message": "Terminal session terminated — reconnecting"}


# ---- Claude Setup Endpoints ---- #


class ClaudeSetupSaveRequest(BaseModel):
    path: str
    content: str


@router.get("/api/claude-setup")
async def get_claude_setup(request: Request):
    """Read all Claude Code integration files from the project directory."""
    service = ClaudeCodeFileService(Path(request.app.state.project_cwd))
    return {"files": service.list_files()}


@router.put("/api/claude-setup")
async def save_claude_setup(request: Request, body: ClaudeSetupSaveRequest):
    """Save an existing Claude Code file."""
    service = ClaudeCodeFileService(Path(request.app.state.project_cwd))
    try:
        return service.write_file(body.path, body.content)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/api/claude-setup")
async def create_claude_setup(request: Request, body: ClaudeSetupSaveRequest):
    """Create a new Claude Code file in an allowed .claude/ subdirectory."""
    service = ClaudeCodeFileService(Path(request.app.state.project_cwd))
    try:
        return service.create_file(body.path, body.content)
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


# ---- Prompt Gallery Endpoints ---- #


class PromptOverrideRequest(BaseModel):
    content: str


def _prompt_service(request: Request) -> PromptGalleryService:
    """Construct a PromptGalleryService from the request's project dir."""
    return PromptGalleryService(Path(request.app.state.project_cwd))


@router.get("/api/prompts")
async def list_prompts(request: Request):
    """List all prompt artifacts with status and summary counts."""
    service = _prompt_service(request)
    artifacts = service.list_artifacts()
    framework_count = sum(1 for a in artifacts if a["status"] == "framework")
    user_owned_count = sum(1 for a in artifacts if a["status"] == "user-owned")
    return {
        "artifacts": artifacts,
        "summary": {
            "total": len(artifacts),
            "framework": framework_count,
            "user_owned": user_owned_count,
        },
    }


class UntrackedRegisterRequest(BaseModel):
    name: str


@router.get("/api/prompts/untracked")
async def list_untracked_prompts(request: Request):
    """Detect files active in Claude Code but not managed by OSPREY."""
    service = _prompt_service(request)
    untracked = service.scan_untracked()
    return {"untracked": untracked, "count": len(untracked)}


@router.post("/api/prompts/untracked/register")
async def register_untracked_prompt(body: UntrackedRegisterRequest, request: Request):
    """Register an untracked file by adding it to config.yml."""
    service = _prompt_service(request)
    try:
        return service.register_untracked(body.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.delete("/api/prompts/untracked/{name:path}")
async def delete_untracked_prompt(name: str, request: Request):
    """Delete an untracked file from disk."""
    service = _prompt_service(request)
    try:
        return service.delete_untracked(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/api/prompts/{name:path}/framework")
async def get_prompt_framework(name: str, request: Request):
    """Get the framework-rendered content for an artifact."""
    service = _prompt_service(request)
    try:
        content = service.get_framework_content(name)
        return {"name": name, "content": content, "source": "framework"}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/api/prompts/{name:path}/diff")
async def get_prompt_diff(name: str, request: Request):
    """Get unified diff between framework and override."""
    service = _prompt_service(request)
    try:
        return service.compute_diff(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/api/prompts/{name:path}/scaffold")
async def scaffold_prompt(name: str, request: Request):
    """Scaffold an override from the framework template."""
    service = _prompt_service(request)
    try:
        return service.scaffold_override(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.put("/api/prompts/{name:path}/override")
async def save_prompt_override(name: str, body: PromptOverrideRequest, request: Request):
    """Save content to an existing override file."""
    service = _prompt_service(request)
    try:
        return service.save_override(name, body.content)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.delete("/api/prompts/{name:path}/override")
async def delete_prompt_override(name: str, request: Request):
    """Remove an override, restoring framework management."""
    delete_file = request.query_params.get("delete_file", "false").lower() == "true"
    service = _prompt_service(request)
    try:
        return service.unoverride(name, delete_file=delete_file)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/api/prompts/{name:path}")
async def get_prompt(name: str, request: Request):
    """Get artifact content (auto-resolves framework vs override)."""
    service = _prompt_service(request)
    try:
        result = service.get_content(name)
        result["name"] = name
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# ---- Claude Memory Gallery Endpoints ---- #


class MemoryFileRequest(BaseModel):
    content: str
    filename: str | None = None


def _memory_service(request: Request) -> ClaudeMemoryService:
    """Construct a ClaudeMemoryService from the request's project dir."""
    return ClaudeMemoryService(request.app.state.project_cwd)


@router.get("/api/claude-memory")
async def list_memory_files(request: Request):
    """List all memory files with metadata."""
    service = _memory_service(request)
    files = service.list_files()
    return {"files": files, "count": len(files)}


@router.get("/api/claude-memory/{filename}")
async def get_memory_file(filename: str, request: Request):
    """Read a single memory file."""
    service = _memory_service(request)
    try:
        return service.read_file(filename)
    except MemoryFileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except MemoryValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/api/claude-memory")
async def create_memory_file(body: MemoryFileRequest, request: Request):
    """Create a new memory file."""
    if not body.filename:
        raise HTTPException(status_code=422, detail="filename is required")
    service = _memory_service(request)
    try:
        return service.create_file(body.filename, body.content)
    except MemoryFileExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except MemoryValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.put("/api/claude-memory/{filename}")
async def update_memory_file(filename: str, body: MemoryFileRequest, request: Request):
    """Update an existing memory file."""
    service = _memory_service(request)
    try:
        return service.update_file(filename, body.content)
    except MemoryFileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except MemoryValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.delete("/api/claude-memory/{filename}")
async def delete_memory_file(filename: str, request: Request):
    """Delete a memory file."""
    service = _memory_service(request)
    try:
        return service.delete_file(filename)
    except MemoryFileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except MemoryValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
