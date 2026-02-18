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
from osprey.interfaces.web_terminal.operator_session import build_clean_env
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery, SessionRegistry

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "web_terminal"}


@router.get("/api/artifact-server")
async def artifact_server_config(request: Request):
    """Return the artifact gallery server URL for iframe embedding."""
    url = getattr(request.app.state, "artifact_server_url", "http://127.0.0.1:8086")
    return {"url": url}


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
            full_url = f"{base_url.rstrip('/')}/{default_page.lstrip('/')}" if default_page else base_url
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
    known = {"artifacts", "ariel", "tuning"}
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
    registry = SessionRegistry(request.app.state.workspace_dir)
    discovery = SessionDiscovery(request.app.state.project_cwd)
    sessions = discovery.list_sessions(allowed_ids=registry.known_ids())
    return {"sessions": [asdict(s) for s in sessions]}


# ---- Workspace resolution helper ---- #

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


@router.websocket("/ws/terminal")
async def terminal_ws(websocket: WebSocket):
    """WebSocket bridge for terminal I/O.

    Protocol:
    - Client → Server text frames: raw terminal input (keystrokes)
    - Client → Server JSON: {"type": "resize", "cols": N, "rows": N}
    - Server → Client binary frames: raw PTY output
    - Server → Client JSON: {"type": "exit", "code": N}
    """
    await websocket.accept()

    registry = websocket.app.state.pty_registry
    base_shell_command = websocket.app.state.shell_command

    # Parse session params from query string
    req_session_id = websocket.query_params.get("session_id")
    mode = websocket.query_params.get("mode", "new")

    # Build the command and determine the PTY key
    if mode == "resume" and req_session_id:
        # Resume existing session — pass --resume flag
        command: str | list[str] = [base_shell_command, "--resume", req_session_id]
        claude_session_id = req_session_id
        # Ensure resumed session is in the local registry
        SessionRegistry(websocket.app.state.workspace_dir).register(req_session_id)
    else:
        # New session — Claude auto-generates a UUID
        command = base_shell_command
        claude_session_id = None

    # Each WebSocket connection gets a unique PTY key
    pty_key = f"terminal-{uuid.uuid4().hex[:8]}"

    # Wait for the client's initial resize message before spawning the PTY.
    # The browser measures the actual terminal container dimensions and sends
    # a {"type": "resize", "cols": N, "rows": N} as its very first message.
    # Spawning the PTY with the correct size prevents garbled output from
    # TUI programs (like Claude Code) that render based on the PTY dimensions.
    initial_cols, initial_rows = 80, 24  # fallback if client sends something else first
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
    discovery = SessionDiscovery(websocket.app.state.project_cwd)
    snapshot: set[str] | None = None
    if claude_session_id is None:
        snapshot = discovery.snapshot_session_ids()

    # Build extra env for session scoping
    extra_env: dict[str, str] = {}
    if claude_session_id:
        extra_env["OSPREY_SESSION_ID"] = claude_session_id

    session = registry.create_session(
        pty_key,
        command,
        initial_rows=initial_rows,
        initial_cols=initial_cols,
        extra_env=extra_env if extra_env else None,
    )

    # For new sessions, discover the Claude-generated UUID asynchronously
    async def _discover_and_notify():
        if snapshot is None:
            return
        loop = asyncio.get_event_loop()
        new_id = await loop.run_in_executor(
            None, discovery.discover_new_session, snapshot
        )
        if new_id:
            # Register in the local session registry
            reg = SessionRegistry(websocket.app.state.workspace_dir)
            reg.register(new_id)
            try:
                await websocket.send_text(
                    json.dumps({"type": "session_info", "session_id": new_id})
                )
            except Exception:
                pass

    if snapshot is not None:
        asyncio.create_task(_discover_and_notify())

    # Task to read PTY output and send to client
    async def send_output():
        try:
            async for data in session.read_output():
                await websocket.send_bytes(data)
        except Exception:
            pass
        finally:
            # Process exited — notify client
            code = session.exit_code
            try:
                await websocket.send_text(json.dumps({"type": "exit", "code": code}))
            except Exception:
                pass

    output_task = asyncio.create_task(send_output())

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                text = message["text"]
                # Check if it's a JSON control message
                try:
                    msg = json.loads(text)
                    if msg.get("type") == "resize":
                        logger.debug("PTY resize: %dx%d", msg["cols"], msg["rows"])
                        session.resize(msg["rows"], msg["cols"])
                        continue
                except (json.JSONDecodeError, KeyError):
                    pass
                # Otherwise treat as raw terminal input
                session.write_input(text.encode("utf-8"))

            elif "bytes" in message:
                session.write_input(message["bytes"])

    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError can be raised when Starlette's WebSocket is
        # already disconnected and we attempt to receive.
        pass
    finally:
        output_task.cancel()
        # Only terminate if this WS still owns the session.
        # A page reload creates a new WS that replaces the session;
        # the stale WS's cleanup must not kill the replacement.
        registry.terminate_session_if_owner(pty_key, session)


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
