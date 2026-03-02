"""Panel configuration, server URL, health, and MCP introspection routes."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

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
        from osprey.utils.workspace import load_osprey_config

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
                    data = json.loads(resp.read())
                    auth_token = data.get("authToken")
            except Exception:
                pass
    return {"url": url, "available": healthy, "authToken": auth_token}


@router.get("/api/agentsview-server")
async def agentsview_server_config(request: Request):
    """Return the agentsview session analytics server URL for iframe embedding."""
    url = getattr(request.app.state, "agentsview_server_url", None)
    project = getattr(request.app.state, "agentsview_project", None)
    result: dict = {"url": url, "available": url is not None}
    if project:
        result["project"] = project
    return result


@router.get("/api/panels")
async def get_panels(request: Request):
    """Return panel configuration: enabled built-in panels + custom panels."""
    enabled = list(getattr(request.app.state, "enabled_panels", set()))
    custom = getattr(request.app.state, "custom_panels", [])
    return {"enabled": enabled, "custom": custom}


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
    known = set(getattr(request.app.state, "enabled_panels", set()))
    known |= {p["id"] for p in getattr(request.app.state, "custom_panels", [])}
    if body.panel not in known:
        raise HTTPException(status_code=422, detail=f"Unknown panel: {body.panel}")
    request.app.state.active_panel = body.panel
    event: dict = {"type": "panel_focus", "panel": body.panel}
    if body.url:
        event["url"] = body.url
    request.app.state.broadcaster.broadcast(event)
    return {"status": "ok", "active_panel": body.panel}


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


# ---- MCP Server Introspection ---- #


@router.get("/api/mcp-servers")
async def get_mcp_servers(request: Request):
    """Return enriched MCP server metadata with tool lists."""
    from osprey.mcp_server.introspect import get_mcp_servers_cached

    project_dir = request.app.state.project_cwd
    mcp_json_path = __import__("pathlib").Path(project_dir) / ".mcp.json"

    if not mcp_json_path.exists():
        return []

    try:
        servers = await get_mcp_servers_cached(mcp_json_path, project_dir, timeout=10.0)
        return servers
    except Exception:
        logger.warning("mcp-servers: introspection failed", exc_info=True)
        return []
