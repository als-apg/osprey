"""Single source of truth for resolving the Bluesky bridge connection.

Two independent OSPREY components talk to the same facility-side Bluesky bridge
and must agree, exactly, on *which* bridge instance and *which* launch token
they are using:

- the Bluesky MCP server (``osprey.mcp_server.bluesky.server_context``), the
  agent's path to the bridge, and
- the operator panels sidecar (``osprey.services.bluesky_panels``), the
  browser's path to the bridge.

If those two ever drift in how they resolve the bridge URL or the launch token
-- a different env-var name, a different config key, a different fallback order
-- the agent and the panel silently arm (or fail to arm) *different* bridge
instances. That is a safety-relevant bug class: the human could approve a
launch in a panel that targets one bridge while the agent's writes-arming
token belongs to another. Keeping the resolution logic here, imported by both,
makes that drift impossible by construction.

This is a top-level leaf module (the precedent is
``osprey.bluesky_tool_names``): it imports **nothing** from
``osprey.mcp_server`` or ``osprey.services`` so both may import it without a
cycle. ``osprey.utils.workspace`` is imported lazily, inside the resolver
functions, only when a config fallback is actually needed.
"""

from __future__ import annotations

import os

DEFAULT_BRIDGE_URL = "http://127.0.0.1:8090"


def resolve_bridge_url() -> str:
    """Resolve the Bluesky bridge base URL.

    Resolution order:

    1. ``BLUESKY_BRIDGE_URL`` env var (full URL) -- set by the framework server
       definition per bridge instance; wins outright.
    2. ``bluesky.bridge_url`` in config.yml.
    3. ``http://127.0.0.1:8090`` default.

    The returned URL has any trailing slash stripped so callers can append a
    path verbatim.
    """
    full = os.environ.get("BLUESKY_BRIDGE_URL")
    if full:
        return full.rstrip("/")

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    url = config.get("bluesky", {}).get("bridge_url", DEFAULT_BRIDGE_URL)
    return str(url).rstrip("/")


def resolve_launch_token() -> str | None:
    """Resolve the Bluesky bridge launch token.

    Resolution order:

    1. ``BLUESKY_LAUNCH_TOKEN`` env var -- minted fail-closed per bridge
       instance by the framework server definition; wins outright.
    2. ``bluesky.launch_token`` in config.yml (local/dev convenience only).
    3. ``None`` -- launch is refused client-side, before contacting the bridge,
       when no token is resolved.
    """
    token = os.environ.get("BLUESKY_LAUNCH_TOKEN")
    if token:
        return token

    from osprey.utils.workspace import load_osprey_config

    config = load_osprey_config()
    token = config.get("bluesky", {}).get("launch_token")
    return str(token) if token else None


def bridge_error_message(body: object, status: int) -> str:
    """Extract the bridge's FastAPI ``detail`` message, falling back to the status."""
    if isinstance(body, dict) and body.get("detail"):
        return str(body["detail"])
    return f"Bluesky bridge returned HTTP {status}."


def unwrap_bridge_conflict_detail(body: object) -> dict | None:
    """Unwrap the bridge's nested 409 ``detail`` dict, or ``None`` when absent.

    A structured 409 from ``POST /draft/run`` nests a ``{"code", "detail",
    "revision"}`` payload under FastAPI's top-level ``detail`` key
    (``{"detail": {"code": ..., ...}}``). Return that nested dict so callers can
    read the discriminator, or ``None`` for a 409 whose ``detail`` is a plain
    string (e.g. a validation-gate rejection) -- in which case the caller falls
    back to its own default rendering. This captures only the shared
    unwrap-or-fallback decision; the divergent downstream rendering (the MCP
    server's ``run_launch_conflict`` error envelope vs the panel sidecar's raw
    409 JSON) stays at each call site.
    """
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            return detail
    return None
