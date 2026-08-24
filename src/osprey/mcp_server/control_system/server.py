"""OSPREY Control System MCP Server.

FastMCP server exposing channel_read, channel_write, archiver_read and the
control-system target switch.

Startup does two things beyond registering tools, both of them about the target
this session is pointed at:

* :func:`create_server` **resets** the target state file to the deployment
  baseline and kills the connector-host children a dead predecessor left
  behind. It is synchronous and runs before the event loop exists, which is
  exactly right: no target selection may survive the process that made it, and
  an orphaned child holding a gateway must not outlive the server that spawned
  it.
* the server **lifespan** runs the background endpoint prober, because a task
  needs a running loop and ``create_server()`` is called before ``run()``
  starts one. The prober only produces the roster's reachability rows, so every
  part of it is guarded: a server whose prober will not start is a server whose
  roster reports fewer rows, and that is a far better outcome than a controls
  server that will not start at all.

Usage:
    python -m osprey.mcp_server.control_system
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.control_system")

#: The running endpoint prober, or ``None`` when one was never started. Held
#: here rather than on the context because it is a property of this server
#: process's lifespan, and shutdown reaches it through :func:`stop_background`.
_prober: Any = None


def get_endpoint_prober() -> Any:
    """The running endpoint prober, or ``None``.

    The roster reads its reachability rows from this; ``None`` means the rows
    are unavailable, which the roster reports rather than papers over.
    """
    return _prober


async def start_background() -> Any:
    """Start the endpoint prober. Returns it, or ``None`` when it could not start.

    Guarded end to end: reachability rows are a convenience the roster degrades
    without, so nothing here may prevent the server from serving.
    """
    global _prober

    if _prober is not None:
        return _prober
    try:
        from osprey.mcp_server.control_system.endpoint_prober import EndpointProber
        from osprey.mcp_server.control_system.server_context import get_server_context

        prober = EndpointProber(get_server_context().config.raw)
        await prober.start()
    except Exception:
        logger.warning(
            "Could not start the endpoint prober; the roster runs without reachability rows",
            exc_info=True,
        )
        return None
    _prober = prober
    return prober


async def stop_background() -> None:
    """Stop the endpoint prober, if one is running. Never raises."""
    global _prober

    prober, _prober = _prober, None
    if prober is None:
        return
    try:
        await prober.stop()
    except Exception:  # pragma: no cover - defensive
        logger.debug("Error stopping the endpoint prober (ignored)", exc_info=True)


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Own the things that need a running event loop.

    The connector-host child is deliberately NOT started here: a deployment
    that never switches targets serves its tools from the in-process connector
    and spawns nothing, and the supervisor is created on first use.
    """
    await start_background()
    try:
        yield {}
    finally:
        await stop_background()
        try:
            from osprey.mcp_server.control_system.server_context import get_server_context

            await get_server_context().shutdown()
        except Exception:
            logger.debug("Error during control-system shutdown (ignored)", exc_info=True)


mcp = FastMCP(
    "controls",
    instructions="Read and write control-system channels and query archiver history",
    lifespan=_lifespan,
)


def _reset_target_state() -> None:
    """Publish the deployment baseline and kill inherited connector hosts.

    Guarded: a state file that cannot be written costs the prompt hook its
    identity line and the roster its display metadata, both of which degrade to
    "unknown". Refusing to start the server over it would cost the operator
    every control-system tool instead.
    """
    try:
        from osprey.mcp_server.control_system.server_context import get_server_context

        orphans = get_server_context().connector_hosts.reset_state()
    except Exception:
        logger.warning(
            "Could not reset the control-target state file; the session target is unpublished",
            exc_info=True,
        )
        return
    if orphans:
        logger.warning("Killed %d orphaned connector-host child process(es)", len(orphans))


def create_server() -> FastMCP:
    """Initialize the registry and import tool modules, then return the server."""
    from osprey.mcp_server.control_system.server_context import initialize_server_context
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
        startup_timer,
    )
    from osprey.utils.workspace import resolve_workspace_root

    prime_config_builder()

    with startup_timer("server_context"):
        initialize_server_context()

    # Session working root used by other tools at call time; the artifact
    # store itself is rooted at the shared data root inside
    # initialize_workspace_singletons().
    logger.info("Workspace root: %s", resolve_workspace_root())
    initialize_workspace_singletons()

    # The state file is a reset, not a merge: a fresh server always starts on
    # the deployment baseline, and any connector host a dead predecessor left
    # running is killed before this one can spawn its own.
    with startup_timer("target_state"):
        _reset_target_state()

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.control_system.tools import (  # noqa: F401
            archiver_read,
            channel_limits,
            channel_read,
            channel_write,
            control_target,
        )

    logger.info("Control System MCP server initialised with all tools registered")
    return mcp
