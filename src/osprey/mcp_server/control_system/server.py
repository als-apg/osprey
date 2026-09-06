"""OSPREY Control System MCP Server.

FastMCP server exposing channel_read, channel_write, archiver_read and the
control-system target switch.

Startup does two things beyond registering tools, both of them about the target
this deployment is pointed at:

* :func:`create_server` **starts this server from the deployment's record**: it
  publishes this process's report, kills the connector-host children a dead
  predecessor left behind, and claims the control-context record when nothing
  alive owns it. It is synchronous and runs before the event loop exists, which
  is exactly right: an orphaned child holding a gateway must not outlive the
  server that spawned it, and the target the deployment is pointed at has to be
  known before anything can be served on it. What this step no longer does is
  publish a baseline — the target belongs to the deployment and survives every
  process that reads it, so a fresh server adopts it rather than resetting it.
* the server **lifespan** runs two background tasks, because a task needs a
  running loop and ``create_server()`` is called before ``run()`` starts one:
  the endpoint prober, which produces the roster's reachability rows, and the
  session-control reconciler, which turns the desired state an operator writes
  from the web terminal into something that has happened to this server's
  connector. Both are guarded end to end and guarded *separately*: a server
  whose prober will not start is a server whose roster reports fewer rows, and
  one whose reconciler will not start is a server the header chip's buttons do
  not reach — either is a far better outcome than a controls server that will
  not start at all, and neither may cost the other.

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

#: The running session-control reconciler, or ``None`` when one was never
#: started. Held beside the prober, for the same reason and with the same
#: lifetime: both belong to this server process's lifespan.
_reconciler: Any = None


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


def get_session_reconciler() -> Any:
    """The running session-control reconciler, or ``None``.

    ``None`` means an operator's posture toggles and Switch button reach this
    server's connector through nothing — the agent's own tool still works, and
    the state file still says what is true.
    """
    return _reconciler


async def start_session_control() -> Any:
    """Start the reconciler. Returns it, or ``None`` when it could not start.

    Guarded end to end and separately from the prober: reconciling the desired
    state an operator wrote is a service this server offers, and serving the
    control-system tools is the job it exists for. Neither may cost the other.
    """
    global _reconciler

    if _reconciler is not None:
        return _reconciler
    try:
        from osprey.mcp_server.control_system.session_control import SessionControlReconciler

        reconciler = SessionControlReconciler()
        await reconciler.start()
    except Exception:
        logger.warning(
            "Could not start the session-control reconciler; the header chip's posture and "
            "switch gestures will not reach this server",
            exc_info=True,
        )
        return None
    _reconciler = reconciler
    return reconciler


async def stop_session_control() -> None:
    """Stop the reconciler, if one is running. Never raises."""
    global _reconciler

    reconciler, _reconciler = _reconciler, None
    if reconciler is None:
        return
    try:
        await reconciler.stop()
    except Exception:  # pragma: no cover - defensive
        logger.debug("Error stopping the session-control reconciler (ignored)", exc_info=True)


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Own the things that need a running event loop.

    The connector-host child is deliberately NOT started here: a deployment
    that never switches targets serves its tools from the in-process connector
    and spawns nothing, and the supervisor is created on first use.
    """
    await start_background()
    await start_session_control()
    try:
        yield {}
    finally:
        await stop_session_control()
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


def _start_from_record() -> None:
    """Publish this server's report, reap inherited children, claim the record.

    Guarded in two halves that fail apart, because they cost different things.
    A report that cannot be written costs the roster this server's row and the
    prompt hook its identity line, both of which degrade to "unknown"; a claim
    that cannot be made leaves the record owned by whoever holds it, which is
    the ordinary state of a controls server running beside a web terminal.
    Refusing to start the server over either would cost the operator every
    control-system tool instead — and a report this server failed to publish is
    no reason to leave the deployment's record ownerless.
    """
    try:
        from osprey.mcp_server.control_system.server_context import get_server_context

        orphans = get_server_context().connector_hosts.reset_state()
    except Exception:
        logger.warning(
            "Could not write this server's control-target report; the roster and the prompt "
            "hook will not see this server",
            exc_info=True,
        )
    else:
        if orphans:
            logger.warning("Killed %d orphaned connector-host child process(es)", len(orphans))

    try:
        from osprey.mcp_server.control_system.server_context import (
            claim_control_context,
            get_server_context,
        )

        claim_control_context(baseline=get_server_context().baseline)
    except Exception:
        logger.warning(
            "Could not resolve the deployment baseline to claim the control context with; "
            "this server follows whatever owns the record",
            exc_info=True,
        )


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

    # A fresh server adopts the target the deployment is already pointed at
    # rather than resetting it, publishes its own report, and kills any
    # connector host a dead predecessor left running before it can spawn one.
    with startup_timer("target_state"):
        _start_from_record()

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
