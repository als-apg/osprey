"""MCP tools for the event dispatcher — trigger inspection and manual firing."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers

from osprey.dispatch.pool import DispatchPool, QueueFullError
from osprey.dispatch.registry import TriggerRegistry
from osprey.dispatch.trigger_config import TriggerConfig
from osprey.utils.owner_header import OWNER_HEADER, owner_from_header

# The dispatcher's fire callback: (trigger, payload, owner) -> dispatch_id, or
# None if the trigger is disabled. ``owner`` names the human the fire is
# attributed to, and is None when the fire carries no attribution. Raises
# QueueFullError when the pool is saturated.
#
# ``osprey.dispatch.sources.base.FireCallback`` is the same callback spelled
# with two arguments, deliberately: a trigger source fires on nobody's behalf,
# so it has no owner to name. The server's concrete callback defaults the owner
# and therefore satisfies both spellings.
FireCallback = Callable[[TriggerConfig, dict[str, Any], str | None], Awaitable[str | None]]


def register_tools(
    mcp: FastMCP,
    registry: TriggerRegistry,
    pool: DispatchPool,
    fire_callback: FireCallback | None = None,
) -> None:
    """Register all event dispatcher MCP tools on the given FastMCP instance.

    Uses closures so each tool has access to registry and pool without
    relying on module-level globals.

    ``fire_callback`` is the server's real dispatch entry point — the same
    callback the webhook/cron sources use. ``manual_fire`` routes through it so
    a manual fire honors the disabled short-circuit and the per-trigger tool
    allowlist (rather than merely recording an event). When omitted (e.g. a
    bare inspection-only server), ``manual_fire`` reports that dispatch is
    unavailable instead of silently recording without dispatching.
    """

    @mcp.tool()
    async def list_triggers() -> str:
        """List all registered triggers with their current status.

        Returns a JSON array of trigger summaries including name, source,
        status, and last_fired timestamp.
        """
        triggers = await registry.list_triggers()
        return json.dumps(triggers, indent=2)

    @mcp.tool()
    async def trigger_history(name: str, limit: int = 20) -> str:
        """Return recent dispatch events for a trigger.

        Args:
            name: Trigger name to query.
            limit: Maximum number of recent events to return (default 20).

        Returns:
            JSON array of event records (timestamp, event_data, result),
            newest-last order.
        """
        try:
            history = await registry.get_history(name, limit=limit)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(history, indent=2)

    @mcp.tool()
    async def trigger_status(name: str) -> str:
        """Return detailed status for a trigger including dispatch pool position.

        Args:
            name: Trigger name to query.

        Returns:
            JSON object with trigger status fields and current pool metrics.
        """
        try:
            status = await registry.get_status(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        pool_status = pool.get_pool_status()
        return json.dumps({**status, "pool": pool_status}, indent=2)

    @mcp.tool()
    async def manual_fire(name: str, payload: dict[str, Any] | None = None) -> str:
        """Dispatch a trigger on demand regardless of its configured source type.

        Useful for testing triggers without waiting for the real event source
        (webhook, cron, etc.) to fire.

        The fire is attributed to whoever the ``X-Osprey-Owner`` request header
        names; there is no way to fire on another human's behalf.

        Args:
            name: Trigger name to fire.
            payload: Optional payload dict passed to the trigger action.

        Returns:
            JSON object with dispatch_id on success, or error details.
        """
        if payload is None:
            payload = {}

        try:
            trigger_info = await registry.get_status(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        if fire_callback is None:
            return json.dumps({"error": "dispatch is not available on this server"})

        trigger_cfg = registry._triggers.get(name)
        if trigger_cfg is None:  # pragma: no cover - get_status above already guards this
            return json.dumps({"error": f"Trigger '{name}' not registered"})

        event_payload: dict[str, Any] = {"manual": True, **payload}

        # The owner is whoever the request says it is, never whoever the caller
        # says it is: an argument would let an agent holding this tool credit
        # its fire to any human. ``get_http_headers`` lower-cases every name and
        # returns an empty mapping when no HTTP request is in scope, so a
        # direct in-process call is simply owner-less. A value that names nobody
        # costs the fire its owner, never the dispatch: the run goes ahead
        # owner-less, so its writes are checked against no one's narrowing and
        # the attribution is lost with it — see ``owner_from_header``.
        owner = owner_from_header(get_http_headers().get(OWNER_HEADER.lower()))

        # Route through the real dispatch path so a manual fire honors the
        # disabled short-circuit and the per-trigger allowlist, exactly like a
        # webhook/cron fire. fire_callback returns None when the trigger is
        # disabled (and records "ignored: disabled" itself) and raises
        # QueueFullError when the pool is saturated.
        try:
            dispatch_id = await fire_callback(trigger_cfg, event_payload, owner)
        except QueueFullError as exc:
            return json.dumps({"error": str(exc)})

        if dispatch_id is None:
            return json.dumps(
                {"dispatched": False, "reason": "disabled", "trigger": name},
                indent=2,
            )

        return json.dumps(
            {
                "dispatched": True,
                "dispatch_id": dispatch_id,
                "trigger": name,
                "source": trigger_info["source"],
                "payload": event_payload,
            },
            indent=2,
        )
