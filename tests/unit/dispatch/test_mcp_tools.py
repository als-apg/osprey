"""Unit tests for the event-dispatcher MCP tools (``osprey.dispatch.mcp_tools``).

Covers all four tools — ``list_triggers``, ``trigger_history``,
``trigger_status``, and ``manual_fire``. The tools are registered as closures on
a FastMCP instance; we register them on a throwaway ``FastMCP`` and pull each
tool's raw coroutine via ``await mcp.get_tool(name)`` → ``.fn``.

``manual_fire`` is the behavioral focus: it must route through the server's real
``fire_callback`` (honoring the disabled short-circuit and surfacing
``QueueFullError``) rather than merely recording an event.
"""

from __future__ import annotations

import json

from osprey.dispatch.mcp_tools import register_tools
from osprey.dispatch.pool import DispatchPool, QueueFullError
from osprey.dispatch.registry import TriggerRegistry
from osprey.dispatch.trigger_config import TriggerConfig


def _trigger(name: str = "deploy", source: str = "webhook") -> TriggerConfig:
    return TriggerConfig(
        name=name,
        source=source,
        action={"prompt": "do it", "allowed_tools": []},
    )


async def _registry_with(*triggers: TriggerConfig) -> TriggerRegistry:
    registry = TriggerRegistry()
    for t in triggers:
        await registry.register(t)
    return registry


async def _get_tools(registry, pool, fire_callback=None):
    """Register tools on a fresh FastMCP and return {name: raw coroutine fn}."""
    from fastmcp import FastMCP

    mcp = FastMCP("test-dispatcher")
    register_tools(mcp, registry, pool, fire_callback)
    names = ("list_triggers", "trigger_history", "trigger_status", "manual_fire")
    return {name: (await mcp.get_tool(name)).fn for name in names}


# ---------------------------------------------------------------------------
# list_triggers
# ---------------------------------------------------------------------------


async def test_list_triggers_returns_all():
    registry = await _registry_with(_trigger("a"), _trigger("b", source="cron"))
    tools = await _get_tools(registry, DispatchPool(max_concurrent=1, max_queue_depth=1))

    result = json.loads(await tools["list_triggers"]())

    names = {t["name"] for t in result}
    assert names == {"a", "b"}
    assert all("status" in t and "source" in t for t in result)


# ---------------------------------------------------------------------------
# trigger_history
# ---------------------------------------------------------------------------


async def test_trigger_history_returns_events():
    registry = await _registry_with(_trigger("a"))
    await registry.record_event("a", {"x": 1}, "dispatched")
    tools = await _get_tools(registry, DispatchPool(max_concurrent=1, max_queue_depth=1))

    result = json.loads(await tools["trigger_history"]("a"))

    assert len(result) == 1
    assert result[0]["result"] == "dispatched"


async def test_trigger_history_unknown_trigger_errors():
    registry = await _registry_with(_trigger("a"))
    tools = await _get_tools(registry, DispatchPool(max_concurrent=1, max_queue_depth=1))

    result = json.loads(await tools["trigger_history"]("nope"))

    assert "error" in result


# ---------------------------------------------------------------------------
# trigger_status
# ---------------------------------------------------------------------------


async def test_trigger_status_includes_pool():
    registry = await _registry_with(_trigger("a"))
    tools = await _get_tools(registry, DispatchPool(max_concurrent=3, max_queue_depth=5))

    result = json.loads(await tools["trigger_status"]("a"))

    assert result["name"] == "a"
    assert result["pool"]["max"] == 3


async def test_trigger_status_unknown_trigger_errors():
    registry = await _registry_with(_trigger("a"))
    tools = await _get_tools(registry, DispatchPool(max_concurrent=1, max_queue_depth=1))

    result = json.loads(await tools["trigger_status"]("nope"))

    assert "error" in result


# ---------------------------------------------------------------------------
# manual_fire — routes through fire_callback
# ---------------------------------------------------------------------------


async def test_manual_fire_invokes_fire_callback_and_returns_dispatch_id():
    """An enabled trigger fires through fire_callback and returns its dispatch_id."""
    registry = await _registry_with(_trigger("deploy"))
    seen: dict = {}

    async def spy_fire(trigger, payload):
        seen["trigger"] = trigger
        seen["payload"] = payload
        return "dispatch-123"

    tools = await _get_tools(registry, DispatchPool(max_concurrent=1, max_queue_depth=1), spy_fire)

    result = json.loads(await tools["manual_fire"]("deploy", {"k": "v"}))

    assert result["dispatched"] is True
    assert result["dispatch_id"] == "dispatch-123"
    assert result["source"] == "webhook"
    # The real TriggerConfig (not a status dict) is handed to fire_callback,
    # with the manual marker folded into the payload.
    assert seen["trigger"].name == "deploy"
    assert seen["payload"] == {"manual": True, "k": "v"}


async def test_manual_fire_disabled_trigger_is_refused():
    """fire_callback returns None for a disabled trigger → manual_fire reports it."""
    registry = await _registry_with(_trigger("deploy"))
    await registry.set_status("deploy", "disabled")

    async def fire_honoring_disabled(trigger, payload):
        # Mirror the server's fire_callback disabled short-circuit.
        if registry._status.get(trigger.name) == "disabled":
            await registry.record_event(trigger.name, payload, "ignored: disabled")
            return None
        return "dispatch-xyz"

    tools = await _get_tools(
        registry, DispatchPool(max_concurrent=1, max_queue_depth=1), fire_honoring_disabled
    )

    result = json.loads(await tools["manual_fire"]("deploy"))

    assert result["dispatched"] is False
    assert result["reason"] == "disabled"
    # The disabled fire was recorded as ignored, not dispatched.
    history = await registry.get_history("deploy")
    assert history[-1]["result"] == "ignored: disabled"


async def test_manual_fire_unknown_trigger_errors():
    registry = await _registry_with(_trigger("deploy"))
    called = {"n": 0}

    async def spy_fire(trigger, payload):
        called["n"] += 1
        return "x"

    tools = await _get_tools(registry, DispatchPool(max_concurrent=1, max_queue_depth=1), spy_fire)

    result = json.loads(await tools["manual_fire"]("ghost"))

    assert "error" in result
    assert called["n"] == 0  # never reaches the dispatch path


async def test_manual_fire_queue_full_errors():
    registry = await _registry_with(_trigger("deploy"))

    async def fire_queue_full(trigger, payload):
        raise QueueFullError("Queue depth 1 exceeded for trigger 'deploy'")

    tools = await _get_tools(
        registry, DispatchPool(max_concurrent=1, max_queue_depth=1), fire_queue_full
    )

    result = json.loads(await tools["manual_fire"]("deploy"))

    assert "error" in result
    assert "Queue depth" in result["error"]


async def test_manual_fire_without_fire_callback_reports_unavailable():
    """A server wired without a fire_callback must not silently record-only."""
    registry = await _registry_with(_trigger("deploy"))
    tools = await _get_tools(registry, DispatchPool(max_concurrent=1, max_queue_depth=1), None)

    result = json.loads(await tools["manual_fire"]("deploy"))

    assert "error" in result
    assert "not available" in result["error"]
