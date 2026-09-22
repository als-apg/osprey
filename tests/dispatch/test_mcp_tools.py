"""Unit tests for the event-dispatcher MCP tools (``osprey.dispatch.mcp_tools``).

Covers all four tools — ``list_triggers``, ``trigger_history``,
``trigger_status``, and ``manual_fire``. The tools are registered as closures on
a FastMCP instance; we register them on a throwaway ``FastMCP`` and pull each
tool's raw coroutine via ``await mcp.get_tool(name)`` → ``.fn``.

``manual_fire`` is the behavioral focus: it must route through the server's real
``fire_callback`` (honoring the disabled short-circuit and surfacing
``QueueFullError``) rather than merely recording an event, and it must credit
the fire to the human named by the ``X-Osprey-Owner`` request header. The owner
rows supply that header directly; the last row instead posts a real MCP call at
the bearer-gated transport, which is what would notice ``fastmcp`` changing the
spelling it surfaces headers under.
"""

from __future__ import annotations

import inspect
import json
import logging

import httpx
import pytest

from osprey.dispatch import mcp_tools
from osprey.dispatch.mcp_tools import register_tools
from osprey.dispatch.pool import DispatchPool, QueueFullError
from osprey.dispatch.registry import TriggerRegistry
from osprey.dispatch.trigger_config import TriggerConfig
from osprey.utils.owner_header import OWNER_HEADER


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

    async def spy_fire(trigger, payload, _owner):
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

    async def fire_honoring_disabled(trigger, payload, _owner):
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

    async def spy_fire(_trigger, _payload, _owner):
        called["n"] += 1
        return "x"

    tools = await _get_tools(registry, DispatchPool(max_concurrent=1, max_queue_depth=1), spy_fire)

    result = json.loads(await tools["manual_fire"]("ghost"))

    assert "error" in result
    assert called["n"] == 0  # never reaches the dispatch path


async def test_manual_fire_queue_full_errors():
    registry = await _registry_with(_trigger("deploy"))

    async def fire_queue_full(_trigger, _payload, _owner):
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


# ---------------------------------------------------------------------------
# manual_fire — owner attribution from the X-Osprey-Owner header
# ---------------------------------------------------------------------------

#: The logger the shared header guard refuses through. Asserting on the logger
#: rather than the message keeps the guard's wording in its own test file
#: (``tests/utils/test_owner_header.py``), which is where it is pinned.
_GUARD_LOGGER = "osprey.utils.owner_header"


def _with_owner_header(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    """Put *value* in scope as the request's owner header, or nothing for None.

    The raw tool coroutine runs with no HTTP request in scope, so the header
    dict has to be supplied here. ``fastmcp`` lower-cases every header name
    before ``manual_fire`` sees it, so the key is spelled that way.
    """
    headers = {} if value is None else {OWNER_HEADER.lower(): value}
    monkeypatch.setattr(mcp_tools, "get_http_headers", lambda: headers)


def _guard_refusals(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == _GUARD_LOGGER and record.levelno >= logging.WARNING
    ]


async def _fire_capturing_owner(registry, header: str | None, monkeypatch):
    """Fire 'deploy' with *header* in scope; return (result dict, owner seen)."""
    seen: dict = {}

    async def spy_fire(_trigger, _payload, owner):
        seen["owner"] = owner
        return "dispatch-123"

    _with_owner_header(monkeypatch, header)
    tools = await _get_tools(registry, DispatchPool(max_concurrent=1, max_queue_depth=1), spy_fire)

    result = json.loads(await tools["manual_fire"]("deploy"))
    return result, seen["owner"]


async def test_manual_fire_credits_the_owner_named_by_the_header(monkeypatch, caplog):
    """A roster-shaped header value reaches fire_callback verbatim."""
    registry = await _registry_with(_trigger("deploy"))

    with caplog.at_level(logging.DEBUG):
        result, owner = await _fire_capturing_owner(registry, "alice", monkeypatch)

    assert result["dispatched"] is True
    assert owner == "alice"
    assert _guard_refusals(caplog) == []


@pytest.mark.parametrize("header", ["${OSPREY_TERMINAL_USER}", "", "bob/../x"])
async def test_manual_fire_with_a_malformed_header_is_owner_less(header, monkeypatch, caplog):
    """A header that names nobody costs the owner, never the dispatch.

    The run still goes out, but owner-less — so its writes are checked against
    no one's narrowing, which is the real cost of a refused header; the lost
    attribution is only the visible half of it.
    """
    registry = await _registry_with(_trigger("deploy"))

    with caplog.at_level(logging.DEBUG):
        result, owner = await _fire_capturing_owner(registry, header, monkeypatch)

    assert result["dispatched"] is True
    assert owner is None
    assert len(_guard_refusals(caplog)) == 1


async def test_manual_fire_without_the_header_is_owner_less_and_silent(monkeypatch, caplog):
    """An absent header is the ordinary owner-less fire — a cron tick, say."""
    registry = await _registry_with(_trigger("deploy"))

    with caplog.at_level(logging.DEBUG):
        result, owner = await _fire_capturing_owner(registry, None, monkeypatch)

    assert result["dispatched"] is True
    assert owner is None
    assert _guard_refusals(caplog) == []


async def test_manual_fire_takes_no_owner_argument():
    """The owner is read from the request, so a caller cannot choose one.

    An agent holding this tool must not be able to name the human its fire is
    credited to, which it could do if the owner were a tool parameter.
    """
    registry = await _registry_with(_trigger("deploy"))
    tools = await _get_tools(registry, DispatchPool(max_concurrent=1, max_queue_depth=1))

    parameters = inspect.signature(tools["manual_fire"]).parameters

    assert list(parameters) == ["name", "payload"]


async def test_manual_fire_over_the_mcp_transport_credits_the_header_owner(monkeypatch):
    """The owner survives a real request, not just a patched header mapping.

    Every row above supplies the header dict directly, so none of them would
    notice if ``fastmcp`` stopped surfacing request headers to a tool, or
    surfaced them under their sent spelling rather than lower-cased — either
    would make every fire from a terminal owner-less on the wire while the unit
    rows stayed green. This one drives the bearer-gated transport the panel
    proxy actually posts to.
    """
    from fastmcp import FastMCP

    from osprey.dispatch import DISPATCHER_MCP_PATH
    from osprey.dispatch.server import _DispatcherTokenVerifier

    token = "dispatcher-transport-secret"
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", token)
    seen: dict = {}

    async def spy_fire(_trigger, _payload, owner):
        seen["owner"] = owner
        return "dispatch-123"

    registry = await _registry_with(_trigger("deploy"))
    mcp = FastMCP("test-dispatcher", auth=_DispatcherTokenVerifier())
    register_tools(mcp, registry, DispatchPool(max_concurrent=1, max_queue_depth=1), spy_fire)
    # Stateless, JSON-bodied answers: one self-contained POST per call, with no
    # session to carry between requests. Nothing about the header read changes.
    app = mcp.http_app(path=DISPATCHER_MCP_PATH, stateless_http=True, json_response=True)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://dispatcher.invalid"
        ) as client:
            response = await client.post(
                DISPATCHER_MCP_PATH,
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                    # Sent in the canonical mixed case a proxy mints it in; the
                    # tool reads it lower-cased, which is the point of the row.
                    OWNER_HEADER: "alice",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "manual_fire", "arguments": {"name": "deploy"}},
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert "error" not in body, body
    assert seen["owner"] == "alice"
