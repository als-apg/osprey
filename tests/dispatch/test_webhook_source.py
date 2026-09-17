"""Unit tests for the webhook trigger source."""

from __future__ import annotations

import pytest

from osprey.dispatch.pool import QueueFullError
from osprey.dispatch.sources.webhook import WebhookSource
from osprey.dispatch.trigger_config import TriggerConfig


def _make_trigger(name: str = "deploy") -> TriggerConfig:
    return TriggerConfig(
        name=name,
        source="webhook",
        action={"prompt": "do the thing", "allowed_tools": []},
    )


class _RecordingCallback:
    """Fake fire callback that records calls and returns a configured result."""

    def __init__(self, result: str | None = None, raises: Exception | None = None) -> None:
        self._result = result
        self._raises = raises
        self.calls: list[tuple[TriggerConfig, dict]] = []

    async def __call__(self, trigger: TriggerConfig, payload: dict) -> str | None:
        self.calls.append((trigger, payload))
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.fixture
def started_source(monkeypatch):
    """A WebhookSource with one trigger and a fire callback installed."""
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret")
    source = WebhookSource()
    trigger = _make_trigger("deploy")
    source._triggers = {trigger.name: trigger}
    callback = _RecordingCallback(result="dispatch-123")
    source._fire_callback = callback
    return source, trigger, callback


@pytest.mark.asyncio
async def test_handle_unauthorized_wrong_token(started_source, monkeypatch):
    source, _, callback = started_source
    body, status = await source._handle("deploy", "Bearer wrong", {})
    assert status == 401
    assert body == {"detail": "Unauthorized"}
    assert callback.calls == []


@pytest.mark.asyncio
async def test_handle_unauthorized_missing_header(started_source):
    source, _, callback = started_source
    body, status = await source._handle("deploy", "", {})
    assert status == 401
    assert callback.calls == []


@pytest.mark.asyncio
async def test_handle_unconfigured_token_fails_closed(started_source, monkeypatch):
    """With no EVENT_DISPATCHER_TOKEN set, auth fails closed (503), never open."""
    source, _, callback = started_source
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)
    # An empty-token bearer must NOT slip through when the server is unconfigured.
    body, status = await source._handle("deploy", "Bearer ", {})
    assert status == 503
    assert body == {"detail": "Server misconfigured"}
    assert callback.calls == []


@pytest.mark.asyncio
async def test_handle_not_started_returns_503(monkeypatch):
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret")
    source = WebhookSource()  # never started: _fire_callback is None
    body, status = await source._handle("deploy", "Bearer secret", {})
    assert status == 503
    assert body == {"detail": "Source not started"}


@pytest.mark.asyncio
async def test_handle_unknown_trigger_returns_404(started_source):
    source, _, callback = started_source
    body, status = await source._handle("nope", "Bearer secret", {})
    assert status == 404
    assert body == {"detail": "Trigger 'nope' not found"}
    assert callback.calls == []


@pytest.mark.asyncio
async def test_handle_disabled_returns_409(monkeypatch):
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret")
    source = WebhookSource()
    trigger = _make_trigger("deploy")
    source._triggers = {trigger.name: trigger}
    callback = _RecordingCallback(result=None)  # None => disabled
    source._fire_callback = callback

    body, status = await source._handle("deploy", "Bearer secret", {"x": 1})
    assert status == 409
    assert body == {"detail": "Trigger 'deploy' is disabled"}
    assert callback.calls == [(trigger, {"x": 1})]


@pytest.mark.asyncio
async def test_handle_queue_full_returns_429(monkeypatch):
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret")
    source = WebhookSource()
    trigger = _make_trigger("deploy")
    source._triggers = {trigger.name: trigger}
    callback = _RecordingCallback(raises=QueueFullError("queue depth 100 exceeded"))
    source._fire_callback = callback

    body, status = await source._handle("deploy", "Bearer secret", {})
    assert status == 429
    assert body == {"detail": "queue depth 100 exceeded"}


@pytest.mark.asyncio
async def test_handle_success_returns_202(started_source):
    source, trigger, callback = started_source
    payload = {"ref": "main", "sha": "abc"}
    body, status = await source._handle("deploy", "Bearer secret", payload)
    assert status == 202
    assert body == {"dispatched": True, "dispatch_id": "dispatch-123"}
    # fire_callback was awaited with the right (trigger, payload)
    assert callback.calls == [(trigger, payload)]


@pytest.mark.asyncio
async def test_stop_clears_state_so_handler_rejects(started_source):
    source, _, _ = started_source
    await source.stop()
    assert source._triggers == {}
    assert source._fire_callback is None
    # After stop, even a valid token + known name would now 503 (callback gone).
    body, status = await source._handle("deploy", "Bearer secret", {})
    assert status == 503


@pytest.mark.asyncio
async def test_register_routes_registers_webhook_route(monkeypatch):
    """register_routes() registers /webhook/{trigger_name} on a real FastMCP app."""
    from fastmcp import FastMCP

    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret")
    app = FastMCP("test-dispatcher")
    source = WebhookSource()
    trigger = _make_trigger("deploy")
    callback = _RecordingCallback(result="d-1")

    # Factory phase: route must be registered before the app is built.
    source.register_routes(app)

    # Inspect the underlying Starlette app's routes for our path template.
    starlette_app = app.http_app()
    paths = [getattr(r, "path", None) for r in starlette_app.routes]
    assert "/webhook/{trigger_name}" in paths, f"route not found; saw {paths}"

    # Lifespan phase: after start(), _handle dispatches successfully.
    await source.start([trigger], callback)
    body, status = await source._handle("deploy", "Bearer secret", {"ref": "main"})
    assert status == 202
    assert body == {"dispatched": True, "dispatch_id": "d-1"}


@pytest.mark.asyncio
async def test_handle_503_between_register_routes_and_start(monkeypatch):
    """After register_routes() but BEFORE start(), _handle returns 503 not-started."""
    from fastmcp import FastMCP

    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret")
    app = FastMCP("test-dispatcher")
    source = WebhookSource()
    source.register_routes(app)  # factory phase only; start() not yet called

    body, status = await source._handle("deploy", "Bearer secret", {})
    assert status == 503
    assert body == {"detail": "Source not started"}
