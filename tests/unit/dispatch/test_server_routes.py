"""Route-level tests for the event dispatcher server.

These exercise the FastMCP app through a Starlette ``TestClient``. Using the
``with TestClient(app) as client:`` form runs the ASGI lifespan, which registers
triggers into the registry and starts the trigger sources — required for the
``/webhook`` route to dispatch.

Entry-point discovery for trigger sources lands in a later migration task, so the
``osprey.trigger_sources`` group is empty at runtime today. To make ``/webhook``
route-registration work here, ``osprey.dispatch.source_registry.entry_points`` is
monkeypatched to yield a fake entry point loading ``WebhookSource`` (the same
technique used in ``test_source_registry.py``).

``create_server()`` mutates a module-level FastMCP singleton (``server.mcp``) and
appends routes to it. In production it is called exactly once; in tests we call it
per case, so an autouse fixture snapshots and restores ``_additional_http_routes``
to keep each test's route set clean (otherwise a stale, never-started webhook route
from an earlier ``create_server()`` would shadow the live one and 503).
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from osprey.dispatch import server
from osprey.dispatch.sources.webhook import WebhookSource


class _FakeEntryPoint:
    """Minimal EntryPoint stand-in exposing .name and .load()."""

    def __init__(self, name: str, cls: type) -> None:
        self.name = name
        self._cls = cls

    def load(self) -> type:
        return self._cls


@pytest.fixture(autouse=True)
def _reset_mcp_routes():
    """Reset the shared FastMCP singleton's routes around each test.

    Keeps only the module-level baseline routes (/health, /dispatch/...) so each
    ``create_server()`` call starts from a clean slate.
    """
    baseline = list(server.mcp._additional_http_routes)
    yield
    server.mcp._additional_http_routes = baseline


@pytest.fixture
def triggers_yml(tmp_path):
    """Write a minimal triggers.yml with one webhook trigger."""
    path = tmp_path / "triggers.yml"
    path.write_text(
        "dispatcher:\n"
        "  dispatch_target: http://localhost:9999\n"
        "  max_concurrent_runs: 2\n"
        "  max_queue_depth: 10\n"
        "triggers:\n"
        "  - name: deploy\n"
        "    source: webhook\n"
        "    action:\n"
        "      prompt: do the thing\n"
        "      allowed_tools: []\n"
    )
    return path


@pytest.fixture
def app(triggers_yml, monkeypatch):
    """Build the dispatcher FastMCP ASGI app with the webhook source discoverable.

    Patches entry-point discovery so ``SourceRegistry.discover()`` finds
    ``WebhookSource`` (real entry points are registered in a later task).
    """
    monkeypatch.setenv("TRIGGERS_YML", str(triggers_yml))
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret")

    def fake_entry_points(*, group):
        assert group == "osprey.trigger_sources"
        return [_FakeEntryPoint("webhook", WebhookSource)]

    monkeypatch.setattr("osprey.dispatch.source_registry.entry_points", fake_entry_points)

    return server.create_server().http_app()


def test_health_returns_status_and_pool(app):
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["trigger_count"] == 1
    assert "pool" in body
    assert {"running", "max", "queued"} <= set(body["pool"])


def test_webhook_success_returns_202(app):
    with TestClient(app) as client:
        resp = client.post(
            "/webhook/deploy",
            json={"ref": "main"},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["dispatched"] is True
    assert isinstance(body["dispatch_id"], str) and body["dispatch_id"]


def test_webhook_wrong_token_returns_401(app):
    with TestClient(app) as client:
        resp = client.post(
            "/webhook/deploy",
            json={},
            headers={"Authorization": "Bearer wrong"},
        )
    assert resp.status_code == 401


def test_webhook_unknown_trigger_returns_404(app):
    with TestClient(app) as client:
        resp = client.post(
            "/webhook/nope",
            json={},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status_code == 404


def test_webhook_unconfigured_token_fails_closed_503(triggers_yml, monkeypatch):
    """With EVENT_DISPATCHER_TOKEN unset, the webhook fails closed (503)."""
    monkeypatch.setenv("TRIGGERS_YML", str(triggers_yml))
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)

    def fake_entry_points(*, group):
        return [_FakeEntryPoint("webhook", WebhookSource)]

    monkeypatch.setattr("osprey.dispatch.source_registry.entry_points", fake_entry_points)
    app = server.create_server().http_app()
    with TestClient(app) as client:
        resp = client.post("/webhook/deploy", json={}, headers={"Authorization": "Bearer "})
    assert resp.status_code == 503
    assert resp.json() == {"detail": "Server misconfigured"}


def test_create_server_fails_loud_on_malformed_triggers(tmp_path, monkeypatch):
    """A malformed triggers.yml must crash ``create_server()`` at boot, not start
    the dispatcher with silently-dropped triggers.

    ``load_triggers`` raises ``ValueError`` on a bad file, and ``create_server``
    only guards ``FileNotFoundError`` — so the ValueError must propagate. This is
    the container-boot contract: fail loud rather than come up degraded. (The
    per-field validation itself is covered in test_trigger_config.py; this pins
    that create_server does not swallow it.)
    """
    bad = tmp_path / "triggers.yml"
    # A trigger missing the required action.prompt -> ValueError from load_triggers.
    bad.write_text(
        "dispatcher:\n"
        "  dispatch_target: http://localhost:9999\n"
        "triggers:\n"
        "  - name: broken\n"
        "    source: webhook\n"
        "    action: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRIGGERS_YML", str(bad))

    with pytest.raises(ValueError, match="broken"):
        server.create_server()


def test_create_server_degrades_when_triggers_missing(tmp_path, monkeypatch):
    """A missing triggers.yml must NOT crash the dispatcher: it starts with zero
    triggers, health is OK, and every /webhook resolves to 404.

    This degraded-startup branch (``except FileNotFoundError`` -> empty trigger
    list, DISPATCH_TARGET-env dispatcher config) lets the dispatcher boot for a
    health check even before a triggers file is mounted.
    """
    monkeypatch.setenv("TRIGGERS_YML", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret")

    app = server.create_server().http_app()
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["trigger_count"] == 0

        resp = client.post(
            "/webhook/anything",
            json={},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status_code == 404


def test_webhook_disabled_returns_409(app):
    with TestClient(app) as client:
        # Disable the trigger via the status route first.
        put = client.put(
            "/trigger/deploy/status",
            json={"status": "disabled"},
            headers={"Authorization": "Bearer secret"},
        )
        assert put.status_code == 200
        assert put.json() == {"name": "deploy", "status": "disabled"}

        resp = client.post(
            "/webhook/deploy",
            json={},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status_code == 409


def test_dashboard_state_shape(app):
    """/dashboard/state returns the aggregate snapshot keys; runs is [] (no worker)."""
    with TestClient(app) as client:
        resp = client.get("/dashboard/state", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"pool", "triggers", "runs", "timeline", "server_time_iso"}
    # No worker is reachable, so fetch_worker_runs fails and is caught -> [].
    assert body["runs"] == []
    assert any(t["name"] == "deploy" for t in body["triggers"])


# ---------------------------------------------------------------------------
# Dashboard READ endpoints are now bearer-gated (they surface agent output).
# ---------------------------------------------------------------------------

_GATED_READ_ROUTES = [
    "/dashboard/triggers",
    "/dashboard/runs",
    "/dashboard/state",
    "/dashboard/stream/some-run-id",
    "/dispatch/some-dispatch-id",
]


@pytest.mark.parametrize("path", _GATED_READ_ROUTES)
def test_gated_read_route_requires_auth(app, path):
    """Each gated read route → 401 without a credential."""
    with TestClient(app) as client:
        resp = client.get(path)
    assert resp.status_code == 401


@pytest.mark.parametrize("path", ["/dashboard/triggers", "/dashboard/state"])
def test_gated_read_route_allows_valid_token(app, path):
    """A valid bearer token is accepted on the gated read routes."""
    with TestClient(app) as client:
        resp = client.get(path, headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200


def test_dashboard_html_shell_is_ungated(app):
    """The HTML shell stays open so the standalone #token= handoff can run."""
    with TestClient(app) as client:
        resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_stream_accepts_header_token(app, monkeypatch):
    """The stream route is header-gated; a valid bearer header reaches the SSE proxy.

    The worker is stubbed (none is reachable in tests); the point is that a valid
    Authorization header passes the auth gate (200), not 401.
    """

    async def _fake_stream(url, token, run_id):
        yield b"data: {}\n\n"

    monkeypatch.setattr(server, "proxy_worker_stream", _fake_stream)
    with TestClient(app) as client:
        resp = client.get(
            "/dashboard/stream/some-run-id", headers={"Authorization": "Bearer secret"}
        )
    assert resp.status_code == 200


def test_stream_rejects_query_token(app):
    """A ?token= query must NOT authenticate — the bearer is header-only (no URL leak)."""
    with TestClient(app) as client:
        resp = client.get("/dashboard/stream/some-run-id?token=secret")
    assert resp.status_code == 401


def test_get_dispatch_result_omits_dispatch_target(app):
    """The poll response must not echo the internal worker URL."""
    with TestClient(app) as client:
        # Unknown id is fine — we only assert the field never appears.
        resp = client.get("/dispatch/does-not-exist", headers={"Authorization": "Bearer secret"})
    assert "dispatch_target" not in resp.json()


def test_check_auth_routes_reject_unconfigured_token(triggers_yml, monkeypatch):
    """Auth-guarded routes fail closed (503) when the token is unset."""
    monkeypatch.setenv("TRIGGERS_YML", str(triggers_yml))
    monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)

    def fake_entry_points(*, group):
        return [_FakeEntryPoint("webhook", WebhookSource)]

    monkeypatch.setattr("osprey.dispatch.source_registry.entry_points", fake_entry_points)
    app = server.create_server().http_app()
    with TestClient(app) as client:
        resp = client.put(
            "/trigger/deploy/status",
            json={"status": "disabled"},
            headers={"Authorization": "Bearer anything"},
        )
    assert resp.status_code == 503
    assert resp.json() == {"detail": "Server misconfigured"}


def test_status_route_wrong_token_returns_401(app):
    with TestClient(app) as client:
        resp = client.put(
            "/trigger/deploy/status",
            json={"status": "disabled"},
            headers={"Authorization": "Bearer wrong"},
        )
    assert resp.status_code == 401


def test_status_route_bad_status_returns_400(app):
    with TestClient(app) as client:
        resp = client.put(
            "/trigger/deploy/status",
            json={"status": "bogus"},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status_code == 400


def test_retry_known_trigger_returns_202(app):
    with TestClient(app) as client:
        resp = client.post(
            "/retry/deploy",
            json={"payload": {"ref": "main"}},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["dispatched"] is True
    assert isinstance(body["dispatch_id"], str) and body["dispatch_id"]


def test_retry_unknown_trigger_returns_404(app):
    with TestClient(app) as client:
        resp = client.post(
            "/retry/nope",
            json={},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status_code == 404


def test_retry_wrong_token_returns_401(app):
    with TestClient(app) as client:
        resp = client.post(
            "/retry/deploy",
            json={},
            headers={"Authorization": "Bearer wrong"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_dispatch_with_policy_injects_payload_into_prompt(monkeypatch):
    """A non-empty event payload is folded into the dispatched prompt as JSON."""
    from osprey.dispatch.registry import TriggerRegistry
    from osprey.dispatch.trigger_config import TriggerConfig

    captured: dict = {}

    async def fake_dispatch(url, prompt, allowed_tools, token, timeout=30.0, **kwargs):
        captured["prompt"] = prompt
        return {"run_id": "r1", "status": "ok"}

    monkeypatch.setattr(server, "dispatch_to_worker", fake_dispatch)

    reg = TriggerRegistry()
    trig = TriggerConfig(name="t", source="webhook", action={"prompt": "base prompt"})
    await reg.register(trig)
    await server._dispatch_with_policy(trig, {"ref": "main", "n": 3}, reg, "http://w", "tok")

    assert "base prompt" in captured["prompt"]
    assert "Event payload (JSON):" in captured["prompt"]
    assert '"ref": "main"' in captured["prompt"]


@pytest.mark.asyncio
async def test_dispatch_with_policy_empty_payload_no_injection(monkeypatch):
    """An empty payload leaves the prompt untouched (no trailing payload section)."""
    from osprey.dispatch.registry import TriggerRegistry
    from osprey.dispatch.trigger_config import TriggerConfig

    captured: dict = {}

    async def fake_dispatch(url, prompt, allowed_tools, token, timeout=30.0, **kwargs):
        captured["prompt"] = prompt
        return {"run_id": "r1", "status": "ok"}

    monkeypatch.setattr(server, "dispatch_to_worker", fake_dispatch)

    reg = TriggerRegistry()
    trig = TriggerConfig(name="t", source="webhook", action={"prompt": "base prompt"})
    await reg.register(trig)
    await server._dispatch_with_policy(trig, {}, reg, "http://w", "tok")

    assert captured["prompt"] == "base prompt"


# ---------------------------------------------------------------------------
# Surface prompt / per-surface tool scope: read at the trigger and forwarded
# to the worker via dispatch_to_worker. See TriggerConfig.surface_prompt
# (Task 2.1) and DispatchRequest.surface_prompt/surface_tools (dispatch_api).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_with_policy_forwards_surface_prompt(monkeypatch):
    """A trigger with ``action.surface_prompt`` set forwards it to the worker."""
    from osprey.dispatch.registry import TriggerRegistry
    from osprey.dispatch.trigger_config import TriggerConfig

    captured: dict = {}

    async def fake_dispatch(url, prompt, allowed_tools, token, timeout=30.0, **kwargs):
        captured.update(kwargs)
        return {"run_id": "r1", "status": "ok"}

    monkeypatch.setattr(server, "dispatch_to_worker", fake_dispatch)

    reg = TriggerRegistry()
    trig = TriggerConfig(
        name="t",
        source="webhook",
        action={"prompt": "base prompt", "surface_prompt": "You are the deploy-bot."},
        surface_prompt="You are the deploy-bot.",
    )
    await reg.register(trig)
    await server._dispatch_with_policy(trig, {}, reg, "http://w", "tok")

    assert captured["surface_prompt"] == "You are the deploy-bot."


@pytest.mark.asyncio
async def test_dispatch_with_policy_forwards_surface_tools(monkeypatch):
    """A per-surface tool scope on the trigger's action is forwarded to the worker."""
    from osprey.dispatch.registry import TriggerRegistry
    from osprey.dispatch.trigger_config import TriggerConfig

    captured: dict = {}

    async def fake_dispatch(url, prompt, allowed_tools, token, timeout=30.0, **kwargs):
        captured.update(kwargs)
        return {"run_id": "r1", "status": "ok"}

    monkeypatch.setattr(server, "dispatch_to_worker", fake_dispatch)

    reg = TriggerRegistry()
    trig = TriggerConfig(
        name="t",
        source="webhook",
        action={
            "prompt": "base prompt",
            "allowed_tools": ["read_pv"],
            "surface_tools": ["read_pv", "mcp__osprey_workspace__list_files"],
        },
    )
    await reg.register(trig)
    await server._dispatch_with_policy(trig, {}, reg, "http://w", "tok")

    assert captured["surface_tools"] == ["read_pv", "mcp__osprey_workspace__list_files"]


@pytest.mark.asyncio
async def test_dispatch_with_policy_absent_surface_fields_forward_as_none(monkeypatch):
    """No ``surface`` config on the trigger -> the forwarded surface fields are None.

    Existing prompt/allowed_tools behavior must be identical to a pre-surface
    trigger; only the new, additive kwargs carry ``None``.
    """
    from osprey.dispatch.registry import TriggerRegistry
    from osprey.dispatch.trigger_config import TriggerConfig

    captured: dict = {}

    async def fake_dispatch(url, prompt, allowed_tools, token, timeout=30.0, **kwargs):
        captured["prompt"] = prompt
        captured["allowed_tools"] = allowed_tools
        captured.update(kwargs)
        return {"run_id": "r1", "status": "ok"}

    monkeypatch.setattr(server, "dispatch_to_worker", fake_dispatch)

    reg = TriggerRegistry()
    trig = TriggerConfig(
        name="t",
        source="webhook",
        action={"prompt": "base prompt", "allowed_tools": ["read_pv"]},
    )
    await reg.register(trig)
    await server._dispatch_with_policy(trig, {}, reg, "http://w", "tok")

    assert captured["prompt"] == "base prompt"
    assert captured["allowed_tools"] == ["read_pv"]
    assert captured["surface_prompt"] is None
    assert captured["surface_tools"] is None


@pytest.mark.asyncio
async def test_dispatch_with_policy_retries_with_backoff_on_dispatch_error(monkeypatch):
    """``on_error: retry`` re-dispatches up to ``max_retries`` with backoff between attempts.

    Automatic retry/backoff fires ONLY when the dispatcher cannot reach the
    worker (i.e. ``dispatch_to_worker`` raises). That is the path the retired
    ``retry-demo`` tutorial trigger could never actually exercise via a ``curl``
    against a healthy stack, so it is covered here as a deterministic,
    token-free check instead of as a (misleading) demo trigger.
    """
    from osprey.dispatch.registry import TriggerRegistry
    from osprey.dispatch.trigger_config import TriggerConfig
    from osprey.dispatch.worker_client import DispatchError

    attempts = {"dispatch": 0}

    async def always_fails(url, prompt, allowed_tools, token, timeout=30.0, **kwargs):
        attempts["dispatch"] += 1
        raise DispatchError("worker unreachable")

    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    # ``server`` calls ``await asyncio.sleep(...)`` against its module-level
    # ``asyncio`` import; patch that so the backoff is asserted without delay.
    monkeypatch.setattr(server, "dispatch_to_worker", always_fails)
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    reg = TriggerRegistry()
    trig = TriggerConfig(
        name="retry-trigger",
        source="webhook",
        action={"prompt": "do a thing"},
        on_error={"action": "retry", "max_retries": 2, "backoff_sec": 0.01},
    )
    await reg.register(trig)

    result = await server._dispatch_with_policy(trig, {}, reg, "http://w", "tok")

    # Retries exhausted -> dropped -> returns None.
    assert result is None
    # Initial attempt + max_retries retries == 3 dispatch calls.
    assert attempts["dispatch"] == 3
    # Backoff slept once before each of the 2 retries (never on the final drop).
    assert slept == [0.01, 0.01]
    # Every failed attempt records an error event in the trigger history.
    history = await reg.get_history("retry-trigger")
    error_events = [e for e in history if str(e["result"]).startswith("error:")]
    assert len(error_events) == 3


def test_sanitize_source_config_redacts_secret_keys():
    """Secret-like source_config keys are redacted before dashboard exposure."""
    cfg = {"interval_sec": 300, "signing_secret": "shhh", "API_KEY": "abc", "pv": "X:Y"}
    sanitized = server._sanitize_source_config(cfg)
    assert sanitized["interval_sec"] == 300
    assert sanitized["pv"] == "X:Y"
    assert sanitized["signing_secret"] == "***"
    assert sanitized["API_KEY"] == "***"  # case-insensitive match


@pytest.mark.asyncio
async def test_dispatch_with_policy_alert_records_and_returns_none(monkeypatch):
    """``on_error: alert`` logs + records the error and returns None (no retry)."""
    from osprey.dispatch.registry import TriggerRegistry
    from osprey.dispatch.trigger_config import TriggerConfig
    from osprey.dispatch.worker_client import DispatchError

    async def always_fails(url, prompt, allowed_tools, token, timeout=30.0, **kwargs):
        raise DispatchError("worker unreachable")

    monkeypatch.setattr(server, "dispatch_to_worker", always_fails)

    reg = TriggerRegistry()
    trig = TriggerConfig(
        name="alert-trigger",
        source="webhook",
        action={"prompt": "x"},
        on_error={"action": "alert", "max_retries": 0, "backoff_sec": 0.0},
    )
    await reg.register(trig)

    result = await server._dispatch_with_policy(trig, {}, reg, "http://w", "tok")
    assert result is None
    history = await reg.get_history("alert-trigger")
    assert any(str(e["result"]).startswith("error:") for e in history)


@pytest.mark.asyncio
async def test_dispatch_with_policy_auth_error_flows_through_policy(monkeypatch):
    """An AuthError is handled by the policy (recorded + dropped), not raised."""
    from osprey.dispatch.registry import TriggerRegistry
    from osprey.dispatch.trigger_config import TriggerConfig
    from osprey.dispatch.worker_client import AuthError

    async def auth_fails(url, prompt, allowed_tools, token, timeout=30.0, **kwargs):
        raise AuthError("Unauthorized (401)")

    monkeypatch.setattr(server, "dispatch_to_worker", auth_fails)

    reg = TriggerRegistry()
    trig = TriggerConfig(name="auth-trigger", source="webhook", action={"prompt": "x"})
    await reg.register(trig)

    result = await server._dispatch_with_policy(trig, {}, reg, "http://w", "tok")
    assert result is None
    history = await reg.get_history("auth-trigger")
    assert any("Unauthorized" in str(e["result"]) for e in history)


@pytest.mark.asyncio
async def test_dispatch_with_policy_generic_exception_propagates(monkeypatch):
    """A genuine bug (non-Dispatch/Auth) is NOT swallowed by the policy."""
    from osprey.dispatch.registry import TriggerRegistry
    from osprey.dispatch.trigger_config import TriggerConfig

    async def boom(url, prompt, allowed_tools, token, timeout=30.0, **kwargs):
        raise RuntimeError("genuine bug")

    monkeypatch.setattr(server, "dispatch_to_worker", boom)

    reg = TriggerRegistry()
    trig = TriggerConfig(name="bug-trigger", source="webhook", action={"prompt": "x"})
    await reg.register(trig)

    with pytest.raises(RuntimeError, match="genuine bug"):
        await server._dispatch_with_policy(trig, {}, reg, "http://w", "tok")


def test_dashboard_state_surfaces_worker_error(app, monkeypatch):
    """When the worker is unreachable, /dashboard/state carries a worker_error marker."""
    from osprey.dispatch.worker_client import DispatchError

    async def fetch_fails(url, token, timeout=10.0):
        raise DispatchError("Connection error")

    monkeypatch.setattr(server, "fetch_worker_runs", fetch_fails)
    with TestClient(app) as client:
        resp = client.get("/dashboard/state", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["worker_error"] and "Connection error" in body["worker_error"]
    assert body["runs"] == []


def test_dashboard_runs_worker_down_returns_502(app, monkeypatch):
    from osprey.dispatch.worker_client import DispatchError

    async def fetch_fails(url, token, timeout=10.0):
        raise DispatchError("Connection error")

    monkeypatch.setattr(server, "fetch_worker_runs", fetch_fails)
    with TestClient(app) as client:
        resp = client.get("/dashboard/runs", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 502
    assert "worker_error" in resp.json()


def test_retry_non_dict_payload_returns_400(app):
    with TestClient(app) as client:
        resp = client.post(
            "/retry/deploy",
            json={"payload": "not-an-object"},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status_code == 400
    assert "payload must be an object" in resp.json()["detail"]


def test_retry_queue_full_returns_429(app, monkeypatch):
    """When the pool is saturated, /retry surfaces 429 from QueueFullError."""
    from osprey.dispatch.pool import QueueFullError

    async def _full(*a, **k):
        raise QueueFullError("Queue depth 10 exceeded")

    # Patch the pool's submit on the live server instance.
    server.mcp._dispatcher_pool.submit = _full  # type: ignore[attr-defined]
    with TestClient(app) as client:
        resp = client.post(
            "/retry/deploy",
            json={"payload": {}},
            headers={"Authorization": "Bearer secret"},
        )
    assert resp.status_code == 429


def test_dashboard_cancel_requires_auth(app):
    with TestClient(app) as client:
        resp = client.post("/dashboard/cancel/some-run-id")
    assert resp.status_code == 401


def test_dashboard_cancel_proxies_to_worker(app, monkeypatch):
    """A valid cancel proxies to the worker client and returns its result."""

    async def fake_cancel(url, token, run_id):
        return {"run_id": run_id, "cancelled": True}

    monkeypatch.setattr(server, "cancel_worker_run", fake_cancel)
    with TestClient(app) as client:
        resp = client.post("/dashboard/cancel/run-1", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    assert resp.json()["cancelled"] is True


def test_dashboard_cancel_worker_auth_failure_returns_502(app, monkeypatch):
    from osprey.dispatch.worker_client import AuthError

    async def fake_cancel(url, token, run_id):
        raise AuthError("nope")

    monkeypatch.setattr(server, "cancel_worker_run", fake_cancel)
    with TestClient(app) as client:
        resp = client.post("/dashboard/cancel/run-1", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 502
