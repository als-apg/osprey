"""Unit tests for the dispatch-worker FastAPI app.

Exercises the HTTP surface of ``osprey.mcp_server.dispatch_worker.dispatch_api``
with a sync ``TestClient`` (lifespan runs inside the ``with`` block). The real
``sdk_runner.run_dispatch`` calls the Claude Agent SDK, so it is monkeypatched
with a fast canned coroutine — the background task then completes without the
SDK and tests never assert a timing-dependent terminal status.
"""

from __future__ import annotations

import os
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from osprey.mcp_server.dispatch_worker import dispatch_api

_TOKEN = "test-secret-token"

_CANNED_RESULT: dict[str, Any] = {
    "status": "completed",
    "text_output": "ok",
    "tool_calls": [],
    "error": None,
    "duration_sec": 0.01,
    "cost_usd": 0.0,
    "num_turns": 1,
}


@pytest.fixture
def client(monkeypatch):
    """A TestClient with auth configured and run_dispatch stubbed out.

    Resets the module-level in-memory stores so tests do not leak state into
    each other, and patches ``sdk_runner.run_dispatch`` with a fast async stub.
    """
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", _TOKEN)

    # Isolate global state between tests.
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})

    async def _fake_run_dispatch(
        *, prompt, allowed_tools, max_turns, event_queue, denied_tools=(), run_id=None
    ):
        if event_queue is not None:
            await event_queue.put({"type": "done"})
        return dict(_CANNED_RESULT)

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _fake_run_dispatch)
    # Avoid touching disk during the background task.
    monkeypatch.setattr(dispatch_api, "_persist_run", lambda run_id, run: None)

    with TestClient(dispatch_api.app) as c:
        yield c


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {_TOKEN}"}


def _wait_for_terminal(client: TestClient, run_id: str, timeout: float = 5.0) -> dict:
    """Poll a run until it leaves the ``pending`` state (or time out)."""
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        resp = client.get(f"/dispatch/{run_id}", headers=_auth())
        assert resp.status_code == 200
        last = resp.json()
        if last.get("status") != "pending":
            return last
        time.sleep(0.02)
    return last


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_no_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    for key in ("pending_runs", "completed_runs", "error_runs", "total_runs"):
        assert key in body
        assert isinstance(body[key], int)


# ---------------------------------------------------------------------------
# POST /dispatch
# ---------------------------------------------------------------------------


def test_dispatch_accepts_benign_tools(client):
    resp = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers=_auth(),
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert isinstance(body["run_id"], str)
    assert body["run_id"]


def test_dispatch_rejects_denied_tool(client):
    resp = client.post(
        "/dispatch",
        json={"prompt": "fetch", "allowed_tools": ["Read", "WebFetch"]},
        headers=_auth(),
    )
    assert resp.status_code == 403
    assert "WebFetch" in resp.json()["detail"]


def test_dispatch_rejects_wildcard_denied_tool(client):
    """Denylist '*'-suffix entries block by prefix (e.g. all playwright tools)."""
    tool = "mcp__plugin_playwright_playwright__browser_click"
    resp = client.post(
        "/dispatch",
        json={"prompt": "click", "allowed_tools": [tool]},
        headers=_auth(),
    )
    assert resp.status_code == 403
    assert tool in resp.json()["detail"]


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("WebFetch", True),
        ("WebSearch", True),
        ("Bash", True),
        ("BashOutput", True),
        ("KillShell", True),
        ("KillBash", True),
        ("mcp__plugin_playwright_playwright__browser_click", True),
        ("mcp__plugin_playwright_playwright__", True),  # bare prefix still matches
        ("Read", False),
        ("Write", False),
        ("mcp__osprey_workspace__write_channel", False),
        ("WebFetcher", False),  # not an exact match, not a wildcard entry
        ("", False),
    ],
)
def test_is_denied_matrix(tool, expected):
    """The server-side denylist matcher: exact entries + '*'-suffix prefixes."""
    assert dispatch_api._is_denied(tool) is expected


def test_dispatch_denied_tool_schedules_no_run(client):
    """A denied tool 403s AND never creates a run (nothing enters the store)."""
    monkey_before = len(dispatch_api._runs)
    resp = client.post(
        "/dispatch",
        json={"prompt": "shell out", "allowed_tools": ["Read", "Bash"]},
        headers=_auth(),
    )
    assert resp.status_code == 403
    assert "Bash" in resp.json()["detail"]
    # No run was created and no task was scheduled.
    assert len(dispatch_api._runs) == monkey_before
    assert dispatch_api._tasks == {}


def test_dispatch_wrong_token(client):
    resp = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers={"Authorization": "Bearer wrong"},
    )
    assert resp.status_code == 401


def test_dispatch_no_auth_header(client):
    # HTTPBearer auto-error rejects a missing Authorization header. The exact
    # code depends on the FastAPI version (older: 403, current: 401); accept
    # either so the test tracks behavior without pinning a version-specific code.
    resp = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
    )
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# GET /dispatch/{run_id}
# ---------------------------------------------------------------------------


def test_get_dispatch_unknown_id_404(client):
    resp = client.get("/dispatch/does-not-exist", headers=_auth())
    assert resp.status_code == 404


def test_get_dispatch_returns_status(client):
    post = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers=_auth(),
    )
    run_id = post.json()["run_id"]

    result = _wait_for_terminal(client, run_id)
    assert result.get("status") in ("pending", "completed", "error")
    # With the fast stub the run should reach completion.
    assert result["status"] == "completed"
    assert result["text_output"] == "ok"


# ---------------------------------------------------------------------------
# DELETE /dispatch/{run_id}
# ---------------------------------------------------------------------------


def test_cancel_unknown_id_404(client):
    resp = client.delete("/dispatch/does-not-exist", headers=_auth())
    assert resp.status_code == 404


def test_cancel_finished_run(client):
    post = client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers=_auth(),
    )
    run_id = post.json()["run_id"]
    # Let the stub finish so the run is no longer pending.
    _wait_for_terminal(client, run_id)

    resp = client.delete(f"/dispatch/{run_id}", headers=_auth())
    assert resp.status_code == 200
    body = resp.json()
    assert "cancelled" in body
    # A finished run cannot be cancelled.
    assert body["cancelled"] is False


# ---------------------------------------------------------------------------
# GET /dashboard/runs
# ---------------------------------------------------------------------------


def test_dashboard_runs_requires_auth(client):
    # The runs feed leaks full text_output/error, so it is token-gated like the
    # other worker endpoints. HTTPBearer auto-error rejects a missing header
    # (401 current FastAPI, 403 older) — accept either.
    resp = client.get("/dashboard/runs")
    assert resp.status_code in (401, 403)


def test_dashboard_runs_with_auth(client):
    # Seed one run via the API.
    client.post(
        "/dispatch",
        json={"prompt": "do it", "allowed_tools": ["Read"]},
        headers=_auth(),
    )
    resp = client.get("/dashboard/runs", headers=_auth())
    assert resp.status_code == 200
    runs = resp.json()
    assert isinstance(runs, list)
    assert len(runs) >= 1
    assert "run_id" in runs[0]
    assert "status" in runs[0]


# ---------------------------------------------------------------------------
# Startup Claude-artifact provisioning
# ---------------------------------------------------------------------------
#
# The deployed worker mounts only config.yml into the project dir, so it must
# regenerate .mcp.json / .claude/ at startup; otherwise dispatched agents have
# no project MCP servers (e.g. osprey_workspace) and no safety hooks.


def test_provision_skips_when_no_config(tmp_path, monkeypatch):
    """No config.yml in the project dir → provisioning is a no-op (no regen call)."""
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))
    called = {"n": 0}

    class _FakeTM:
        def regenerate_claude_code(self, *a, **k):
            called["n"] += 1
            return {"changed": []}

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", _FakeTM, raising=True)
    dispatch_api._provision_claude_artifacts_once()  # must not raise
    assert called["n"] == 0, "regen must not run without a config.yml"


def test_provision_regenerates_when_mcp_json_absent(tmp_path, monkeypatch):
    """config.yml present but .mcp.json absent (the container case) → regen runs."""
    (tmp_path / "config.yml").write_text("project_name: t\n", encoding="utf-8")
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))
    seen = {}

    class _FakeTM:
        def regenerate_claude_code(self, project_dir, *a, **k):
            seen["project_dir"] = project_dir
            return {"changed": [".mcp.json", "CLAUDE.md"]}

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", _FakeTM, raising=True)
    dispatch_api._provision_claude_artifacts_once()
    assert str(seen.get("project_dir")) == str(tmp_path)


def test_provision_skips_when_already_provisioned(tmp_path, monkeypatch):
    """A project that already has .mcp.json must NOT be regenerated.

    The subprocess path and non-container deploys run in the real project dir,
    which ships container-correct artifacts and may carry user customizations
    (e.g. via ``osprey eject``). Regenerating would clobber them.
    """
    (tmp_path / "config.yml").write_text("project_name: t\n", encoding="utf-8")
    (tmp_path / ".mcp.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))
    called = {"n": 0}

    class _FakeTM:
        def regenerate_claude_code(self, *a, **k):
            called["n"] += 1
            return {"changed": []}

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", _FakeTM, raising=True)
    dispatch_api._provision_claude_artifacts_once()
    assert called["n"] == 0, "must not regenerate when .mcp.json already exists"


def test_provision_swallows_regen_errors(tmp_path, monkeypatch):
    """A regen failure must not crash the worker (it still serves no-tool triggers)."""
    (tmp_path / "config.yml").write_text("project_name: t\n", encoding="utf-8")
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))

    class _FakeTM:
        def regenerate_claude_code(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr("osprey.cli.templates.manager.TemplateManager", _FakeTM, raising=True)
    dispatch_api._provision_claude_artifacts_once()  # must not raise


# ---------------------------------------------------------------------------
# Stale-run sweep cancels orphaned tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_marks_stale_run_error_and_cancels_task(monkeypatch):
    """A run pending past the cutoff is marked error AND its task is cancelled."""
    import asyncio

    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})

    async def _long_runner():
        await asyncio.sleep(30)

    task = asyncio.create_task(_long_runner())
    run_id = "stale-1"
    stale_cutoff = dispatch_api.DISPATCH_TIMEOUT_SEC + 30
    dispatch_api._runs[run_id] = {
        "status": "pending",
        "created_at": time.time() - (stale_cutoff + 60),
    }
    dispatch_api._tasks[run_id] = task

    dispatch_api._sweep_stale_runs()

    assert dispatch_api._runs[run_id]["status"] == "error"
    assert "Timed out" in dispatch_api._runs[run_id]["error"]
    # Let the cancellation propagate.
    await asyncio.sleep(0)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_sweep_leaves_fresh_pending_run_alone(monkeypatch):
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})

    dispatch_api._runs["fresh"] = {"status": "pending", "created_at": time.time()}
    dispatch_api._sweep_stale_runs()
    assert dispatch_api._runs["fresh"]["status"] == "pending"


# ---------------------------------------------------------------------------
# Worker auth on the remaining gated endpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/dispatch/some-id"),
        ("get", "/dispatch/some-id/stream"),
        ("delete", "/dispatch/some-id"),
        ("get", "/dashboard/runs"),
    ],
)
def test_gated_endpoint_missing_auth_rejected(client, method, path):
    """Missing Authorization header is rejected (401 current FastAPI, 403 older)."""
    resp = getattr(client, method)(path)
    assert resp.status_code in (401, 403)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/dispatch/some-id"),
        ("delete", "/dispatch/some-id"),
        ("get", "/dashboard/runs"),
    ],
)
def test_gated_endpoint_wrong_token_401(client, method, path):
    resp = getattr(client, method)(path, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_unconfigured_worker_token_fails_closed_500(monkeypatch):
    """With DISPATCH_WORKER_TOKEN unset, the worker fails closed (500) on a token check."""
    monkeypatch.delenv("DISPATCH_WORKER_TOKEN", raising=False)
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    with TestClient(dispatch_api.app) as c:
        resp = c.get("/dispatch/some-id", headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 500
    assert "DISPATCH_WORKER_TOKEN" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# _inject_provider_env_once: ${VAR} expansion + proxy-start for the worker (#307)
# ---------------------------------------------------------------------------

_ARGO_CONFIG = """\
api:
  providers:
    argo:
      base_url: ${ARGO_PROD_URL}
claude_code:
  provider: argo
"""

_CBORG_CONFIG = """\
claude_code:
  provider: cborg
"""


def _isolated_environ(monkeypatch, tmp_path, **extra):
    """Swap os.environ for a throwaway dict so the function's mutations don't leak."""
    fake = dict(os.environ)
    fake["OSPREY_PROJECT_DIR"] = str(tmp_path)
    fake.pop("ANTHROPIC_BASE_URL", None)
    fake.update(extra)
    monkeypatch.setattr(os, "environ", fake)
    return fake


def test_inject_provider_env_expands_and_starts_proxy(tmp_path, monkeypatch):
    """Custom provider: ${VAR} base_url is expanded, proxy started, base URL repointed."""
    (tmp_path / "config.yml").write_text(_ARGO_CONFIG)
    (tmp_path / ".env").write_text("ARGO_PROD_URL=https://argo.example/v1\nARGO_API_KEY=sk-argo\n")
    fake = _isolated_environ(monkeypatch, tmp_path)
    proxy = MagicMock(return_value=7777)
    monkeypatch.setattr("osprey.infrastructure.proxy.lifecycle.start_proxy", proxy)

    dispatch_api._inject_provider_env_once()

    proxy.assert_called_once()
    upstream, api_key = proxy.call_args[0]
    assert upstream == "https://argo.example/v1"
    assert api_key == "sk-argo"
    assert fake["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:7777"


def test_inject_provider_env_no_proxy_for_native(tmp_path, monkeypatch):
    """Native provider (cborg): env injected but no translation proxy started."""
    (tmp_path / "config.yml").write_text(_CBORG_CONFIG)
    _isolated_environ(monkeypatch, tmp_path, CBORG_API_KEY="sk-cborg")
    proxy = MagicMock(return_value=1)
    monkeypatch.setattr("osprey.infrastructure.proxy.lifecycle.start_proxy", proxy)

    dispatch_api._inject_provider_env_once()

    proxy.assert_not_called()


def test_inject_provider_env_refuses_on_managed_policy_conflict(tmp_path, monkeypatch):
    """A managed-policy env override aborts worker startup rather than starting
    the agent against a backend the project did not configure (#355).

    The refusal must propagate — it is raised before the broad ``except`` that
    otherwise swallows provider-injection errors."""
    (tmp_path / "config.yml").write_text(_CBORG_CONFIG)
    _isolated_environ(monkeypatch, tmp_path, CBORG_API_KEY="sk-cborg")
    monkeypatch.setattr(
        "osprey.cli.claude_code_resolver.detect_managed_policy_conflicts",
        lambda: {"ANTHROPIC_BASE_URL": ("https://evil.example", "/etc/.../managed-settings.json")},
    )

    with pytest.raises(RuntimeError, match="Refusing to start the dispatch worker"):
        dispatch_api._inject_provider_env_once()
