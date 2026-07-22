"""Dispatcher-side input_files passthrough hygiene, fatal-4xx error_code
surfacing, /health capability advertisement, and webhook body-limit.

The direct-call tests exercise ``_dispatch_with_policy`` with a stubbed
``dispatch_to_worker`` and an AsyncMock registry; the route tests drive the real
FastMCP app through a Starlette ``TestClient`` (same harness as
``test_server_routes``)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from osprey.dispatch import server
from osprey.dispatch.sources.webhook import WebhookSource
from osprey.dispatch.trigger_config import TriggerConfig
from osprey.dispatch.worker_client import FatalDispatchError

_FILES = [
    {"filename": "plot.png", "mime": "image/png", "content_b64": "QUJD" * 100, "ingest": True},
]


def _trigger() -> TriggerConfig:
    return TriggerConfig(
        name="deploy", source="webhook", action={"prompt": "do it", "allowed_tools": []}
    )


# ---------------------------------------------------------------------------
# _dispatch_with_policy: input_files passthrough hygiene
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_passthrough_pops_input_files_before_fold_and_record():
    registry = AsyncMock()
    captured: dict[str, Any] = {}

    async def _fake_dispatch(**kwargs):
        captured.update(kwargs)
        return {"run_id": "r1", "status": "accepted"}

    payload = {"question": "hello", "input_files": _FILES}
    with patch.object(server, "dispatch_to_worker", _fake_dispatch):
        result = await server._dispatch_with_policy(
            _trigger(), payload, registry, "http://worker", "tok"
        )

    assert result == {"run_id": "r1", "status": "accepted"}
    # Forwarded to the worker...
    assert captured["input_files"] == _FILES
    # ...but stripped from the payload, so it never lands in the prompt fold...
    assert "input_files" not in payload
    assert "content_b64" not in captured["prompt"]
    # ...nor in the recorded history event (record_event got the popped payload).
    recorded_payload = registry.record_event.await_args.args[1]
    assert "input_files" not in recorded_payload


@pytest.mark.asyncio
async def test_dispatch_passthrough_no_input_files_forwards_none():
    registry = AsyncMock()
    captured: dict[str, Any] = {}

    async def _fake_dispatch(**kwargs):
        captured.update(kwargs)
        return {"run_id": "r1", "status": "accepted"}

    with patch.object(server, "dispatch_to_worker", _fake_dispatch):
        await server._dispatch_with_policy(
            _trigger(), {"question": "hi"}, registry, "http://worker", "tok"
        )
    assert captured["input_files"] is None


# ---------------------------------------------------------------------------
# _dispatch_with_policy: fatal 4xx -> sentinel error dict carrying error_code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_error_code_fatal_4xx_returns_sentinel_not_none():
    registry = AsyncMock()

    async def _fake_dispatch(**kwargs):
        raise FatalDispatchError("HTTP 400 from worker", error_code="input_files_cap_exceeded")

    with patch.object(server, "dispatch_to_worker", _fake_dispatch):
        result = await server._dispatch_with_policy(
            _trigger(), {"input_files": _FILES}, registry, "http://worker", "tok"
        )

    # Never drop-to-None for a 4xx: a sentinel error dict carries the code.
    assert result is not None
    assert result["status"] == "error"
    assert result["error_code"] == "input_files_cap_exceeded"
    # It is NOT routed through the retry/drop policy (recorded as a rejection).
    recorded_status = registry.record_event.await_args.args[2]
    assert recorded_status.startswith("rejected")


@pytest.mark.asyncio
async def test_dispatch_error_code_generic_4xx_returns_sentinel_none_code():
    registry = AsyncMock()

    async def _fake_dispatch(**kwargs):
        raise FatalDispatchError("HTTP 403 from worker", error_code=None)

    with patch.object(server, "dispatch_to_worker", _fake_dispatch):
        result = await server._dispatch_with_policy(
            _trigger(), {}, registry, "http://worker", "tok"
        )
    assert result["status"] == "error"
    assert result["error_code"] is None


@pytest.mark.asyncio
async def test_dispatch_passthrough_retry_rethreads_input_files():
    """A retryable failure then success must re-forward the popped input_files."""
    registry = AsyncMock()
    calls: list = []

    async def _fake_dispatch(**kwargs):
        calls.append(kwargs["input_files"])
        if len(calls) == 1:
            from osprey.dispatch.worker_client import DispatchError

            raise DispatchError("transient")
        return {"run_id": "r2", "status": "accepted"}

    trigger = TriggerConfig(
        name="deploy",
        source="webhook",
        action={"prompt": "p", "allowed_tools": []},
        on_error={"action": "retry", "max_retries": 1, "backoff_sec": 0.0},
    )
    with patch.object(server, "dispatch_to_worker", _fake_dispatch):
        result = await server._dispatch_with_policy(
            trigger, {"input_files": _FILES}, registry, "http://worker", "tok"
        )
    assert result == {"run_id": "r2", "status": "accepted"}
    # Both the initial attempt and the retry carried the same files (payload was
    # mutated by the pop, so only the threaded value keeps them alive).
    assert calls == [_FILES, _FILES]


# ---------------------------------------------------------------------------
# Route-level: /health capability, get_dispatch_result reshape, webhook body_limit
# ---------------------------------------------------------------------------


class _FakeEntryPoint:
    def __init__(self, name: str, cls: type) -> None:
        self.name = name
        self._cls = cls

    def load(self) -> type:
        return self._cls


@pytest.fixture(autouse=True)
def _reset_mcp_routes():
    baseline = list(server.mcp._additional_http_routes)
    yield
    server.mcp._additional_http_routes = baseline


@pytest.fixture
def app(tmp_path, monkeypatch):
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
    monkeypatch.setenv("TRIGGERS_YML", str(path))
    monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "secret")

    def fake_entry_points(*, group):
        assert group == "osprey.trigger_sources"
        return [_FakeEntryPoint("webhook", WebhookSource)]

    monkeypatch.setattr("osprey.dispatch.source_registry.entry_points", fake_entry_points)
    return server.create_server().http_app()


def test_health_capability_advertised(app):
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["capabilities"] == ["input_files"]
    assert isinstance(body["boot_nonce"], (int, float))


def test_get_dispatch_result_error_code_surfaced_top_level(app):
    """A pool result wrapping a fatal sentinel surfaces top-level status+error_code."""
    with TestClient(app) as client:
        pool = server.mcp._dispatcher_pool
        pool._results["did-1"] = {
            "status": "completed",
            "result": {
                "status": "error",
                "error_code": "input_files_invalid",
                "error": "HTTP 400 from worker",
            },
            "trigger_name": "deploy",
        }
        resp = client.get("/dispatch/did-1", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["error_code"] == "input_files_invalid"


def test_get_dispatch_result_normal_completion_unchanged(app):
    """A genuine worker completion (status 'accepted', has run_id) is passed through."""
    with TestClient(app) as client:
        pool = server.mcp._dispatcher_pool
        pool._results["did-2"] = {
            "status": "completed",
            "result": {"status": "accepted", "run_id": "run-xyz"},
            "trigger_name": "deploy",
        }
        resp = client.get("/dispatch/did-2", headers={"Authorization": "Bearer secret"})
    body = resp.json()
    assert body["status"] == "completed"
    assert body["result"]["run_id"] == "run-xyz"
    assert "error_code" not in body


def test_webhook_body_limit_rejects_oversize_413(app):
    # A genuinely oversize body so httpx sets a real >32MB Content-Length; the
    # route must reject with 413 from the header before parsing the body.
    oversize = ('{"question":"' + "a" * (33 * 1024 * 1024) + '"}').encode()
    with TestClient(app) as client:
        resp = client.post(
            "/webhook/deploy",
            headers={"Authorization": "Bearer secret"},
            content=oversize,
        )
    assert resp.status_code == 413


def test_webhook_body_limit_allows_normal_body(app):
    with TestClient(app) as client:
        resp = client.post(
            "/webhook/deploy",
            headers={"Authorization": "Bearer secret"},
            content=json.dumps({"question": "hi"}),
        )
    # Small body sails past the limit (202 dispatched, or 409 disabled — never 413).
    assert resp.status_code != 413
