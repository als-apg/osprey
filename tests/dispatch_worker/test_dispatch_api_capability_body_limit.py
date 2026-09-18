"""Worker /health capability advertisement and the 32MB request body-limit guard."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from osprey.mcp_server.dispatch_worker import dispatch_api

_TOKEN = "test-secret-token"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DISPATCH_WORKER_TOKEN", _TOKEN)
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_persist_run", lambda run_id, run: None)

    async def _fake_run_dispatch(*, event_queue=None, **kwargs):
        if event_queue is not None:
            await event_queue.put({"type": "done"})
        return {"status": "completed", "text_output": "ok", "tool_calls": []}

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _fake_run_dispatch)
    with TestClient(dispatch_api.app) as c:
        yield c


def test_health_capability_advertised(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["capabilities"] == ["input_files"]
    # boot_nonce already present on the worker — unchanged, still exposed.
    assert isinstance(body["boot_nonce"], (int, float))


def test_dispatch_body_limit_rejects_oversize_413(client):
    # A real >32MB body so httpx sets a genuine Content-Length; the middleware
    # must reject with 413 before the route (or even auth) reads it.
    oversize = ('{"prompt":"' + "a" * (33 * 1024 * 1024) + '","allowed_tools":[]}').encode()
    resp = client.post(
        "/dispatch",
        headers={"Authorization": f"Bearer {_TOKEN}", "content-type": "application/json"},
        content=oversize,
    )
    assert resp.status_code == 413


def test_dispatch_body_limit_allows_normal_body(client):
    resp = client.post(
        "/dispatch",
        headers={"Authorization": f"Bearer {_TOKEN}"},
        json={"prompt": "hi", "allowed_tools": []},
    )
    # A small body is never blocked by the size guard (202 accepted here).
    assert resp.status_code != 413
