"""L2b — integration test of the proxy FastAPI app against a MOCK upstream.

Exercises create_proxy_app end to end (auth extraction, upstream URL building,
request+response translation through the /v1/messages route) without touching
the network. The upstream OpenAI server is faked by patching httpx.AsyncClient.

Experiment branch: experiment/cborg-claude-code (issue #259).
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import osprey.infrastructure.proxy.app as app_module
from osprey.infrastructure.proxy.app import create_proxy_app


class _FakeResp:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("upstream error", request=httpx.Request("POST", "http://x"), response=self)  # type: ignore[arg-type]


def _install_fake_upstream(monkeypatch, payload: dict):
    """Patch app_module.httpx.AsyncClient so the proxy talks to a canned OpenAI."""
    captured: dict = {}

    class _FakeAsyncClient:
        instances = 0

        def __init__(self, *a, **k):
            type(self).instances += 1

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def aclose(self):
            return None

        async def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _FakeResp(payload)

    _FakeAsyncClient.instances = 0
    monkeypatch.setattr(app_module.httpx, "AsyncClient", _FakeAsyncClient)
    captured["_client_cls"] = _FakeAsyncClient
    return captured


def test_proxy_translates_text_request_end_to_end(monkeypatch):
    captured = _install_fake_upstream(
        monkeypatch,
        {
            "choices": [{"message": {"role": "assistant", "content": "PONG"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    )
    app = create_proxy_app("https://api.cborg.lbl.gov/v1", upstream_api_key="secret-key")
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "cborg-coder",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # Response came back in Anthropic shape:
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "PONG"}]
    assert body["model"] == "cborg-coder"

    # Proxy hit the right upstream URL with the right auth and translated body:
    assert captured["url"] == "https://api.cborg.lbl.gov/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["json"]["model"] == "cborg-coder"
    assert captured["json"]["messages"][-1] == {"role": "user", "content": "ping"}


def test_proxy_translates_tool_call_request_end_to_end(monkeypatch):
    captured = _install_fake_upstream(
        monkeypatch,
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read_pv", "arguments": '{"name": "BPM:01"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 9},
        },
    )
    app = create_proxy_app("https://api.cborg.lbl.gov/v1", upstream_api_key="secret-key")
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "cborg-coder",
            "messages": [{"role": "user", "content": "read BPM:01"}],
            "tools": [
                {
                    "name": "read_pv",
                    "description": "Read a PV.",
                    "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                }
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    tool_block = body["content"][0]
    assert tool_block == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read_pv",
        "input": {"name": "BPM:01"},
    }
    # The tool definition was forwarded to the upstream in OpenAI function shape:
    assert captured["json"]["tools"][0]["function"]["name"] == "read_pv"


def test_proxy_falls_back_to_request_bearer_when_no_upstream_key(monkeypatch):
    """When the app has no baked-in key, it must read the caller's Bearer token."""
    captured = _install_fake_upstream(
        monkeypatch,
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    )
    app = create_proxy_app("https://api.cborg.lbl.gov/v1", upstream_api_key=None)
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        headers={"Authorization": "Bearer caller-token"},
        json={"model": "cborg-coder", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert captured["headers"]["Authorization"] == "Bearer caller-token"


def test_proxy_reuses_single_pooled_client_across_requests(monkeypatch):
    """Regression for the #259 outage (2026-06-18): a fresh httpx.AsyncClient per
    request opened/closed an upstream TCP connection every call, exhausting the
    host's ephemeral port pool via tens of thousands of TIME_WAIT sockets. The
    proxy must build exactly ONE pooled client and reuse it for every request.
    """
    captured = _install_fake_upstream(
        monkeypatch,
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    )
    app = create_proxy_app("https://api.cborg.lbl.gov/v1", upstream_api_key="secret-key")
    client = TestClient(app)

    for _ in range(5):
        resp = client.post(
            "/v1/messages",
            json={"model": "cborg-coder", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200

    # Five requests, but only one upstream client ever constructed.
    assert captured["_client_cls"].instances == 1


def test_health_endpoint_reports_upstream():
    app = create_proxy_app("https://api.cborg.lbl.gov/v1")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "upstream": "https://api.cborg.lbl.gov/v1"}
