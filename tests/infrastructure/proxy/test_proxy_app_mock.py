"""L2b — integration test of the proxy FastAPI app against a MOCK upstream.

Exercises create_proxy_app end to end (auth extraction, upstream URL building,
request+response translation through the /v1/messages route) without touching
the network. The upstream OpenAI server is faked by patching httpx.AsyncClient.

Experiment branch: experiment/cborg-claude-code (issue #259).
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

import osprey.infrastructure.proxy.app as app_module
from osprey.infrastructure.proxy.app import create_proxy_app
from osprey.models.providers.openai import OpenAIProviderAdapter


class _FakeResp:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "upstream error", request=httpx.Request("POST", "http://x"), response=self
            )  # type: ignore[arg-type]


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
            "choices": [
                {"message": {"role": "assistant", "content": "PONG"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    )
    app = create_proxy_app("https://api.example.com/v1", upstream_api_key="secret-key")
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
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
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
    app = create_proxy_app("https://api.example.com/v1", upstream_api_key="secret-key")
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
    app = create_proxy_app("https://api.example.com/v1", upstream_api_key=None)
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
    app = create_proxy_app("https://api.example.com/v1", upstream_api_key="secret-key")
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
    app = create_proxy_app("https://api.example.com/v1")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "upstream": "https://api.example.com/v1"}


_PONG = {
    "choices": [{"message": {"role": "assistant", "content": "PONG"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
}

_PING = {
    "model": "some-model",
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 16,
}


def test_proxy_forwards_exactly_the_declared_headers(monkeypatch):
    """The headers the launch declared reach the upstream; no other client header does."""
    captured = _install_fake_upstream(monkeypatch, _PONG)
    app = create_proxy_app(
        "https://gw.example/v1",
        upstream_api_key="secret-key",
        forward_headers={"x-litellm-end-user-id", "x-litellm-tags", "X-Corp-Trace"},
    )
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json=_PING,
        headers={
            "X-LiteLLM-End-User-Id": "alice",
            "x-litellm-tags": "osprey,surface:terminal",
            "x-corp-trace": "abc123",
            "X-Other": "1",
            "x-litellm-extra": "1",
            "Authorization": "Bearer client-key",
        },
    )
    assert resp.status_code == 200

    sent = {k.lower(): v for k, v in captured["headers"].items()}
    assert sent["x-litellm-end-user-id"] == "alice"
    assert sent["x-litellm-tags"] == "osprey,surface:terminal"
    assert sent["x-corp-trace"] == "abc123"
    assert "x-other" not in sent
    assert "x-litellm-extra" not in sent
    assert sent["authorization"] == "Bearer secret-key"


def test_proxy_without_a_declaration_forwards_no_client_header(monkeypatch):
    captured = _install_fake_upstream(monkeypatch, _PONG)
    app = create_proxy_app("https://gw.example/v1", upstream_api_key="secret-key")
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json=_PING,
        headers={"x-litellm-end-user-id": "alice", "X-Corp-Trace": "abc123"},
    )
    assert resp.status_code == 200

    assert {k.lower() for k in captured["headers"]} == {"authorization", "content-type"}


def test_a_declared_header_the_proxy_owns_is_refused_and_named(monkeypatch, caplog):
    captured = _install_fake_upstream(monkeypatch, _PONG)
    with caplog.at_level(logging.WARNING, logger="osprey.infrastructure.proxy"):
        app = create_proxy_app(
            "https://gw.example/v1",
            upstream_api_key="secret-key",
            forward_headers={"Authorization", "host", "X-Corp-Trace"},
        )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "authorization" in warnings[0].getMessage()
    assert "host" in warnings[0].getMessage()

    client = TestClient(app)
    resp = client.post(
        "/v1/messages",
        json=_PING,
        headers={"Authorization": "Bearer client-key", "X-Corp-Trace": "abc123"},
    )
    assert resp.status_code == 200

    sent = {k.lower(): v for k, v in captured["headers"].items()}
    assert sent["authorization"] == "Bearer secret-key"
    assert sent["x-corp-trace"] == "abc123"
    assert "host" not in sent


def test_client_key_fallback_is_unchanged(monkeypatch):
    captured = _install_fake_upstream(monkeypatch, _PONG)
    app = create_proxy_app("https://gw.example/v1", upstream_api_key=None)
    client = TestClient(app)

    resp = client.post("/v1/messages", json=_PING, headers={"x-api-key": "client-key"})
    assert resp.status_code == 200

    sent = {k.lower(): v for k, v in captured["headers"].items()}
    assert sent["authorization"] == "Bearer client-key"
    assert "x-api-key" not in sent


def test_proxy_sends_the_upstream_its_declared_request_shape(monkeypatch):
    """An upstream that takes max_completion_tokens and no temperature gets exactly that."""
    captured = _install_fake_upstream(
        monkeypatch,
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    )
    app = create_proxy_app(
        "https://api.example.com/v1",
        upstream_api_key="secret-key",
        max_tokens_param="max_completion_tokens",
        accepts_temperature=lambda _model: False,
    )
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "gpt-6-sol",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    assert captured["json"]["max_completion_tokens"] == 16
    assert "max_tokens" not in captured["json"]
    assert "temperature" not in captured["json"]


@pytest.mark.parametrize(("model", "sent"), [("gpt-4o", True), ("gpt-6-sol", False)])
def test_proxy_asks_the_provider_per_request_model_whether_a_temperature_is_sent(
    monkeypatch, model, sent
):
    """One proxy fronts every model its provider serves, and each request's model decides."""
    captured = _install_fake_upstream(
        monkeypatch,
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    )
    app = create_proxy_app(
        "https://api.example.com/v1",
        upstream_api_key="secret-key",
        max_tokens_param="max_completion_tokens",
        accepts_temperature=OpenAIProviderAdapter.accepts_temperature,
    )
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    if sent:
        assert captured["json"]["temperature"] == 0.0
    else:
        assert "temperature" not in captured["json"]
