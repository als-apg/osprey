"""Mock-upstream tests for the proxy's streaming and error paths.

``test_proxy_app_mock.py`` covers the non-streaming happy path, health, auth
fallback and client pooling. This module extends coverage to the parts it
leaves out: the SSE streaming translation (``_stream_proxy``), upstream HTTP
error translation (``_translate_error``), the streaming error frame, and the
request-error 502 path. No network — ``httpx.AsyncClient`` is faked.
"""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

import osprey.infrastructure.proxy.app as app_module
from osprey.infrastructure.proxy.app import create_proxy_app


class _FakeStreamResp:
    """Async-context-manager stand-in for ``client.stream(...)``'s response."""

    def __init__(self, status_code: int, lines: list[str] | None = None, body: bytes = b""):
        self.status_code = status_code
        self._lines = lines or []
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aiter_bytes(self):
        yield self._body


def _install_fake_stream_client(monkeypatch, *, stream_resp=None, post_behavior=None):
    """Patch ``app_module.httpx.AsyncClient`` with a streaming-aware fake."""
    captured: dict = {}

    class _FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def aclose(self):
            return None

        def stream(self, method, url, json=None, headers=None):
            captured["stream_url"] = url
            captured["stream_json"] = json
            return stream_resp

        async def post(self, url, json=None, headers=None):
            captured["post_url"] = url
            return post_behavior()

    monkeypatch.setattr(app_module.httpx, "AsyncClient", _FakeAsyncClient)
    return captured


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    """Parse an SSE stream body into a list of (event, data-dict) pairs."""
    events: list[tuple[str, dict]] = []
    event = None
    for line in raw.splitlines():
        if line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            events.append((event, json.loads(line[len("data: ") :])))
    return events


def _sse_line(payload: dict) -> str:
    return "data: " + json.dumps(payload)


class TestStreaming:
    def test_text_stream_translated_to_anthropic_sse(self, monkeypatch):
        lines = [
            _sse_line({"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]}),
            _sse_line({"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]}),
            _sse_line(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 2},
                }
            ),
            "data: [DONE]",
        ]
        _install_fake_stream_client(monkeypatch, stream_resp=_FakeStreamResp(200, lines=lines))
        app = create_proxy_app("https://up.example/v1", upstream_api_key="k")
        client = TestClient(app)

        resp = client.post(
            "/v1/messages",
            json={
                "model": "m",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        names = [e for e, _ in events]

        assert names[0] == "message_start"
        assert "content_block_start" in names
        # Both text fragments were forwarded as deltas.
        text_deltas = [
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
        ]
        assert "".join(text_deltas) == "Hello"
        assert names[-1] == "message_stop"
        # completion_tokens surfaced in the message_delta usage.
        deltas = [d for e, d in events if e == "message_delta"]
        assert deltas and deltas[-1]["usage"]["output_tokens"] == 2

    def test_tool_call_stream_emits_tool_use_block(self, monkeypatch):
        lines = [
            _sse_line(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "read_pv", "arguments": ""},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            _sse_line(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [{"index": 0, "function": {"arguments": '{"n": 1}'}}]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            _sse_line({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            "data: [DONE]",
        ]
        _install_fake_stream_client(monkeypatch, stream_resp=_FakeStreamResp(200, lines=lines))
        app = create_proxy_app("https://up.example/v1", upstream_api_key="k")
        client = TestClient(app)

        resp = client.post(
            "/v1/messages",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}]},
        )
        events = _parse_sse(resp.text)

        starts = [d for e, d in events if e == "content_block_start"]
        assert any(s["content_block"].get("type") == "tool_use" for s in starts)
        tool_start = next(s for s in starts if s["content_block"]["type"] == "tool_use")
        assert tool_start["content_block"]["name"] == "read_pv"
        # tool_calls finish reason maps to tool_use stop_reason.
        deltas = [d for e, d in events if e == "message_delta"]
        assert deltas[-1]["delta"]["stop_reason"] == "tool_use"

    def test_stream_without_finish_reason_still_closes_cleanly(self, monkeypatch):
        lines = [
            _sse_line({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}),
            "data: [DONE]",
        ]
        _install_fake_stream_client(monkeypatch, stream_resp=_FakeStreamResp(200, lines=lines))
        app = create_proxy_app("https://up.example/v1", upstream_api_key="k")
        client = TestClient(app)

        resp = client.post(
            "/v1/messages",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}]},
        )
        events = _parse_sse(resp.text)
        names = [e for e, _ in events]
        # [DONE] with no finish_reason → fallback close (block stop + end_turn).
        assert names[-1] == "message_stop"
        deltas = [d for e, d in events if e == "message_delta"]
        assert deltas[-1]["delta"]["stop_reason"] == "end_turn"

    def test_upstream_non_200_stream_yields_error_frame(self, monkeypatch):
        _install_fake_stream_client(
            monkeypatch,
            stream_resp=_FakeStreamResp(500, body=b"upstream boom"),
        )
        app = create_proxy_app("https://up.example/v1", upstream_api_key="k")
        client = TestClient(app)

        resp = client.post(
            "/v1/messages",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}]},
        )
        events = _parse_sse(resp.text)
        assert events[0][0] == "error"
        assert "upstream boom" in events[0][1]["error"]["message"]

    def test_malformed_sse_line_is_skipped(self, monkeypatch):
        lines = [
            "event: ignored-non-data-line",
            "data: not-json-at-all",
            _sse_line({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}),
            "data: [DONE]",
        ]
        _install_fake_stream_client(monkeypatch, stream_resp=_FakeStreamResp(200, lines=lines))
        app = create_proxy_app("https://up.example/v1", upstream_api_key="k")
        client = TestClient(app)

        resp = client.post(
            "/v1/messages",
            json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}]},
        )
        # The un-parseable line is dropped; the valid one still produces output.
        events = _parse_sse(resp.text)
        text_deltas = [
            d["delta"]["text"]
            for e, d in events
            if e == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
        ]
        assert "".join(text_deltas) == "ok"


class TestNonStreamErrorTranslation:
    def _error_resp(self, status: int, payload: dict):
        return httpx.Response(
            status_code=status,
            json=payload,
            request=httpx.Request("POST", "http://up.example/v1/chat/completions"),
        )

    def test_401_maps_to_authentication_error(self, monkeypatch):
        resp = self._error_resp(401, {"error": {"message": "bad key"}})

        def _post():
            r = _NonRaisingResp(resp)
            return r

        _install_fake_stream_client(monkeypatch, post_behavior=_post)
        app = create_proxy_app("https://up.example/v1", upstream_api_key="k")
        client = TestClient(app)

        out = client.post(
            "/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "x"}]}
        )
        assert out.status_code == 401
        body = out.json()
        assert body["error"]["type"] == "authentication_error"
        assert body["error"]["message"] == "bad key"

    def test_429_maps_to_rate_limit_error(self, monkeypatch):
        resp = self._error_resp(429, {"error": {"message": "slow down"}})
        _install_fake_stream_client(monkeypatch, post_behavior=lambda: _NonRaisingResp(resp))
        app = create_proxy_app("https://up.example/v1", upstream_api_key="k")
        client = TestClient(app)

        out = client.post(
            "/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "x"}]}
        )
        assert out.status_code == 429
        assert out.json()["error"]["type"] == "rate_limit_error"

    def test_404_maps_to_not_found_error(self, monkeypatch):
        resp = self._error_resp(404, {"error": {"message": "no model"}})
        _install_fake_stream_client(monkeypatch, post_behavior=lambda: _NonRaisingResp(resp))
        app = create_proxy_app("https://up.example/v1", upstream_api_key="k")
        client = TestClient(app)

        out = client.post(
            "/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "x"}]}
        )
        assert out.status_code == 404
        assert out.json()["error"]["type"] == "not_found_error"

    def test_request_error_maps_to_502(self, monkeypatch):
        def _post():
            raise httpx.ConnectError("connection refused")

        _install_fake_stream_client(monkeypatch, post_behavior=_post)
        app = create_proxy_app("https://up.example/v1", upstream_api_key="k")
        client = TestClient(app)

        out = client.post(
            "/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "x"}]}
        )
        assert out.status_code == 502
        assert out.json()["error"]["type"] == "api_error"


class _NonRaisingResp:
    """Wrap a real httpx.Response so raise_for_status raises HTTPStatusError.

    The proxy calls ``resp.raise_for_status()``; we delegate to the wrapped
    response's status so 4xx/5xx codes trigger the error-translation branch.
    """

    def __init__(self, response: httpx.Response):
        self._response = response
        self.status_code = response.status_code
        self.text = response.text

    def json(self):
        return self._response.json()

    def raise_for_status(self):
        raise httpx.HTTPStatusError("err", request=self._response.request, response=self._response)
