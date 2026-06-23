"""Tests for worker_client using httpx.MockTransport."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from osprey.dispatch.worker_client import (
    AuthError,
    DispatchError,
    cancel_worker_run,
    dispatch_to_worker,
    fetch_worker_runs,
    proxy_worker_stream,
)


def _patched_client(transport: httpx.MockTransport):
    """Return a patch() ctx that injects the mock transport into AsyncClient."""
    original_cls = httpx.AsyncClient

    class PatchedClient(original_cls):
        def __init__(self, **kwargs):
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    return patch("osprey.dispatch.worker_client.httpx.AsyncClient", PatchedClient)


def _mock_transport(status_code: int = 200, body: dict | None = None) -> httpx.MockTransport:
    """Return a MockTransport that always responds with the given status and JSON body."""
    if body is None:
        body = {"run_id": "abc123", "status": "accepted"}
    encoded = json.dumps(body).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            content=encoded,
            headers={"content-type": "application/json"},
            request=request,
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_successful_dispatch_returns_response_dict():
    expected = {"run_id": "abc123", "status": "accepted"}
    transport = _mock_transport(200, expected)

    with patch(
        "httpx.AsyncClient",
        lambda **kw: httpx.AsyncClient(
            transport=transport, **{k: v for k, v in kw.items() if k != "transport"}
        ),
    ):
        # Patch AsyncClient to inject our mock transport
        pass

    # Patch at the module level
    original_cls = httpx.AsyncClient

    class PatchedClient(original_cls):
        def __init__(self, **kwargs):
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    with patch("osprey.dispatch.worker_client.httpx.AsyncClient", PatchedClient):
        result = await dispatch_to_worker(
            url="http://dispatch-worker-1:9190",
            prompt="show beam current",
            allowed_tools=["read_pv"],
            token="secret-token",
        )

    assert result == expected


@pytest.mark.asyncio
async def test_connection_error_raises_dispatch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(handler)
    original_cls = httpx.AsyncClient

    class PatchedClient(original_cls):
        def __init__(self, **kwargs):
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    with patch("osprey.dispatch.worker_client.httpx.AsyncClient", PatchedClient):
        with pytest.raises(DispatchError, match="Connection error"):
            await dispatch_to_worker(
                url="http://unreachable:9999",
                prompt="test",
                allowed_tools=[],
                token="tok",
            )


@pytest.mark.asyncio
async def test_http_401_raises_auth_error():
    transport = _mock_transport(status_code=401, body={"detail": "invalid token"})
    original_cls = httpx.AsyncClient

    class PatchedClient(original_cls):
        def __init__(self, **kwargs):
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    with patch("osprey.dispatch.worker_client.httpx.AsyncClient", PatchedClient):
        with pytest.raises(AuthError, match="Unauthorized"):
            await dispatch_to_worker(
                url="http://worker:9190",
                prompt="test",
                allowed_tools=[],
                token="bad-token",
            )


@pytest.mark.asyncio
async def test_timeout_raises_dispatch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    transport = httpx.MockTransport(handler)
    original_cls = httpx.AsyncClient

    class PatchedClient(original_cls):
        def __init__(self, **kwargs):
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    with patch("osprey.dispatch.worker_client.httpx.AsyncClient", PatchedClient):
        with pytest.raises(DispatchError, match="Timeout"):
            await dispatch_to_worker(
                url="http://worker:9190",
                prompt="test",
                allowed_tools=[],
                token="tok",
                timeout=1.0,
            )


@pytest.mark.asyncio
async def test_bearer_token_sent_in_headers():
    captured_headers: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            status_code=200,
            content=json.dumps({"run_id": "x", "status": "ok"}).encode(),
            headers={"content-type": "application/json"},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    original_cls = httpx.AsyncClient

    class PatchedClient(original_cls):
        def __init__(self, **kwargs):
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    with patch("osprey.dispatch.worker_client.httpx.AsyncClient", PatchedClient):
        await dispatch_to_worker(
            url="http://worker:9190",
            prompt="test",
            allowed_tools=[],
            token="my-secret-token",
        )

    assert captured_headers.get("authorization") == "Bearer my-secret-token"


@pytest.mark.asyncio
async def test_non_401_http_error_raises_typed_dispatch_error_without_body():
    """A 5xx from the worker → DispatchError carrying only the status, not the body.

    The worker's response body can contain a stack trace or token-bearing detail;
    it must never be echoed into the dispatcher's registry history.
    """
    leaky_body = {"detail": "Traceback: secret-token leaked in error"}
    transport = _mock_transport(500, leaky_body)
    original_cls = httpx.AsyncClient

    class PatchedClient(original_cls):
        def __init__(self, **kwargs):
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    with patch("osprey.dispatch.worker_client.httpx.AsyncClient", PatchedClient):
        with pytest.raises(DispatchError) as exc_info:
            await dispatch_to_worker(
                url="http://dispatch-worker-1:9190",
                prompt="x",
                allowed_tools=[],
                token="tok",
            )

    msg = str(exc_info.value)
    assert "HTTP 500" in msg
    assert "secret-token" not in msg
    assert "Traceback" not in msg


@pytest.mark.asyncio
async def test_401_still_raises_auth_error():
    transport = _mock_transport(401, {"detail": "Invalid bearer token"})
    original_cls = httpx.AsyncClient

    class PatchedClient(original_cls):
        def __init__(self, **kwargs):
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    with patch("osprey.dispatch.worker_client.httpx.AsyncClient", PatchedClient):
        with pytest.raises(AuthError):
            await dispatch_to_worker(
                url="http://dispatch-worker-1:9190",
                prompt="x",
                allowed_tools=[],
                token="bad",
            )


# ---------------------------------------------------------------------------
# fetch_worker_runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_worker_runs_returns_list():
    runs = [{"run_id": "r1", "status": "completed"}]
    transport = _mock_transport(200, body=runs)  # type: ignore[arg-type]
    with _patched_client(transport):
        result = await fetch_worker_runs("http://worker:9190", token="tok")
    assert result == runs


@pytest.mark.asyncio
async def test_fetch_worker_runs_connection_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with _patched_client(httpx.MockTransport(handler)):
        with pytest.raises(DispatchError, match="Connection error"):
            await fetch_worker_runs("http://worker:9190", token="tok")


@pytest.mark.asyncio
async def test_fetch_worker_runs_sends_bearer():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            content=json.dumps([]).encode(),
            headers={"content-type": "application/json"},
            request=request,
        )

    with _patched_client(httpx.MockTransport(handler)):
        await fetch_worker_runs("http://worker:9190", token="tok")
    assert captured["auth"] == "Bearer tok"


# ---------------------------------------------------------------------------
# cancel_worker_run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_worker_run_success():
    transport = _mock_transport(200, body={"run_id": "r1", "cancelled": True})
    with _patched_client(transport):
        result = await cancel_worker_run("http://worker:9190", token="tok", run_id="r1")
    assert result["cancelled"] is True


@pytest.mark.asyncio
async def test_cancel_worker_run_401_raises_auth_error():
    transport = _mock_transport(401, body={"detail": "no"})
    with _patched_client(transport):
        with pytest.raises(AuthError):
            await cancel_worker_run("http://worker:9190", token="bad", run_id="r1")


@pytest.mark.asyncio
async def test_cancel_worker_run_404_raises_dispatch_error():
    transport = _mock_transport(404, body={"detail": "missing"})
    with _patched_client(transport):
        with pytest.raises(DispatchError, match="not found"):
            await cancel_worker_run("http://worker:9190", token="tok", run_id="ghost")


# ---------------------------------------------------------------------------
# proxy_worker_stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_worker_stream_yields_chunks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data: {}\n\n", request=request)

    chunks = []
    with _patched_client(httpx.MockTransport(handler)):
        async for chunk in proxy_worker_stream("http://worker:9190", "tok", "r1"):
            chunks.append(chunk)
    assert b"".join(chunks) == b"data: {}\n\n"


@pytest.mark.asyncio
async def test_proxy_worker_stream_401_raises_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"", request=request)

    with _patched_client(httpx.MockTransport(handler)):
        with pytest.raises(AuthError):
            async for _ in proxy_worker_stream("http://worker:9190", "bad", "r1"):
                pass


@pytest.mark.asyncio
async def test_proxy_worker_stream_non_200_raises_dispatch_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"", request=request)

    with _patched_client(httpx.MockTransport(handler)):
        with pytest.raises(DispatchError, match="HTTP 503"):
            async for _ in proxy_worker_stream("http://worker:9190", "tok", "r1"):
                pass
