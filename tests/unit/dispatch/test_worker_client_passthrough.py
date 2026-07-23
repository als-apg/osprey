"""input_files passthrough, fatal-4xx error_code extraction, and upload-sized
timeouts for ``dispatch_to_worker`` (worker_client), driven with httpx.MockTransport."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from osprey.dispatch.worker_client import (
    AuthError,
    DispatchError,
    FatalDispatchError,
    dispatch_to_worker,
)


def _capturing_transport(
    captured: dict[str, Any], status_code: int = 200, body: dict | None = None
) -> httpx.MockTransport:
    """MockTransport that records the request body/JSON then returns status+body."""
    if body is None:
        body = {"run_id": "abc123", "status": "accepted"}
    encoded = json.dumps(body).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content"] = request.content
        try:
            captured["json"] = json.loads(request.content)
        except ValueError:
            captured["json"] = None
        return httpx.Response(
            status_code=status_code,
            content=encoded,
            headers={"content-type": "application/json"},
            request=request,
        )

    return httpx.MockTransport(handler)


def _patch_asyncclient(transport: httpx.MockTransport, captured_kwargs: dict | None = None):
    """Patch worker_client's AsyncClient to use ``transport`` and record its kwargs."""
    original_cls = httpx.AsyncClient

    class PatchedClient(original_cls):
        def __init__(self, **kwargs):
            if captured_kwargs is not None:
                captured_kwargs.update(kwargs)
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    return patch("osprey.dispatch.worker_client.httpx.AsyncClient", PatchedClient)


_FILES = [
    {"filename": "plot.png", "mime": "image/png", "content_b64": "AAECAwQF", "ingest": True},
]


@pytest.mark.asyncio
async def test_dispatch_passthrough_forwards_input_files():
    captured: dict[str, Any] = {}
    transport = _capturing_transport(captured)
    with _patch_asyncclient(transport):
        result = await dispatch_to_worker(
            url="http://worker:9190",
            prompt="hi",
            allowed_tools=[],
            token="tok",
            input_files=_FILES,
        )
    assert result == {"run_id": "abc123", "status": "accepted"}
    assert captured["json"]["input_files"] == _FILES


@pytest.mark.asyncio
async def test_dispatch_passthrough_omits_input_files_when_absent():
    """No input_files -> the key is absent, so a worker predating the field is unaffected."""
    captured: dict[str, Any] = {}
    transport = _capturing_transport(captured)
    with _patch_asyncclient(transport):
        await dispatch_to_worker(
            url="http://worker:9190", prompt="hi", allowed_tools=[], token="tok"
        )
    assert "input_files" not in captured["json"]


@pytest.mark.asyncio
async def test_dispatch_passthrough_empty_input_files_is_omitted():
    captured: dict[str, Any] = {}
    transport = _capturing_transport(captured)
    with _patch_asyncclient(transport):
        await dispatch_to_worker(
            url="http://worker:9190",
            prompt="hi",
            allowed_tools=[],
            token="tok",
            input_files=[],
        )
    assert "input_files" not in captured["json"]


@pytest.mark.asyncio
async def test_dispatch_upload_timeout_widens_connect_and_write():
    """An input_files upload can be ~24MB; connect/write timeouts must be generous."""
    captured_kwargs: dict[str, Any] = {}
    transport = _capturing_transport({})
    with _patch_asyncclient(transport, captured_kwargs):
        await dispatch_to_worker(
            url="http://worker:9190",
            prompt="hi",
            allowed_tools=[],
            token="tok",
            timeout=30.0,
            input_files=_FILES,
        )
    timeout = captured_kwargs["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 10.0
    assert timeout.write == 120.0
    # Read stays at the base timeout — the worker returns 202 immediately.
    assert timeout.read == 30.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "detail, expected_code",
    [
        ("input_files_cap_exceeded", "input_files_cap_exceeded"),
        ("input_files_invalid", "input_files_invalid"),
        ("some_other_internal_detail", None),
        (None, None),
    ],
)
async def test_dispatch_fatal_400_extracts_whitelisted_error_code(detail, expected_code):
    body = {"detail": detail} if detail is not None else {}
    transport = _capturing_transport({}, status_code=400, body=body)
    with _patch_asyncclient(transport):
        with pytest.raises(FatalDispatchError) as exc_info:
            await dispatch_to_worker(
                url="http://worker:9190", prompt="x", allowed_tools=[], token="tok"
            )
    assert exc_info.value.error_code == expected_code
    # Body text is never echoed into the exception message — only the status.
    assert "HTTP 400" in str(exc_info.value)


@pytest.mark.asyncio
async def test_dispatch_fatal_413_body_too_large_is_fatal():
    transport = _capturing_transport({}, status_code=413, body={"detail": "Request body too large"})
    with _patch_asyncclient(transport):
        with pytest.raises(FatalDispatchError) as exc_info:
            await dispatch_to_worker(
                url="http://worker:9190", prompt="x", allowed_tools=[], token="tok"
            )
    # 413 is not a whitelisted rejection code -> generic (None), still fatal.
    assert exc_info.value.error_code is None


@pytest.mark.asyncio
async def test_dispatch_fatal_403_denied_tools_is_fatal_generic():
    transport = _capturing_transport({}, status_code=403, body={"detail": "Tools blocked"})
    with _patch_asyncclient(transport):
        with pytest.raises(FatalDispatchError):
            await dispatch_to_worker(
                url="http://worker:9190", prompt="x", allowed_tools=[], token="tok"
            )


@pytest.mark.asyncio
async def test_dispatch_401_still_auth_error_not_fatal():
    """401 stays AuthError (dispatcher<->worker token), distinct from the 4xx fatal path."""
    transport = _capturing_transport({}, status_code=401, body={"detail": "no"})
    with _patch_asyncclient(transport):
        with pytest.raises(AuthError):
            await dispatch_to_worker(
                url="http://worker:9190", prompt="x", allowed_tools=[], token="bad"
            )


@pytest.mark.asyncio
async def test_dispatch_500_still_retryable_dispatch_error():
    """5xx remains a retryable DispatchError, not a FatalDispatchError."""
    transport = _capturing_transport({}, status_code=500, body={"detail": "boom"})
    with _patch_asyncclient(transport):
        with pytest.raises(DispatchError) as exc_info:
            await dispatch_to_worker(
                url="http://worker:9190", prompt="x", allowed_tools=[], token="tok"
            )
    assert not isinstance(exc_info.value, FatalDispatchError)
    assert "HTTP 500" in str(exc_info.value)
