"""Unit tests for the bluesky panels sidecar's plan-draft relay (task 3.1).

Exercises `draft_relay.router` mounted on a LOCAL FastAPI app (never the
package-level `osprey.services.bluesky_panels.app.app`, which is covered
separately in `test_app_integration.py`). The bridge HTTP layer is faked with
`httpx.MockTransport` so no real network call is made and no real Bluesky
bridge process needs to be running. This module never touches OSPREY config
resolution directly (the bridge URL is injected onto `app.state` by the test
harness, mirroring `test_read_proxy.py`), so no config-isolation fixture is
needed here.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.services.bluesky_panels import draft_relay

_BRIDGE_URL = "http://bridge.test"


def _build_app(handler: Callable[[httpx.Request], httpx.Response]) -> FastAPI:
    """Build a local FastAPI app with the draft-relay router mounted, backed
    by a mock-transport client standing in for the real bridge.
    """
    app = FastAPI()
    app.include_router(draft_relay.router)
    app.state.bridge_url = _BRIDGE_URL
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return app


def _json_response(status_code: int, body: object) -> httpx.Response:
    return httpx.Response(status_code, json=body)


class _DelayedFrames(httpx.AsyncByteStream):
    """An async byte stream that yields ``frames`` with ``delay`` between them.

    Used to prove the SSE relay is a genuine streaming pass-through (not a
    buffer-then-forward), without a real multi-second sleep.
    """

    def __init__(self, frames: list[bytes], delay: float = 0.05) -> None:
        self._frames = frames
        self._delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i, frame in enumerate(self._frames):
            if i > 0:
                await asyncio.sleep(self._delay)
            yield frame


# ---------------------------------------------------------------------------
# GET/PATCH/DELETE passthrough
# ---------------------------------------------------------------------------


def test_get_draft_round_trips_body_and_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/draft"
        return _json_response(
            200, {"draft": {"plan_name": "grid_scan", "plan_args": {}}, "revision": 3}
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/draft")

    assert response.status_code == 200
    assert response.json() == {"draft": {"plan_name": "grid_scan", "plan_args": {}}, "revision": 3}


def test_get_draft_null_draft_round_trips() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"draft": None, "revision": 0})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/draft")

    assert response.status_code == 200
    assert response.json() == {"draft": None, "revision": 0}


def test_patch_draft_forwards_body_verbatim() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.method == "PATCH"
        assert request.url.path == "/draft"
        return _json_response(200, {"revision": 4, "changed": ["detectors"], "plan_name": "orm"})

    payload = {
        "plan_args_patch": {"detectors": ["bpm1"]},
        "client_id": "tab-1",
    }
    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.patch("/draft", json=payload)

    assert response.status_code == 200
    assert response.json() == {"revision": 4, "changed": ["detectors"], "plan_name": "orm"}
    assert len(seen) == 1
    import json as _json

    assert _json.loads(seen[0].content) == payload


def test_patch_draft_no_op_response_round_trips() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(200, {"revision": 4, "changed": [], "plan_name": "orm"})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.patch("/draft", json={"plan_args_patch": {}, "client_id": "tab-1"})

    assert response.status_code == 200
    assert response.json()["changed"] == []


def test_delete_draft_round_trips() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/draft"
        return _json_response(200, {"revision": 5, "changed": [], "plan_name": None})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.delete("/draft")

    assert response.status_code == 200
    assert response.json() == {"revision": 5, "changed": [], "plan_name": None}


# ---------------------------------------------------------------------------
# Query-param forwarding on PATCH (forward-compatibility, mirrors read_proxy)
# ---------------------------------------------------------------------------


def test_patch_draft_forwards_query_params() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _json_response(200, {"revision": 1, "changed": [], "plan_name": "orm"})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.patch("/draft", params={"debug": "1"}, json={"client_id": "tab-1"})

    assert response.status_code == 200
    assert seen[0].url.params["debug"] == "1"


# ---------------------------------------------------------------------------
# Error passthrough (verbatim body + status, never recomputed)
# ---------------------------------------------------------------------------


def test_patch_draft_no_draft_409_passes_through_verbatim() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(409, {"code": "no_draft"})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.patch("/draft", json={"plan_args_patch": {}, "client_id": "tab-1"})

    assert response.status_code == 409
    assert response.json() == {"code": "no_draft"}


def test_patch_draft_expected_plan_name_mismatch_409_passes_through() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(409, {"detail": "expected_plan_name mismatch"})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.patch(
            "/draft",
            json={"expected_plan_name": "orm", "client_id": "tab-1"},
        )

    assert response.status_code == 409
    assert response.json() == {"detail": "expected_plan_name mismatch"}


def test_patch_draft_field_validation_422_passes_through() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            422, {"detail": [{"loc": ["detectors"], "msg": "min_length", "type": "value_error"}]}
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.patch(
            "/draft",
            json={"plan_args_patch": {"detectors": []}, "client_id": "tab-1"},
        )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "min_length"


def test_patch_draft_unknown_plan_name_422_passes_through() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(422, {"detail": "unknown plan"})

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.patch("/draft", json={"plan_name": "nope", "client_id": "tab-1"})

    assert response.status_code == 422
    assert response.json() == {"detail": "unknown plan"}


# ---------------------------------------------------------------------------
# Bridge unreachable -> 502, never an uncaught 500 (all four verbs)
# ---------------------------------------------------------------------------


def _refusing_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def test_get_draft_bridge_unreachable_returns_502() -> None:
    app = _build_app(_refusing_handler)
    with TestClient(app) as client:
        response = client.get("/draft")

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}


def test_patch_draft_bridge_unreachable_returns_502() -> None:
    app = _build_app(_refusing_handler)
    with TestClient(app) as client:
        response = client.patch("/draft", json={"client_id": "tab-1"})

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}


def test_delete_draft_bridge_unreachable_returns_502() -> None:
    app = _build_app(_refusing_handler)
    with TestClient(app) as client:
        response = client.delete("/draft")

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}


def test_draft_events_bridge_unreachable_returns_502() -> None:
    app = _build_app(_refusing_handler)
    with TestClient(app) as client:
        response = client.get("/draft/events")

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}


# ---------------------------------------------------------------------------
# SSE relay: streaming, timeout override, hop-by-hop header stripping
# ---------------------------------------------------------------------------


def test_draft_events_streams_multiple_frames_incl_delayed_second_frame() -> None:
    frames = [b"event: hello\ndata: {}\n\n", b"event: change\ndata: {}\n\n"]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/draft/events"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DelayedFrames(frames),
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        with client.stream("GET", "/draft/events") as response:
            assert response.status_code == 200
            body = b"".join(response.iter_bytes())

    assert body == b"".join(frames)


def test_draft_events_disables_read_timeout() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.extensions.get("timeout", {}))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DelayedFrames([b"data: 1\n\n"]),
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        with client.stream("GET", "/draft/events") as response:
            assert response.status_code == 200
            list(response.iter_bytes())

    assert len(captured) == 1
    assert captured[0]["read"] is None
    assert captured[0]["connect"] == 5.0


def test_draft_events_strips_hop_by_hop_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "connection": "keep-alive",
                "transfer-encoding": "chunked",
                "x-custom-marker": "present",
            },
            stream=_DelayedFrames([b"data: 1\n\n"]),
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        with client.stream("GET", "/draft/events") as response:
            assert response.status_code == 200
            list(response.iter_bytes())
            headers = response.headers

    assert headers.get("x-custom-marker") == "present"
    assert headers.get("connection") != "keep-alive"
    assert "chunked" not in (headers.get("transfer-encoding") or "")


def test_draft_events_relays_non_200_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, headers={"content-type": "text/event-stream"}, stream=_DelayedFrames([])
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        with client.stream("GET", "/draft/events") as response:
            status = response.status_code
            list(response.iter_bytes())

    assert status == 503


@pytest.mark.parametrize("path", ["/draft"])
def test_all_verbs_present_on_draft_path(path: str) -> None:
    app = FastAPI()
    app.include_router(draft_relay.router)
    operations = app.openapi()["paths"][path]
    assert {"get", "patch", "delete"} <= set(operations.keys())
