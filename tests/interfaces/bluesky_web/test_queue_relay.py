"""Unit tests for the bluesky-web sidecar's plan-queue relay.

Exercises ``osprey.interfaces.bluesky_web.queue_relay.router`` mounted on a
LOCAL FastAPI app, with the bridge HTTP layer faked by ``httpx.MockTransport``
(respx is not installed here), mirroring ``test_draft_relay.py`` /
``test_launch.py``. The composed-app assertions at the bottom are the
exception: they import the package-level app to prove the queue paths are
actually wired and collide with nothing, and to pin the ``/results`` ->
BLUESKY-panel mount alias.

Two contracts carry most of the weight here, and both are asserted on BODIES,
not just status codes -- a status-code-only assertion is how a wire contract
rots without a test noticing:

- **Refusals pass through verbatim.** The bridge mints every queue refusal as
  ``{"detail": {"code": <machine-readable>, "detail": <sentence>, ...extras}}``.
  The sidecar holds no queue state and no queue policy, so these tests assert
  the relayed body is deep-equal to the bridge's, ``detail.code`` and every
  extra field included. Nothing is unwrapped, reshaped, or re-synthesized.
- **The launch token is server-side only.** It is resolved in-process (env var
  or config, via the shared resolver), rides every queue WRITE, never rides a
  read, is never accepted from the incoming request, and is never echoed back.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.bluesky_web import queue_relay

_BRIDGE_URL = "http://bridge.test"
TOKEN = "s3cr3t-launch-token"  # noqa: S105 - test fixture value, not a real secret


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Never let ambient repo/user config leak a real launch token in.

    Mirrors ``test_launch.py``'s fixture: config resolution is pointed at a
    file that does not exist, so a test asserting the unarmed path cannot pass
    (or fail) because of the developer's own ``config.yml``. Tests that need a
    token set ``BLUESKY_LAUNCH_TOKEN`` themselves.
    """
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)


def _build_app(handler: Callable[[httpx.Request], httpx.Response]) -> FastAPI:
    app = FastAPI()
    app.include_router(queue_relay.router)
    app.state.bridge_url = _BRIDGE_URL
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return app


def _recording_app(
    status_code: int, body: object
) -> tuple[FastAPI, list[httpx.Request], list[bytes]]:
    """An app whose mock bridge always answers ``(status_code, body)``.

    Returns the requests it saw and their raw bodies (read eagerly, since
    ``httpx.Request.content`` is not readable after the response is built).
    """
    seen: list[httpx.Request] = []
    contents: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        contents.append(request.content)
        return httpx.Response(status_code, json=body)

    return _build_app(handler), seen, contents


def _refusing_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


class _DelayedFrames(httpx.AsyncByteStream):
    """Yields ``frames`` with ``delay`` between them, to prove the SSE relay
    genuinely streams rather than buffering upstream to completion first."""

    def __init__(self, frames: list[bytes], delay: float = 0.05) -> None:
        self._frames = frames
        self._delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i, frame in enumerate(self._frames):
            if i > 0:
                await asyncio.sleep(self._delay)
            yield frame


# The bridge's own success shapes, used verbatim so a change to the bridge
# contract shows up here as a diff rather than as a quietly-passing test.
_QUEUE_SNAPSHOT = {
    "status": {
        "available": True,
        "manager_state": "idle",
        "plan_queue_uid": "uid-1",
        "items_in_queue": 2,
    },
    "items": [
        {"item_uid": "item-a", "name": "grid_scan", "kwargs": {"num": 5}},
        {"item_uid": "item-b", "name": "orm", "kwargs": {}},
    ],
    "running_item": None,
}


# ---------------------------------------------------------------------------
# Reads: GET /queue relays body + status verbatim, and carries no token
# ---------------------------------------------------------------------------


def test_get_queue_round_trips_body_and_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/queue"
        return httpx.Response(200, json=_QUEUE_SNAPSHOT)

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/queue")

    assert response.status_code == 200
    assert response.json() == _QUEUE_SNAPSHOT


def test_get_queue_relays_a_running_item_with_progress_verbatim() -> None:
    """``running_item.progress`` is the bridge's to compute -- including the
    indeterminate ``fraction: null`` a plan with an unknown denominator
    yields. The relay must not substitute a 0."""
    snapshot = {
        "status": {"available": True, "manager_state": "executing_queue"},
        "items": [],
        "running_item": {
            "item_uid": "item-a",
            "name": "orm",
            "progress": {"fraction": None, "rows_seen": 12, "complete": False},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=snapshot)

    app = _build_app(handler)
    with TestClient(app) as client:
        response = client.get("/queue")

    assert response.json() == snapshot
    assert response.json()["running_item"]["progress"]["fraction"] is None


def test_get_queue_never_sends_the_launch_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """No read is ever arming-gated, so the credential has no business on one."""
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_QUEUE_SNAPSHOT)

    app = _build_app(handler)
    with TestClient(app) as client:
        client.get("/queue")

    assert "x-launch-token" not in seen[0].headers


def test_get_queue_bridge_unreachable_returns_502() -> None:
    app = _build_app(_refusing_handler)
    with TestClient(app) as client:
        response = client.get("/queue")

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}


# ---------------------------------------------------------------------------
# Writes: bodies forwarded byte-for-byte, success shapes relayed verbatim
# ---------------------------------------------------------------------------


def test_add_item_forwards_body_verbatim_and_relays_success() -> None:
    success = {
        "run_id": "run-abc123",
        "revision": 7,
        "item": {"item_uid": "item-c", "name": "grid_scan", "kwargs": {"num": 5}},
    }
    app, seen, contents = _recording_app(200, success)

    with TestClient(app) as client:
        response = client.post("/queue/items", json={"draft_revision": 7})

    assert response.status_code == 200
    assert response.json() == success
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/queue/items"
    assert json.loads(contents[0]) == {"draft_revision": 7}


def test_move_item_forwards_body_and_relays_success() -> None:
    app, seen, contents = _recording_app(200, {"moved": True, "item": {"item_uid": "item-a"}})

    with TestClient(app) as client:
        response = client.post("/queue/items/item-a/move", json={"pos_dest": "front"})

    assert response.status_code == 200
    assert response.json() == {"moved": True, "item": {"item_uid": "item-a"}}
    assert seen[0].url.path == "/queue/items/item-a/move"
    assert json.loads(contents[0]) == {"pos_dest": "front"}


def test_remove_item_relays_success() -> None:
    app, seen, _ = _recording_app(200, {"removed": True, "item": {"item_uid": "item-a"}})

    with TestClient(app) as client:
        response = client.delete("/queue/items/item-a")

    assert response.status_code == 200
    assert response.json() == {"removed": True, "item": {"item_uid": "item-a"}}
    assert seen[0].method == "DELETE"
    assert seen[0].url.path == "/queue/items/item-a"


def test_remove_item_sends_no_body_and_no_content_type() -> None:
    """An empty body must reach the bridge as "no body", not as an empty JSON
    document: FastAPI's ``Body | None = None`` parameters distinguish the two,
    and a stray ``content-type: application/json`` on an empty body is a 422."""
    app, seen, contents = _recording_app(200, {"removed": True, "item": None})

    with TestClient(app) as client:
        client.delete("/queue/items/item-a")

    assert contents[0] == b""
    assert "content-type" not in seen[0].headers


def test_start_queue_relays_success() -> None:
    app, seen, _ = _recording_app(200, {"started": True, "msg": ""})

    with TestClient(app) as client:
        response = client.post("/queue/start")

    assert response.status_code == 200
    assert response.json() == {"started": True, "msg": ""}
    assert seen[0].url.path == "/queue/start"


def test_stop_queue_relays_success_and_forwards_cancel_verbatim() -> None:
    app, seen, contents = _recording_app(200, {"stop_pending": False, "msg": ""})

    with TestClient(app) as client:
        response = client.post("/queue/stop", json={"cancel": True})

    assert response.status_code == 200
    assert response.json() == {"stop_pending": False, "msg": ""}
    assert json.loads(contents[0]) == {"cancel": True}


def test_stop_queue_with_no_body_forwards_no_body() -> None:
    """The bridge's ``POST /queue/stop`` takes an OPTIONAL body; a bodyless
    plain stop must stay bodyless rather than being filled in here."""
    app, _, contents = _recording_app(200, {"stop_pending": True, "msg": ""})

    with TestClient(app) as client:
        response = client.post("/queue/stop")

    assert response.status_code == 200
    assert contents[0] == b""


def test_item_uid_is_re_escaped_into_the_path_never_into_the_query() -> None:
    """A uid is request-supplied data, so it is quoted before interpolation:
    ``?``/``=``/space stay inside the path segment on the bridge rather than
    opening a query string the bridge would then read as parameters."""
    app, seen, _ = _recording_app(200, {"removed": True, "item": None})

    with TestClient(app) as client:
        response = client.delete("/queue/items/a%20b%3Fcancel%3Dtrue")

    assert response.status_code == 200
    # ``raw_path`` is the wire form; ``url.path`` would show it decoded and
    # hide whether the escaping survived.
    assert seen[0].url.raw_path == b"/queue/items/a%20b%3Fcancel%3Dtrue"
    assert seen[0].url.params.get("cancel") is None


def test_a_uid_bearing_an_encoded_slash_never_reaches_the_bridge() -> None:
    """Belt to the quoting brace: a ``%2F`` uid cannot open a second path
    segment because Starlette's ``{uid}`` convertor refuses it outright, so the
    request is a sidecar 404 and no bridge call is made at all."""
    app, seen, _ = _recording_app(200, {"removed": True, "item": None})

    with TestClient(app) as client:
        response = client.delete("/queue/items/weird%2Fstart")

    assert response.status_code == 404
    assert seen == []


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/queue/items", {"draft_revision": 1}),
        ("post", "/queue/items/item-a/move", {"pos_dest": "front"}),
        ("delete", "/queue/items/item-a", None),
        ("post", "/queue/start", None),
        ("post", "/queue/stop", {"cancel": False}),
    ],
)
def test_every_write_returns_502_when_the_bridge_is_unreachable(
    method: str, path: str, payload: dict | None
) -> None:
    app = _build_app(_refusing_handler)
    with TestClient(app) as client:
        response = client.request(method.upper(), path, json=payload)

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}


# ---------------------------------------------------------------------------
# Launch token: in-process only, on every write, never from the request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/queue/items", {"draft_revision": 1}),
        ("post", "/queue/items/item-a/move", {"pos_dest": "front"}),
        ("delete", "/queue/items/item-a", None),
        ("post", "/queue/start", None),
        ("post", "/queue/stop", {"cancel": True}),
        # The bridge gates the abort on nothing and ignores this header. It is
        # still on the list: an exception carved out here would be the same
        # per-route policy copy, pointed the other way.
        ("post", "/queue/abort", None),
    ],
)
def test_every_write_carries_the_resolved_token(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str, payload: dict | None
) -> None:
    """The token rides the WHOLE write surface, not just the routes the bridge
    currently gates -- keeping the gating policy in exactly one place (the
    bridge) instead of mirrored here, where it would rot."""
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    app, seen, _ = _recording_app(200, {})

    with TestClient(app) as client:
        client.request(method.upper(), path, json=payload)

    assert seen[0].headers["x-launch-token"] == TOKEN


def test_token_resolves_from_config_when_no_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Second resolver tier: ``bluesky.launch_token`` in config.yml, same as
    the launch route and the Bluesky MCP server use."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(f"bluesky:\n  launch_token: {TOKEN}\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)
    app, seen, _ = _recording_app(200, {"started": True, "msg": ""})

    with TestClient(app) as client:
        client.post("/queue/start")

    assert seen[0].headers["x-launch-token"] == TOKEN


def test_a_browser_supplied_token_is_never_forwarded() -> None:
    """The forwarded header set is built from scratch, so a token in the
    incoming request is dropped rather than proxied -- the browser cannot arm
    the bridge through this sidecar even by guessing the header name."""
    app, seen, _ = _recording_app(200, {"started": True, "msg": ""})

    with TestClient(app) as client:
        client.post("/queue/start", headers={"X-Launch-Token": "forged-by-the-browser"})

    assert "x-launch-token" not in seen[0].headers


def test_a_browser_supplied_token_cannot_override_the_resolved_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    app, seen, _ = _recording_app(200, {"started": True, "msg": ""})

    with TestClient(app) as client:
        client.post("/queue/start", headers={"X-Launch-Token": "forged-by-the-browser"})

    assert seen[0].headers["x-launch-token"] == TOKEN


def test_unarmed_deployment_still_reaches_the_bridge() -> None:
    """No local short-circuit. An unarmed enqueue onto
    an idle queue is a legitimate bridge-permitted flow (compose and queue now,
    arm and start later); refusing it here would break it. When the bridge does
    refuse, its own body is the answer the panel renders."""
    app, seen, _ = _recording_app(200, {"run_id": "run-1", "revision": 3, "item": {}})

    with TestClient(app) as client:
        response = client.post("/queue/items", json={"draft_revision": 3})

    assert response.status_code == 200
    assert len(seen) == 1
    assert "x-launch-token" not in seen[0].headers


def test_the_token_is_never_echoed_back_to_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    app, _, _ = _recording_app(200, {"started": True, "msg": ""})

    with TestClient(app) as client:
        response = client.post("/queue/start")

    assert TOKEN not in response.text
    assert TOKEN not in str(dict(response.headers))


# ---------------------------------------------------------------------------
# Refusals pass through verbatim -- body AND status, ``detail.code`` included
# ---------------------------------------------------------------------------


def test_start_unarmed_503_passes_through_with_its_code() -> None:
    refusal = {
        "detail": {
            "code": "launch_token_required",
            "detail": "starting the queue requires a launch token",
            "manager_state": "idle",
        }
    }
    app = _build_app(lambda request: httpx.Response(503, json=refusal))

    with TestClient(app) as client:
        response = client.post("/queue/start")

    assert response.status_code == 503
    assert response.json() == refusal
    assert response.json()["detail"]["code"] == "launch_token_required"


def test_start_token_mismatch_403_passes_through_with_its_code() -> None:
    refusal = {
        "detail": {
            "code": "launch_token_required",
            "detail": "the supplied launch token does not match this bridge",
            "manager_state": "executing_queue",
        }
    }
    app = _build_app(lambda request: httpx.Response(403, json=refusal))

    with TestClient(app) as client:
        response = client.post("/queue/start")

    assert response.status_code == 403
    assert response.json() == refusal
    assert response.json()["detail"]["code"] == "launch_token_required"


def test_abort_relays_the_bridge_success_body_verbatim() -> None:
    """The abort's body is richer than the other writes' and the panel reads
    every key of it (`abort_pending` drives what the operator is told), so a
    deep-equal assertion is what keeps the relay from quietly reshaping it."""
    body = {
        "aborted": True,
        "abort_pending": True,
        "paused_first": True,
        "manager_state": "paused",
        "msg": "plan aborted",
    }
    app, seen, contents = _recording_app(200, body)

    with TestClient(app) as client:
        response = client.post("/queue/abort", json={})

    assert response.status_code == 200
    assert response.json() == body
    assert seen[0].url.path == "/queue/abort"


def test_abort_nothing_running_409_passes_through_with_its_code() -> None:
    refusal = {
        "detail": {
            "code": "nothing_running",
            "detail": "No plan is running, so there is nothing to abort (manager state 'idle').",
        }
    }
    app = _build_app(lambda request: httpx.Response(409, json=refusal))

    with TestClient(app) as client:
        response = client.post("/queue/abort", json={})

    assert response.status_code == 409
    assert response.json() == refusal


def test_abort_pause_timeout_503_reaches_the_panel_saying_nothing_stopped() -> None:
    """The one refusal that must not be softened anywhere on the path: the
    plan may still be running, and the bridge's sentence is what says so."""
    refusal = {
        "detail": {
            "code": "abort_pause_timeout",
            "detail": (
                "The Run Engine did not reach the paused state an abort requires within "
                "10s, so NOTHING WAS ABORTED and the plan may still be running."
            ),
        }
    }
    app = _build_app(lambda request: httpx.Response(503, json=refusal))

    with TestClient(app) as client:
        response = client.post("/queue/abort", json={})

    assert response.status_code == 503
    assert response.json() == refusal
    assert "NOTHING WAS ABORTED" in response.json()["detail"]["detail"]


def _timeout_capturing_app(
    shared_timeout: float,
) -> tuple[FastAPI, list[dict]]:
    """An app whose shared client mirrors production's ``httpx.AsyncClient(timeout=15.0)``.

    ``_build_app`` leaves the client on httpx's own 5s default, which would make
    a "the abort is longer than the others" assertion pass for the wrong reason.
    Pinning the shared value here is what makes the comparison meaningful.
    """
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(dict(request.extensions.get("timeout", {})))
        return httpx.Response(200, json={"ok": True})

    app = FastAPI()
    app.include_router(queue_relay.router)
    app.state.bridge_url = _BRIDGE_URL
    app.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=shared_timeout
    )
    return app, captured


def test_abort_is_budgeted_above_the_bridges_composed_worst_case() -> None:
    """The shared 15s read timeout is shorter than the abort the bridge composes.

    The bridge's abort is ~10s of poll sleeps PLUS ~43 manager round trips, each
    bounded by the queueserver client's 2.0s recv timeout -- ~96s worst case (see
    ``queue_relay._ABORT_TIMEOUT``). Under the shared default, a slow-but-answering
    manager makes this relay answer 502 "bridge unreachable" while the bridge is
    still aborting: a false "the bridge is down" on the emergency path, which
    invites a retry race.
    """
    app, captured = _timeout_capturing_app(15.0)

    with TestClient(app) as client:
        assert client.post("/queue/abort", json={}).status_code == 200

    assert len(captured) == 1
    assert captured[0]["read"] == 120.0
    assert captured[0]["connect"] == 5.0


def test_other_queue_writes_keep_the_shared_timeout() -> None:
    """Negative control for the override above: it is per-request, not global.

    Every other queue write is a single manager call at the bridge, and must keep
    failing fast on a dead bridge rather than inheriting the abort's budget.
    """
    app, captured = _timeout_capturing_app(15.0)

    with TestClient(app) as client:
        assert client.post("/queue/start", json={}).status_code == 200
        assert client.post("/queue/stop", json={"cancel": False}).status_code == 200
        assert client.post("/queue/items", json={"revision": 1}).status_code == 200

    assert len(captured) == 3
    for seen in captured:
        assert seen["read"] == 15.0


def test_abort_still_relays_when_the_launch_token_cannot_be_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolution failure must not fail the halt.

    ``resolve_launch_token`` reads project config on the miss path; letting it
    raise would turn an unreadable config into a 500 on the emergency abort.
    The header is omitted instead, which IS the documented unarmed path.
    """

    def _explode() -> str:
        raise RuntimeError("no project config")

    monkeypatch.setattr(queue_relay, "resolve_launch_token", _explode)
    app, seen, _ = _recording_app(200, {"aborted": True, "abort_pending": False, "msg": ""})

    with TestClient(app) as client:
        response = client.post("/queue/abort", json={})

    assert response.status_code == 200
    assert "x-launch-token" not in {key.lower() for key in seen[0].headers}


def test_stop_cancel_unarmed_503_passes_through_with_its_code() -> None:
    """Withdrawing a pending stop is an arming action at the bridge; the
    sidecar never reads ``cancel`` and never anticipates the refusal."""
    refusal = {
        "detail": {
            "code": "launch_token_required",
            "detail": "withdrawing a pending stop requires a launch token",
            "manager_state": "executing_queue",
        }
    }
    app = _build_app(lambda request: httpx.Response(503, json=refusal))

    with TestClient(app) as client:
        response = client.post("/queue/stop", json={"cancel": True})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "launch_token_required"


def test_enqueue_browse_only_capability_409_passes_through_with_its_capability() -> None:
    """The whole capability record -- ``reason`` code and operator-facing
    ``detail`` with the flip command -- is what the panel banner renders. It
    survives the hop intact."""
    refusal = {
        "detail": {
            "code": "browse_only_connector",
            "detail": (
                "This deployment uses the mock connector, which cannot move hardware, so "
                "plans can be composed and validated but not executed. To execute "
                "plans, run `osprey set connector=virtual_accelerator` and "
                "redeploy."
            ),
            "capability": {
                "can_execute": False,
                "reason": "browse_only_connector",
                "detail": "This deployment uses the mock connector...",
            },
        }
    }
    app = _build_app(lambda request: httpx.Response(409, json=refusal))

    with TestClient(app) as client:
        response = client.post("/queue/items", json={"draft_revision": 3})

    assert response.status_code == 409
    assert response.json() == refusal
    assert response.json()["detail"]["capability"]["can_execute"] is False


def test_enqueue_stale_revision_409_passes_through_without_unwrapping() -> None:
    """Deliberately NOT unwrapped to a top-level ``code`` the way the retired
    launch relay did: the queue panel classifies on the bridge's own envelope
    (``detail.code``), so the relay keeps exactly one copy of it."""
    refusal = {
        "detail": {
            "code": "stale_draft_revision",
            "detail": "the draft moved on since revision 3",
            "revision": 5,
        }
    }
    app = _build_app(lambda request: httpx.Response(409, json=refusal))

    with TestClient(app) as client:
        response = client.post("/queue/items", json={"draft_revision": 3})

    assert response.status_code == 409
    assert response.json() == refusal
    assert "code" not in response.json()


@pytest.mark.parametrize(
    ("status_code", "code"),
    [
        (409, "draft_revision_already_launched"),
        (409, "session_plan_unvalidated"),
        (409, "session_plan_not_in_namespace"),
        (409, "queue_request_rejected"),
        (409, "unsupported_connector"),
        (409, "config_unreadable"),
        (503, "manager_not_configured"),
        (503, "manager_unreachable"),
        (503, "environment_unavailable"),
    ],
)
def test_every_refusal_code_reaches_the_caller_intact(status_code: int, code: str) -> None:
    refusal = {"detail": {"code": code, "detail": f"refused: {code}"}}
    app = _build_app(lambda request: httpx.Response(status_code, json=refusal))

    with TestClient(app) as client:
        response = client.post("/queue/items", json={"draft_revision": 3})

    assert response.status_code == status_code
    assert response.json() == refusal


def test_enqueue_refusal_extras_survive_the_hop() -> None:
    """``item_left_behind``/``item_uid`` are the ONLY witness that a refused
    unarmed enqueue failed to withdraw its own item. A relay that projected the
    refusal onto a fixed shape would drop them silently."""
    refusal = {
        "detail": {
            "code": "launch_token_required",
            "detail": "the queue became active during the add",
            "manager_state": "executing_queue",
            "item_left_behind": True,
            "item_uid": "item-stranded",
        }
    }
    app = _build_app(lambda request: httpx.Response(503, json=refusal))

    with TestClient(app) as client:
        response = client.post("/queue/items", json={"draft_revision": 3})

    assert response.json()["detail"]["item_left_behind"] is True
    assert response.json()["detail"]["item_uid"] == "item-stranded"


def test_successful_enqueue_carries_no_refusal_keys() -> None:
    """Negative control for the refusal assertions above: the relay does not
    manufacture a ``code`` on a success, so the panel's "is this a refusal?"
    test cannot be satisfied incidentally."""
    app = _build_app(
        lambda request: httpx.Response(200, json={"run_id": "run-1", "revision": 3, "item": {}})
    )

    with TestClient(app) as client:
        response = client.post("/queue/items", json={"draft_revision": 3})

    body = response.json()
    assert "code" not in body
    assert "detail" not in body


def test_a_non_json_bridge_body_does_not_become_a_500() -> None:
    app = _build_app(lambda request: httpx.Response(502, text="<html>gateway</html>"))

    with TestClient(app) as client:
        response = client.post("/queue/start")

    assert response.status_code == 502
    assert response.json() is None


# ---------------------------------------------------------------------------
# SSE relay: streams, disables the read timeout, strips hop-by-hop headers
# ---------------------------------------------------------------------------


def test_queue_events_streams_multiple_frames_incl_delayed_second_frame() -> None:
    frames = [
        b'event: message\ndata: {"type":"hello"}\n\n',
        b'event: message\ndata: {"type":"queue"}\n\n',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/queue/events"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DelayedFrames(frames),
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        with client.stream("GET", "/queue/events") as response:
            assert response.status_code == 200
            body = b"".join(response.iter_bytes())

    assert body == b"".join(frames)


def test_queue_events_disables_read_timeout() -> None:
    """The shared client's 15s read timeout would kill the stream during the
    bridge's ~15s heartbeat idle gaps; the per-request override is what makes
    the relay survive an idle queue."""
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
        with client.stream("GET", "/queue/events") as response:
            list(response.iter_bytes())

    assert len(captured) == 1
    assert captured[0]["read"] is None
    assert captured[0]["connect"] == 5.0


def test_queue_events_strips_hop_by_hop_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "connection": "keep-alive",
                "transfer-encoding": "chunked",
                "x-accel-buffering": "no",
            },
            stream=_DelayedFrames([b"data: 1\n\n"]),
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        with client.stream("GET", "/queue/events") as response:
            list(response.iter_bytes())
            headers = response.headers

    assert headers.get("x-accel-buffering") == "no"
    assert headers.get("connection") != "keep-alive"
    assert "chunked" not in (headers.get("transfer-encoding") or "")


def test_queue_events_relays_non_200_status() -> None:
    app = _build_app(
        lambda request: httpx.Response(
            503, headers={"content-type": "text/event-stream"}, stream=_DelayedFrames([])
        )
    )

    with TestClient(app) as client:
        with client.stream("GET", "/queue/events") as response:
            status = response.status_code
            list(response.iter_bytes())

    assert status == 503


def test_queue_events_never_sends_the_launch_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_DelayedFrames([b"data: 1\n\n"]),
        )

    app = _build_app(handler)
    with TestClient(app) as client:
        with client.stream("GET", "/queue/events") as response:
            list(response.iter_bytes())

    assert "x-launch-token" not in seen[0].headers


def test_queue_events_bridge_unreachable_returns_502() -> None:
    app = _build_app(_refusing_handler)
    with TestClient(app) as client:
        response = client.get("/queue/events")

    assert response.status_code == 502
    assert response.json() == {"detail": "bluesky bridge unreachable"}


# ---------------------------------------------------------------------------
# Route surface
# ---------------------------------------------------------------------------


def test_router_exposes_exactly_the_bridge_queue_surface() -> None:
    """Asserted through the OpenAPI schema, not ``app.routes``: since Starlette
    1.0 ``include_router`` does not flatten child routes into the parent's
    route list, so an ``app.routes`` scan would see none of these and pass
    vacuously."""
    app = FastAPI()
    app.include_router(queue_relay.router)
    paths = {path: set(operations.keys()) for path, operations in app.openapi()["paths"].items()}

    assert paths == {
        "/queue": {"get"},
        "/queue/items": {"post"},
        "/queue/items/{uid}/move": {"post"},
        "/queue/items/{uid}": {"delete"},
        "/queue/start": {"post"},
        "/queue/stop": {"post"},
        "/queue/abort": {"post"},
        "/queue/events": {"get"},
    }


# ---------------------------------------------------------------------------
# Composed app: queue paths wired, no collisions, /results mount alias
# ---------------------------------------------------------------------------


def test_queue_paths_are_wired_onto_the_composed_app() -> None:
    from osprey.interfaces.bluesky_web.app import app as composed_app

    paths = composed_app.openapi()["paths"]
    for path in (
        "/queue",
        "/queue/items",
        "/queue/items/{uid}/move",
        "/queue/items/{uid}",
        "/queue/start",
        "/queue/stop",
        "/queue/events",
    ):
        assert path in paths, f"queue relay path missing from the composed app: {path}"


def test_the_queue_relay_shadows_no_pre_existing_sidecar_route() -> None:
    """The sidecar already owns ``/health`` (its container healthcheck),
    ``/plans``, ``/runs*`` and ``/draft*``. A queue path that collided with one
    of those would silently take it over.

    ``/runs/launch`` is deliberately not on this list: there is no launch
    relay, and every bridge primitive it would call answers an unconditional
    410."""
    from osprey.interfaces.bluesky_web.app import app as composed_app

    paths = composed_app.openapi()["paths"]
    for path in ("/health", "/bridge/health", "/plans", "/runs", "/draft"):
        assert path in paths, f"queue relay displaced a pre-existing route: {path}"
    assert set(paths["/health"].keys()) == {"get"}


# ---------------------------------------------------------------------------
# The PLAN LANE axis on the write surface and the SSE stream
# ---------------------------------------------------------------------------
#
# One vocabulary with the read proxy, by import: the same `?lane=` parameter,
# the same resolver against `app.state.bridge_urls`, the same 404 body for a
# lane this deployment does not render. The claims pinned here are the two
# that matter for arming hardware: a write addressed to a lane reaches THAT
# lane's bridge carrying THAT lane's token (URL and credential travel as one
# decision), and an unknown lane is refused without ever falling back to lane
# 1 -- the wrong-machine relay the axis exists to remove.

_VA_BRIDGE_URL = "http://bridge-va.test"
VA_TOKEN = "va-launch-token"  # noqa: S105 - test fixture value, not a real secret


def _two_lane_app(
    handler: Callable[[httpx.Request], httpx.Response],
) -> FastAPI:
    """A sidecar app shaped like a two-lane deployment's after its lifespan."""
    app = _build_app(handler)
    app.state.bridge_urls = {"bluesky": _BRIDGE_URL, "bluesky_va": _VA_BRIDGE_URL}
    return app


def _two_lane_recording_app() -> tuple[FastAPI, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"success": True})

    return _two_lane_app(handler), seen


def test_a_write_addressed_to_a_lane_reaches_that_lanes_bridge_with_its_own_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """URL and credential travel as one decision.

    The request must never reach one lane's bridge carrying another lane's
    arming proof -- a launch a human approved against one machine, replayed
    against the other, is exactly what per-lane tokens exist to prevent.
    """
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    monkeypatch.setenv("BLUESKY_VA_LAUNCH_TOKEN", VA_TOKEN)
    app, seen = _two_lane_recording_app()

    with TestClient(app) as client:
        response = client.post("/queue/start?lane=bluesky_va")

    assert response.status_code == 200
    assert len(seen) == 1
    assert str(seen[0].url).startswith(_VA_BRIDGE_URL)
    assert seen[0].headers["X-Launch-Token"] == VA_TOKEN


def test_a_lane_one_write_still_carries_lane_ones_token_on_a_two_lane_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", TOKEN)
    monkeypatch.setenv("BLUESKY_VA_LAUNCH_TOKEN", VA_TOKEN)
    app, seen = _two_lane_recording_app()

    with TestClient(app) as client:
        bare = client.post("/queue/start")
        named = client.post("/queue/start?lane=bluesky")

    assert bare.status_code == named.status_code == 200
    for request in seen:
        assert str(request.url).startswith(_BRIDGE_URL)
        assert request.headers["X-Launch-Token"] == TOKEN


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/queue/items"),
        ("POST", "/queue/items/item-a/move"),
        ("DELETE", "/queue/items/item-a"),
        ("POST", "/queue/start"),
        ("POST", "/queue/stop"),
        ("POST", "/queue/abort"),
    ],
)
def test_every_write_refuses_an_unknown_lane_without_touching_a_bridge(
    method: str, path: str
) -> None:
    """404 with the read proxy's body -- never a relay from lane 1.

    The emergency abort is deliberately in this list: an abort addressed to a
    lane that does not exist aborts NOTHING, and relaying it to lane 1 would
    halt a different machine than the caller named.
    """
    from osprey.interfaces.bluesky_web.read_proxy import UNKNOWN_LANE_BODY

    app, seen, _ = _recording_app(200, {"success": True})
    with TestClient(app) as client:
        response = client.request(method, f"{path}?lane=bluesky_va")

    assert response.status_code == 404
    assert response.json() == UNKNOWN_LANE_BODY
    assert seen == []


def test_a_two_lane_app_still_refuses_the_lane_it_does_not_render() -> None:
    from osprey.interfaces.bluesky_web.read_proxy import UNKNOWN_LANE_BODY

    app, seen = _two_lane_recording_app()
    with TestClient(app) as client:
        response = client.post("/queue/start?lane=bluesky_live")

    assert response.status_code == 404
    assert response.json() == UNKNOWN_LANE_BODY
    assert seen == []


def test_the_lane_parameter_is_consumed_never_forwarded_to_the_bridge() -> None:
    """The bridge has no parameter by that name; every other param survives."""
    app, seen = _two_lane_recording_app()

    with TestClient(app) as client:
        response = client.post("/queue/stop?lane=bluesky_va&foo=1")

    assert response.status_code == 200
    assert seen[0].url.params.get("foo") == "1"
    assert "lane" not in seen[0].url.params


def test_an_empty_lane_parameter_names_no_lane_on_the_write_surface() -> None:
    """``?lane=`` behaves as ``?lane`` omitted, exactly as the read proxy reads it."""
    app, seen, _ = _recording_app(200, {"success": True})
    with TestClient(app) as client:
        response = client.post("/queue/start?lane=")

    assert response.status_code == 200
    assert str(seen[0].url).startswith(_BRIDGE_URL)


def test_queue_events_streams_from_the_named_lanes_bridge() -> None:
    """A queue stream labelled with the wrong machine is a wrong-machine write
    waiting to happen -- the SSE hop takes the lane like every write does."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"type": "hello"}\n\n',
        )

    app = _two_lane_app(handler)
    with TestClient(app) as client:
        response = client.get("/queue/events?lane=bluesky_va")

    assert response.status_code == 200
    assert str(seen[0].url) == f"{_VA_BRIDGE_URL}/queue/events"


def test_queue_events_refuses_an_unknown_lane() -> None:
    from osprey.interfaces.bluesky_web.read_proxy import UNKNOWN_LANE_BODY

    app, seen, _ = _recording_app(200, {})
    with TestClient(app) as client:
        response = client.get("/queue/events?lane=bluesky_va")

    assert response.status_code == 404
    assert response.json() == UNKNOWN_LANE_BODY
    assert seen == []
