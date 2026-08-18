"""Sidecar relay for the Bluesky bridge's plan-queue endpoints.

This module defines ``router`` only; ``osprey.interfaces.bluesky_web.app``
mounts it alongside ``read_proxy`` and ``draft_relay``, so every route here
reads the shared ``httpx.AsyncClient`` and resolved bridge base URL from
``request.app.state`` at request time (set in the app's lifespan) -- the same
pattern as those two.

The relayed surface mirrors the bridge's queue contract one-for-one:

- ``GET /queue`` -> ``{status, items, running_item}``
- ``POST /queue/items`` -> enqueue the shared draft at a pinned revision
- ``POST /queue/items/{uid}/move`` -> reorder one queued item
- ``DELETE /queue/items/{uid}`` -> drop one queued item
- ``POST /queue/start`` -> start draining the queue
- ``POST /queue/stop`` -> stop after the running item (``cancel: true`` withdraws)
- ``POST /queue/abort`` -> abort the plan already in motion
- ``GET /queue/events`` -> the queue's Server-Sent-Events stream

**This sidecar holds no queue state and makes no queue policy.** It does not
know which routes arm hardware, does not pre-check capability, and does not
reshape a refusal. The bridge owns all of that; every response -- success or
refusal -- is relayed with its body AND status code untouched. Concretely,
the bridge's refusal envelope ``{"detail": {"code": ..., "detail": ...,
...extras}}`` arrives at the panel exactly as the bridge minted it, so a panel
classifies on ``detail.code`` (``launch_token_required``,
``stale_draft_revision``, ``draft_revision_already_launched``,
``session_plan_unvalidated``, ``browse_only_connector``, ``queue_request_rejected``,
...). Nothing here unwraps that envelope: the discriminator a panel branches on
is the bridge's, and a relay that reshaped it would be a second, drifting copy
of the contract.

The launch token is the one thing this module adds to a request. It is
resolved in-process on every write via the shared
``osprey.bluesky_bridge_connection.resolve_launch_token`` -- the same resolver
the Bluesky MCP server uses, so the two agent-facing paths to a bridge never
drift on which token arms which one. It is never accepted from the incoming request
(the forwarded header set is built from scratch, so a browser-supplied
``X-Launch-Token`` is dropped, not proxied) and never echoed back.

The token rides EVERY queue write, not only the two the bridge currently gates
(``POST /queue/start``, and ``POST /queue/stop`` with ``cancel: true``).
Deciding per-route which writes need arming would put a copy of the bridge's
gating policy here, to rot the first time the bridge tightens a gate; sending
the credential on the whole write surface and letting the bridge decide keeps
exactly one copy of that policy. That includes ``POST /queue/abort``, which the
bridge never gates and simply ignores the header on: carving out an exception
for it would be the same per-route policy copy, pointed the other way. Reads
(``GET /queue``, ``GET /queue/events``) never carry it -- no read is ever
gated, so there is nothing to arm.

Because the token rides the abort too, RESOLVING it must not be able to fail
the request: ``resolve_launch_token`` reads the project config, and an
unreadable config would otherwise turn every queue write -- including the
emergency halt -- into a 500. A resolution failure omits the header instead,
which is exactly the documented no-token path, and the bridge answers for
itself.

An unarmed deployment is NOT short-circuited here, deliberately: this relay
never refuses a write on its own reading of the arming state. A missing local
token means the header is simply omitted and the bridge answers for itself,
because an unarmed enqueue onto an idle queue is a legitimate, bridge-permitted
flow (compose and queue now, arm and start later) -- refusing it locally would
break the intended browse-and-queue path AND put a second, drifting copy of the
bridge's gating policy here. When the bridge does refuse, its ``503``/``403``
``launch_token_required`` body reaches the panel intact -- a far better signal
than a locally-synthesized one.

A connection-level failure to reach the bridge (refused connection, DNS
failure, timeout, ...) becomes a 502 with the same fixed detail body the other
relays use -- never an uncaught 500. HTTP-level error responses from the
bridge itself (403, 409, 503, ...) are not exceptions to httpx at all; they
flow straight through.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from osprey.bluesky_bridge_connection import resolve_launch_token
from osprey.interfaces.bluesky_web._shared import UNREACHABLE_BODY, safe_json

# The queue's read routes reuse ``read_proxy``'s single GET passthrough rather
# than growing a second one: that helper IS the sidecar's GET relay semantics
# (verbatim body + status, query params forwarded, 502 on an unreachable
# bridge), and a private-but-same-package import keeps the two from drifting.
# The write routes live here instead of there because ``read_proxy`` is
# GET-only by documented invariant.
from osprey.interfaces.bluesky_web.read_proxy import _forward_get
from osprey.utils.http_proxy import HOP_BY_HOP

router = APIRouter()

_LAUNCH_TOKEN_HEADER = "X-Launch-Token"

logger = logging.getLogger("osprey.interfaces.bluesky_web.queue_relay")


def _resolve_token_or_none() -> str | None:
    """``resolve_launch_token`` that degrades to "no token" instead of raising.

    The resolver reads the project config on the miss path, so an unreadable or
    absent config raises. Letting that propagate would fail EVERY queue write
    on a config problem -- including ``POST /queue/abort``, the emergency halt,
    which needs no token at all. Omitting the header is the already-documented
    unarmed path: the bridge refuses what needs arming and permits what does
    not, which is the correct answer either way.
    """
    try:
        return resolve_launch_token()
    except Exception:
        logger.warning(
            "could not resolve the bridge launch token; relaying this queue write "
            "without one and letting the bridge answer",
            exc_info=True,
        )
        return None


# The abort is the one queue write whose bridge-side work is a COMPOSITION of
# manager calls, so the shared client's 15s read timeout is far too short for
# it. Worst case, from the bridge's own constants
# (``QueueBackend.__init__``: ``abort_pause_polls=20``,
# ``abort_poll_interval=0.5``) and the queueserver zmq client's per-call recv
# timeout (``bluesky_queueserver_api`` default, 2.0s):
#
#     sleep budget      20 polls x 0.5s                              = 10s
#     manager calls     1 initial status + 1 initial re_pause
#                       + 20 poll statuses + 19 re_pause re-issues
#                       + 1 re_abort + 1 final status  = 43 round trips
#                       43 x 2.0s                                    = 86s
#     ------------------------------------------------------------------
#     true bound                                                     ~96s
#
# That is the pathological "manager answers, but only just inside its own
# timeout, every time" case; a manager that actually stops answering raises on
# the first call and the abort fails fast. 120s sits ~25% above the bound, so a
# slow-but-answering manager can never make this relay report the bridge as
# unreachable (502) while the bridge is mid-abort -- a false "bridge is down"
# on the emergency path, which invites exactly the retry race an abort must not
# have. Keep this in step with ``queue_backend.QueueBackend.abort``'s docstring
# and with the same override in the MCP ``stop_run`` tool.
_ABORT_TIMEOUT = httpx.Timeout(120.0, connect=5.0)


async def _forward_write(
    request: Request,
    method: str,
    path: str,
    *,
    timeout: httpx.Timeout | None = None,
) -> JSONResponse:
    """Relay ``method path`` onto the bridge and return its body/status verbatim.

    ``path`` must already be a properly-escaped bridge-relative path (path
    segments taken from the incoming request are quoted by the caller). The
    incoming request's raw body is forwarded byte-for-byte, never
    re-serialized, and its query params are forwarded unchanged.

    The forwarded header set is built from scratch: exactly the body's
    content-type (only when there IS a body -- an empty body must reach the
    bridge as "no body", not as an empty JSON document) plus the in-process
    launch token when one resolves. Nothing from the incoming request's
    headers is proxied, which is what makes a browser-supplied
    ``X-Launch-Token`` unusable rather than merely discouraged.

    ``timeout`` overrides the shared client's default for this one request, the
    way ``queue_events`` does for the SSE stream. Only the abort needs it; every
    other queue write is a single manager call and keeps the shared 15s.
    """
    client: httpx.AsyncClient = request.app.state.client
    bridge_url: str = request.app.state.bridge_url

    content = await request.body()
    headers: dict[str, str] = {}
    if content:
        headers["content-type"] = request.headers.get("content-type", "application/json")
    token = _resolve_token_or_none()
    if token:
        headers[_LAUNCH_TOKEN_HEADER] = token

    # Passed through only when set: httpx reads an explicit ``timeout=None`` as
    # "no timeout at all", not as "use the client's default".
    overrides: dict[str, Any] = {} if timeout is None else {"timeout": timeout}

    try:
        response = await client.request(
            method,
            f"{bridge_url}{path}",
            params=request.query_params,
            content=content or None,
            headers=headers,
            **overrides,
        )
    except httpx.RequestError:
        return JSONResponse(content=UNREACHABLE_BODY, status_code=502)

    return JSONResponse(content=safe_json(response), status_code=response.status_code)


@router.get("/queue")
async def get_queue(request: Request) -> JSONResponse:
    """Relay the queue snapshot: status summary, pending items, running item."""
    return await _forward_get(request, "/queue")


@router.post("/queue/items")
async def add_queue_item(request: Request) -> JSONResponse:
    """Relay an enqueue of the shared draft at a pinned revision.

    The enqueued plan comes from the bridge's own draft snapshot at the
    revision this body pins; nothing here reads or reshapes it. A capability
    refusal (browse-only deployment), a stale revision, or an unvalidated
    session plan is the bridge's 409/503 body, relayed intact.
    """
    return await _forward_write(request, "POST", "/queue/items")


@router.post("/queue/items/{uid}/move")
async def move_queue_item(request: Request, uid: str) -> JSONResponse:
    """Relay a reorder of one queued item."""
    return await _forward_write(request, "POST", f"/queue/items/{quote(uid, safe='')}/move")


@router.delete("/queue/items/{uid}")
async def remove_queue_item(request: Request, uid: str) -> JSONResponse:
    """Relay a removal of one queued item."""
    return await _forward_write(request, "DELETE", f"/queue/items/{quote(uid, safe='')}")


@router.post("/queue/start")
async def start_queue(request: Request) -> JSONResponse:
    """Relay a queue start -- the bridge's arming action.

    The resolved launch token rides along; whether it is accepted, and what a
    rejection looks like, is entirely the bridge's answer (``503``
    ``launch_token_required`` unarmed, ``403`` on a mismatch), relayed here
    with its body intact.
    """
    return await _forward_write(request, "POST", "/queue/start")


@router.post("/queue/stop")
async def stop_queue(request: Request) -> JSONResponse:
    """Relay a queue stop, including the token-gated ``cancel: true`` withdrawal.

    The sidecar does not read ``cancel`` -- the bridge's asymmetry (a plain
    stop is ungated, withdrawing a pending stop is an arming action) is the
    bridge's to enforce.
    """
    return await _forward_write(request, "POST", "/queue/stop")


@router.post("/queue/abort")
async def abort_running_plan(request: Request) -> JSONResponse:
    """Relay the emergency abort of the plan already in motion.

    The bridge gates this on nothing at all, and neither does the sidecar: the
    resolved token rides along like every other queue write (see the module
    docstring for why there is no per-route exception here) and the bridge
    ignores it. Every outcome is the bridge's -- the 409 ``nothing_running``,
    the 503 ``abort_pause_timeout`` that says plainly nothing was aborted, and
    the success body -- relayed with body and status untouched.

    The one thing this route does NOT share with its siblings is the read
    timeout: the bridge composes an abort out of many manager calls, so this
    request carries ``_ABORT_TIMEOUT`` (see its derivation above) instead of the
    shared client's 15s. Under the shared default a slow-but-answering manager
    would trip a 502 "bridge unreachable" while the bridge was still aborting.
    """
    return await _forward_write(request, "POST", "/queue/abort", timeout=_ABORT_TIMEOUT)


@router.get("/queue/events")
async def queue_events(request: Request) -> Response:
    """Relay the bridge's ``/queue/events`` SSE stream to the browser.

    The browser never talks to the bridge directly -- this sidecar is the sole
    hop. Uses a per-request ``httpx.Timeout(None, connect=5.0)`` to disable the
    shared client's 15s default read timeout, since the stream idles between
    the bridge's ~15s heartbeat comment frames. Mirrors
    ``draft_relay.draft_events``; a buffering relay would defeat the stream.
    """
    client: httpx.AsyncClient = request.app.state.client
    bridge_url: str = request.app.state.bridge_url

    try:
        upstream = await client.send(
            client.build_request(
                "GET",
                f"{bridge_url}/queue/events",
                timeout=httpx.Timeout(None, connect=5.0),
            ),
            stream=True,
        )
    except httpx.RequestError:
        return JSONResponse(content=UNREACHABLE_BODY, status_code=502)

    async def _stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP}

    return StreamingResponse(
        _stream(),
        status_code=upstream.status_code,
        headers=resp_headers,
        media_type="text/event-stream",
    )
