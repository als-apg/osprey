"""The dispatch worker on the bus: its HTTP routes, answered over ZeroMQ.

Each bus request is served by the very route function that serves it over
HTTP (:mod:`~osprey.mcp_server.dispatch_worker.dispatch_api`), so the denylist,
the input-file policy, the agent record and every status body stay in one
place. An ``HTTPException`` a route raises becomes an ``ERR`` with the same
status and detail, which the dispatcher maps to the same exception the HTTP
client would raise.

The bearer token does not apply here: on the bus the dispatcher is
authenticated by its CURVE key before any request arrives.

A worker on the bus opens no port, so its container healthcheck reads the
file named by :data:`ALIVE_FILE_ENV`, which every heartbeat touches.
"""

from __future__ import annotations

import asyncio
import functools
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from osprey.dispatch.bus import protocol
from osprey.dispatch.bus.link import Handler, WorkerLink
from osprey.dispatch.bus.settings import WorkerSettings
from osprey.mcp_server.dispatch_worker import dispatch_api

#: The file every heartbeat touches, for the container healthcheck.
ALIVE_FILE_ENV = "DISPATCH_BUS_ALIVE_FILE"
#: The most runs this worker takes at once; unset means no bound of its own.
CAPACITY_ENV = "DISPATCH_WORKER_CAPACITY"


def _translating(
    fn: Callable[[dict[str, Any], bytes | None], Awaitable[tuple[Any, bytes | None]]],
) -> Handler:
    """Answer a route's ``HTTPException`` as ``ERR`` with its status and detail."""

    @functools.wraps(fn)
    async def handler(body: dict[str, Any], blob: bytes | None) -> tuple[Any, bytes | None]:
        try:
            return await fn(body, blob)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else ""
            raise protocol.BusError(exc.status_code, detail) from None

    return handler


def _run_id(body: dict[str, Any]) -> str:
    run_id = body.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise protocol.BusError(400, "run_id is required")
    return run_id


@_translating
async def _dispatch(body: dict[str, Any], _blob: bytes | None) -> tuple[Any, None]:
    try:
        request = dispatch_api.DispatchRequest.model_validate(body.get("request"))
    except ValidationError:
        # FastAPI answers an invalid body 422; so does the bus, without echoing it.
        raise protocol.BusError(422, "invalid dispatch request") from None
    response = await dispatch_api.dispatch(request)
    return response.model_dump(), None


@_translating
async def _status(body: dict[str, Any], _blob: bytes | None) -> tuple[Any, None]:
    return await dispatch_api.get_dispatch_result(_run_id(body)), None


@_translating
async def _cancel(body: dict[str, Any], _blob: bytes | None) -> tuple[Any, None]:
    return await dispatch_api.cancel_dispatch(_run_id(body)), None


@_translating
async def _runs(_body: dict[str, Any], _blob: bytes | None) -> tuple[Any, None]:
    return await dispatch_api.dashboard_runs(), None


@_translating
async def _artifacts(body: dict[str, Any], _blob: bytes | None) -> tuple[Any, None]:
    return await dispatch_api.list_dispatch_artifacts(_run_id(body)), None


@_translating
async def _artifact(body: dict[str, Any], _blob: bytes | None) -> tuple[Any, bytes]:
    artifact_id = body.get("artifact_id")
    if not isinstance(artifact_id, str):
        raise protocol.BusError(400, "artifact_id is required")
    ref = await dispatch_api.resolve_single_run_artifact(_run_id(body), artifact_id)
    if ref is None:
        # The route's constant 404, for every deny reason alike.
        raise protocol.BusError(404, "artifact not found for run")
    data = await asyncio.to_thread(Path(ref.abs_path).read_bytes)
    meta = {"mime": ref.mime_type or "application/octet-stream", "filename": ref.filename}
    return meta, data


HANDLERS: dict[bytes, Handler] = {
    protocol.DISPATCH: _dispatch,
    protocol.STATUS: _status,
    protocol.CANCEL: _cancel,
    protocol.RUNS: _runs,
    protocol.ARTIFACTS: _artifacts,
    protocol.ARTIFACT: _artifact,
}


def _capacity() -> int | None:
    raw = (os.environ.get(CAPACITY_ENV) or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


def _hello() -> dict[str, Any]:
    """The state each heartbeat carries; touching the alive file on the way."""
    alive = (os.environ.get(ALIVE_FILE_ENV) or "").strip()
    if alive:
        Path(alive).touch()
    return {
        "boot_nonce": dispatch_api._BOOT_NONCE,
        "capabilities": list(dispatch_api._CAPABILITIES),
        "in_flight": sorted(dispatch_api._in_flight_run_ids()),
        "capacity": _capacity(),
    }


def make_link(settings: WorkerSettings) -> WorkerLink:
    """This worker's link to the dispatcher, serving :data:`HANDLERS`."""
    return WorkerLink(
        settings.connect,
        settings.name,
        HANDLERS,
        hello=_hello,
        keys=settings.keys,
        max_request_bytes=dispatch_api.MAX_REQUEST_BYTES,
    )


async def serve(settings: WorkerSettings) -> None:
    """Run the worker on the bus until cancelled: the app's lifespan, then the link."""
    link = make_link(settings)
    dispatch_api.event_tap = link.emit
    try:
        async with dispatch_api.app.router.lifespan_context(dispatch_api.app):
            await link.run()
    finally:
        dispatch_api.event_tap = None
