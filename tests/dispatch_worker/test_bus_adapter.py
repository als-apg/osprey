"""The dispatch worker's own routes, served over the bus.

The worker side is the real :mod:`~osprey.mcp_server.dispatch_worker.bus_adapter`
over the real route functions of ``dispatch_api``; only the agent run itself
(``sdk_runner.run_dispatch``) is a canned coroutine, as in ``test_dispatch_api``.
The dispatcher side is a real broker and client on loopback TCP. So these cases
pin that a run dispatched over the bus is validated, recorded, streamed,
cancelled and read back exactly as one posted over HTTP.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from osprey.agent_runner.tool_names import DISPATCH_DENIED_TOOLS
from osprey.dispatch.bus import protocol
from osprey.dispatch.bus.broker import WorkerBus
from osprey.dispatch.bus.client import BusWorkerClient
from osprey.dispatch.bus.settings import WorkerSettings
from osprey.dispatch.worker_client import WorkerRejectedRequestError
from osprey.mcp_server.dispatch_worker import bus_adapter, dispatch_api

_CANNED_RESULT: dict[str, Any] = {
    "status": "completed",
    "text_output": "ok",
    "tool_calls": [],
    "error": None,
    "duration_sec": 0.01,
    "cost_usd": 0.0,
    "num_turns": 1,
}


async def until(probe, timeout: float = 5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        value = await probe() if inspect.iscoroutinefunction(probe) else probe()
        if value:
            return value
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met in time")


@pytest.fixture
async def on_bus(monkeypatch, worker_record):
    """A worker on a live bus; yields the dispatcher's client and the run's release gate."""
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    release = asyncio.Event()
    release.set()

    async def fake_run_dispatch(*, event_queue, session_id=None, **_kwargs):
        await event_queue.put({"type": "text", "content": "working"})
        await release.wait()
        if session_id is not None:
            worker_record.answer(session_id, _CANNED_RESULT["text_output"])
        await event_queue.put({"type": "done"})
        return dict(_CANNED_RESULT)

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", fake_run_dispatch)
    monkeypatch.setattr(dispatch_api, "describe_run_artifacts", lambda run_id: [])
    monkeypatch.setattr(dispatch_api, "describe_run_input_artifacts", lambda run_id: [])

    bus = WorkerBus("tcp://127.0.0.1:*", keys=None, heartbeat_sec=0.1)
    await bus.start()
    link = bus_adapter.make_link(WorkerSettings(bus.endpoint, "dispatch-worker-1", None))
    monkeypatch.setattr(dispatch_api, "event_tap", link.emit)
    task = asyncio.create_task(link.run())
    try:
        await until(lambda: bus.live() == ["dispatch-worker-1"])
        yield BusWorkerClient(bus, worker_wait_sec=1.0), release
    finally:
        release.set()
        for run in list(dispatch_api._tasks.values()):
            run.cancel()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await bus.stop()


async def _events(client: BusWorkerClient, run_id: str) -> list[bytes]:
    return [c async for c in client.stream(run_id) if not c.startswith(b":")]


async def test_a_run_dispatched_over_the_bus_streams_and_reads_back_from_the_record(on_bus):
    client, _ = on_bus
    answer = await client.dispatch(prompt="do it", allowed_tools=["Read"])
    assert answer["status"] == "accepted"
    run_id = answer["run_id"]

    events = await asyncio.wait_for(_events(client, run_id), 5)
    assert events == [
        b'data: {"type": "text", "content": "working"}\n\n',
        b'data: {"type": "done"}\n\n',
    ]

    async def terminal():
        body = await client.status(run_id)
        return body if body.get("status") != "pending" else None

    body = await until(terminal)
    assert body["status"] == "completed"
    assert body["text_output"] == "ok"


async def test_the_run_list_and_a_runs_artifacts_read_back_over_the_bus(on_bus):
    client, _ = on_bus
    run_id = (await client.dispatch(prompt="do it", allowed_tools=["Read"]))["run_id"]
    await asyncio.wait_for(_events(client, run_id), 5)

    rows = await until(client.runs)
    assert run_id in [row["run_id"] for row in rows]
    assert await client.artifacts(run_id) == []


@pytest.mark.parametrize(
    ("kind", "body"),
    [
        (protocol.STATUS, {}),
        (protocol.CANCEL, {"run_id": ""}),
        (protocol.ARTIFACTS, {"run_id": 7}),
        (protocol.ARTIFACT, {"run_id": "r1"}),
    ],
    ids=["status", "cancel", "artifacts", "artifact"],
)
async def test_a_request_without_its_ids_is_a_bad_request(kind, body):
    with pytest.raises(protocol.BusError) as caught:
        await bus_adapter.HANDLERS[kind](body, None)
    assert caught.value.status == 400


def test_a_failing_event_tap_loses_no_event(monkeypatch):
    def unreachable(_run_id: str, _event: dict[str, Any]) -> None:
        raise RuntimeError("bus down")

    monkeypatch.setattr(dispatch_api, "event_tap", unreachable)
    queue = dispatch_api._RunEventQueue("r1")
    queue.put_nowait({"type": "text", "content": "kept"})
    assert queue.get_nowait() == {"type": "text", "content": "kept"}


async def test_a_denied_tool_is_refused_and_starts_no_run(on_bus):
    client, _ = on_bus
    denied = sorted(DISPATCH_DENIED_TOOLS)[0]
    with pytest.raises(WorkerRejectedRequestError):
        await client.dispatch(prompt="do it", allowed_tools=[denied])
    assert dispatch_api._runs == {}


async def test_a_malformed_dispatch_is_refused_without_echoing_it(on_bus):
    client, _ = on_bus
    with pytest.raises(WorkerRejectedRequestError) as caught:
        await client.dispatch(prompt="do it", allowed_tools="Read")  # type: ignore[arg-type]
    assert "Read" not in str(caught.value)
    assert dispatch_api._runs == {}


async def test_a_run_in_flight_is_in_the_heartbeat_and_cancels_over_the_bus(on_bus):
    client, release = on_bus
    release.clear()
    run_id = (await client.dispatch(prompt="do it", allowed_tools=["Read"]))["run_id"]
    viewer = asyncio.create_task(_events(client, run_id))
    await until(lambda: dispatch_api._tasks.get(run_id))

    hello = bus_adapter._hello()
    assert hello["in_flight"] == [run_id]
    assert hello["capabilities"] == dispatch_api._CAPABILITIES

    assert await client.cancel(run_id) == {"run_id": run_id, "cancelled": True}
    events = await asyncio.wait_for(viewer, 5)
    assert events[-1] == b'data: {"type": "error", "message": "cancelled by user"}\n\n'


async def test_an_artifact_is_served_from_the_store_and_a_foreign_one_is_not(
    on_bus, monkeypatch, tmp_path
):
    client, _ = on_bus
    png = tmp_path / "plot.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nbody")

    async def resolve(run_id, artifact_id):
        if (run_id, artifact_id) != ("r1", "a1"):
            return None
        return SimpleNamespace(abs_path=str(png), mime_type="image/png", filename="plot.png")

    monkeypatch.setattr(dispatch_api, "resolve_single_run_artifact", resolve)
    meta, data = await client.artifact("r1", "a1")
    assert meta == {"mime": "image/png", "filename": "plot.png"}
    assert data == png.read_bytes()
    with pytest.raises(WorkerRejectedRequestError):
        await client.artifact("r2", "a1")


async def test_the_alive_file_is_touched_by_every_heartbeat(monkeypatch, tmp_path):
    alive = tmp_path / "alive"
    monkeypatch.setenv(bus_adapter.ALIVE_FILE_ENV, str(alive))
    bus_adapter._hello()
    assert alive.exists()
