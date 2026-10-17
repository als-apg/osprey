"""The trajectory service's graceful shutdown, through a real in-process uvicorn server.

The server is :class:`~osprey.services.trajectory.__main__.TrajectoryServer`
over the real ``create_app()``, bound to a loopback socket and served on the
test's own loop, so the app's objects can be awaited directly. The connector
is the doors' stand-in machine with one write held open by a gate: the write
in flight when the shutdown begins.
"""

from __future__ import annotations

import asyncio
import json
import math
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from tests.services.trajectory.test_doors import (  # noqa: F401  (the autouse deployment)
    ALICE_SECRET,
    HCM1,
    HCM2,
    TOKEN,
    TOOL,
    _app,
    _as,
    _deployment,
    _Machine,
)

from osprey.services.trajectory import __main__ as entry
from osprey.services.trajectory.__main__ import TrajectoryServer
from osprey.services.trajectory.actions import SERVICE_STOPPING, ActionRefused
from osprey.services.trajectory.app import CLIENT_HEADER, LAUNCH_TOKEN_ENV
from osprey.services.trajectory.state import ShotResult

pytestmark = pytest.mark.no_auth_seam


class _Gate:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()


class _GatedMachine(_Machine):
    """A machine whose write to a gated address lands, then holds until released."""

    def __init__(self) -> None:
        super().__init__()
        self.gates: dict[str, _Gate] = {}

    async def write_channel_checked(self, address: str, value: Any, **kwargs: Any) -> Any:
        result = await super().write_channel_checked(address, value, **kwargs)
        gate = self.gates.get(address)
        if gate is not None:
            gate.started.set()
            await gate.release.wait()
        return result


async def _until(check: Any, timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not check():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


class _Served:
    def __init__(self, app: FastAPI, server: TrajectoryServer, task: asyncio.Task[Any], port: int):
        self.app = app
        self.server = server
        self.task = task
        self.base_url = f"http://127.0.0.1:{port}"


@pytest.fixture
def machine() -> _GatedMachine:
    return _GatedMachine()


@pytest.fixture
async def served(
    machine: _GatedMachine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[_Served]:
    monkeypatch.setenv(LAUNCH_TOKEN_ENV, TOKEN)
    from osprey.services.trajectory.store import STATE_DIR_ENV_VAR

    monkeypatch.setenv(STATE_DIR_ENV_VAR, str(tmp_path / "state"))
    app = _app(machine, tool=TOOL)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = TrajectoryServer(config, app)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        await _until(lambda: server.started or task.done(), 10)
        assert server.started, "the server did not start"
        yield _Served(app, server, task, port)
    finally:
        server.should_exit = True
        for gate in machine.gates.values():
            gate.release.set()
        await asyncio.wait_for(task, 30)
        sock.close()


def _put_back(*names: str) -> dict[str, Any]:
    return {"model": "SR", "values": [{"name": name, "before": 0.25} for name in names]}


def _code(response: httpx.Response) -> str:
    return response.json()["detail"]["code"]


async def test_the_drain_is_bounded_by_the_stop_budget(served: _Served) -> None:
    budget = served.app.state.sessions.budgets().stop_budget_s
    assert served.server.config.timeout_graceful_shutdown == math.ceil(budget)


async def test_shutdown_fails_queued_work_ends_streams_and_writes_nothing_after_the_in_flight_write(
    served: _Served, machine: _GatedMachine
) -> None:
    sessions = served.app.state.sessions
    gate = machine.gates.setdefault(HCM1, _Gate())
    async with (
        httpx.AsyncClient(base_url=served.base_url, timeout=30) as http,
        httpx.AsyncClient(base_url=served.base_url, timeout=30) as streaming,
    ):
        grant = (await http.post("/api/session", headers=_as(ALICE_SECRET, "/api"))).json()
        holder = _as(ALICE_SECRET, "/api", **{CLIENT_HEADER: grant["client_id"]})

        # An open state stream.
        lines: list[str] = []
        first_event = asyncio.Event()

        async def follow() -> None:
            async with streaming.stream(
                "GET", "/panel/api/events", headers=_as(ALICE_SECRET, "/panel/api")
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    lines.append(line)
                    if line.startswith("data: "):
                        first_event.set()

        stream = asyncio.create_task(follow())
        await asyncio.wait_for(first_event.wait(), 5)

        # The write in flight: a put-back of two settings, held after its first write.
        in_flight = asyncio.create_task(
            http.post("/api/apply", headers=holder, json=_put_back(HCM1, HCM2))
        )
        await asyncio.wait_for(gate.started.wait(), 5)

        # A continuous run, its first shot queued behind the write.
        shots_run: list[str] = []

        def build(_hook: Any) -> Any:
            async def shoot(_admission: Any, _stop: asyncio.Event) -> ShotResult:
                shots_run.append("shot")
                raise AssertionError("a shot queued at shutdown must not run")

            return shoot

        run = sessions.start_continuous(
            "alice", "agent", grant["client_id"], served.app.state.store, build
        )
        await _until(lambda: run.shots == 1)

        # A second apply, queued behind the write.
        queued = asyncio.create_task(http.post("/api/apply", headers=holder, json=_put_back(HCM2)))
        await _until(lambda: sessions.layer._queue.qsize() >= 2)

        served.server.should_exit = True
        await _until(lambda: sessions.layer.closed)

        # Submitted during the shutdown: refused at once, the write still held.
        during = await http.post("/api/apply", headers=holder, json=_put_back(HCM2))
        assert during.status_code == 503, during.text
        assert _code(during) == SERVICE_STOPPING

        refused = await asyncio.wait_for(queued, 5)
        assert refused.status_code == 503, refused.text
        assert _code(refused) == SERVICE_STOPPING

        await asyncio.wait_for(stream, 5)
        ended = await asyncio.wait_for(run.wait(), 5)
        assert ended.end == "shutdown"
        assert not sessions.continuous_running
        assert not in_flight.done()

        gate.release.set()
        answered = await asyncio.wait_for(in_flight, 10)
        assert answered.status_code == 200, answered.text
        assert answered.json()["stopped"] is True

    await asyncio.wait_for(served.task, 30)
    assert machine.writes == [(HCM1, 0.25)], "nothing is written after the in-flight write"
    assert shots_run == []
    data = [json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")]
    assert data, "the stream carried events before it ended"


async def test_shutdown_is_idempotent_and_refuses_a_continuous_start(served: _Served) -> None:
    sessions = served.app.state.sessions
    async with httpx.AsyncClient(base_url=served.base_url, timeout=30) as http:
        grant = (await http.post("/api/session", headers=_as(ALICE_SECRET, "/api"))).json()
    await asyncio.gather(sessions.shutdown(), sessions.shutdown())
    await sessions.shutdown()
    assert sessions.shutdown_signal.is_set()
    assert served.app.state.shutdown_signal is sessions.shutdown_signal
    assert sessions.layer.stop_event.is_set()
    with pytest.raises(ActionRefused) as caught:
        sessions.start_continuous(
            "alice", "agent", grant["client_id"], served.app.state.store, lambda _hook: None
        )
    assert caught.value.code == SERVICE_STOPPING
    assert caught.value.status == 503


def test_main_serves_the_app_through_the_trajectory_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[tuple[uvicorn.Config, FastAPI]] = []

    def run(self: TrajectoryServer) -> None:
        served.append((self.config, self.app))

    sentinel = FastAPI()
    monkeypatch.setattr(TrajectoryServer, "run", run)
    monkeypatch.setattr("osprey.services.trajectory.app.create_app", lambda: sentinel)
    monkeypatch.setattr("osprey_connectors.logger.configure_logging", lambda: None)
    monkeypatch.setattr(
        "osprey_connectors.workspace.load_osprey_config",
        lambda: {"services": {"trajectory": {"port": 18123}}},
    )
    monkeypatch.setenv(entry.BIND_ENV_VAR, "0.0.0.0")
    entry.main()
    ((config, app),) = served
    assert app is sentinel
    assert config.app is sentinel
    assert (config.host, config.port) == ("0.0.0.0", 18123)
