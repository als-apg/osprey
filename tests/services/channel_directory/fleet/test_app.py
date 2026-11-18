"""The fleet service app: supervision of the refreshers (D3) and the read-only C5 routes."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("osprey_connectors.service_answers")

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from osprey.services.channel_directory.fleet import __main__ as fleet_main
from osprey.services.channel_directory.fleet.app import (
    RECYCLE_AFTER,
    Backoff,
    Supervisor,
    SupervisorStatus,
    create_app,
    refresher_command,
    unanswered,
)
from osprey.services.channel_directory.fleet.refresher import (
    EXIT_CANNOT_BUILD,
    EXIT_CONNECTOR_LOST,
    EXIT_NOT_REFRESHED,
)
from osprey.services.channel_directory.fleet.settings import NOT_REFRESHED_IN_PROCESS
from osprey.services.channel_directory.fleet.snapshot import (
    Liveness,
    Outcome,
    ReadPathStatus,
    ServerFleetRow,
    Snapshot,
    write_snapshot,
)
from osprey.services.channel_directory.fleet.states import LivenessKind, ReadPath, State
from osprey.services.channel_directory.resolve import SectionError

#: A live machine on EPICS with a served virtual accelerator and a stand-in beside it.
_LIVE_CS = {
    "type": "epics",
    "connector": {
        "epics": {"timeout_s": 4},
        "virtual_accelerator": {"port": 5064, "timeout_s": 2},
        "live_standin": {"port": 5070},
    },
}

START = 1_000_000.0


def _config(fleet: dict[str, Any] | None = None, control_system: Any = None) -> dict[str, Any]:
    section: dict[str, Any] = {}
    if fleet is not None:
        section["fleet"] = fleet
    return {
        "control_system": _LIVE_CS if control_system is None else control_system,
        "channel_directory": section,
    }


def _row(name: str, state: State) -> ServerFleetRow:
    return ServerFleetRow(
        name=name,
        host="host-1",
        class_name="default",
        group=None,
        state=state,
        since=START,
        reason="test",
        findings=(),
        directory_status=None,
        listed=True,
        sources=(),
        liveness=Liveness(channel=f"{name}:HEARTBEAT", kind=LivenessKind.COUNTER),
    )


def _snapshot(
    target: str = "live",
    *,
    generated_at: float = START,
    probe: ReadPath = ReadPath.OK,
    states: tuple[State, ...] = (State.ALIVE,),
) -> Snapshot:
    return Snapshot.ok(
        target=target,
        generated_at=generated_at,
        cycle=1,
        period_s=10.0,
        cycle_duration_s=0.5,
        read_path=ReadPathStatus(
            status=probe, channel=None if probe is ReadPath.UNVERIFIED else "SR:PROBE"
        ),
        sources=(),
        group_by="host",
        servers=[_row(f"ioc-{i}", state) for i, state in enumerate(states)],
    )


# --- Fakes ------------------------------------------------------------------------------


class FakeProcess:
    """A refresher process the test ends by hand, or that exits with a set status."""

    _pids = iter(range(1000, 100_000))

    def __init__(self, exit_code: int | None = None) -> None:
        self.pid = next(self._pids)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._done = asyncio.Event()
        if exit_code is not None:
            self.exit(exit_code)

    def exit(self, code: int) -> None:
        self.returncode = code
        self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.exit(-15)

    def kill(self) -> None:
        self.killed = True
        self.exit(-9)


class Spawner:
    """Hands out scripted processes: each entry is an exit code, or ``None`` to stay up."""

    def __init__(self, script: list[int | None]) -> None:
        self.script = list(script)
        self.started: list[FakeProcess] = []
        self.targets: list[str] = []

    async def __call__(self, target: str) -> FakeProcess:
        self.targets.append(target)
        code = self.script.pop(0) if self.script else None
        process = FakeProcess(code)
        self.started.append(process)
        return process


class Sleeps:
    """Records each backoff delay; stops the supervisor after *limit* of them."""

    def __init__(self, limit: int) -> None:
        self.delays: list[float] = []
        self.limit = limit

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        if len(self.delays) >= self.limit:
            raise asyncio.CancelledError
        await asyncio.sleep(0)


def _supervisor(tmp_path: Path, spawner: Spawner, sleeps: Sleeps, **kw: Any) -> Supervisor:
    options: dict[str, Any] = {
        "spawn": spawner,
        "state_dir": tmp_path,
        "poll_s": 0.01,
        "backoff_initial_s": 2.0,
        "backoff_cap_s": 16.0,
        "restart_delay_s": 1.0,
        "grace_s": 0.05,
        "sleep": sleeps,
        "clock": lambda: START - 100.0,
    }
    options.update(kw)
    return Supervisor("live", **options)


async def _run(supervisor: Supervisor, timeout: float = 5.0) -> None:
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(supervisor.run(), timeout)


# --- Backoff and the unanswered rule ---------------------------------------------------


def test_backoff_doubles_to_its_cap_and_resets() -> None:
    backoff = Backoff(2.0, 10.0)
    assert [backoff.next() for _ in range(5)] == [2.0, 4.0, 8.0, 10.0, 10.0]
    backoff.reset()
    assert backoff.next() == 2.0


class TestUnanswered:
    def test_probe_down_is_unanswered(self) -> None:
        assert unanswered(_snapshot(probe=ReadPath.DOWN, states=(State.UNKNOWN,)))

    def test_probe_ok_is_answered_even_when_every_server_is_quiet(self) -> None:
        assert not unanswered(_snapshot(states=(State.SILENT, State.UNREACHED)))

    def test_without_probe_every_remembered_server_quiet_is_unanswered(self) -> None:
        quiet = (State.SILENT, State.UNKNOWN, State.UNREACHED)
        assert unanswered(_snapshot(probe=ReadPath.UNVERIFIED, states=quiet))

    def test_without_probe_one_answering_server_is_answered(self) -> None:
        states = (State.SILENT, State.STALLED)
        assert not unanswered(_snapshot(probe=ReadPath.UNVERIFIED, states=states))

    def test_without_probe_and_without_servers_nothing_is_judged(self) -> None:
        assert not unanswered(_snapshot(probe=ReadPath.UNVERIFIED, states=()))

    def test_a_non_ok_snapshot_is_not_judged(self) -> None:
        not_ok = Snapshot.not_ok(Outcome.CANNOT_LIST, target="live", reason="no listing")
        assert not unanswered(not_ok)


# --- Supervision (D3) -----------------------------------------------------------------------


class TestSupervision:
    async def test_connector_loss_restarts_the_refresher(self, tmp_path: Path) -> None:
        spawner = Spawner([EXIT_CONNECTOR_LOST, EXIT_CONNECTOR_LOST])
        sleeps = Sleeps(limit=2)
        supervisor = _supervisor(tmp_path, spawner, sleeps)

        await _run(supervisor)

        assert spawner.targets == ["live", "live"]
        assert sleeps.delays == [1.0, 1.0]
        assert supervisor.last_exit == EXIT_CONNECTOR_LOST

    async def test_a_crash_restarts_with_exponential_backoff(self, tmp_path: Path) -> None:
        spawner = Spawner([1, 1, 1])
        sleeps = Sleeps(limit=3)
        supervisor = _supervisor(tmp_path, spawner, sleeps)

        await _run(supervisor)

        assert len(spawner.started) == 3
        assert sleeps.delays == [2.0, 4.0, 8.0]

    async def test_cannot_build_backs_off_exponentially_to_the_cap(self, tmp_path: Path) -> None:
        reason = "cannot build the read-only connector: OSError: no route"
        write_snapshot(
            tmp_path, Snapshot.not_ok(Outcome.NOT_REFRESHED, target="live", reason=reason)
        )
        spawner = Spawner([EXIT_CANNOT_BUILD] * 5)
        sleeps = Sleeps(limit=5)
        supervisor = _supervisor(tmp_path, spawner, sleeps)

        await _run(supervisor)

        assert sleeps.delays == [2.0, 4.0, 8.0, 16.0, 16.0]
        assert supervisor.health()["reason"] == reason

    async def test_not_refreshed_exit_is_not_restarted(self, tmp_path: Path) -> None:
        write_snapshot(
            tmp_path, Snapshot.not_ok(Outcome.NOT_REFRESHED, target="live", reason="in process")
        )
        spawner = Spawner([EXIT_NOT_REFRESHED])
        sleeps = Sleeps(limit=1)
        supervisor = _supervisor(tmp_path, spawner, sleeps)

        await asyncio.wait_for(supervisor.run(), 5.0)

        assert len(spawner.started) == 1
        assert sleeps.delays == []
        assert supervisor.status is SupervisorStatus.NOT_REFRESHED
        assert supervisor.reason == "in process"

    async def test_a_spawn_failure_backs_off_and_retries(self, tmp_path: Path) -> None:
        calls: list[str] = []

        async def failing(target: str) -> FakeProcess:
            calls.append(target)
            raise OSError("no python")

        sleeps = Sleeps(limit=2)
        supervisor = _supervisor(tmp_path, Spawner([]), sleeps, spawn=failing)

        await _run(supervisor)

        assert calls == ["live", "live"]
        assert sleeps.delays == [2.0, 4.0]

    async def _publish_cycles(
        self, tmp_path: Path, supervisor: Supervisor, snapshots: list[Snapshot]
    ) -> None:
        for snapshot in snapshots:
            write_snapshot(tmp_path, snapshot)
            for _ in range(200):
                await asyncio.sleep(0.005)
                if supervisor._last_seen == snapshot.generated_at:
                    break

    async def test_probe_unanswered_for_several_cycles_recycles_with_backoff(
        self, tmp_path: Path
    ) -> None:
        spawner = Spawner([None, None, None])
        sleeps = Sleeps(limit=2)
        supervisor = _supervisor(tmp_path, spawner, sleeps)
        task = asyncio.create_task(supervisor.run())
        down = ReadPath.DOWN
        unknown = (State.UNKNOWN,)

        await asyncio.sleep(0.02)
        await self._publish_cycles(
            tmp_path,
            supervisor,
            [
                _snapshot(generated_at=START + i, probe=down, states=unknown)
                for i in range(RECYCLE_AFTER)
            ],
        )
        await asyncio.sleep(0.05)
        first = spawner.started[0]
        assert first.terminated
        assert len(spawner.started) == 2
        assert sleeps.delays == [2.0]

        await self._publish_cycles(
            tmp_path,
            supervisor,
            [
                _snapshot(generated_at=START + 10 + i, probe=down, states=unknown)
                for i in range(RECYCLE_AFTER)
            ],
        )
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5.0)

        assert spawner.started[1].terminated
        assert sleeps.delays == [2.0, 4.0]

    async def test_an_answered_cycle_resets_the_count(self, tmp_path: Path) -> None:
        spawner = Spawner([None])
        sleeps = Sleeps(limit=1)
        supervisor = _supervisor(tmp_path, spawner, sleeps)
        task = asyncio.create_task(supervisor.run())
        down = ReadPath.DOWN

        await asyncio.sleep(0.02)
        cycles = [
            _snapshot(generated_at=START + 1, probe=down, states=(State.UNKNOWN,)),
            _snapshot(generated_at=START + 2, probe=down, states=(State.UNKNOWN,)),
            _snapshot(generated_at=START + 3),
            _snapshot(generated_at=START + 4, probe=down, states=(State.UNKNOWN,)),
            _snapshot(generated_at=START + 5, probe=down, states=(State.UNKNOWN,)),
        ]
        await self._publish_cycles(tmp_path, supervisor, cycles)
        await asyncio.sleep(0.05)

        assert not spawner.started[0].terminated
        assert supervisor.unanswered_cycles == 2
        assert supervisor.health()["status"] == "running"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert spawner.started[0].terminated

    async def test_without_probe_all_quiet_servers_recycle(self, tmp_path: Path) -> None:
        spawner = Spawner([None, None])
        sleeps = Sleeps(limit=1)
        supervisor = _supervisor(tmp_path, spawner, sleeps)
        task = asyncio.create_task(supervisor.run())

        await asyncio.sleep(0.02)
        quiet = (State.SILENT, State.UNKNOWN)
        await self._publish_cycles(
            tmp_path,
            supervisor,
            [
                _snapshot(generated_at=START + i, probe=ReadPath.UNVERIFIED, states=quiet)
                for i in range(RECYCLE_AFTER)
            ],
        )
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5.0)

        assert spawner.started[0].terminated
        assert sleeps.delays == [2.0]

    async def test_a_cycle_from_before_the_process_started_is_not_judged(
        self, tmp_path: Path
    ) -> None:
        write_snapshot(
            tmp_path,
            _snapshot(generated_at=START - 500, probe=ReadPath.DOWN, states=(State.UNKNOWN,)),
        )
        spawner = Spawner([None])
        supervisor = _supervisor(tmp_path, spawner, Sleeps(limit=1))
        task = asyncio.create_task(supervisor.run())
        await asyncio.sleep(0.05)

        assert supervisor.unanswered_cycles == 0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_a_refresher_that_ignores_terminate_is_killed(self, tmp_path: Path) -> None:
        class Stubborn(FakeProcess):
            def terminate(self) -> None:
                self.terminated = True

        process = Stubborn()

        async def spawn(_target: str) -> FakeProcess:
            return process

        supervisor = _supervisor(tmp_path, Spawner([]), Sleeps(limit=1), spawn=spawn)
        task = asyncio.create_task(supervisor.run())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert process.terminated and process.killed
        assert supervisor.status is SupervisorStatus.STOPPED


def test_refresher_command_runs_the_refresher_module(tmp_path: Path) -> None:
    command = refresher_command(tmp_path / "config.yml", "va", tmp_path / "state")

    assert command[:3] == [
        sys.executable,
        "-m",
        "osprey.services.channel_directory.fleet.refresher",
    ]
    assert command[3:] == [
        "--config",
        str(tmp_path / "config.yml"),
        "--target",
        "va",
        "--state-dir",
        str(tmp_path / "state"),
    ]


# --- The app ------------------------------------------------------------------------------


def _client(
    tmp_path: Path,
    fleet: dict[str, Any],
    *,
    control_system: Any = None,
    spawner: Spawner | None = None,
    supervise: bool = False,
) -> TestClient:
    app = create_app(
        _config(fleet, control_system),
        config_path=tmp_path / "config.yml",
        state_dir=tmp_path,
        spawn=spawner or Spawner([]),
        supervise=supervise,
        poll_s=0.01,
        supervisor_options={"grace_s": 0.05, "backoff_initial_s": 60.0},
    )
    return TestClient(app)


class TestRoutes:
    def test_a_named_target_with_no_file_yet_is_no_snapshot(self, tmp_path: Path) -> None:
        with _client(tmp_path, {"targets": ["live"]}) as client:
            answer = client.get("/v1/snapshot", params={"target": "live"})

        assert answer.status_code == 200
        body = answer.json()
        assert (body["outcome"], body["target"]) == ("no_snapshot", "live")
        assert body["reason"]

    def test_a_named_target_serves_its_file_unchanged(self, tmp_path: Path) -> None:
        snapshot = _snapshot("va", states=(State.ALIVE, State.SILENT))
        write_snapshot(tmp_path, snapshot)

        with _client(tmp_path, {"targets": ["live", "va"], "period_s": 10}) as client:
            body = client.get("/v1/snapshot", params={"target": "va"}).json()

        assert body == snapshot.to_json()
        assert Snapshot.from_json(body) == snapshot

    def test_a_cannot_build_file_is_served_as_written(self, tmp_path: Path) -> None:
        written = Snapshot.not_ok(Outcome.NOT_REFRESHED, target="live", reason="no connector")
        write_snapshot(tmp_path, written)

        with _client(tmp_path, {"targets": ["live"]}) as client:
            body = client.get("/v1/snapshot", params={"target": "live"}).json()

        assert body == written.to_json()

    def test_a_configured_target_not_named_is_not_refreshed(self, tmp_path: Path) -> None:
        # A file left over from an earlier configuration is not served.
        write_snapshot(tmp_path, _snapshot("standin"))

        with _client(tmp_path, {"targets": ["live"]}) as client:
            body = client.get("/v1/snapshot", params={"target": "standin"}).json()

        assert body["outcome"] == "not_refreshed"
        assert body["target"] == "standin"
        assert "channel_directory.fleet.targets" in body["reason"]

    def test_a_target_the_deployment_lacks_is_not_refreshed(self, tmp_path: Path) -> None:
        with _client(tmp_path, {"targets": ["live"]}) as client:
            body = client.get("/v1/snapshot", params={"target": "../etc"}).json()

        assert body["outcome"] == "not_refreshed"
        assert "not a target" in body["reason"]

    def test_a_named_in_process_target_is_not_refreshed_with_its_reason(
        self, tmp_path: Path
    ) -> None:
        with _client(tmp_path, {}, control_system={}) as client:
            body = client.get("/v1/snapshot", params={"target": "va"}).json()
            health = client.get("/health").json()

        assert (body["outcome"], body["reason"]) == ("not_refreshed", NOT_REFRESHED_IN_PROCESS)
        assert health["targets"]["va"] == {
            "status": "not_refreshed",
            "reason": NOT_REFRESHED_IN_PROCESS,
        }

    def test_the_target_parameter_is_required(self, tmp_path: Path) -> None:
        with _client(tmp_path, {"targets": ["live"]}) as client:
            assert client.get("/v1/snapshot").status_code == 422

    def test_no_route_accepts_a_write_method(self, tmp_path: Path) -> None:
        write_methods = {"POST", "PUT", "PATCH", "DELETE"}
        client = _client(tmp_path, {"targets": ["live"]})
        routes = [route for route in client.app.routes if isinstance(route, APIRoute)]

        assert {route.path for route in routes} >= {"/v1/snapshot", "/health"}
        for route in routes:
            assert not (route.methods & write_methods), route.path
        with client:
            for method in write_methods:
                for path in ("/v1/snapshot?target=live", "/health"):
                    assert client.request(method, path).status_code == 405, (method, path)


class TestHealth:
    def test_health_is_200_and_states_each_target(self, tmp_path: Path) -> None:
        spawner = Spawner([None, EXIT_CANNOT_BUILD])
        fleet = {"targets": ["live", "va"], "period_s": 10}

        with _client(tmp_path, fleet, spawner=spawner, supervise=True) as client:
            for _ in range(200):
                body = client.get("/health").json()
                statuses = {name: entry["status"] for name, entry in body["targets"].items()}
                if statuses == {"live": "running", "va": "cannot_build"}:
                    break
                time.sleep(0.01)
            answer = client.get("/health")

        assert answer.status_code == 200
        body = answer.json()
        assert body["status"] == "ok"
        assert body["targets"]["live"]["status"] == "running"
        assert body["targets"]["live"]["pid"] == spawner.started[0].pid
        assert body["targets"]["va"]["status"] == "cannot_build"
        assert sorted(spawner.targets[:2]) == ["live", "va"]
        # Leaving the app stops every refresher it started.
        assert spawner.started[0].terminated

    def test_health_without_supervision_states_targets_as_starting(self, tmp_path: Path) -> None:
        with _client(tmp_path, {"targets": ["live"]}) as client:
            answer = client.get("/health")

        assert answer.status_code == 200
        assert answer.json()["targets"]["live"]["status"] == "starting"


class TestConfig:
    def test_a_config_without_the_fleet_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SectionError) as raised:
            create_app(_config(None), config_path=tmp_path / "c.yml", state_dir=tmp_path)
        assert raised.value.key == "channel_directory.fleet"

    def test_a_malformed_fleet_is_refused_naming_its_key(self, tmp_path: Path) -> None:
        with pytest.raises(SectionError) as raised:
            create_app(
                _config({"targets": ["nowhere"]}),
                config_path=tmp_path / "c.yml",
                state_dir=tmp_path,
            )
        assert raised.value.key == "channel_directory.fleet.targets"

    def test_only_refreshed_targets_are_supervised(self, tmp_path: Path) -> None:
        app = create_app(
            _config({}, control_system={}), config_path=tmp_path / "c.yml", state_dir=tmp_path
        )
        assert app.state.supervisors == {}


class TestMain:
    def test_the_port_is_the_configured_one_else_the_layout_slot(self) -> None:
        from osprey.server_fleet_connection import DEFAULT_SERVER_FLEET_PORT

        assert fleet_main.service_port({"services": {"server_fleet": {"port": 9123}}}) == 9123
        assert fleet_main.service_port({}) == DEFAULT_SERVER_FLEET_PORT

    def test_main_serves_the_app_on_the_resolved_port(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import uvicorn

        import osprey_connectors.config as connectors_config

        config = _config({"targets": ["live"]})
        served: dict[str, Any] = {}
        monkeypatch.setattr(connectors_config, "load_config", lambda path: config)
        monkeypatch.setattr(
            uvicorn,
            "run",
            lambda app, host, port, log_level: served.update(app=app, host=host, port=port),
        )
        (tmp_path / "config.yml").write_text("{}\n")

        fleet_main.main(["--config", str(tmp_path / "config.yml"), "--state-dir", str(tmp_path)])

        assert served["host"] == "127.0.0.1"
        assert served["port"] == fleet_main.service_port(config)
        assert set(served["app"].state.supervisors) == {"live"}

    def test_main_exits_2_on_a_config_without_the_fleet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import osprey_connectors.config as connectors_config

        monkeypatch.setattr(connectors_config, "load_config", lambda path: _config(None))

        with pytest.raises(SystemExit) as raised:
            fleet_main.main(["--config", str(tmp_path / "c.yml"), "--state-dir", str(tmp_path)])
        assert raised.value.code == 2
