"""A notebook kernel takes its connector from a connector-host pool.

A kernel is re-stamped from the deployment's record before every cell, so its
stamp moves under it when the control target is switched. A Channel Access
client keeps the first context it created in a process, so a connector rebuilt
in the same process after a switch reports the new target while it still
reaches the old one's server. These tests pin the fix: a process that opted in
(:func:`osprey.runtime._route_connector_through_pool`) on a deployment that can
switch is served each stamp by a connector-host child of its own.

The failure shape is reproduced first with a fake whose servers name themselves,
so the regression below cannot pass vacuously. No Channel Access server and no
``epics`` import are involved; the last class drives real connector-host
children serving the pool tests' mock connector.
"""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import osprey.runtime as runtime
from osprey.runtime import ControlTargetChangedError, SwitchInProgressError
from osprey_connectors import control_context, posture_store
from osprey_connectors.control_system.base import ChannelWriteResult, WriteOutcome
from osprey_connectors.types import baseline_target, configured_targets, resolve_target
from tests._control_context_fixtures import state_dir_under
from tests.connectors.ipc.test_pool import PYTHONPATH, SERVED, SLOW
from tests.facility.served_tree import mock_config, served_tree
from tests.runtime.test_executor_target_stamp import stamp_env, write_record

#: A deployment that can switch: a virtual accelerator baseline and a stand-in.
#: Each block names the server a connector configured from it would reach.
SWITCHABLE = {
    "type": "virtual_accelerator",
    "connector": {
        "virtual_accelerator": {"server": "va-server"},
        "live_standin": {"server": "standin-server"},
    },
}

#: A deployment with one target, which cannot switch.
SINGLE = {"type": "mock", "connector": {"mock": {"server": "mock-server"}}}


def _serve(monkeypatch, section: dict[str, Any]) -> None:
    """Serve *section* as ``control_system`` to every reader in this process."""

    def get_config_value(path, default=None, _config_path=None):
        return section if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)


@pytest.fixture(autouse=True)
def runtime_state(monkeypatch):
    """A process that has not opted in, holds no connector and carries no stamp."""
    for name in (
        runtime.ENV_CONTROL_TARGET,
        runtime.ENV_CONTROL_TARGET_GENERATION,
        runtime.ENV_CONTROL_TARGET_REFUSAL,
        runtime.ENV_IN_CELL,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)

    def reset():
        runtime._runtime_connector = None
        runtime._connector_stamp = None
        runtime._connector_launch_pin = None
        runtime._connector_pool = None
        runtime._pool_routing = False
        runtime._cell_marker = None
        runtime._limits_validator = None

    reset()
    yield
    reset()
    loop = runtime._pool_loop
    runtime._pool_loop = None
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    """A throwaway agent-data root the write pin reads its record from."""
    root = tmp_path / "var" / "agent_data"
    state_dir_under(root).mkdir(parents=True)
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    control_context.invalidate_cache()
    yield root
    control_context.invalidate_cache()


# ---------------------------------------------------------------------------
# An in-process connector under the first-context rule
# ---------------------------------------------------------------------------

#: The server the first connector built in this "process" reached; every later
#: instance reads from it, whatever target it was built for.
_FIRST_SERVER: dict[str, str] = {}


class _FirstContextConnector:
    """A connector whose client keeps the first context the process created."""

    disconnected: list["_FirstContextConnector"] = []

    async def connect(self, config: dict[str, Any]) -> None:
        self.server = config["server"]
        _FIRST_SERVER.setdefault("server", self.server)

    async def read_channel(self, channel_address: str, timeout: float | None = None):  # noqa: ARG002 - a connector's read_channel takes a timeout
        return SimpleNamespace(value=f"{_FIRST_SERVER['server']}/{channel_address}")

    async def disconnect(self) -> None:
        type(self).disconnected.append(self)


@pytest.fixture
def in_process_registry():
    """Register the first-context fake under every type the sections select."""
    from osprey_connectors.factory import ConnectorFactory, isolated_connector_registries

    with isolated_connector_registries():
        for name in ("mock", "virtual_accelerator", "live_standin"):
            ConnectorFactory.register_control_system(name, _FirstContextConnector)
        _FIRST_SERVER.clear()
        _FirstContextConnector.disconnected = []
        yield
        _FIRST_SERVER.clear()
        _FirstContextConnector.disconnected = []


# ---------------------------------------------------------------------------
# A pool whose children each have a first context of their own
# ---------------------------------------------------------------------------


class _FakeHandle:
    """A pooled connector: one child, which reached its own target's server."""

    def __init__(self, pool: "_FakePool", target: str, server: str) -> None:
        self.pool = pool
        self.target = target
        self.server = server
        self.disconnected = False
        self.writes: list[tuple[str, Any, dict[str, Any]]] = []
        self.batches: list[tuple[list[tuple[str, Any]], dict[str, Any]]] = []

    async def read_channel(self, channel_address: str, timeout: float | None = None):  # noqa: ARG002 - a connector's read_channel takes a timeout
        self.pool.record_loop()
        return SimpleNamespace(value=f"{self.server}/{channel_address}")

    async def write_channel_checked(self, channel_address: str, value: Any, **kwargs: Any):
        self.pool.record_loop()
        self.writes.append((channel_address, value, kwargs))
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.CONFIRMED,
            observed_value=value,
        )

    async def write_multiple_channels(self, operations, timeout=None, confirm=None):
        self.pool.record_loop()
        self.batches.append((list(operations), {"timeout": timeout, "confirm": confirm}))
        return [
            ChannelWriteResult(
                channel_address=address, value_written=value, outcome=WriteOutcome.CONFIRMED
            )
            for address, value in operations
        ]

    async def disconnect(self) -> None:
        self.pool.record_loop()
        self.disconnected = True


class _FakePool:
    """Stands in for ``ConnectorHostPool``: a fresh child per ``connector()``."""

    instances: list["_FakePool"] = []

    def __init__(self, control_system, *, config_file=None) -> None:
        self.section = dict(control_system)
        self.config_file = config_file
        self.asked: list[tuple[str, str | None]] = []
        self.handles: list[_FakeHandle] = []
        self.loops: list[asyncio.AbstractEventLoop] = []
        self.closed_on: asyncio.AbstractEventLoop | None = None
        type(self).instances.append(self)

    def record_loop(self) -> None:
        self.loops.append(asyncio.get_running_loop())

    async def connector(self, target: str, *, execution_mode: str | None = None) -> _FakeHandle:
        self.record_loop()
        self.asked.append((target, execution_mode))
        connector_type = resolve_target(self.section, target)
        handle = _FakeHandle(self, target, self.section["connector"][connector_type]["server"])
        self.handles.append(handle)
        return handle

    async def close(self) -> None:
        self.record_loop()
        self.closed_on = asyncio.get_running_loop()


@pytest.fixture
def fake_pool(monkeypatch):
    """Replace the pool at the runtime's import seam; forbid an in-process build."""
    from osprey_connectors.factory import ConnectorFactory
    from osprey_connectors.ipc import pool as pool_module

    async def no_in_process_build(*args, **kwargs):
        raise AssertionError("an opted-in kernel built a connector in its own process")

    _FakePool.instances = []
    monkeypatch.setattr(pool_module, "ConnectorHostPool", _FakePool)
    monkeypatch.setattr(ConnectorFactory, "create_control_system_connector", no_in_process_build)
    yield _FakePool.instances
    _FakePool.instances = []


def inside_a_running_loop(call):
    """Run *call* synchronously from inside a coroutine, as a notebook cell runs."""

    async def driver():
        return call()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(driver())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# The failure shape, and the fix
# ---------------------------------------------------------------------------


class TestTheFirstContextRule:
    """Without the pool, a rebuilt connector reports one target and reads another."""

    @pytest.mark.usefixtures("in_process_registry")
    def test_a_rebuilt_connector_still_reads_the_first_server(self, monkeypatch):
        _serve(monkeypatch, SWITCHABLE)

        stamp_env(monkeypatch, target="standin", generation="0")
        assert runtime.read_channel("SR:X") == "standin-server/SR:X"

        stamp_env(monkeypatch, target="va", generation="1")
        value = runtime.read_channel("SR:X")

        assert runtime._connector_stamp == ("va", 1)
        assert runtime._runtime_connector._control_target == "va"
        assert value == "standin-server/SR:X"


class TestTheKernelPool:
    """Opted in, each stamp is served by a child of its own."""

    def test_a_switch_reads_the_new_targets_server(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()

        stamp_env(monkeypatch, target="standin", generation="0")
        assert inside_a_running_loop(lambda: runtime.read_channel("SR:X")) == (
            "standin-server/SR:X"
        )
        standin = runtime._runtime_connector

        stamp_env(monkeypatch, target="va", generation="1")
        value = inside_a_running_loop(lambda: runtime.read_channel("SR:X"))

        (pool,) = fake_pool
        assert runtime._runtime_connector.target == "va"
        assert value == "va-server/SR:X"
        assert standin.disconnected
        assert pool.asked == [("standin", None), ("va", None)]
        assert len(set(pool.loops)) == 1
        assert pool.loops[0] is runtime._pool_loop

    def test_both_loop_contexts_reach_the_same_pool_loop(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        stamp_env(monkeypatch, target="va", generation="0")

        runtime.read_channel("SR:A")
        inside_a_running_loop(lambda: runtime.read_channel("SR:B"))

        (pool,) = fake_pool
        assert len(pool.handles) == 1
        assert set(pool.loops) == {runtime._pool_loop}

    def test_the_pool_reads_the_kernels_own_config_file(self, monkeypatch, tmp_path, fake_pool):
        config = tmp_path / "config.yml"
        monkeypatch.setenv("OSPREY_CONFIG", str(config))
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()

        runtime.read_channel("SR:X")

        (pool,) = fake_pool
        assert pool.config_file == str(config)
        assert pool.section == SWITCHABLE

    def test_a_generation_move_on_the_same_target_respawns(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()

        stamp_env(monkeypatch, target="va", generation="0")
        runtime.read_channel("SR:X")
        first = runtime._runtime_connector
        stamp_env(monkeypatch, target="va", generation="1")
        runtime.read_channel("SR:X")

        assert first.disconnected
        assert runtime._runtime_connector is not first
        assert fake_pool[0].asked == [("va", None), ("va", None)]

    @pytest.mark.usefixtures("fake_pool")
    def test_a_launch_pin_move_alone_respawns(self, monkeypatch):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        stamp_env(monkeypatch, target="va", generation="0")

        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "va=readonly")
        runtime.read_channel("SR:X")
        first = runtime._runtime_connector
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "va=writes")
        runtime.read_channel("SR:X")

        assert first.disconnected
        assert runtime._runtime_connector is not first
        assert runtime._connector_launch_pin == "va=writes"

    def test_the_same_stamp_and_pin_reuse_the_handle(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        stamp_env(monkeypatch, target="va", generation="0")
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "va=writes")

        runtime.read_channel("SR:X")
        first = runtime._runtime_connector
        runtime.read_channel("SR:X")

        assert runtime._runtime_connector is first
        assert not first.disconnected
        assert len(fake_pool[0].asked) == 1

    def test_an_unstamped_cell_reads_the_baseline_target(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()

        assert runtime.read_channel("SR:X") == "va-server/SR:X"

        assert fake_pool[0].asked == [(baseline_target(SWITCHABLE), None)]

    def test_a_switch_in_flight_refuses_before_the_pool_is_built(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        monkeypatch.setenv(runtime.ENV_CONTROL_TARGET_REFUSAL, "switch_in_progress:4242")

        with pytest.raises(SwitchInProgressError, match="switch_in_progress:4242"):
            inside_a_running_loop(lambda: runtime.read_channel("SR:X"))

        assert fake_pool == []

    @pytest.mark.usefixtures("in_process_registry")
    def test_a_deployment_that_cannot_switch_builds_in_process(self, monkeypatch):
        _serve(monkeypatch, SINGLE)
        runtime._route_connector_through_pool()

        assert runtime.read_channel("SR:X") == "mock-server/SR:X"

        assert isinstance(runtime._runtime_connector, _FirstContextConnector)
        assert runtime._connector_pool is None
        assert runtime._pool_loop is None


class TestPooledWrites:
    """The pin still goes first; the write reaches the child's checked call."""

    def test_a_moved_record_refuses_before_the_pool_is_touched(
        self, monkeypatch, state_root, fake_pool
    ):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        write_record(state_root, target="va", generation=2)
        stamp_env(monkeypatch, target="va", generation="1")

        with pytest.raises(ControlTargetChangedError):
            runtime.write_channel("SR:SP", 1.0)
        with pytest.raises(ControlTargetChangedError):
            runtime.write_channels({"SR:SP1": 1.0, "SR:SP2": 2.0})

        assert fake_pool == []

    def test_a_permitted_write_reaches_write_channel_checked(
        self, monkeypatch, state_root, fake_pool
    ):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        write_record(state_root, target="va", generation=1)
        stamp_env(monkeypatch, target="va", generation="1")

        inside_a_running_loop(
            lambda: runtime.write_channel("SR:SP", 1.5, timeout=2.0, confirm=True)
        )

        (handle,) = fake_pool[0].handles
        assert handle.writes == [("SR:SP", 1.5, {"timeout": 2.0, "confirm": True})]

    def test_a_multi_channel_write_reaches_write_multiple_channels(
        self, monkeypatch, state_root, fake_pool
    ):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        write_record(state_root, target="va", generation=1)
        stamp_env(monkeypatch, target="va", generation="1")

        runtime.write_channels({"SR:SP1": 1.0, "SR:SP2": 2.0}, timeout=3.0)

        (handle,) = fake_pool[0].handles
        assert handle.batches == [
            ([("SR:SP1", 1.0), ("SR:SP2", 2.0)], {"timeout": 3.0, "confirm": None})
        ]


class TestPoolCleanup:
    """``cleanup_runtime`` closes the pool on its own loop, from anywhere."""

    def test_cleanup_from_a_foreign_loop_closes_the_pool_on_its_loop(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        runtime.read_channel("SR:X")
        handle = runtime._runtime_connector

        asyncio.run(runtime.cleanup_runtime())

        (pool,) = fake_pool
        assert handle.disconnected
        assert pool.closed_on is runtime._pool_loop
        assert runtime._connector_pool is None
        assert runtime._runtime_connector is None

        runtime.read_channel("SR:X")

        assert len(fake_pool) == 2

    def test_cleanup_on_the_pool_loop_closes_it_there(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        runtime.read_channel("SR:X")

        asyncio.run_coroutine_threadsafe(runtime.cleanup_runtime(), runtime._pool_loop).result(
            timeout=10
        )

        assert fake_pool[0].closed_on is runtime._pool_loop
        assert runtime._connector_pool is None

    def test_the_exit_hook_closes_a_pool_holding_no_handle(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        runtime.read_channel("SR:X")
        runtime._run_on_pool_loop(runtime._runtime_connector.disconnect())
        runtime._runtime_connector = None

        runtime._cleanup_on_exit()

        assert fake_pool[0].closed_on is runtime._pool_loop
        assert runtime._connector_pool is None


# ---------------------------------------------------------------------------
# Real connector-host children
# ---------------------------------------------------------------------------


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture(scope="module")
def pool_view(tmp_path_factory) -> Path:
    """The simulator view the pool tests' mock connector serves."""
    return served_tree(tmp_path_factory.mktemp("kernel_pool_served"), SERVED)


class TestRealChildren:
    """The kernel path against real ``osprey_connectors.ipc.host`` children."""

    def test_reads_share_a_child_and_a_generation_move_respawns_it(
        self, monkeypatch, tmp_path, pool_view
    ):
        view = pool_view
        section = {
            "type": SLOW,
            "writes_enabled": False,
            "connector": {
                SLOW: mock_config(view, response_delay_ms=1),
                "live_standin": {"timeout_s": 1.0},
            },
        }
        assert configured_targets(section) == ["live", "standin"]
        config = tmp_path / "config.yml"
        config.write_text(yaml.safe_dump({"control_system": section}))
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        monkeypatch.setenv("OSPREY_CONFIG", str(config))
        monkeypatch.setenv("PYTHONPATH", PYTHONPATH)
        _serve(monkeypatch, section)
        runtime._route_connector_through_pool()

        stamp_env(monkeypatch, target="live", generation="0")
        try:
            first = inside_a_running_loop(lambda: runtime.read_channel("SR:DCCT", timeout=10))
            second = inside_a_running_loop(lambda: runtime.read_channel("SR:A", timeout=10))
            assert first is not None and second is not None
            pool = runtime._connector_pool
            assert len(pool.pids()) == 1
            first_pid = runtime._runtime_connector.pid

            stamp_env(monkeypatch, target="live", generation="1")
            inside_a_running_loop(lambda: runtime.read_channel("SR:DCCT", timeout=10))
            second_pid = runtime._runtime_connector.pid

            assert second_pid is not None and second_pid != first_pid
            assert not _alive(first_pid)
            assert list(pool.pids()) == [("live", None)]
        finally:
            asyncio.run(runtime.cleanup_runtime())

        assert not _alive(second_pid)
        assert runtime._connector_pool is None
