"""A notebook kernel takes its connector from a connector-host pool.

A kernel is re-stamped from the deployment's record before every cell, so its
stamp moves under it when the control target is switched. A Channel Access
client keeps the first context it created in a process, so a connector rebuilt
in the same process after a switch reports the new target while it still
reaches the old one's server. These tests pin the fix: every process that opted
in (:func:`osprey.runtime._route_connector_through_pool`) is served each stamp by
a connector-host child of its own, whatever shape the deployment has.

The failure shape is reproduced first with a fake whose servers name themselves,
so the regression below cannot pass vacuously. No Channel Access server and no
``epics`` import are involved; the last class drives real connector-host
children serving the pool tests' mock connector.
"""

import asyncio
import inspect
import os
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import osprey.runtime as runtime
from osprey.runtime import ControlTargetChangedError, SwitchInProgressError
from osprey_connectors import control_context, posture_store
from osprey_connectors.control_system._base import ChannelWriteResult, WriteOutcome
from osprey_connectors.errors import ChannelWriteBlockedError
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors.types import baseline_target, configured_targets, resolve_target
from osprey_connectors_tests._harness import state_dir_under, write_control_context
from osprey_connectors_tests._write_fakes import arm_switch, turn_switch
from osprey_connectors_tests.ipc._pool_connectors import WRITE_LOG_ENV
from tests._live import wait_until
from tests._processes import pid_alive
from tests.connectors.ipc.test_pool import CHILD_START_TIMEOUT_S, PYTHONPATH, SERVED, SLOW
from tests.facility.served_tree import in_process_config, served_tree
from tests.runtime._loop import inside_a_running_loop
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
SINGLE = {"type": "epics", "connector": {"epics": {"server": "epics-server"}}}

#: A deployment serving the simulator in process; its baseline target is ``va``.
IN_PROCESS = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": {"serving": "in_process", "server": "va-server"}},
}


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
        runtime._connector_writes = None
        runtime._connector_pool = None
        runtime._pool_routing = False
        runtime._cell_marker = None
        runtime._limits_validator = None

    reset()
    yield
    reset()
    runtime._stop_pool_loop()


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
    from osprey_connectors.factory import ConnectorFactory
    from osprey_connectors_tests._harness import isolated_connector_registries

    with isolated_connector_registries():
        for name in ("epics", "virtual_accelerator", "live_standin"):
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
    """A pooled connector: one child, which reached its own target's server.

    ``role`` is the gateway a real child would have selected: the write gateway
    when the write verdict allowed writes to its target at spawn time, as the
    real pool derives it.
    """

    def __init__(self, pool: "_FakePool", target: str, server: str, role: str) -> None:
        self.pool = pool
        self.target = target
        self.server = server
        self.role = role
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

    def __init__(self, control_system, *, config_file=None, **options) -> None:
        self.section = dict(control_system)
        self.config_file = config_file
        self.options = options
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
        allowed = posture_store.write_verdict(self.section, target).allowed
        handle = _FakeHandle(
            self,
            target,
            self.section["connector"][connector_type]["server"],
            "write_access" if allowed else "read_only",
        )
        self.handles.append(handle)
        return handle

    async def close(self) -> None:
        self.record_loop()
        self.closed_on = asyncio.get_running_loop()


@pytest.fixture
def fake_pool(monkeypatch):
    """Replace the pool at the runtime's import seam; forbid an in-process build."""
    from osprey_connectors.factory import ConnectorFactory
    from osprey_connectors.ipc import _pool as pool_module

    async def no_in_process_build(*args, **kwargs):
        raise AssertionError("an opted-in kernel built a connector in its own process")

    _FakePool.instances = []
    monkeypatch.setattr(pool_module, "ConnectorHostPool", _FakePool)
    monkeypatch.setattr(ConnectorFactory, "create_control_system_connector", no_in_process_build)
    yield _FakePool.instances
    _FakePool.instances = []


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

    def test_the_pool_keeps_its_own_start_bound_when_the_runtime_names_none(
        self, monkeypatch, fake_pool
    ):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()

        runtime.read_channel("SR:X")

        (pool,) = fake_pool
        assert pool.options == {}

    def test_the_pool_takes_the_runtimes_start_bound_read_when_it_is_built(
        self, monkeypatch, fake_pool
    ):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        monkeypatch.setattr(runtime, "_pool_start_timeout_s", 7.5)

        runtime.read_channel("SR:X")

        (pool,) = fake_pool
        assert pool.options == {"start_timeout_s": 7.5}

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

    @pytest.mark.parametrize(
        "section",
        [
            SINGLE,
            {
                "type": "virtual_accelerator",
                "connector": {"virtual_accelerator": {"server": "va-server"}},
            },
            {"type": "live_standin", "connector": {"live_standin": {"server": "standin-server"}}},
            IN_PROCESS,
        ],
        ids=["epics", "va-served", "standin", "in-process"],
    )
    def test_a_single_target_deployment_is_served_by_the_pool(
        self, monkeypatch, fake_pool, section
    ):
        _serve(monkeypatch, section)
        runtime._route_connector_through_pool()

        runtime.read_channel("SR:X")

        (pool,) = fake_pool
        assert pool.asked == [(baseline_target(section), None)]
        assert runtime._runtime_connector is pool.handles[0]

    def test_an_in_process_deployment_beside_a_real_block_asks_for_va_never_live(
        self, monkeypatch, fake_pool
    ):
        section = {
            **IN_PROCESS,
            "connector": {**IN_PROCESS["connector"], "epics": {"server": "epics-server"}},
        }
        _serve(monkeypatch, section)
        runtime._route_connector_through_pool()

        assert runtime.read_channel("SR:X") == "va-server/SR:X"

        assert fake_pool[0].asked == [("va", None)]

    def test_an_unreadable_section_raises_in_the_cell(self, monkeypatch, fake_pool):
        from osprey_connectors.factory import ConnectorFactory

        class SectionUnreadable(Exception):
            pass

        class InProcessBuild(Exception):
            pass

        def get_config_value(path, _default=None, _config_path=None):
            raise SectionUnreadable(path)

        async def in_process_build(*args, **kwargs):
            raise InProcessBuild

        monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)
        monkeypatch.setattr(ConnectorFactory, "create_control_system_connector", in_process_build)
        runtime._route_connector_through_pool()

        with pytest.raises(SectionUnreadable):
            runtime.read_channel("SR:X")

        assert fake_pool == []
        assert runtime._connector_pool is None


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


#: :data:`SWITCHABLE` with a write tier that allows writes, so the switch decides.
SWITCHABLE_ALLOWED = {**SWITCHABLE, "writes": "allowed"}


@pytest.fixture
def switch_record(tmp_path, monkeypatch):
    """An ownerless ``va``/1 record with every switch off, for a kernel stamped ``va``/1."""
    root = tmp_path / "agent-data"
    arm_switch(monkeypatch, root, None, stamp=False)
    write_control_context(root, target="va", generation=1, owned_by=None)
    stamp_env(monkeypatch, target="va", generation="1")
    yield root
    control_context.invalidate_cache()


class TestTheSwitchRespawns:
    """A turn-on or turn-off lands on a fresh child, through the existing respawn check."""

    @pytest.mark.usefixtures("switch_record")
    def test_the_first_write_after_a_turn_on_uses_the_write_gateway(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE_ALLOWED)
        runtime._route_connector_through_pool()

        runtime.read_channel("SR:X")
        spawned_off = runtime._runtime_connector
        turn_switch("va", "on")
        runtime.write_channel("SR:SP", 1.0)
        on = runtime._runtime_connector
        turn_switch("va", "off")
        runtime.write_channel("SR:SP", 2.0)
        off = runtime._runtime_connector

        assert spawned_off.role == "read_only"
        assert spawned_off.disconnected and spawned_off.writes == []
        assert on is not spawned_off
        assert on.role == "write_access"
        assert [address for address, _, _ in on.writes] == ["SR:SP"]
        assert on.disconnected
        assert off.role == "read_only"
        assert fake_pool[0].asked == [("va", None)] * 3

    @pytest.mark.usefixtures("switch_record")
    def test_an_unchanged_switch_reuses_the_child_and_reads_no_record(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE_ALLOWED)
        runtime._route_connector_through_pool()
        turn_switch("va", "on")
        runtime.read_channel("SR:X")
        first = runtime._runtime_connector
        parses: list[Path] = []
        real_read = control_context._read

        def counting_read(path):
            parses.append(path)
            return real_read(path)

        monkeypatch.setattr(control_context, "_read", counting_read)

        runtime.read_channel("SR:X")
        runtime.write_channel("SR:SP", 1.0)

        assert runtime._runtime_connector is first
        assert not first.disconnected
        assert len(fake_pool[0].asked) == 1
        assert parses == []

    @pytest.mark.usefixtures("switch_record", "fake_pool")
    def test_a_tier_that_never_writes_does_not_respawn_on_a_turn_on(self, monkeypatch):
        _serve(monkeypatch, {**SWITCHABLE, "writes": "never"})
        runtime._route_connector_through_pool()

        runtime.read_channel("SR:X")
        first = runtime._runtime_connector
        turn_switch("va", "on")
        runtime.read_channel("SR:X")

        assert runtime._runtime_connector is first
        assert first.role == "read_only"

    @pytest.mark.usefixtures("switch_record", "fake_pool")
    def test_a_launch_pin_holds_the_child_on_read_only_after_a_turn_on(self, monkeypatch):
        _serve(monkeypatch, SWITCHABLE_ALLOWED)
        monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "va=sandbox")
        runtime._route_connector_through_pool()

        runtime.read_channel("SR:X")
        first = runtime._runtime_connector
        turn_switch("va", "on")
        runtime.read_channel("SR:X")

        assert runtime._runtime_connector is first
        assert first.role == "read_only"


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

        loop = runtime._pool_loop

        runtime._cleanup_on_exit()

        assert fake_pool[0].closed_on is loop
        assert runtime._connector_pool is None

    def test_the_exit_hook_closes_the_pool_on_its_loop_then_the_loop(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        runtime.read_channel("SR:X")
        loop = runtime._pool_loop

        runtime._cleanup_on_exit()

        assert fake_pool[0].closed_on is loop
        assert loop.is_closed()
        assert runtime._pool_loop is None

    @pytest.mark.filterwarnings("ignore:.*multi-threaded.*fork:DeprecationWarning")
    def test_a_forked_child_exits_without_touching_the_parents_pool(self, monkeypatch, fake_pool):
        _serve(monkeypatch, SWITCHABLE)
        runtime._route_connector_through_pool()
        runtime.read_channel("SR:X")

        pid = os.fork()
        if pid == 0:
            code = 1
            try:
                runtime._cleanup_on_exit()
                code = 0
            finally:
                os._exit(code)

        status: list[int] = []

        def reaped() -> bool:
            done, code = os.waitpid(pid, os.WNOHANG)
            if done == pid:
                status.append(code)
            return bool(status)

        try:
            wait_until(reaped, timeout=10, what="the forked child's exit")
        finally:
            if not status:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status[0]) == 0
        assert fake_pool[0].closed_on is None
        assert runtime._connector_pool is not None


class TestThePoolLoop:
    """The pool's loop is started once, stopped and closed once, refused in a forked child."""

    def test_the_pool_loop_is_closed_by_its_stop(self):
        loop = runtime._pool_event_loop()

        runtime._stop_pool_loop()

        assert loop.is_closed()
        assert runtime._pool_loop is None

    def test_stop_from_the_pool_loop_is_refused(self):
        loop = runtime._pool_event_loop()

        async def stop_here() -> None:
            runtime._stop_pool_loop()

        with pytest.raises(RuntimeError, match="cannot be stopped from its own thread"):
            asyncio.run_coroutine_threadsafe(stop_here(), loop).result(timeout=10)
        assert runtime._pool_loop is loop
        assert not loop.is_closed()

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_the_pool_loop_refuses_a_forked_child(self, monkeypatch):
        runtime._pool_event_loop()
        ran = []

        async def work() -> None:
            ran.append(True)

        coro = work()
        with monkeypatch.context() as child:
            child.setattr(runtime.os, "getpid", lambda: runtime._pool_loop_pid + 1)
            with pytest.raises(RuntimeError, match="forked process"):
                runtime._run_on_pool_loop(coro)

        assert ran == []
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED


# ---------------------------------------------------------------------------
# Real connector-host children
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pool_view(tmp_path_factory) -> Path:
    """The simulator view the pool tests' mock connector serves."""
    return served_tree(tmp_path_factory.mktemp("kernel_pool_served"), SERVED)


class TestRealChildren:
    """The kernel path against real ``osprey_connectors.ipc._host`` children."""

    @pytest.fixture(autouse=True)
    def child_start_guard(self, monkeypatch):
        """Real children start under a loaded run's hang guard, not the product default."""
        monkeypatch.setattr(runtime, "_pool_start_timeout_s", CHILD_START_TIMEOUT_S)

    def test_reads_share_a_child_and_a_generation_move_respawns_it(
        self, monkeypatch, tmp_path, pool_view
    ):
        view = pool_view
        section = {
            "type": SLOW,
            "writes_enabled": False,
            "connector": {
                SLOW: in_process_config(view, response_delay_ms=1),
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
            assert pool._start_timeout_s == CHILD_START_TIMEOUT_S
            assert len(pool.pids()) == 1
            first_pid = runtime._runtime_connector.pid

            stamp_env(monkeypatch, target="live", generation="1")
            inside_a_running_loop(lambda: runtime.read_channel("SR:DCCT", timeout=10))
            second_pid = runtime._runtime_connector.pid

            assert second_pid is not None and second_pid != first_pid
            assert not pid_alive(first_pid)
            assert list(pool.pids()) == [("live", None)]
        finally:
            asyncio.run(runtime.cleanup_runtime())

        assert not pid_alive(second_pid)
        assert runtime._connector_pool is None

    def test_an_in_process_deployment_reads_through_a_real_child(
        self, monkeypatch, tmp_path, pool_view
    ):
        section = {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": in_process_config(pool_view)},
        }
        config = tmp_path / "config.yml"
        config.write_text(yaml.safe_dump({"control_system": section}))
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        monkeypatch.setenv("OSPREY_CONFIG", str(config))
        monkeypatch.setenv("PYTHONPATH", PYTHONPATH)
        _serve(monkeypatch, section)
        runtime._route_connector_through_pool()

        try:
            value = inside_a_running_loop(lambda: runtime.read_channel("SR:DCCT", timeout=10))
            assert value is not None
            pool = runtime._connector_pool
            assert list(pool.pids()) == [("va", None)]
            pid = runtime._runtime_connector.pid
        finally:
            asyncio.run(runtime.cleanup_runtime())

        assert not pid_alive(pid)


class TestRealChildrenFollowTheSwitch:
    """The kernel against real children: a turn-on and a turn-off each land on a fresh child."""

    @pytest.fixture(autouse=True)
    def child_start_guard(self, monkeypatch):
        monkeypatch.setattr(runtime, "_pool_start_timeout_s", CHILD_START_TIMEOUT_S)

    @pytest.mark.usefixtures("switch_record")
    def test_writes_follow_the_switch_through_fresh_children(
        self, monkeypatch, tmp_path, pool_view
    ):
        section = {
            "type": "virtual_accelerator",
            "writes": "allowed",
            "connector": {"virtual_accelerator": in_process_config(pool_view)},
        }
        config = tmp_path / "config.yml"
        config.write_text(yaml.safe_dump({"control_system": section}))
        log = tmp_path / "writes.log"
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CONFIG_FILE", raising=False)
        monkeypatch.setenv("OSPREY_CONFIG", str(config))
        monkeypatch.setenv("PYTHONPATH", PYTHONPATH)
        monkeypatch.setenv(WRITE_LOG_ENV, str(log))
        _serve(monkeypatch, section)
        runtime._route_connector_through_pool()

        try:
            with pytest.raises(ChannelWriteBlockedError) as off:
                inside_a_running_loop(lambda: runtime.write_channel("SR:SP", 1.0, confirm=False))
            off_pid = runtime._runtime_connector.pid

            turn_switch("va", "on")
            inside_a_running_loop(lambda: runtime.write_channel("SR:SP", 2.0, confirm=False))
            on_pid = runtime._runtime_connector.pid
            on_report = runtime._runtime_connector.report

            turn_switch("va", "off")
            with pytest.raises(ChannelWriteBlockedError):
                inside_a_running_loop(lambda: runtime.write_channel("SR:SP", 3.0, confirm=False))
            again_pid = runtime._runtime_connector.pid
        finally:
            asyncio.run(runtime.cleanup_runtime())

        assert refusal_phrase("switch_off", "simulator") in str(off.value)
        assert on_report["writes_enabled"] is True
        assert len({off_pid, on_pid, again_pid}) == 3
        assert not pid_alive(off_pid) and not pid_alive(on_pid)
