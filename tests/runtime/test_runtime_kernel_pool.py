"""A notebook kernel's Channel Access connectors live in connector-host children.

pvapy binds a process to one gateway at its first channel, so a kernel that
follows a control-target switch cannot rebuild its EPICS connector in-process.
:mod:`osprey.runtime._kernel_host` routes it to a
:class:`~osprey_connectors.ipc.pool.ConnectorHostPool` child instead, on one
loop the kernel keeps for its life. These tests pin the routing decision, the
rebuild on a moved stamp, the error mapping and the limits net against a stand-in
pool; the last test drives the real thing — two soft IOCs serving the same
names, a kernel-shaped process switching between them, and a prompt exit.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.runtime as runtime
from osprey import jupyter_kernel
from osprey.runtime import SwitchInProgressError, _kernel_host
from osprey_connectors import posture_store
from osprey_connectors.control_system.base import ChannelValue, ChannelWriteResult, WriteOutcome
from osprey_connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey_connectors.errors import (
    ChannelLimitsViolationError,
    ChannelWriteBlockedError,
    ChannelWriteFailedError,
)
from osprey_connectors.ipc.pool import (
    READONLY,
    ConnectorHostLostError,
    PooledConnector,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: A deployment whose two targets are both Channel Access machines.
CA_SECTION: dict[str, Any] = {
    "type": "virtual_accelerator",
    "connector": {
        "virtual_accelerator": {"gateways": {"read_only": {"address": "va", "port": 5064}}},
        "live_standin": {"gateways": {"read_only": {"address": "standin", "port": 5065}}},
    },
}


def test_the_kernel_pid_name_is_spelled_the_same_on_both_sides():
    assert _kernel_host.ENV_NOTEBOOK_KERNEL_PID == jupyter_kernel.ENV_NOTEBOOK_KERNEL_PID


def test_the_launcher_stamps_this_process_as_the_kernel(monkeypatch):
    monkeypatch.delenv(_kernel_host.ENV_NOTEBOOK_KERNEL_PID, raising=False)
    env: dict[str, str] = {}
    stamps = jupyter_kernel.compute_stamps("abc", env)
    assert stamps[jupyter_kernel.ENV_NOTEBOOK_KERNEL_PID] == str(os.getpid())
    assert env[jupyter_kernel.ENV_NOTEBOOK_KERNEL_PID] == str(os.getpid())


def test_the_children_log_to_the_kernels_original_stderr(monkeypatch):
    """Descriptor 2 is published into the running cell; the children must not inherit it."""
    import logging

    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", list(root.handlers))
    monkeypatch.setattr(_kernel_host, "_child_stderr", None)
    fd = jupyter_kernel._route_logs_to_process_stderr()
    try:
        assert fd != 2
        assert os.path.sameopenfile(fd, 2)
        handler = root.handlers[-1]
        assert handler.stream.fileno() == fd  # type: ignore[attr-defined]
    finally:
        root.handlers[-1].stream.close()  # type: ignore[attr-defined]

    seen: list[int | None] = []
    monkeypatch.setattr(jupyter_kernel, "_route_logs_to_process_stderr", lambda: 99)
    monkeypatch.setattr(
        jupyter_kernel,
        "_prepare_environment",
        lambda _argv: seen.append(_kernel_host._child_stderr),
    )
    monkeypatch.setattr(jupyter_kernel, "_initialize_registry", lambda: None)

    class _Stop(Exception):
        pass

    def stop(*_args, **_kwargs):
        raise _Stop

    monkeypatch.setattr("ipykernel.kernelapp.IPKernelApp.instance", stop)
    with pytest.raises(_Stop):
        jupyter_kernel.main([])
    assert seen == [99]


class TestWhichProcessIsAKernel:
    def test_the_stamped_pid_is_a_kernel(self, monkeypatch):
        monkeypatch.setenv(_kernel_host.ENV_NOTEBOOK_KERNEL_PID, str(os.getpid()))
        assert _kernel_host.in_notebook_kernel()

    def test_an_inherited_stamp_is_not(self, monkeypatch):
        """A subprocess a cell starts, or a fork, carries the kernel's pid, not its own."""
        monkeypatch.setenv(_kernel_host.ENV_NOTEBOOK_KERNEL_PID, str(os.getpid() + 1))
        assert not _kernel_host.in_notebook_kernel()

    def test_an_executor_sandbox_is_not(self, monkeypatch):
        monkeypatch.delenv(_kernel_host.ENV_NOTEBOOK_KERNEL_PID, raising=False)
        assert not _kernel_host.in_notebook_kernel()


# ---------------------------------------------------------------------------
# A stand-in pool
# ---------------------------------------------------------------------------


class FakePool:
    """Answers :class:`PooledConnector` calls; records which loop thread made them."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str, tuple[Any, ...], dict[str, Any]]] = []
        self.retired: list[Any] = []
        self.threads: set[str] = set()
        self.raise_on: dict[str, BaseException] = {}
        self.value = 7.0

    async def _invoke(self, key: Any, method: str, *args: Any, **kwargs: Any) -> Any:
        self.threads.add(threading.current_thread().name)
        self.calls.append((key, method, args, kwargs))
        if method in self.raise_on:
            raise self.raise_on[method]
        if method == "read_channel":
            return ChannelValue(value=self.value, timestamp=datetime.now())
        if method == "write_channel_checked":
            return ChannelWriteResult(
                channel_address=args[0],
                value_written=args[1],
                outcome=WriteOutcome.UNREQUESTED,
            )
        if method == "write_multiple_channels":
            return [
                ChannelWriteResult(
                    channel_address=c, value_written=v, outcome=WriteOutcome.UNREQUESTED
                )
                for c, v in args[0]
            ]
        raise AssertionError(method)

    async def _retire_key(self, key: Any) -> None:
        self.retired.append(key)


@pytest.fixture
def kernel(monkeypatch):
    """This process, stamped as a notebook kernel on a two-target CA deployment."""
    monkeypatch.setenv(_kernel_host.ENV_NOTEBOOK_KERNEL_PID, str(os.getpid()))
    for name in (
        runtime.ENV_CONTROL_TARGET,
        runtime.ENV_CONTROL_TARGET_GENERATION,
        runtime.ENV_CONTROL_TARGET_REFUSAL,
        runtime.ENV_IN_CELL,
        "OSPREY_EXECUTION_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, "*=sandbox")
    for name in ("_runtime_connector", "_connector_stamp", "_cell_marker", "_limits_validator"):
        monkeypatch.setattr(runtime, name, None)

    section = {"value": CA_SECTION}
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value",
        lambda key, default=None: section["value"] if key == "control_system" else default,
    )
    pool = FakePool()
    routes: list[_kernel_host.PoolRoute] = []

    async def pooled(route):
        assert _kernel_host.on_owner_loop()
        routes.append(route)
        return PooledConnector(pool, (route.target, route.execution_mode))  # type: ignore[arg-type]

    built: list[Any] = []

    async def create(config=None, control_target=None):  # noqa: ARG001 - called by keyword
        built.append(threading.current_thread().name)

        class _InProcess:
            async def read_channel(self, address, timeout=None):  # noqa: ARG002
                return ChannelValue(value="in-process", timestamp=datetime.now())

            async def disconnect(self):
                pass

        return _InProcess()

    monkeypatch.setattr(_kernel_host, "pooled_connector", pooled)
    monkeypatch.setattr(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector", create
    )
    yield monkeypatch, pool, routes, built, section
    _kernel_host.shutdown(runtime.cleanup_runtime())
    runtime._runtime_connector = None
    runtime._connector_stamp = None


def _stamp(monkeypatch, target: str, generation: int, pin: str | None = None) -> None:
    monkeypatch.setenv(runtime.ENV_CONTROL_TARGET, target)
    monkeypatch.setenv(runtime.ENV_CONTROL_TARGET_GENERATION, str(generation))
    monkeypatch.setenv(posture_store.LAUNCH_POSTURE_ENV_VAR, pin or f"{target}=sandbox")


class TestRouting:
    def test_a_ca_target_is_served_from_the_pool_on_the_kernel_loop(self, kernel):
        monkeypatch, pool, routes, built, _ = kernel
        _stamp(monkeypatch, "va", 1)

        assert runtime.read_channel("SR:CH") == 7.0

        assert built == []
        assert [(r.target, r.execution_mode) for r in routes] == [("va", None)]
        assert pool.threads == {"osprey-runtime-kernel-loop"}

    def test_a_switch_retires_the_old_child_and_routes_to_the_new_key(self, kernel):
        monkeypatch, pool, routes, _, _ = kernel
        _stamp(monkeypatch, "va", 1)
        runtime.read_channel("SR:CH")
        _stamp(monkeypatch, "standin", 2)
        runtime.read_channel("SR:CH")

        assert [r.target for r in routes] == ["va", "standin"]
        assert pool.retired == [("va", None)]
        assert [call[0] for call in pool.calls] == [("va", None), ("standin", None)]

    def test_the_same_stamp_reuses_the_child(self, kernel):
        monkeypatch, pool, routes, _, _ = kernel
        _stamp(monkeypatch, "va", 1)
        runtime.read_channel("SR:CH")
        runtime.read_channel("SR:CH")
        assert len(routes) == 1
        assert pool.retired == []

    def test_a_new_launch_pin_respawns_the_child(self, kernel):
        """The child reads the pin from the environment it was spawned with."""
        monkeypatch, pool, routes, _, _ = kernel
        _stamp(monkeypatch, "va", 1, "va=sandbox")
        runtime.read_channel("SR:CH")
        _stamp(monkeypatch, "va", 1, "va=writes")
        runtime.read_channel("SR:CH")
        assert [r.launch_pin for r in routes] == ["va=sandbox", "va=writes"]
        assert pool.retired == [("va", None)]

    def test_a_readonly_run_gets_the_readonly_key(self, kernel):
        monkeypatch, pool, routes, _, _ = kernel
        monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
        _stamp(monkeypatch, "va", 1)
        runtime.read_channel("SR:CH")
        assert routes[0].execution_mode == READONLY
        assert pool.calls[0][0] == ("va", READONLY)

    def test_an_unstamped_cell_reads_the_ca_baseline_through_the_pool(self, kernel):
        _, _, routes, built, _ = kernel
        runtime.read_channel("SR:CH")
        assert built == []
        assert routes[0].target == "va"

    def test_a_mock_deployment_stays_in_process_on_the_kernel_loop(self, kernel):
        _, _, routes, built, section = kernel
        section["value"] = {"type": "mock"}
        assert runtime.read_channel("SR:CH") == "in-process"
        assert routes == []
        assert built == ["osprey-runtime-kernel-loop"]

    def test_a_switch_from_mock_to_ca_moves_to_the_pool(self, kernel):
        monkeypatch, _, routes, built, section = kernel
        section["value"] = {"type": "mock", **{k: v for k, v in CA_SECTION.items() if k != "type"}}
        runtime.read_channel("SR:CH")
        _stamp(monkeypatch, "standin", 2)
        runtime.read_channel("SR:CH")
        assert len(built) == 1
        assert [r.target for r in routes] == ["standin"]

    def test_a_sandbox_keeps_the_in_process_connector(self, kernel):
        """Not a kernel: stamped once, never re-pointed, so nothing is pooled."""
        monkeypatch, _, routes, built, _ = kernel
        monkeypatch.delenv(_kernel_host.ENV_NOTEBOOK_KERNEL_PID)
        _stamp(monkeypatch, "va", 1)
        monkeypatch.setattr(runtime, "_target_connector_config", lambda: None)
        assert runtime.read_channel("SR:CH") == "in-process"
        assert routes == []
        assert len(built) == 1
        assert not _kernel_host.owner_started()

    def test_a_switch_in_flight_is_still_refused_first(self, kernel):
        monkeypatch, pool, routes, _, _ = kernel
        monkeypatch.setenv(runtime.ENV_CONTROL_TARGET_REFUSAL, "switch_in_progress:42")
        with pytest.raises(SwitchInProgressError):
            runtime.read_channel("SR:CH")
        assert routes == [] and pool.calls == []

    def test_the_calls_run_on_the_kernel_loop_from_a_running_loop(self, kernel):
        """A cell runs inside ipykernel's loop; the pool must still see one loop."""
        monkeypatch, pool, _, _, _ = kernel
        _stamp(monkeypatch, "va", 1)

        async def cell():
            runtime.read_channel("SR:CH")
            runtime.read_channel("SR:CH")
            await runtime.cleanup_runtime()

        asyncio.run(cell())
        asyncio.run(cell())
        assert pool.threads == {"osprey-runtime-kernel-loop"}
        assert pool.retired == [("va", None), ("va", None)]


class TestWrites:
    def _armed(self, monkeypatch, tmp_path):
        from tests._control_context_fixtures import write_control_context

        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
        write_control_context(tmp_path, target="va", generation=1)
        _stamp(monkeypatch, "va", 1, "va=writes")

    def test_a_write_goes_through_the_childs_checked_write(self, kernel, tmp_path):
        monkeypatch, pool, _, _, _ = kernel
        self._armed(monkeypatch, tmp_path)
        runtime.write_channel("SR:SP", 1.5, confirm=True)
        assert pool.calls[-1][1:] == ("write_channel_checked", ("SR:SP", 1.5), {"confirm": True})

    def test_a_write_on_a_moved_target_is_refused_before_the_child(self, kernel, tmp_path):
        monkeypatch, pool, _, _, _ = kernel
        self._armed(monkeypatch, tmp_path)
        _stamp(monkeypatch, "va", 0, "va=writes")
        with pytest.raises(runtime.ControlTargetChangedError):
            runtime.write_channel("SR:SP", 1.5)
        assert pool.calls == []

    def test_a_refusal_crosses_unchanged(self, kernel, tmp_path):
        monkeypatch, pool, _, _, _ = kernel
        self._armed(monkeypatch, tmp_path)
        pool.raise_on["write_channel_checked"] = ChannelWriteBlockedError(
            "SR:SP", "WRITES_DISABLED"
        )
        with pytest.raises(ChannelWriteBlockedError):
            runtime.write_channel("SR:SP", 1.5)

    def test_a_child_lost_mid_write_is_an_unconfirmed_write(self, kernel, tmp_path):
        monkeypatch, pool, _, _, _ = kernel
        self._armed(monkeypatch, tmp_path)
        lost = ConnectorHostLostError(
            "gone", target="va", execution_mode=None, pid=1, cause="exited", returncode=-9
        )
        pool.raise_on["write_channel_checked"] = lost
        with pytest.raises(ChannelWriteFailedError) as caught:
            runtime.write_channel("SR:SP", 1.5)
        assert caught.value.reason == "UNCONFIRMED"
        assert caught.value.outcome is WriteOutcome.UNCONFIRMED
        assert caught.value.value_written == 1.5
        assert caught.value.__cause__ is lost

    def test_a_pool_timeout_on_a_batch_is_unconfirmed_too(self, kernel, tmp_path):
        monkeypatch, pool, _, _, _ = kernel
        self._armed(monkeypatch, tmp_path)
        pool.raise_on["write_multiple_channels"] = TimeoutError("child slow")
        with pytest.raises(ChannelWriteFailedError) as caught:
            runtime.write_channels({"SR:A": 1.0, "SR:B": 2.0})
        assert caught.value.reason == "UNCONFIRMED"

    def test_a_childs_own_connection_error_is_not_rewritten(self, kernel, tmp_path):
        """An error the child's connector raised already is the in-process error."""
        from osprey_connectors.ipc import proxy

        monkeypatch, pool, _, _, _ = kernel
        self._armed(monkeypatch, tmp_path)
        own = ConnectionError("channel unreachable")
        setattr(own, proxy._FROM_CHILD, True)
        pool.raise_on["write_channel_checked"] = own
        with pytest.raises(ConnectionError) as caught:
            runtime.write_channel("SR:SP", 1.5)
        assert caught.value is own

    def test_the_limits_net_fails_closed_on_a_pooled_max_step_channel(self, kernel, tmp_path):
        """No synchronous reader over the pool: the net refuses rather than skips."""
        monkeypatch, pool, _, _, _ = kernel
        self._armed(monkeypatch, tmp_path)
        monkeypatch.setattr(
            runtime,
            "_limits_validator",
            LimitsValidator(
                {
                    "SR:SP": ChannelLimitsConfig(
                        channel_address="SR:SP", min_value=0.0, max_value=10.0, max_step=1.0
                    )
                },
                {"allow_unlisted_channels": False},
                {},
            ),
        )
        with pytest.raises(ChannelLimitsViolationError) as caught:
            runtime.write_channel("SR:SP", 1.5)
        assert caught.value.violation_type == "STEP_CHECK_FAILED"
        assert [c for c in pool.calls if c[1].startswith("write")] == []


# ---------------------------------------------------------------------------
# Two soft IOCs, one kernel-shaped process
# ---------------------------------------------------------------------------

SCENARIO_BUDGET_S = 90.0
#: From the last line the scenario prints to its exit: closing the pool and the
#: kernel loop. A pvapy call on a joined thread would hang here instead.
EXIT_BUDGET_S = 10.0

_SCENARIO = textwrap.dedent(
    """
    import asyncio, json, os, sys, time
    from pathlib import Path

    os.environ["OSPREY_NOTEBOOK_KERNEL_PID"] = str(os.getpid())
    import osprey.runtime as runtime
    from tests._control_context_fixtures import write_control_context

    PREFIX = os.environ["SCENARIO_PREFIX"]
    ROOT = Path(os.environ["OSPREY_AGENT_DATA_ROOT"])

    def stamp(target, generation):
        # What pre_run_cell does from the record.
        write_control_context(ROOT, target=target, generation=generation)
        os.environ["OSPREY_CONTROL_TARGET"] = target
        os.environ["OSPREY_CONTROL_TARGET_GENERATION"] = str(generation)
        os.environ["OSPREY_LAUNCH_POSTURE"] = f"{target}=writes"

    def alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    out = {}

    async def cells():
        # Synchronous runtime calls inside a running loop, as in ipykernel.
        stamp("live", 1)
        out["live"] = runtime.read_channel(f"{PREFIX}:RB")
        out["live_pid"] = runtime._runtime_connector.pid
        stamp("standin", 2)
        out["standin"] = runtime.read_channel(f"{PREFIX}:RB")
        out["standin_pid"] = runtime._runtime_connector.pid
        runtime.write_channel(f"{PREFIX}:SP", 42.0, confirm=True)
        out["standin_sp"] = runtime.read_channel(f"{PREFIX}:SP")
        out["live_child_alive_after_switch"] = alive(out["live_pid"])
        stamp("live", 3)
        out["live_again"] = runtime.read_channel(f"{PREFIX}:SP")
        out["last_pid"] = runtime._runtime_connector.pid

    asyncio.run(cells())
    out["pvaccess_in_parent"] = "pvaccess" in sys.modules
    out["printed_at"] = time.time()
    print("RESULT " + json.dumps(out), flush=True)
    """
)


def test_a_kernel_follows_a_switch_between_two_iocs_and_exits_promptly(tmp_path):
    pytest.importorskip("pvaccess")
    pytest.importorskip("epicscorelibs")
    from tests.connectors.ipc.test_pool_soft_ioc import SEED_A, SEED_B, SoftIOC

    prefix = f"KRN{os.urandom(4).hex().upper()}"
    machine = SoftIOC(tmp_path, "machine", prefix, SEED_A)
    try:
        standin = SoftIOC(tmp_path, "standin", prefix, SEED_B)
    except BaseException:
        machine.stop()
        raise
    try:

        def gateways(port):
            endpoint = {"address": "127.0.0.1", "port": port, "use_name_server": False}
            return {"read_only": dict(endpoint), "write_access": dict(endpoint)}

        config = tmp_path / "config.yml"
        config.write_text(
            yaml.safe_dump(
                {
                    "control_system": {
                        "type": "epics",
                        "writes_enabled": True,
                        "connector": {
                            "epics": {"timeout": 5.0, "gateways": gateways(machine.port)},
                            "live_standin": {"timeout": 5.0, "gateways": gateways(standin.port)},
                        },
                    }
                }
            )
        )
        root = tmp_path / "agent_data"
        root.mkdir()
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("EPICS_", "OSPREY_", "PYTEST_", "CONFIG_FILE"))
        }
        env.update(
            PYTHONPATH=os.pathsep.join(
                [
                    str(REPO_ROOT),
                    str(REPO_ROOT / "src"),
                    str(REPO_ROOT / "packages" / "osprey-connectors" / "src"),
                ]
            ),
            CONFIG_FILE=str(config),
            OSPREY_AGENT_DATA_ROOT=str(root),
            OSPREY_AUDIT_IDENTITY="kernel-pool-test",
            SCENARIO_PREFIX=prefix,
        )
        start = time.monotonic()
        try:
            done = subprocess.run(
                [sys.executable, "-c", _SCENARIO],
                capture_output=True,
                text=True,
                env=env,
                cwd=tmp_path,
                timeout=SCENARIO_BUDGET_S,
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(
                f"the kernel-shaped process did not exit within {SCENARIO_BUDGET_S}s. "
                f"stdout: {exc.stdout!r} stderr: {exc.stderr!r}"
            )
        elapsed = time.monotonic() - start
        exited_at = time.time()
    finally:
        machine.stop()
        standin.stop()

    assert done.returncode == 0, done.stderr
    [line] = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    out = json.loads(line[len("RESULT ") :])

    assert out["live"] == SEED_A
    # The read after the switch comes from the NEW IOC, in the same process.
    assert out["standin"] == SEED_B
    assert out["standin_sp"] == 42.0
    # The stand-in's write never reached the machine.
    assert out["live_again"] == SEED_A
    assert out["live_pid"] != out["standin_pid"]
    assert out["live_child_alive_after_switch"] is False
    assert out["pvaccess_in_parent"] is False
    # Normal interpreter shutdown stopped the last child, promptly.
    assert exited_at - out["printed_at"] < EXIT_BUDGET_S
    with pytest.raises(ProcessLookupError):
        os.kill(out["last_pid"], 0)
    assert elapsed < SCENARIO_BUDGET_S
