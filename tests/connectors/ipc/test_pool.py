"""The connector-host pool, driving real child processes.

Every child here is a real ``python -m osprey_connectors.ipc.host`` process
serving a mock connector, loaded by dotted path from
``tests/connectors/ipc/_pool_connectors.py`` so that each test can pick the one
misbehaviour it needs: a hanging channel, a failing ``connect()``, a
``connect()`` that never returns. No network and no EPICS are involved; the
two-IOC Channel Access suite beside this file covers the real protocol.

Each test runs in a scratch directory with ``CONFIG_FILE`` removed from the
environment, so a child reads only the config file its test hands the pool.
"""

import asyncio
import contextlib
import gc
import logging
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

from osprey_connectors.control_system.base import WriteOutcome
from osprey_connectors.ipc import pool as pool_module
from osprey_connectors.ipc.pool import (
    ConnectorHostError,
    ConnectorHostLostError,
    ConnectorHostPool,
    ConnectorHostStartError,
    ConnectorHostUnresponsiveError,
    ConnectorHostWriteTimeoutError,
    PooledConnector,
)
from tests.connectors.ipc._pool_connectors import WRITE_LOG_ENV

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHONPATH = os.pathsep.join(
    [
        str(REPO_ROOT),
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "packages" / "osprey-connectors" / "src"),
    ]
)

_HELPERS = "tests.connectors.ipc._pool_connectors"
SLOW = f"{_HELPERS}.SlowMockConnector"
FAILING = f"{_HELPERS}.FailingConnector"
HANGING = f"{_HELPERS}.HangingConnector"
SLOW_START = f"{_HELPERS}.SlowStartConnector"
TIMING_OUT = f"{_HELPERS}.TimingOutConnector"
EXITING = f"{_HELPERS}.ExitingConnector"

#: Bound for calls that must simply succeed, generous enough for a loaded CI box.
OK_TIMEOUT_S = 10.0


def _section(connector_type: str, *, writes_enabled: bool = False, block=None) -> dict:
    return {
        "type": connector_type,
        "writes_enabled": writes_enabled,
        "connector": {
            connector_type: {"response_delay_ms": 1, "noise_level": 0.0, **(block or {})}
        },
    }


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """A scratch cwd, no ambient config, and children that can import the helpers."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.setenv("PYTHONPATH", PYTHONPATH)
    return tmp_path


@pytest.fixture
def writable(isolated, monkeypatch):
    """A config file on disk that arms the slow mock, and a write log."""
    section = _section(SLOW, writes_enabled=True)
    config_file = isolated / "config.yml"
    config_file.write_text(yaml.safe_dump({"control_system": section}))
    log = isolated / "writes.log"
    monkeypatch.setenv(WRITE_LOG_ENV, str(log))
    return section, config_file, log


@pytest.fixture
async def pools():
    """Pools a test builds; every one is closed however the test ends."""
    made: list[ConnectorHostPool] = []

    def make(*args, **kwargs):
        pool = ConnectorHostPool(*args, **kwargs)
        made.append(pool)
        return pool

    yield make
    for pool in made:
        await pool.close()


@pytest.fixture
def spawns(monkeypatch):
    """Every child the pool spawns, with the loop time it was spawned at."""
    record: list[tuple[float, object]] = []
    real = pool_module.spawn_host

    async def counting(*args, **kwargs):
        process = await real(*args, **kwargs)
        record.append((asyncio.get_running_loop().time(), process))
        return process

    monkeypatch.setattr(pool_module, "spawn_host", counting)
    return record


@pytest.fixture
def asyncio_errors(caplog):
    """ERROR records asyncio logs — abandoned futures, unretrieved exceptions."""
    caplog.set_level(logging.ERROR, logger="asyncio")

    def collected():
        gc.collect()
        return [record.getMessage() for record in caplog.records if record.name == "asyncio"]

    return collected


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# ------------------------------------------------------------ keys and spawn


async def test_a_child_starts_on_first_use_and_is_shared_by_its_key(pools, spawns):
    # A slow connect keeps the start window open for the other four callers.
    pool = pools(_section(SLOW_START))
    assert pool.pids() == {}

    handles = await asyncio.gather(*(pool.connector("live") for _ in range(5)))

    assert len(spawns) == 1
    assert len(pool.pids()) == 1
    assert len({handle.pid for handle in handles}) == 1
    assert (await handles[0].read_channel("SR:DCCT", timeout=OK_TIMEOUT_S)).value is not None


async def test_two_keys_start_side_by_side(pools, spawns):
    pool = pools(_section(SLOW_START))

    async def started(mode):
        await pool.connector("live", execution_mode=mode)
        return asyncio.get_running_loop().time()

    finished = await asyncio.gather(started(None), started("readonly"))

    # Both children were spawned before either finished its slow connect: the
    # second key did not queue behind the first key's start.
    assert len(spawns) == 2
    assert max(when for when, _ in spawns) < min(finished)


async def test_readonly_is_a_separate_child_on_the_same_target(pools):
    pool = pools(_section(SLOW))
    live = await pool.connector("live")
    live_ro = await pool.connector("live", execution_mode="readonly")

    assert live.pid != live_ro.pid
    assert set(pool.pids()) == {("live", None), ("live", "readonly")}
    assert live_ro.report["readonly_run"] is True
    assert live.report["readonly_run"] is False


async def test_a_project_env_file_puts_no_epics_variable_back_into_a_child(pools, writable):
    # Reading the config file is what would load the .env beside it, after
    # the child's own scrub.
    section, config_file, _ = writable
    (config_file.parent / ".env").write_text(
        "EPICS_CA_ADDR_LIST=10.9.9.9\nEPICS_CA_MAX_ARRAY_BYTES=100000000\n"
    )
    pool = pools(section, config_file=config_file)

    live = await pool.connector("live")

    assert live.report["epics_env"] == {}
    assert live.report["mode"] is None


async def test_a_hung_call_on_one_target_does_not_hold_up_another(pools):
    pool = pools(_section(SLOW))
    live = await pool.connector("live")
    live_ro = await pool.connector("live", execution_mode="readonly")

    hung = asyncio.create_task(live.read_channel("SLOW:X"))
    value = await asyncio.wait_for(live_ro.read_channel("SR:DCCT"), 5.0)

    assert value.value is not None
    assert not hung.done()
    hung.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await hung


async def test_an_unknown_execution_mode_is_refused(pools):
    pool = pools(_section(SLOW))
    with pytest.raises(ValueError, match="execution_mode"):
        await pool.connector("live", execution_mode="readwrite")


# ------------------------------------------------------------ write posture


async def test_a_readonly_child_refuses_writes_the_config_arms(pools, writable):
    section, config_file, log = writable
    pool = pools(section, config_file=config_file)
    live = await pool.connector("live")
    live_ro = await pool.connector("live", execution_mode="readonly")

    armed = await live.write_channel("SR:SP", 1.0)
    assert armed.outcome is not WriteOutcome.REFUSED

    refused = await live_ro.write_channel("SR:SP", 2.0)
    assert refused.outcome is WriteOutcome.REFUSED
    assert refused.refusal_reason == "WRITES_DISABLED"

    # Only the armed child's write reached the connector at all.
    assert [line.split() for line in log.read_text().splitlines()] == [["SR:SP", "1.0"]]


async def test_a_section_that_arms_writes_without_a_config_file_is_refused(pools):
    # The child reads its posture from a config file, and there is none: its
    # writes are off. The section says they are armed, and the pool will not
    # hand out a connector whose posture is not the one its caller described.
    pool = pools(_section(SLOW, writes_enabled=True))
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("live")

    assert caught.value.stage == "verify"
    assert "writes off" in str(caught.value) and "config_file" in str(caught.value)
    assert pool.pids() == {}


@pytest.mark.parametrize("via", ["passed", "inherited"])
async def test_a_config_file_that_arms_what_the_section_does_not_is_refused(
    pools, writable, monkeypatch, via
):
    _, config_file, log = writable
    # Gatewayless mock: no gateway role could reveal the disagreement, so only
    # the posture check stands between this child and an armed write. The file
    # reaches the child either handed to the pool or through CONFIG_FILE.
    if via == "passed":
        pool = pools(_section(SLOW, writes_enabled=False), config_file=config_file)
    else:
        monkeypatch.setenv("CONFIG_FILE", str(config_file))
        pool = pools(_section(SLOW, writes_enabled=False))
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("live")

    assert caught.value.stage == "verify"
    assert "writes armed" in str(caught.value)
    assert not log.exists()


# ------------------------------------------------------------ losing a child


async def test_a_killed_child_fails_its_call_once_and_the_next_call_respawns(pools, writable):
    section, config_file, log = writable
    pool = pools(section, config_file=config_file)
    live = await pool.connector("live")
    first_pid = live.pid

    write = asyncio.create_task(live.write_channel("SLOW:SP", 7.0))
    await _wait_for(lambda: log.exists() and log.read_text().strip())
    os.kill(first_pid, signal.SIGKILL)

    with pytest.raises(ConnectorHostLostError) as caught:
        await asyncio.wait_for(write, 10.0)
    lost = caught.value
    assert isinstance(lost, ConnectionError)
    assert (lost.target, lost.execution_mode, lost.pid, lost.cause) == (
        "live",
        None,
        first_pid,
        "exited",
    )
    assert str(first_pid) in str(lost) and "'live'" in str(lost)

    assert pool.pids() == {}
    assert (await live.read_channel("SR:DCCT", timeout=OK_TIMEOUT_S)).value is not None
    assert live.pid not in (None, first_pid)

    # Exactly one write reached any connector: the lost call was not retried.
    assert len(log.read_text().splitlines()) == 1


async def test_a_wedged_child_is_killed_and_replaced(pools):
    pool = pools(_section(SLOW), timeout_grace_s=0.5, ping_timeout_s=0.5, terminate_grace_s=0.5)
    live = await pool.connector("live")
    first_pid = live.pid

    wedging = asyncio.create_task(live.read_channel("WEDGE:X", timeout=0.5))
    # A second call on the wedged child fails the same way once it is killed.
    await asyncio.sleep(0.2)
    bystander = asyncio.create_task(live.read_channel("SR:DCCT", timeout=30.0))

    with pytest.raises(ConnectorHostUnresponsiveError) as caught:
        await asyncio.wait_for(wedging, 10.0)
    assert isinstance(caught.value, TimeoutError)
    assert isinstance(caught.value, ConnectorHostLostError)
    assert caught.value.cause == "unresponsive"
    assert "nor a ping" in str(caught.value)
    with pytest.raises(ConnectorHostUnresponsiveError) as also:
        await asyncio.wait_for(bystander, 10.0)
    assert also.value.cause == "unresponsive"

    assert not _alive(first_pid)
    assert (await live.read_channel("SR:DCCT", timeout=OK_TIMEOUT_S)).value is not None
    assert live.pid != first_pid


async def test_a_slow_but_alive_child_times_out_the_call_and_keeps_running(pools, writable):
    section, config_file, log = writable
    pool = pools(section, config_file=config_file, timeout_grace_s=0.2)
    live = await pool.connector("live")
    pid = live.pid

    # A write that outlasts its own timeout plus grace, in a child that still
    # answers: the batched/confirmed-write case. It must not be killed.
    with pytest.raises(TimeoutError) as caught:
        await live.write_channel("SLOW:SP", 1.0, timeout=0.3)

    assert not isinstance(caught.value, ConnectorHostLostError)
    assert "still answers a ping" in str(caught.value)
    assert live.pid == pid and _alive(pid)
    assert (await live.read_channel("SR:DCCT", timeout=OK_TIMEOUT_S)).value is not None
    assert len(log.read_text().splitlines()) == 1


async def test_with_kill_on_write_timeout_a_write_that_misses_its_deadline_kills_its_child(
    pools, writable
):
    section, config_file, log = writable
    pool = pools(section, config_file=config_file, timeout_grace_s=0.2, kill_on_write_timeout=True)
    live = await pool.connector("live")
    pid = live.pid

    # The same slow write the test above keeps its child for. This child would
    # answer a ping, and is killed anyway: kept, it could still send SP=1 after
    # a newer write to SP had completed.
    with pytest.raises(ConnectorHostWriteTimeoutError) as caught:
        await live.write_channel("SLOW:SP", 1.0, timeout=0.3)

    timed_out = caught.value
    assert isinstance(timed_out, TimeoutError)
    assert isinstance(timed_out, ConnectorHostLostError)
    assert (timed_out.target, timed_out.pid, timed_out.cause) == ("live", pid, "write_timeout")
    assert "may or may not have landed" in str(timed_out)
    assert not _alive(pid)
    assert pool.pids() == {}

    assert (await live.read_channel("SR:DCCT", timeout=OK_TIMEOUT_S)).value is not None
    assert live.pid not in (None, pid)
    # The timed-out write reached the connector once and was not sent again.
    assert [line.split() for line in log.read_text().splitlines()] == [["SLOW:SP", "1.0"]]


async def test_kill_on_write_timeout_leaves_a_read_that_times_out_on_the_ping_rule(pools):
    pool = pools(_section(SLOW), timeout_grace_s=0.2, kill_on_write_timeout=True)
    live = await pool.connector("live")
    pid = live.pid

    with pytest.raises(TimeoutError) as caught:
        await live.read_channel("SLOW:X", timeout=0.3)

    assert not isinstance(caught.value, ConnectorHostLostError)
    assert "still answers a ping" in str(caught.value)
    assert live.pid == pid and _alive(pid)


async def test_a_wedged_child_is_caught_by_the_call_deadline_too(pools):
    pool = pools(_section(SLOW), call_deadline_s=0.5, ping_timeout_s=0.5, terminate_grace_s=0.5)
    live = await pool.connector("live")
    first_pid = live.pid

    with pytest.raises(ConnectorHostUnresponsiveError):
        await live.read_channel("WEDGE:X")

    assert not _alive(first_pid)
    assert (await live.read_channel("SR:DCCT", timeout=OK_TIMEOUT_S)).value is not None
    assert live.pid != first_pid


async def test_a_stale_failure_never_drops_the_replacement_child(pools):
    pool = pools(_section(SLOW))
    live = await pool.connector("live")
    old = pool._children[("live", None)]
    os.kill(old.pid, signal.SIGKILL)
    await _wait_for(lambda: old.proxy.dead_reason is not None)

    # The next call replaces the child ...
    assert (await live.read_channel("SR:DCCT", timeout=OK_TIMEOUT_S)).value is not None
    replacement = live.pid
    assert replacement not in (None, old.pid)

    # ... and a failure for the OLD child processed afterwards leaves it alone.
    await pool._discard(old, "stale failure", "exited")
    assert live.pid == replacement
    assert _alive(replacement)


async def test_a_connection_error_from_a_healthy_child_leaves_it_in_place(pools):
    pool = pools(_section(SLOW))
    live = await pool.connector("live")
    pid = live.pid

    with pytest.raises(ConnectionError, match="GONE:X is unreachable") as caught:
        await live.read_channel("GONE:X")

    assert not isinstance(caught.value, ConnectorHostLostError)
    assert live.pid == pid
    assert (await live.read_channel("SR:DCCT")).value is not None


# ------------------------------------------------------------ start failures


@pytest.mark.parametrize(
    ("connector", "error", "message"),
    [
        (FAILING, ConnectionError, "gateway refused the connection"),
        (TIMING_OUT, TimeoutError, "did not answer in time"),
    ],
)
async def test_a_connect_failure_reaches_the_caller_as_the_childs_own_error(
    pools, asyncio_errors, connector, error, message
):
    pool = pools(_section(connector))
    with pytest.raises(error) as caught:
        await pool.connector("live")

    assert not isinstance(caught.value, ConnectorHostError)
    assert message in str(caught.value)
    assert any("'live'" in note and "pid" in note for note in caught.value.__notes__)
    assert pool.pids() == {}
    assert asyncio_errors() == []


async def test_failed_starts_leave_no_unhandled_asyncio_errors(asyncio_errors):
    # Many concurrent failed starts are what exposed abandoned acknowledgement
    # futures being failed after nobody waited for them.
    async def one():
        pool = ConnectorHostPool(_section(FAILING))
        try:
            with pytest.raises(ConnectionError):
                await pool.connector("live")
        finally:
            await pool.close()

    await asyncio.gather(*(one() for _ in range(16)))
    assert asyncio_errors() == []


async def test_a_child_that_never_finishes_connecting_hits_the_start_timeout(pools):
    pool = pools(_section(HANGING), start_timeout_s=1.0, timeout_grace_s=0.0, terminate_grace_s=0.5)
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("live")

    assert caught.value.stage == "init"
    await _wait_for(lambda: not _alive(caught.value.pid))


async def test_an_unresolved_placeholder_refuses_before_anything_is_spawned(
    pools, spawns, monkeypatch
):
    monkeypatch.delenv("OSPREY_POOL_TEST_UNSET", raising=False)
    pool = pools(_section(SLOW, block={"note": "${OSPREY_POOL_TEST_UNSET}"}))
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("live")

    assert caught.value.stage == "config"
    assert caught.value.pid is None
    assert "${OSPREY_POOL_TEST_UNSET}" in str(caught.value)
    assert spawns == []


async def test_a_placeholder_is_resolved_at_spawn_not_at_construction(pools, monkeypatch):
    monkeypatch.delenv("OSPREY_POOL_TEST_LATE", raising=False)
    pool = pools(_section(SLOW, block={"note": "${OSPREY_POOL_TEST_LATE}"}))
    monkeypatch.setenv("OSPREY_POOL_TEST_LATE", "set after the pool was built")
    live = await pool.connector("live")
    assert live.pid is not None


async def test_an_unknown_target_is_refused_in_this_process(pools):
    pool = pools(_section(SLOW))
    with pytest.raises(ValueError, match="Unknown control target"):
        await pool.connector("production")
    assert pool.pids() == {}


def _ca_gateways(address: str, read_port, write_port=None) -> dict:
    return {
        "read_only": {"address": address, "port": read_port},
        "write_access": {"address": address, "port": write_port or read_port},
    }


@pytest.mark.parametrize(
    ("target", "connector_type", "block"),
    [
        ("live", "epics", {}),
        ("live", "epics", {"gateways": {}}),
        # A write-only table on a run that selects read_only: rows exist, but
        # none for the role connect() will look up, so it configures nothing.
        ("live", "epics", {"gateways": {"write_access": {"address": "10.0.0.1", "port": 5064}}}),
        ("standin", "live_standin", {}),
        ("va", "virtual_accelerator", {}),
    ],
)
async def test_a_channel_access_block_with_no_gateway_to_select_is_refused_before_any_spawn(
    pools, spawns, target, connector_type, block
):
    # Spawned, this child would set no EPICS_CA_* at all and search by
    # broadcast. Refused on config alone, so nothing here touches a network.
    section = {"type": connector_type, "connector": {connector_type: {"timeout": 1.0, **block}}}
    pool = pools(section)
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector(target)

    assert caught.value.stage == "config"
    assert caught.value.pid is None
    assert "Channel Access" in str(caught.value) and "broadcast" in str(caught.value)
    assert f"control_system.connector.{connector_type}.gateways.read_only" in str(caught.value)
    assert spawns == []
    assert pool.pids() == {}


@pytest.mark.parametrize(
    ("live_gateways", "standin_gateways", "role"),
    [
        # The stand-in's port variable set to the live gateway's port.
        (
            _ca_gateways("127.0.0.1", 5064),
            _ca_gateways("127.0.0.1", "${OSPREY_POOL_TEST_STANDIN_PORT}"),
            "read_only",
        ),
        # A live block copied into live_standin, the address case aside.
        (
            _ca_gateways("IOC.example.org", 5064),
            _ca_gateways(" ioc.example.org", "5064"),
            "read_only",
        ),
        # Only the live write gateway matches.
        (_ca_gateways("10.0.0.1", 5064, 5065), _ca_gateways("10.0.0.1", 5065), "write_access"),
    ],
)
async def test_a_standin_that_would_dial_the_live_machine_is_refused_before_any_spawn(
    pools, spawns, monkeypatch, live_gateways, standin_gateways, role
):
    monkeypatch.setenv("OSPREY_POOL_TEST_STANDIN_PORT", "5064")
    section = {
        "type": "epics",
        "connector": {
            "epics": {"gateways": live_gateways},
            "live_standin": {"gateways": standin_gateways},
        },
    }
    pool = pools(section)
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("standin")

    assert caught.value.stage == "config"
    assert caught.value.pid is None
    assert f"the {role!r} gateway the live machine derives" in str(caught.value)
    assert spawns == []
    assert pool.pids() == {}


async def test_a_standin_on_a_named_host_apart_from_the_live_machine_passes_the_config_stage(
    pools, monkeypatch
):
    # The consumer's compose shape: the stand-in is a service name, neither
    # loopback nor resolvable here, on the same port number as the live
    # gateway but on another host. Spawning is made to fail, so reaching the
    # spawn stage is the proof the config stage let it through, and no child
    # ever dials anything.
    async def no_spawn(*args, **kwargs):
        raise OSError("pool test: spawning is disabled")

    monkeypatch.setattr(pool_module, "spawn_host", no_spawn)
    section = {
        "type": "epics",
        "connector": {
            "epics": {"gateways": _ca_gateways("127.0.0.1", 5064)},
            "live_standin": {"gateways": _ca_gateways("tuning-epics-ioc", 5064)},
        },
    }
    pool = pools(section)
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("standin")

    assert caught.value.stage == "spawn"


# ------------------------------------------------------------ lifetime


async def test_close_stops_every_child(pools):
    pool = pools(_section(SLOW))
    await pool.connector("live")
    await pool.connector("live", execution_mode="readonly")
    pids = list(pool.pids().values())

    await pool.close()

    await _wait_for(lambda: not any(_alive(pid) for pid in pids))
    with pytest.raises(RuntimeError, match="closed"):
        await pool.connector("live")


async def test_close_during_a_start_leaves_nothing_running(pools, spawns):
    pool = pools(_section(SLOW_START))
    starting = asyncio.create_task(pool.connector("live"))
    await _wait_for(lambda: spawns)

    await pool.close()

    with pytest.raises(RuntimeError, match="closed"):
        await starting
    assert len(spawns) == 1
    process = spawns[0][1]
    assert process.returncode is not None
    assert pool.pids() == {}


async def test_disconnect_during_a_start_waits_for_it_and_leaves_no_child_running(pools, spawns):
    pool = pools(_section(SLOW_START))
    starting = asyncio.create_task(pool.connector("live"))
    await _wait_for(lambda: spawns)

    # A handle on the key, from before the start has handed one out.
    await PooledConnector(pool, ("live", None)).disconnect()

    # The disconnect waited for the start and stopped the child it produced ...
    process = spawns[0][1]
    assert process.returncode is not None
    assert not _alive(process.pid)
    assert pool.pids() == {}
    # ... and the start itself completed, handing back a handle whose child is
    # already gone; its next call would start a fresh one.
    handle = await starting
    assert handle.pid is None
    assert len(spawns) == 1


async def test_close_with_a_call_in_flight_fails_it_as_stopped(pools):
    pool = pools(_section(SLOW))
    live = await pool.connector("live")
    pid = live.pid
    hung = asyncio.create_task(live.read_channel("SLOW:X"))
    await asyncio.sleep(0.2)

    await pool.close()

    with pytest.raises(ConnectorHostLostError) as caught:
        await asyncio.wait_for(hung, 10.0)
    assert not isinstance(caught.value, ConnectorHostUnresponsiveError)
    assert caught.value.cause == "stopped"
    assert not _alive(pid)


async def test_the_pool_refuses_a_second_event_loop(pools):
    pool = pools(_section(SLOW))
    await pool.connector("live")

    with pytest.raises(RuntimeError, match="event loop"):
        await asyncio.to_thread(asyncio.run, pool.connector("live"))


def test_children_do_not_outlive_a_parent_that_is_killed(isolated):
    script = textwrap.dedent(
        f"""
        import asyncio, sys
        from osprey_connectors.ipc.pool import ConnectorHostPool

        async def main():
            pool = ConnectorHostPool({_section(SLOW)!r})
            await pool.connector("live")
            await pool.connector("live", execution_mode="readonly")
            print(" ".join(str(pid) for pid in pool.pids().values()), flush=True)
            await asyncio.sleep(3600)

        asyncio.run(main())
        """
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": PYTHONPATH},
        cwd=str(isolated),
    )
    try:
        line = parent.stdout.readline()
        pids = [int(pid) for pid in line.split()]
        assert len(pids) == 2, line
        assert all(_alive(pid) for pid in pids)
    finally:
        parent.kill()
        parent.wait()

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
        time.sleep(0.1)
    assert not any(_alive(pid) for pid in pids), "a child outlived its parent"


# ------------------------------------------------------------ handles and batches


def test_a_handle_names_the_key_it_is_bound_to():
    pool = ConnectorHostPool(_section(SLOW))

    live = PooledConnector(pool, ("live", None))
    live_ro = PooledConnector(pool, ("live", "readonly"))

    assert (live.target, live.execution_mode) == ("live", None)
    assert (live_ro.target, live_ro.execution_mode) == ("live", "readonly")


async def test_batched_reads_and_writes_reach_the_child_and_come_back_in_order(pools, writable):
    section, config_file, log = writable
    pool = pools(section, config_file=config_file)
    live = await pool.connector("live")

    values = await live.read_multiple_channels(["SR:A", "SR:B"], timeout=OK_TIMEOUT_S)
    results = await live.write_multiple_channels([("SR:SP1", 1.0), ("SR:SP2", 2.0)])

    assert sorted(values) == ["SR:A", "SR:B"]
    assert all(value.value is not None for value in values.values())
    assert [(r.channel_address, r.value_written) for r in results] == [
        ("SR:SP1", 1.0),
        ("SR:SP2", 2.0),
    ]
    assert all(result.outcome is not WriteOutcome.REFUSED for result in results)
    assert [line.split() for line in log.read_text().splitlines()] == [
        ["SR:SP1", "1.0"],
        ["SR:SP2", "2.0"],
    ]


# ------------------------------------------------------------ calls lost to a write timeout


@pytest.mark.parametrize(
    ("method", "args", "error", "lands"),
    [
        ("read_channel", ("SLOW:RB",), ConnectorHostLostError, False),
        ("write_channel", ("SLOW:SP2", 2.0), ConnectorHostWriteTimeoutError, True),
    ],
)
async def test_a_call_in_flight_on_a_child_killed_for_another_writes_timeout_is_lost_with_it(
    pools, writable, method, args, error, lands
):
    section, config_file, _ = writable
    pool = pools(section, config_file=config_file, timeout_grace_s=0.2, kill_on_write_timeout=True)
    live = await pool.connector("live")
    pid = live.pid

    bystander = asyncio.create_task(getattr(live, method)(*args, timeout=30.0))
    await asyncio.sleep(0.2)
    with pytest.raises(ConnectorHostWriteTimeoutError):
        await live.write_channel("SLOW:SP", 1.0, timeout=0.3)

    with pytest.raises(ConnectorHostLostError) as caught:
        await asyncio.wait_for(bystander, 10.0)
    lost = caught.value
    # Only a write is a write timeout; a read on the same child was just lost.
    assert type(lost) is error
    assert (lost.pid, lost.cause) == (pid, "write_timeout")
    assert "killed because another write on it missed its deadline" in str(lost)
    assert f"while {method!r} was in flight" in str(lost)
    assert ("The write may or may not have landed." in str(lost)) is lands
    assert not _alive(pid)


# ------------------------------------------------------------ start refusals


async def test_a_placeholder_inside_a_list_in_the_block_refuses_before_anything_is_spawned(
    pools, spawns, monkeypatch
):
    monkeypatch.delenv("OSPREY_POOL_TEST_UNSET_LIST", raising=False)
    block = {"channels": ["SR:A", ("SR:B", "${OSPREY_POOL_TEST_UNSET_LIST}")]}
    pool = pools(_section(SLOW, block=block))
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("live")

    assert caught.value.stage == "config"
    assert caught.value.pid is None
    assert "still carries ${OSPREY_POOL_TEST_UNSET_LIST} after environment resolution" in str(
        caught.value
    )
    assert spawns == []


async def test_a_standin_on_a_deployment_with_no_live_machine_passes_the_config_stage(
    pools, monkeypatch
):
    # No block names a real machine, so ``live`` does not resolve here and
    # there is no live endpoint for the stand-in to collide with.
    async def no_spawn(*args, **kwargs):
        raise OSError("pool test: spawning is disabled")

    monkeypatch.setattr(pool_module, "spawn_host", no_spawn)
    section = {
        "type": "live_standin",
        "connector": {"live_standin": {"gateways": _ca_gateways("127.0.0.1", 5064)}},
    }
    pool = pools(section)
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("standin")

    assert caught.value.stage == "spawn"
    assert "pool test: spawning is disabled" in str(caught.value)


async def test_a_child_that_exits_before_answering_init_is_refused_at_the_init_stage(pools, spawns):
    pool = pools(_section(EXITING))
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("live")

    assert caught.value.stage == "init"
    assert "exited before answering its init frame (exit code 3)" in str(caught.value)
    assert spawns[0][1].returncode == 3
    assert pool.pids() == {}


def _doctor_init_reports(monkeypatch, doctor):
    """Pass every child's post-connect report through *doctor* on its way in.

    The child is real and answers honestly; only what the pool is handed is
    changed, so each verify check sees exactly one lie.
    """

    class DoctoringProxy(pool_module.ConnectorHostProxy):
        async def supervisor_request(self, method, kwargs, timeout):
            reply = await super().supervisor_request(method, kwargs, timeout)
            return doctor(reply) if method == "init" else reply

    monkeypatch.setattr(pool_module, "ConnectorHostProxy", DoctoringProxy)


def _with(**changes):
    return lambda report: {**report, **changes}


@pytest.mark.parametrize(
    ("mode", "doctor", "stage", "message"),
    [
        pytest.param(
            None,
            lambda report: [report],
            "init",
            "answered its init frame with list, not the post-connect report",
            id="report-not-a-dict",
        ),
        pytest.param(
            None,
            _with(target="va"),
            "verify",
            "reports target 'va' where 'live' was asked for",
            id="target-mismatch",
        ),
        pytest.param(
            None,
            _with(connector_type="epics"),
            "verify",
            f"reports connector_type 'epics' where {SLOW!r} was asked for",
            id="connector-type-mismatch",
        ),
        pytest.param(
            "readonly",
            _with(readonly_run=False),
            "verify",
            "was asked to run readonly but reports it is not in a readonly run",
            id="readonly-child-not-readonly",
        ),
        pytest.param(
            None,
            _with(mode="gateway", host="10.9.9.9"),
            "verify",
            "came up somewhere other than derived",
            id="endpoint-verification-fails",
        ),
    ],
)
async def test_a_child_whose_report_disagrees_with_the_derivation_is_refused_and_stopped(
    pools, spawns, monkeypatch, mode, doctor, stage, message
):
    _doctor_init_reports(monkeypatch, doctor)
    pool = pools(_section(SLOW))
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("live", execution_mode=mode)

    assert caught.value.stage == stage
    assert message in str(caught.value)
    assert len(spawns) == 1
    process = spawns[0][1]
    assert caught.value.pid == process.pid
    assert process.returncode is not None
    assert not _alive(process.pid)
    assert pool.pids() == {}


# ------------------------------------------------------------ lifetime, continued


async def test_leaving_an_async_with_block_stops_every_child_and_closes_the_pool():
    async with ConnectorHostPool(_section(SLOW)) as pool:
        live = await pool.connector("live")
        pid = live.pid
        assert _alive(pid)

    await _wait_for(lambda: not _alive(pid))
    assert pool.pids() == {}
    with pytest.raises(RuntimeError, match="ConnectorHostPool is closed"):
        await pool.connector("live")


async def test_a_caller_queued_behind_a_start_that_close_interrupts_is_refused_as_closed(
    pools, spawns
):
    pool = pools(_section(SLOW_START))
    starting = asyncio.create_task(pool.connector("live"))
    await _wait_for(lambda: spawns)
    queued = asyncio.create_task(pool.connector("live"))
    # Past the pool's first closed check, and waiting on the key's lock.
    await asyncio.sleep(0.1)

    await pool.close()

    with pytest.raises(RuntimeError, match="closed while this child was starting"):
        await starting
    with pytest.raises(RuntimeError, match=r"^ConnectorHostPool is closed$"):
        await queued
    assert len(spawns) == 1


async def test_disconnecting_a_key_with_no_child_starts_nothing(pools, spawns):
    pool = pools(_section(SLOW))

    await PooledConnector(pool, ("live", None)).disconnect()

    assert spawns == []
    assert pool.pids() == {}


# ------------------------------------------------------------ helpers


async def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.05)
