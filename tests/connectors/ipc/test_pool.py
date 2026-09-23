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


async def test_a_hung_call_on_one_target_does_not_hold_up_another(pools):
    pool = pools(_section(SLOW))
    live = await pool.connector("live")
    live_ro = await pool.connector("live", execution_mode="readonly")

    hung = asyncio.create_task(live.read_channel("SLOW:X"))
    value = await asyncio.wait_for(live_ro.read_channel("SR:DCCT"), 5.0)

    assert value.value is not None
    assert not hung.done()
    hung.cancel()


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


async def test_a_config_file_that_arms_what_the_section_does_not_is_refused(pools, writable):
    _, config_file, log = writable
    # Gatewayless mock: no gateway role could reveal the disagreement, so only
    # the posture check stands between this child and an armed write.
    pool = pools(_section(SLOW, writes_enabled=False), config_file=config_file)
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("live")

    assert caught.value.stage == "verify"
    assert "writes armed" in str(caught.value)
    assert not log.exists()


async def test_an_inherited_config_file_that_arms_writes_is_refused(pools, writable, monkeypatch):
    _, config_file, _ = writable
    monkeypatch.setenv("CONFIG_FILE", str(config_file))
    pool = pools(_section(SLOW, writes_enabled=False))
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("live")
    assert caught.value.stage == "verify"


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


async def test_an_unresolved_placeholder_refuses_before_anything_is_spawned(pools, monkeypatch):
    monkeypatch.delenv("OSPREY_POOL_TEST_UNSET", raising=False)
    pool = pools(_section(SLOW, block={"note": "${OSPREY_POOL_TEST_UNSET}"}))
    with pytest.raises(ConnectorHostStartError) as caught:
        await pool.connector("live")

    assert caught.value.stage == "config"
    assert caught.value.pid is None
    assert "${OSPREY_POOL_TEST_UNSET}" in str(caught.value)


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


# ------------------------------------------------------------ helpers


async def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.05)
