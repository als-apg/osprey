"""The connector-host pool over real Channel Access, against two soft IOCs.

Two IOCs serve the SAME record names with DIFFERENT values on two ports of this
host — the shape a facility's machine and its stand-in simulator have. The
pool's ``live`` target reaches the first through the deployment's ``epics``
block, and its ``standin`` target reaches the second through the
``live_standin`` block, whose port is ``${EPICS_TESTING_PORT}`` and is only
chosen at runtime.

The IOCs are ``python -m epicscorelibs.ioc``: the minimal soft IOC that ships
with ``epicscorelibs``, which the EPICS connector's stack already installs, so
this suite needs no container and no EPICS base build. This test process never
loads a Channel Access client itself: every read and write goes through a pool
child, which is the only way one process can talk to both IOCs at once.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
import yaml

from osprey_connectors.control_system.base import WriteOutcome
from osprey_connectors.errors import ChannelWriteBlockedError
from osprey_connectors.ipc.pool import ConnectorHostPool

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHONPATH = os.pathsep.join(
    [str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src")]
)


#: Seed values, deliberately far apart so a read from the wrong IOC is obvious.
SEED_A = 1.0
SEED_B = 100.0

IOC_READY_TIMEOUT_S = 30.0
CA_TIMEOUT_S = 5.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class SoftIOC:
    """One ``epicscorelibs`` soft IOC on its own port, serving the test records."""

    def __init__(self, directory: Path, name: str, prefix: str, seed: float) -> None:
        self.port = _free_port()
        database = directory / f"{name}.db"
        database.write_text(
            f'record(ao, "{prefix}:SP") {{ field(VAL, "{seed}") field(PREC, "3") }}\n'
            f'record(ai, "{prefix}:RB") {{ field(VAL, "{seed}") field(PREC, "3") }}\n'
        )
        # Output to a file, never an unread pipe: an IOC that fills a pipe
        # nobody drains blocks, and the file is what a failure message quotes.
        self.log = directory / f"{name}.log"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("EPICS_CA", "EPICS_PVA"))}
        env.update(
            {
                "EPICS_CA_SERVER_PORT": str(self.port),
                "EPICS_CAS_INTF_ADDR_LIST": "127.0.0.1",
                "EPICS_CA_AUTO_ADDR_LIST": "NO",
            }
        )
        # stdin stays open: the IOC's interactive shell exits the IOC on EOF.
        with self.log.open("wb") as output:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "epicscorelibs.ioc", "-d", str(database)],
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=env,
            )
        self._wait_until_serving()

    def _wait_until_serving(self) -> None:
        deadline = time.monotonic() + IOC_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"soft IOC exited with {self.process.returncode}: "
                    f"{self.log.read_text(errors='replace')}"
                )
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                    return
            except OSError:
                time.sleep(0.1)
        self.stop()
        raise RuntimeError(f"soft IOC never listened on 127.0.0.1:{self.port}")

    def stop(self) -> None:
        self.process.kill()
        self.process.wait(timeout=10)


@pytest.fixture
def prefix() -> str:
    """One prefix per test, so an IOC this test did not start — another xdist
    worker's, or one that fell back to a different port — serves none of the
    names this test reads, and a mix-up fails as "cannot connect" instead of
    returning somebody else's plausible value."""
    return f"POOL{uuid.uuid4().hex[:8].upper()}"


@pytest.fixture
def iocs(tmp_path, prefix):
    """Two IOCs serving the SAME names — this test's prefix — with different seeds."""
    first = SoftIOC(tmp_path, "machine", prefix, SEED_A)
    try:
        second = SoftIOC(tmp_path, "standin", prefix, SEED_B)
    except BaseException:
        first.stop()
        raise
    yield first, second
    first.stop()
    second.stop()


def _gateways(address: str, port) -> dict:
    endpoint = {"address": address, "port": port, "use_name_server": False}
    return {"read_only": dict(endpoint), "write_access": dict(endpoint)}


def _control_system(machine_port: int, *, arm: str) -> dict:
    """The section tuning_scripts ships: a live machine and a runtime-port stand-in.

    ``arm`` is ``"both"`` (the deployment-wide key) or ``"epics"`` (the live
    machine's block alone).
    """
    return {
        "type": "epics",
        "writes_enabled": arm == "both",
        "connector": {
            "epics": {
                **({"writes_enabled": True} if arm == "epics" else {}),
                "timeout": CA_TIMEOUT_S,
                "gateways": _gateways("127.0.0.1", machine_port),
            },
            "live_standin": {
                "timeout": CA_TIMEOUT_S,
                "gateways": _gateways("127.0.0.1", "${EPICS_TESTING_PORT}"),
            },
        },
    }


@pytest.fixture
def deployment(tmp_path, monkeypatch, iocs):
    """Write the config to disk and return a factory for pools over it."""
    machine, standin = iocs
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    monkeypatch.delenv("EPICS_TESTING_PORT", raising=False)
    monkeypatch.setenv("PYTHONPATH", PYTHONPATH)

    # Each test closes its own pool in its own event loop; see the pool's
    # loop-affinity rule.
    def make(arm: str) -> ConnectorHostPool:
        section = _control_system(machine.port, arm=arm)
        config_file = tmp_path / f"config-{arm}.yml"
        config_file.write_text(yaml.safe_dump({"control_system": section}))
        pool = ConnectorHostPool(section, config_file=config_file)
        # Chosen after the pool exists, the way an integration run picks it.
        monkeypatch.setenv("EPICS_TESTING_PORT", str(standin.port))
        return pool

    return make, machine, standin


# ----------------------------------------------------------------- the wire


async def test_two_targets_each_reach_only_their_own_ioc(deployment, prefix):
    setpoint, readback = f"{prefix}:SP", f"{prefix}:RB"
    make, machine, standin = deployment
    pool = make("both")
    try:
        live, sim = await asyncio.gather(pool.connector("live"), pool.connector("standin"))
        assert live.report["port"] == machine.port
        assert live.report["connector_type"] == "epics"
        assert sim.report["port"] == standin.port
        assert sim.report["connector_type"] == "live_standin"

        seeds = await asyncio.gather(
            live.read_channel(readback, timeout=CA_TIMEOUT_S),
            sim.read_channel(readback, timeout=CA_TIMEOUT_S),
        )
        assert [value.value for value in seeds] == [SEED_A, SEED_B]

        written = await asyncio.gather(
            live.write_channel_checked(setpoint, 11.0, confirm=True, timeout=CA_TIMEOUT_S),
            sim.write_channel_checked(setpoint, 222.0, confirm=True, timeout=CA_TIMEOUT_S),
        )
        assert [result.outcome for result in written] == [WriteOutcome.CONFIRMED] * 2

        after = await asyncio.gather(
            live.read_channel(setpoint, timeout=CA_TIMEOUT_S),
            sim.read_channel(setpoint, timeout=CA_TIMEOUT_S),
        )
        assert [value.value for value in after] == [11.0, 222.0]
        assert await live.validate_channel(setpoint) is True
    finally:
        await pool.close()


async def test_only_the_armed_type_can_write(deployment, prefix):
    setpoint = f"{prefix}:SP"
    make, _, _ = deployment
    pool = make("epics")
    try:
        live = await pool.connector("live")
        live_ro = await pool.connector("live", execution_mode="readonly")
        sim = await pool.connector("standin")
        assert live.report["selected_role"] == "write_access"
        assert live_ro.report["selected_role"] == "read_only"
        assert sim.report["selected_role"] == "read_only"
        assert sim.report["writes_enabled"] is False

        # A readonly child refuses even on the armed target, before the IOC.
        refused = await live_ro.write_channel(setpoint, 42.0, timeout=CA_TIMEOUT_S)
        assert refused.outcome is WriteOutcome.REFUSED
        assert (await live_ro.read_channel(setpoint, timeout=CA_TIMEOUT_S)).value == SEED_A

        landed = await live.write_channel(setpoint, 5.0, confirm=True, timeout=CA_TIMEOUT_S)
        assert landed.outcome is WriteOutcome.CONFIRMED

        refused = await sim.write_channel(setpoint, 999.0, confirm=True, timeout=CA_TIMEOUT_S)
        assert refused.outcome is WriteOutcome.REFUSED
        assert refused.refusal_reason == "WRITES_DISABLED"
        with pytest.raises(ChannelWriteBlockedError):
            await sim.write_channel_checked(setpoint, 999.0, confirm=True)

        assert (await sim.read_channel(setpoint, timeout=CA_TIMEOUT_S)).value == SEED_B
        assert (await live.read_channel(setpoint, timeout=CA_TIMEOUT_S)).value == 5.0
    finally:
        await pool.close()
