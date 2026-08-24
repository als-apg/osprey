"""The target switch against a real machine, not a second simulator.

``test_target_switch.py`` proves the switch's *mechanics* — processes, drains,
generations — by pointing both targets at the same virtual-accelerator image.
That is the honest way to test a supervisor, and it is deliberately not what
this module does. Here the two ends are different *kinds* of Channel Access
server: the deployment's ``va`` target is the virtual accelerator, and its
``live`` target is a stock EPICS ``softIoc`` (``docker/bench-ioc/``) holding
seeded constants, an access-security file that refuses one write, and a handful
of records the virtual accelerator's manifest has never heard of.

The invariant this module exists to pin: **a session pointed at the live target
is talking to the live machine, and everything that follows from that machine's
own behaviour reaches the operator unchanged.** Not "a switch was reported" —
every claim below is settled by something only the bench IOC can produce:

* a shared channel that reads ``7.25`` here and something else there;
* a ``caput`` the *IOC* denies (its ASG withholds WRITE), arriving as
  ``blocked=True`` with ``refusal_reason="CONTROL_SYSTEM_REFUSED"`` — which is a
  different claim from the reference monitor's own limits refusal, and is
  asserted to be distinguishable from it;
* records that exist on one target and not the other, in both directions;
* record textures a simulator does not have to reproduce: an ``mbbi`` reporting
  its index unconverted, a ``calc`` reporting its derived value, and a readback
  driven past its ``HIHI`` threshold reporting the alarm by name;
* the machine going away mid-session, and the session saying so.

The deployment's baseline is the virtual accelerator (``control_system.type:
virtual_accelerator``), which is what makes the live gate testable at all: FR-8
waives the operator acknowledgment for a session *returning* to its baseline, so
a live-baseline deployment could never show the acknowledgment doing anything.
Here every switch to ``live`` is a switch *toward* the real machine and is
judged as one.

Process boundaries, restated because they are load-bearing
----------------------------------------------------------
This pytest process never becomes a Channel Access client. Reads and writes go
through real ``python -m osprey_connectors.ipc.host`` children, exactly as a
deployment's do — which is also why the refusal above can be asserted at all:
``tests/fixtures/bench_ioc.py``'s convenience ``caput`` collapses its result to
four scalars, and ``refusal_reason`` is not among them. The structured
``ChannelWriteResult`` only exists on the connector path, so the connector path
is what these tests drive. Container lifecycle and the seeded constants come
from that fixture module; nothing about the record set is restated here.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.mcp_server.control_system import error_handling, target_state
from osprey.mcp_server.control_system import server_context as server_context_mod
from osprey.mcp_server.control_system import target_eligibility as te
from osprey.mcp_server.control_system.connector_host_manager import ConnectorHostManager
from osprey.mcp_server.control_system.endpoint_prober import (
    STATUS_OK,
    STATUS_UNREACHABLE,
    EndpointProber,
)
from osprey.mcp_server.control_system.server_context import MCPServerConfig
from osprey.mcp_server.control_system.tools import control_target
from osprey.mcp_server.control_system.tools.channel_read import channel_read
from osprey_connectors.control_system.base import ChannelValue, raise_for_write_result
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelWriteBlockedError
from tests.fixtures import bench_ioc as bench
from tests.fixtures.control_context import context_for
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn
from tests.va.e2e import conftest as e2e_conftest

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_PATHS = (str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src"))

#: The virtual accelerator this module boots for its ``va`` target, and the
#: simulation data it serves. Same image and same data directory as
#: ``test_target_switch.py``; a container of its own, on an ephemeral port,
#: because writing setpoints into the directory's shared session container would
#: move a lattice other modules read.
VA_IMAGE = os.environ.get("OSPREY_VA_E2E_IMAGE", "osprey-va-full:latest")
VA_DATA_DIR = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation"
VA_CONTAINER_PREFIX = "osprey-va-e2e-bench-live"

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 15

# -- what each target serves ------------------------------------------------

#: Served by both, with different values. The switch's probe channel too.
PROBE_CHANNEL = bench.PROBE_CHANNEL
BENCH_PROBE_VALUE = bench.PROBE_VALUE
#: Two more shared channels, so "the values differ" is not one measurement.
SHARED_BPM_X = "SR:DIAG:BPM:01:POSITION:X"
BENCH_BPM_X_VALUE = 0.412
#: Read by anyone, written by no one: the bench IOC's ASG withholds WRITE.
PROTECTED_SP = bench.PROTECTED_SP
BENCH_PROTECTED_VALUE = bench.PROTECTED_VALUE
#: What this module tries to write there. Well inside the channel's shipped
#: +-12 A band, and load-bearingly so: a value outside it would be refused by
#: OSPREY's own limits validator BEFORE any caput was issued, and the refusal
#: this module is about -- the control system's own -- would never be reached.
PROTECTED_WRITE_VALUE = -4.0
#: Writable on both, and listed in the shipped limits database at +-12 A --
#: which is what lets a deliberately out-of-range write be refused by the
#: reference monitor rather than by the machine.
WRITABLE_SP = bench.WRITABLE_SP
LIMITS_MAX = 12.0
OUT_OF_LIMITS_VALUE = 40.0
VA_WRITE_VALUE = 1.25

#: Bench-only: the virtual accelerator's manifest has no SR:DIAG:STRIPLINE.
BENCH_ONLY_AMPLITUDE = "SR:DIAG:STRIPLINE:01:AMPLITUDE:RB"
BENCH_ONLY_AMPLITUDE_VALUE = 12.75
MODE_STATE = bench.MODE_STATE
MODE_STATE_VALUE = bench.MODE_STATE_VALUE
SCALED_AMPLITUDE = bench.SCALED_AMPLITUDE
SCALED_AMPLITUDE_VALUE = bench.SCALED_AMPLITUDE_VALUE

#: Virtual-accelerator-only: the bench database deliberately omits the DCCT.
VA_ONLY_CHANNEL = "SR:DIAG:DCCT:01:CURRENT:RB"

#: The one bench record carrying alarm limits (``HIHI 11.0``, ``HHSV MAJOR``).
#: The alarm texture has to be driven here rather than on a setpoint because no
#: setpoint in ``bench.db`` sets any alarm field -- ``SR:MAG:HCM:01:CURRENT:SP``
#: carries drive limits and nothing else, so it can never alarm. That this is a
#: readback is also why :func:`limits_database` exists: the shipped limits mark
#: readbacks unwritable, correctly, for a deployment.
ALARM_CHANNEL = "SR:MAG:VCM:02:CURRENT:RB"
ALARM_SEEDED_VALUE = -3.5
ALARM_HIHI_THRESHOLD = 11.0
ALARM_DRIVE_VALUE = 11.5
#: EPICS ``epicsSevMAJOR``. The record's ``HHSV`` names MAJOR, so a HIHI here is
#: a severity-2 alarm and not merely a status string.
ALARM_MAJOR_SEVERITY = 2

# -- bounds -----------------------------------------------------------------

#: The connector's own per-operation ceiling, and so the cost of a read of a
#: channel nothing serves when no explicit timeout is passed.
CONNECTOR_TIMEOUT_S = 20.0
#: Bound on a read or write this module expects to be answered.
CALL_TIMEOUT_S = 20.0
#: Bound on a read this module expects to FAIL. Short on purpose: it is paid in
#: full every time an absent channel or a stopped container is asked for.
ABSENT_TIMEOUT_S = 6.0
#: Bound on "spawned and answered its init frame" -- a cold pyepics import is
#: not fast.
SPAWN_TIMEOUT_S = 60.0
PROBE_TIMEOUT_S = 15.0
DRAIN_TIMEOUT_S = 2.0
#: Boot deadline for the virtual accelerator: pinned linux/amd64, emulated here.
VA_BOOT_TIMEOUT_S = 180.0
#: Per-endpoint TCP connect timeout for the reachability prober.
PROBE_CONNECT_TIMEOUT_S = 3.0


# ---------------------------------------------------------------------------
# The virtual accelerator container
# ---------------------------------------------------------------------------


def _require_va_image() -> None:
    """Fail loudly unless the VA image can serve on a port other than 5064.

    The same precondition ``test_target_switch.py`` states, for the same reason:
    the CA *server* library reads ``EPICS_CAS_SERVER_PORT`` and does not fall
    back to the client-side variable, so an image whose entry point does not
    derive one from the other keeps binding its build-time default while telling
    this module's children some other port. The symptom would be an unexplained
    boot timeout; this turns it into a sentence naming the fix.
    """
    inspected = bench.docker(
        "image", "inspect", VA_IMAGE, "--format", "{{.Architecture}}|{{.Config.Cmd}}", timeout=60
    )
    if inspected.returncode != 0:
        pytest.fail(
            f"image {VA_IMAGE!r} is not present. Build it with "
            f"scripts/va/build_and_boot_check.sh, or name another with OSPREY_VA_E2E_IMAGE."
        )
    architecture, _, command = inspected.stdout.strip().partition("|")
    if "EPICS_CAS_SERVER_PORT" not in command:
        pytest.fail(
            f"image {VA_IMAGE!r} ({architecture}) does not derive EPICS_CAS_SERVER_PORT from "
            f"EPICS_CA_SERVER_PORT, so it cannot serve on any port but its baked default. "
            f"Rebuild it (scripts/va/build_and_boot_check.sh) or point OSPREY_VA_E2E_IMAGE at "
            f"a current build. Its entry point is: {command}"
        )


def _va_served(port: int) -> bool:
    """Whether a virtual accelerator answers the probe channel on *port*.

    Out of process, for the reason this directory's ``conftest.py`` gives: libca
    latches ``EPICS_CA_*`` at initialisation and its contexts are per-thread, so
    a main-thread pyepics call in this process would deadlock the connector-host
    children these tests spend their time talking to.
    """
    code = (
        "import sys, epics\n"
        f"v = epics.caget({PROBE_CHANNEL!r}, timeout=1.0, connection_timeout=1.0)\n"
        "sys.stdout.write('SERVED' if v is not None else 'NONE')\n"
        "sys.stdout.flush()\n"
        "import os; os._exit(0)\n"
    )
    environment = {
        **os.environ,
        "EPICS_CA_NAME_SERVERS": f"localhost:{port}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }
    for stale in ("EPICS_CA_ADDR_LIST", "EPICS_CA_SERVER_PORT", "EPICS_CAS_SERVER_PORT"):
        environment.pop(stale, None)
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return False
    return probe.stdout.strip() == "SERVED"


@pytest.fixture(scope="module")
def bench_endpoint() -> Iterator[bench.BenchIOC]:
    """One bench IOC, up and answering, for the life of this module.

    ``tests.fixtures.bench_ioc`` ships this same fixture and it would normally be
    imported rather than restated; it is spelled out here only because a test
    that takes an imported fixture by name shadows the import, which this repo's
    lint refuses. The boot contract itself is not restated -- the image
    precondition and the container lifecycle are the fixture module's, called.

    Module-scoped because booting an emulated container per test would dominate
    the run. The record set is reseeded on every boot, so the only state a test
    can see is what an earlier one wrote: the one test that writes to the bench
    restores the seeded value, and the one successful write anywhere in this
    module lands on the virtual accelerator and restores the simulator's
    quiescent 0.0. Every other write here is refused and reaches no record.
    """
    bench.require_bench_image()
    with bench.bench_ioc() as ioc:
        yield ioc


@pytest.fixture(scope="module")
def va_endpoint() -> Iterator[int]:
    """One virtual accelerator, up and serving, for the life of this module.

    The published port and the server's own port are the same number by
    construction -- a CA search reply carries the server's port, so a remap would
    hand every client an address nothing answers on -- and that number also names
    the container, which is what makes concurrent runs safe.
    """
    _require_va_image()
    port = bench.free_port()
    name = f"{VA_CONTAINER_PREFIX}-{port}"
    # Stale cleanup only: the port is this run's alone, so this can name nothing
    # a concurrent run is using.
    bench.docker("rm", "-f", name, timeout=60)

    started = bench.docker(
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"EPICS_CA_SERVER_PORT={port}",
        "-p",
        f"127.0.0.1:{port}:{port}/tcp",
        "-v",
        f"{VA_DATA_DIR}:/data/simulation:ro",
        VA_IMAGE,
    )
    if started.returncode != 0:
        with contextlib.suppress(subprocess.SubprocessError):
            bench.docker("rm", "-f", name, timeout=60)
        raise RuntimeError(f"docker run failed for {name}:\n{started.stdout}\n{started.stderr}")

    try:
        deadline = time.monotonic() + VA_BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            if _va_served(port):
                break
            time.sleep(1.0)
        else:
            logs = bench.docker("logs", "--tail", "40", name, timeout=60)
            raise RuntimeError(
                f"{name} never served {PROBE_CHANNEL} within {VA_BOOT_TIMEOUT_S}s.\n"
                f"{logs.stdout}\n{logs.stderr}"
            )
        yield port
    finally:
        with contextlib.suppress(subprocess.SubprocessError):
            bench.docker("rm", "-f", name, timeout=60)


# ---------------------------------------------------------------------------
# The deployment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def limits_database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The shipped limits database, plus the one entry this module has to add.

    Everything these tests write is judged against the *shipped* channel limits,
    because a switch toward the live machine is only judged on real terms if the
    limits it is judged against are the real ones. The single exception is the
    alarm-limit carrier: it is a readback, and the shipped database says
    readbacks are not writable -- correctly, for a deployment. Driving a bench
    record into its HIHI band is not a deployment operation, and the bench IOC
    has no other record carrying alarm limits, so this file grants that one
    channel a write band and changes nothing else.

    Derived from the shipped file at run time rather than copied into the tree,
    so the other 2900-odd entries cannot drift away from the ones a deployment
    actually enforces.
    """
    database = json.loads(e2e_conftest.LIMITS_DB_PATH.read_text(encoding="utf-8"))
    database[ALARM_CHANNEL] = {"writable": True, "min_value": -50.0, "max_value": 50.0}
    path = tmp_path_factory.mktemp("bench_live_limits") / "channel_limits.json"
    path.write_text(json.dumps(database), encoding="utf-8")
    return path


def raw_config(
    *,
    bench_port: int,
    va_port: int,
    limits_db: Path,
    acknowledged: bool = True,
) -> dict[str, Any]:
    """A switch-capable deployment: ``va`` the simulator, ``live`` the bench IOC.

    ``control_system.type`` is ``virtual_accelerator``, so ``va`` is this
    deployment's baseline and ``live`` resolves through the connector table to
    the one non-simulated block configured -- ``epics``, pointed at the bench
    IOC. Every switch to ``live`` is therefore a switch *toward* the real
    machine, which is the only direction FR-8's live gate applies to.

    Both blocks name an explicit gateway port: the containers bind ephemeral
    ports and nothing may guess them. Both gateway roles name the same endpoint
    because there is one server at each end; the write-capable role is the one
    the child selects, so a refused write is refused by the IOC rather than by a
    client that never asked.

    The FR-8 posture is set for real -- strict limits against a database derived
    from the shipped one, plus the operator acknowledgment -- so that
    ``acknowledged=False`` isolates exactly one missing thing.
    """

    def block(port: int) -> dict[str, Any]:
        gateway = {"address": "localhost", "port": port, "use_name_server": True}
        return {
            "timeout": CONNECTOR_TIMEOUT_S,
            "probe_channel": PROBE_CHANNEL,
            "gateways": {"read_only": dict(gateway), "write_access": dict(gateway)},
        }

    target_switch: dict[str, Any] = {}
    if acknowledged:
        target_switch[te.ACK_LEAF] = f"localhost:{bench_port}"

    return {
        "control_system": {
            "type": "virtual_accelerator",
            "writes_enabled": True,
            "limits_checking": {
                "enabled": True,
                "allow_unlisted_channels": False,
                "database_path": str(limits_db),
            },
            "target_switch": target_switch,
            "connector": {
                "epics": block(bench_port),
                "virtual_accelerator": block(va_port),
            },
        },
        # Not the mock: pointing a session at the virtual accelerator while the
        # archiver synthesises history is the pairing eligibility refuses.
        # Nothing in this module builds an archiver connector.
        "archiver": {"type": "mongodb_archiver"},
        "agent_data": {"base_dir": "var/agent_data"},
    }


@pytest.fixture(autouse=True)
def state_root(tmp_path, monkeypatch):
    """Anchor the target-state directory in ``tmp_path``, not a real deployment."""
    root = tmp_path / "var" / "agent_data"
    (root / target_state.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
    return root


@pytest.fixture(autouse=True)
def child_environment(monkeypatch):
    """Children see this worktree, and no ambient project config.

    ``PYTHONPATH`` is set explicitly rather than inherited: the interpreter
    running these tests belongs to another checkout's virtualenv, and a child
    that resolved ``osprey`` there would be a child of a different repository.
    """
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(REPO_PATHS))
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)


@pytest.fixture
def written_config(tmp_path):
    """Stage a deployment on disk and hand back its path.

    A connector-host child is configured from the section it is handed on the
    wire, but ``control_system.writes_enabled`` and the limits database are read
    by the connector itself out of the config file -- they are launch-time
    deployment posture, not values the launch payload may grant. So a deployment
    that means to write, under real limits, has to exist as a file.
    """
    written: list[Path] = []

    def write(raw: dict[str, Any]) -> Path:
        path = tmp_path / f"config_{len(written)}.yml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        written.append(path)
        return path

    return write


@pytest.fixture
def deployment(bench_endpoint, va_endpoint, limits_database) -> dict[str, Any]:  # noqa: F811
    """This module's deployment, pointed at the two containers it booted."""
    return raw_config(
        bench_port=bench_endpoint.port, va_port=va_endpoint, limits_db=limits_database
    )


@pytest.fixture
async def make_manager(written_config, state_root):
    """Managers whose children are all reaped when the test ends."""
    created: list[ConnectorHostManager] = []

    def factory(raw: dict[str, Any]) -> ConnectorHostManager:
        manager = ConnectorHostManager(
            MCPServerConfig(raw=raw, config_path=written_config(raw)),
            drain_timeout_s=DRAIN_TIMEOUT_S,
            probe_timeout_s=PROBE_TIMEOUT_S,
            spawn_timeout_s=SPAWN_TIMEOUT_S,
            terminate_grace_s=2.0,
        )
        manager.reset_state()
        created.append(manager)
        return manager

    yield factory

    for manager in created:
        with contextlib.suppress(Exception):
            await manager.shutdown()


@pytest.fixture
def served_context(monkeypatch):
    """Install a context as the controls server's singleton, for the tools."""

    def install(manager: ConnectorHostManager):
        context = context_for(manager)
        monkeypatch.setattr(server_context_mod, "_registry", context)
        return context

    yield install
    server_context_mod.reset_server_context()


@pytest.fixture
def quiet_switch_notifications(monkeypatch):
    """Silence the operator-facing switch emit.

    It is fire-and-forget over HTTP to a web terminal that is not running here.
    """
    recorded: list[dict[str, Any]] = []

    async def record(**kwargs: Any) -> None:
        recorded.append(kwargs)

    monkeypatch.setattr(control_target, "notify_target_switch_async", record)
    return recorded


async def started_on(factory, raw: dict[str, Any], target: str) -> ConnectorHostManager:
    """A manager with a live connector-host child on *target*."""
    manager = factory(raw)
    await manager.start(target)
    assert manager.has_child()
    assert manager.active_target() == target
    return manager


async def reading(
    manager: ConnectorHostManager, address: str, *, timeout: float = CALL_TIMEOUT_S
) -> ChannelValue:
    """One reading off the wire, through whichever child is serving right now."""
    value = await manager.active_proxy().read_channel(address, timeout=timeout)
    assert isinstance(value, ChannelValue)
    return value


async def settled_reading(
    manager: ConnectorHostManager,
    address: str,
    *,
    expected: float,
    alarm_status: str | None = None,
    timeout: float = CALL_TIMEOUT_S,
) -> ChannelValue:
    """Re-read *address* until it reports *expected*, then hand that reading back.

    Every post-write read in this module goes through here, and not out of
    caution: the connector's read path is pyepics' ``pv.get()``, which defaults
    to ``use_monitor=True`` and will answer from a monitor event that has not
    been dispatched yet -- returning the value the channel held BEFORE the write.
    That is the stale-cache class ``scripts/va/build_and_boot_check.sh`` states
    its ``use_monitor=False`` doctrine for, and a bare read immediately after a
    write is exactly the shape that flakes under it.

    The deadline is a bound, not a wait: a channel that already reflects the
    write returns on the first read. Timing out fails naming the last reading
    actually observed, so a genuinely unwritten channel is distinguishable from
    a slow one.
    """
    deadline = time.monotonic() + timeout
    while True:
        last = await reading(manager, address, timeout=timeout)
        matched = last.value == pytest.approx(expected, rel=1e-3) and (
            alarm_status is None or last.metadata.alarm_status == alarm_status
        )
        if matched or time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.1)

    wanted = f"{expected}" + (f" with alarm {alarm_status}" if alarm_status else "")
    assert last.value == pytest.approx(expected, rel=1e-3), (
        f"{address} never settled at {wanted} within {timeout}s; "
        f"the last reading was {last.value} (alarm {last.metadata.alarm_status})"
    )
    if alarm_status is not None:
        assert last.metadata.alarm_status == alarm_status, (
            f"{address} settled at {last.value} but never reported alarm "
            f"{alarm_status} within {timeout}s; the last reading said "
            f"{last.metadata.alarm_status}"
        )
    return last


async def envelope_for(exc: BaseException, tool_name: str = "channel_write") -> dict[str, Any]:
    """The agent-facing envelope ``connector_error_handler`` renders for *exc*."""
    with assert_raises_error() as captured:
        async with error_handling.connector_error_handler(tool_name):
            raise exc
    return captured["envelope"]


# ---------------------------------------------------------------------------
# 1. Switching onto the bench machine
# ---------------------------------------------------------------------------


class TestASwitchToTheBenchMachine:
    """The switch an operator asks for, and the machine they land on.

    Driven through ``control_target_set`` rather than the supervisor, because
    the acknowledgment gate lives in the tool: the supervisor switches whatever
    it is told to switch to, and FR-8 is what decides whether it is told.
    """

    async def test_the_tool_switches_onto_the_bench_and_the_reads_follow(
        self, make_manager, deployment, served_context, quiet_switch_notifications
    ):
        manager = await started_on(make_manager, deployment, "va")
        served_context(manager)
        on_the_simulator = (await reading(manager, PROBE_CHANNEL)).value
        assert on_the_simulator != pytest.approx(BENCH_PROBE_VALUE, abs=1e-3), (
            "the virtual accelerator already answered the bench IOC's seeded value, so "
            "nothing below could tell the two machines apart"
        )

        payload = json.loads(await get_tool_fn(control_target.control_target_set)(target="live"))

        summary = payload["summary"]
        assert summary["target"] == "live"
        assert summary["previous_target"] == "va"
        assert summary["target_changed"] is True
        assert summary["generation"] == 1
        assert summary["connector_type"] == "epics"
        assert summary["probe_channel"] == PROBE_CHANNEL
        assert payload["access_details"]["baseline_target"] == "va"
        # The evidence, off the wire: the same address, the value only the
        # bench IOC serves.
        assert (await reading(manager, PROBE_CHANNEL)).value == pytest.approx(BENCH_PROBE_VALUE)
        assert manager.active_target() == "live"
        assert target_state.read()["target"] == "live"

    async def test_the_two_machines_answer_the_shared_channels_differently(
        self, make_manager, deployment
    ):
        """Routing proven by value, on three channels both targets serve.

        Every value read here is read from a machine at rest: nothing in this
        test writes anywhere, so the only thing separating the two readings is
        which server answered.
        """
        manager = await started_on(make_manager, deployment, "live")

        bench_readings = {
            address: (await reading(manager, address)).value
            for address in (PROBE_CHANNEL, SHARED_BPM_X, PROTECTED_SP)
        }
        await manager.switch("va")
        va_readings = {
            address: (await reading(manager, address)).value
            for address in (PROBE_CHANNEL, SHARED_BPM_X, PROTECTED_SP)
        }

        assert bench_readings[PROBE_CHANNEL] == pytest.approx(BENCH_PROBE_VALUE)
        assert bench_readings[SHARED_BPM_X] == pytest.approx(BENCH_BPM_X_VALUE)
        assert bench_readings[PROTECTED_SP] == pytest.approx(BENCH_PROTECTED_VALUE)
        for address, bench_value in bench_readings.items():
            # The bench constants are all far from a quiescent lattice's, so the
            # margin is about being unmistakable rather than about tolerance.
            assert abs(va_readings[address] - bench_value) > 0.1, (
                f"{address} read {va_readings[address]} on the virtual accelerator and "
                f"{bench_value} on the bench IOC, which is too close to tell the two apart"
            )


# ---------------------------------------------------------------------------
# 2. The acknowledgment, without spawning anything
# ---------------------------------------------------------------------------


class TestTheOperatorAcknowledgmentGatesTheLiveMachine:
    """FR-8's live gate, asked of the config alone.

    Eligibility is a question about configuration, and answering it spawns no
    process, opens no socket and writes nothing -- so this pair of tests takes no
    manager and no child. The configs are built by the same function that builds
    the deployment the containers actually serve, differing in exactly one key,
    which is what makes the reason attributable to that key.
    """

    def rows(self, config: dict[str, Any]) -> dict[str, Any]:
        """The roster's own rows, from config alone: no manager, no child."""
        return control_target.target_rows(config, session_target="va", baseline="va")

    async def test_without_the_acknowledgment_the_live_machine_is_refused_by_name(
        self, bench_endpoint, va_endpoint, limits_database
    ):
        config = raw_config(
            bench_port=bench_endpoint.port,
            va_port=va_endpoint,
            limits_db=limits_database,
            acknowledged=False,
        )

        row = self.rows(config)["live"]

        assert row["available_now"] is False
        assert row["reason"] == te.REASON_OPERATOR_ACK_MISSING
        assert te.ACK_KEY in row["detail"]
        # The refusal is about the acknowledgment and not about the endpoint:
        # the row still names the gateway a switch would have used.
        assert row["endpoints"]["write_access"]["port"] == bench_endpoint.port
        assert row["connector_type"] == "epics"

    async def test_with_the_acknowledgment_the_same_deployment_is_available(self, deployment):
        """Anti-vacuous control: one key is the whole difference.

        This is the very config the containers below are served from, so a
        deployment that were ineligible for some other reason would fail here
        rather than quietly making the test above prove nothing.
        """
        row = self.rows(deployment)["live"]

        assert row["available_now"] is True
        assert row["reason"] is None
        assert row["eligible_from_baseline"] is True
        assert self.rows(deployment)["va"]["reason"] == te.REASON_ALREADY_ACTIVE


# ---------------------------------------------------------------------------
# 3. A write the machine itself refuses
# ---------------------------------------------------------------------------


class TestTheControlSystemsOwnWriteRefusal:
    """The bench IOC's access security says no, and the operator hears whose no.

    The refusal travels the full deployment path: a real connector in a real
    connector-host child issues a real ``caput``, the IOC's CA server answers
    "write access denied" before any record processing, and the structured
    result crosses the IPC boundary intact. Nothing here is a client-side check,
    which is precisely what the last test in this class establishes by showing
    the client-side check produces a *different* answer.
    """

    async def test_the_write_is_blocked_with_the_control_system_as_the_refuser(
        self, make_manager, deployment
    ):
        manager = await started_on(make_manager, deployment, "live")

        # Inside the channel's own limits, so the reference monitor passes it
        # through and the caput really reaches the IOC. A client-side refusal
        # here would look identical at the call site and prove nothing.
        assert abs(PROTECTED_WRITE_VALUE) < LIMITS_MAX
        result = await manager.active_proxy().write_channel(
            PROTECTED_SP, PROTECTED_WRITE_VALUE, timeout=CALL_TIMEOUT_S
        )

        assert result.blocked is True
        assert result.success is False
        assert result.refusal_reason == "CONTROL_SYSTEM_REFUSED"
        # The vocabulary is the shared one, so a reason the error type would
        # reject can never reach an operator through this path.
        assert result.refusal_reason in ChannelWriteBlockedError._VALID_REASONS
        assert result.channel_address == PROTECTED_SP
        assert PROTECTED_SP in (result.error_message or "")
        assert "access security" in (result.error_message or "")
        # No value was written, which is the claim `blocked` makes.
        assert (await reading(manager, PROTECTED_SP)).value == pytest.approx(BENCH_PROTECTED_VALUE)

    async def test_the_refusal_is_the_typed_error_and_not_a_limits_violation(
        self, make_manager, deployment
    ):
        """Two refusals, two refusers, told apart by their reason codes.

        Both leave the channel unwritten, so "the write did not land" cannot
        distinguish them -- and an operator told the reference monitor refused a
        write the *control system* refused would be sent to fix OSPREY's policy
        settings for something only the IOC can grant.
        """
        manager = await started_on(make_manager, deployment, "live")

        refused = await manager.active_proxy().write_channel(
            PROTECTED_SP, PROTECTED_WRITE_VALUE, timeout=CALL_TIMEOUT_S
        )
        with pytest.raises(ChannelWriteBlockedError) as blocked:
            raise_for_write_result(refused)

        with pytest.raises(ChannelLimitsViolationError) as violated:
            await manager.active_proxy().write_channel(
                WRITABLE_SP, OUT_OF_LIMITS_VALUE, timeout=CALL_TIMEOUT_S
            )

        assert blocked.value.reason == "CONTROL_SYSTEM_REFUSED"
        assert blocked.value.channel_address == PROTECTED_SP
        assert type(violated.value) is ChannelLimitsViolationError
        assert violated.value.channel_address == WRITABLE_SP
        assert violated.value.max_value == pytest.approx(LIMITS_MAX)

        # The distinction as the agent meets it: two error types, two accounts
        # of who refused.
        refusal_envelope = await envelope_for(blocked.value)
        limits_envelope = await envelope_for(violated.value)
        assert refusal_envelope["error_type"] == "write_refused"
        assert refusal_envelope["details"]["reason"] == "CONTROL_SYSTEM_REFUSED"
        assert "control system" in refusal_envelope["error_message"]
        assert limits_envelope["error_type"] == "limits_violation"
        # The limits refusal never reached the machine, so the writable setpoint
        # is untouched and the bench IOC's own value still stands.
        assert (await reading(manager, WRITABLE_SP)).value != pytest.approx(OUT_OF_LIMITS_VALUE)

    async def test_the_same_address_accepts_a_write_on_the_virtual_accelerator(
        self, make_manager, deployment
    ):
        """The refusal is the bench machine's, not the address's.

        Same session, same connector stack, same channel name, same value band --
        only the target moves. A deployment that refused this write everywhere
        would satisfy every assertion above while proving nothing about the IOC.
        """
        manager = await started_on(make_manager, deployment, "live")
        on_the_bench = await manager.active_proxy().write_channel(
            PROTECTED_SP, VA_WRITE_VALUE, timeout=CALL_TIMEOUT_S
        )
        assert on_the_bench.blocked is True

        await manager.switch("va")
        try:
            on_the_simulator = await manager.active_proxy().write_channel(
                PROTECTED_SP, VA_WRITE_VALUE, timeout=CALL_TIMEOUT_S
            )

            assert on_the_simulator.blocked is False
            assert on_the_simulator.refusal_reason is None
            assert on_the_simulator.success is True
            await settled_reading(manager, PROTECTED_SP, expected=VA_WRITE_VALUE)
        finally:
            # Every other test in this module reads a machine at rest.
            with contextlib.suppress(Exception):
                await manager.active_proxy().write_channel(
                    PROTECTED_SP, 0.0, timeout=CALL_TIMEOUT_S
                )
        # And the bench IOC still holds its own seeded value: the write that
        # succeeded went to the other machine.
        await manager.switch("live")
        assert (await reading(manager, PROTECTED_SP)).value == pytest.approx(BENCH_PROTECTED_VALUE)


# ---------------------------------------------------------------------------
# 4. What each machine does not serve
# ---------------------------------------------------------------------------


class TestTheTwoMachinesServeDifferentNamespaces:
    """A switch narrows what is reachable, in both directions.

    Shared channels with different values prove the session moved. Channels that
    exist on one target and nowhere on the other prove something a value cannot:
    that the answers are coming from that machine's own database rather than
    from anything on this side holding a copy.
    """

    async def test_the_bench_only_records_are_unreachable_on_the_virtual_accelerator(
        self, make_manager, deployment
    ):
        manager = await started_on(make_manager, deployment, "live")
        assert (await reading(manager, BENCH_ONLY_AMPLITUDE)).value == pytest.approx(
            BENCH_ONLY_AMPLITUDE_VALUE
        )
        assert (await reading(manager, MODE_STATE)).value == MODE_STATE_VALUE

        await manager.switch("va")

        with pytest.raises(ConnectionError) as raised:
            await reading(manager, BENCH_ONLY_AMPLITUDE, timeout=ABSENT_TIMEOUT_S)
        assert BENCH_ONLY_AMPLITUDE in str(raised.value)

    async def test_the_virtual_accelerators_own_records_are_unreachable_on_the_bench(
        self, make_manager, deployment
    ):
        manager = await started_on(make_manager, deployment, "va")
        assert isinstance((await reading(manager, VA_ONLY_CHANNEL)).value, float)

        await manager.switch("live")

        with pytest.raises(ConnectionError) as raised:
            await reading(manager, VA_ONLY_CHANNEL, timeout=ABSENT_TIMEOUT_S)
        assert VA_ONLY_CHANNEL in str(raised.value)


# ---------------------------------------------------------------------------
# 5. Record textures a simulator does not have to reproduce
# ---------------------------------------------------------------------------


class TestTheBenchRecordsKeepTheirEpicsTextures:
    """Reads of a real IOC carry more than a number, and it survives the seam.

    Each of these is a property of stock EPICS record support rather than of
    OSPREY: an ``mbbi`` reports its index, a ``calc`` reports what it computed,
    and a value driven past ``HIHI`` alarms by name. What is under test is that
    all three cross the connector-host boundary unflattened.
    """

    async def test_the_mbbi_reports_its_index_with_the_enum_type_intact(
        self, make_manager, deployment
    ):
        manager = await started_on(make_manager, deployment, "live")

        mode = await reading(manager, MODE_STATE)
        scaled = await reading(manager, SCALED_AMPLITUDE)

        # The index, unconverted -- not "ACQUIRING". A client that received the
        # string would have lost the machine-readable half of the reading.
        assert mode.value == MODE_STATE_VALUE
        assert int(mode.value) == MODE_STATE_VALUE
        assert not isinstance(mode.value, str)
        assert "enum" in str(mode.metadata.raw_metadata["type"])
        assert mode.metadata.alarm_status == "NO_ALARM"
        # The contrast that makes the type assertion mean something: an
        # analogue record on the same IOC is not typed as an enumeration.
        assert "enum" not in str(scaled.metadata.raw_metadata["type"])

    async def test_the_calc_record_reports_its_derived_value(self, make_manager, deployment):
        manager = await started_on(make_manager, deployment, "live")

        scaled = await reading(manager, SCALED_AMPLITUDE)

        assert scaled.value == pytest.approx(SCALED_AMPLITUDE_VALUE)
        assert scaled.metadata.units == "mV"

    async def test_driving_the_readback_past_hihi_reports_the_alarm_by_name(
        self, make_manager, deployment
    ):
        """The alarm is the record's, and it reaches the reader as a name.

        The channel driven here is the one bench record carrying alarm limits
        (``HIHI 11.0``, ``HHSV MAJOR``); its seeded value sits well inside the
        healthy band, which is what makes the "before" reading a control rather
        than an assumption.
        """
        manager = await started_on(make_manager, deployment, "live")
        before = await reading(manager, ALARM_CHANNEL)
        assert before.value == pytest.approx(ALARM_SEEDED_VALUE)
        assert before.metadata.alarm_status == "NO_ALARM"

        try:
            written = await manager.active_proxy().write_channel(
                ALARM_CHANNEL, ALARM_DRIVE_VALUE, timeout=CALL_TIMEOUT_S
            )
            assert written.blocked is False, written.error_message
            assert ALARM_DRIVE_VALUE > ALARM_HIHI_THRESHOLD

            alarming = await settled_reading(
                manager, ALARM_CHANNEL, expected=ALARM_DRIVE_VALUE, alarm_status="HIHI"
            )

            assert alarming.value == pytest.approx(ALARM_DRIVE_VALUE)
            # Exactly HIHI: HIGH would mean the record alarmed on the wrong
            # threshold, and MAJOR alone would not say which one.
            assert alarming.metadata.alarm_status == "HIHI"
            assert alarming.metadata.raw_metadata["severity"] == ALARM_MAJOR_SEVERITY
        finally:
            # Restore the seeded value: the bench IOC is shared by every test in
            # this module, and an IOC left in a MAJOR alarm is noise in the logs
            # of all of them.
            with contextlib.suppress(Exception):
                await manager.active_proxy().write_channel(
                    ALARM_CHANNEL, ALARM_SEEDED_VALUE, timeout=CALL_TIMEOUT_S
                )


# ---------------------------------------------------------------------------
# 6. The machine goes away
# ---------------------------------------------------------------------------


class TestWhenTheBenchMachineGoesAway:
    """A live machine stopping is reported, not absorbed.

    Both tests boot a bench IOC of their own and stop it: the module-scoped one
    is shared by everything above, and a suite that killed it would be a suite
    whose later results depended on its own ordering.

    What must survive the outage is the session's own account of itself. The
    reads fail, the roster's reachability row goes down, and through all of it
    the session still names ``live`` as its target -- because a machine that
    stopped answering is not a switch, and claiming one would invent it.
    """

    async def test_a_read_after_the_machine_stops_names_the_channel_and_the_timeout(
        self, make_manager, va_endpoint, limits_database, served_context, monkeypatch
    ):
        with bench.bench_ioc(prefix="osprey-bench-ioc-outage") as ioc:
            config = raw_config(bench_port=ioc.port, va_port=va_endpoint, limits_db=limits_database)
            manager = await started_on(make_manager, config, "live")
            served_context(manager)
            assert (await reading(manager, PROBE_CHANNEL)).value == pytest.approx(BENCH_PROBE_VALUE)
            # The generation the session is standing on. It is 1 rather than 0
            # because this deployment's baseline is the virtual accelerator, so
            # coming up on `live` is itself a move; what the outage must not do
            # is move it again.
            generation = manager.active_generation()

            stopped = bench.docker("stop", ioc.container, timeout=60)
            assert stopped.returncode == 0, stopped.stderr

            # The recovery seam is recorded rather than executed: a ConnectionError
            # is what makes the tool handler respawn the child, and letting that
            # run would race a spawn against a gateway nothing is listening on
            # while these assertions are being made. That it was *asked* for is
            # asserted below; the respawn itself is pinned in test_target_switch.py.
            invalidated: list[str] = []

            async def record(connector_name: str) -> None:
                invalidated.append(connector_name)

            monkeypatch.setattr(error_handling, "invalidate_active_connector", record)

            with assert_raises_error(error_type="connection_error") as captured:
                await get_tool_fn(channel_read)(channels=[PROBE_CHANNEL], include_metadata=False)

            message = captured["envelope"]["error_message"]
            assert PROBE_CHANNEL in message, message
            assert "timeout after" in message, message
            assert invalidated == ["control_system"]
            # A dead machine is not a switch.
            assert manager.active_target() == "live"
            assert manager.active_generation() == generation
            assert target_state.read()["target"] == "live"

    async def test_the_endpoint_probe_and_the_roster_report_the_gateway_down(
        self, make_manager, va_endpoint, limits_database, served_context, monkeypatch
    ):
        """The roster's reachability half, measured before and after.

        ``endpoint_tcp`` is the one thing in the roster that comes from a
        measurement rather than from config, so it is the one thing that can
        report a machine that was configured correctly and then stopped
        answering. The healthy sweep is what makes the second one evidence.
        """
        from osprey.mcp_server.control_system import server as controls_server

        with bench.bench_ioc(prefix="osprey-bench-ioc-outage") as ioc:
            config = raw_config(bench_port=ioc.port, va_port=va_endpoint, limits_db=limits_database)
            manager = await started_on(make_manager, config, "live")
            served_context(manager)
            generation = manager.active_generation()
            prober = EndpointProber(config, connect_timeout_s=PROBE_CONNECT_TIMEOUT_S)
            monkeypatch.setattr(controls_server, "get_endpoint_prober", lambda: prober)

            await prober.sweep_once()
            healthy = prober.snapshot()["live"]["write_access"]["endpoint_tcp"]

            stopped = bench.docker("stop", ioc.container, timeout=60)
            assert stopped.returncode == 0, stopped.stderr
            await prober.sweep_once()

            assert healthy == STATUS_OK
            assert prober.snapshot()["live"]["write_access"]["endpoint_tcp"] == STATUS_UNREACHABLE
            # The virtual accelerator is untouched by the bench IOC's outage,
            # which is what makes the row above about one endpoint.
            assert prober.snapshot()["va"]["write_access"]["endpoint_tcp"] == STATUS_OK

            roster = json.loads(await get_tool_fn(control_target.control_target)())

            live_row = roster["access_details"]["targets"]["live"]
            assert live_row["endpoints"]["write_access"]["endpoint_tcp"] == STATUS_UNREACHABLE
            assert live_row["endpoints"]["write_access"]["port"] == ioc.port
            assert live_row["active"] is True
            # The session still names the machine it is on, unreachable or not,
            # and on the generation it was already standing on: an endpoint that
            # stopped answering is not a switch.
            assert roster["summary"]["target"] == "live"
            assert roster["summary"]["generation"] == generation
            assert roster["summary"]["connector_host_alive"] is True
            assert roster["summary"]["baseline_target"] == "va"


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_bench_ioc_live.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
