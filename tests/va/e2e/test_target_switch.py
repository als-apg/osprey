"""The runtime target switch, driven against two real Channel Access machines.

Everything the switch promises is a claim about *processes and wires*: that a
connector-host child really dies, that the session really lands on a different
server, that a value approved for one machine never reaches the other, that a
read still in flight on a retired child really ends and says why. None of that
is decidable from inside one process, and none of it is decidable against a
connector fake — so this module spawns real children, points them at real
soft-IOCs, and asserts on what came back over Channel Access.

Two endpoints, and why they are two containers
----------------------------------------------
A switch is only a switch if the two ends are different machines. This module
therefore boots **two** ``osprey-va-full`` containers on two ephemeral ports and
gives the deployment one target each:

* ``va``  -> the ``virtual_accelerator`` connector block, pointed at container V;
* ``live`` -> the ``epics`` connector block, pointed at container L.

That makes ``control_system.type: epics`` the deployment's own type, so ``live``
is its baseline, ``va`` is the other target, and
:func:`~osprey_connectors.types.switch_capable` holds. The ``live`` end is a
simulator wearing the live target's connector type, which is exactly what the
e2e harness can honestly provide: what it gives these tests is a *second real CA
endpoint under the harness's control*, which is the property every scenario here
actually rests on. Nothing in this file claims to have touched hardware, and
nothing here is a statement about a real facility's gateways.

The two are told apart on the wire rather than by configuration. Container L is
booted with one BPM's readout error seeded (``VA_BPM_ERRORS``) and container V is
not, so :data:`SEEDED_BPM` reads a different number on each — with no client
write anywhere, from a machine at rest, and reproducibly. A round trip that
reported the same value at both ends would be a switch that moved nothing.

Ports, names and 5064
---------------------
Each container binds an ephemeral port and publishes it unchanged: a Channel
Access search reply carries the server's own port, so a remap would hand every
client an address nothing answers on. That port also *names* the container, for
the reason ``test_serving_parity.py`` states at length — a fixed name is
mutually destructive between concurrent runs. Nothing here goes near 5064, which
belongs to whatever the operator is running.

Every Channel Access operation in this file happens in **another process**: in a
connector-host child, in the readiness probe, or in the sandbox-side subprocess
of the executor test. This process never becomes a CA client, which is the same
rule ``conftest.py`` states — libca latches ``EPICS_CA_*`` on initialisation and
its contexts are per-thread, so a main-thread pyepics call here would deadlock
the very children under test.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from osprey.mcp_server.control_system import server_context as server_context_mod
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.connector_host_manager import (
    ConnectorHostManager,
    NoConnectorHostError,
    SwitchError,
)
from osprey.mcp_server.control_system.server_context import (
    ControlSystemContext,
    MCPServerConfig,
)
from osprey.mcp_server.control_system.tools import channel_write as channel_write_module
from osprey.mcp_server.control_system.tools import control_target
from osprey.mcp_server.python_executor import executor as host_executor
from osprey_connectors.control_system.base import ChannelValue
from tests.fixtures.control_context import context_for
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn
from tests.va.e2e import conftest as e2e_conftest

# The Bluesky MCP tools pull in the queueserver client stack. A checkout without
# it must still collect this module — the lane scenario is the only thing that
# needs it, and it skips with the reason rather than failing the file.
try:  # pragma: no cover - import-availability branch
    from osprey.mcp_server.bluesky import server_context as bluesky_context
    from osprey.mcp_server.bluesky.tools import queue as bluesky_queue
    from osprey.mcp_server.control_system import target_banner
    from osprey.services.bluesky_bridge import queue_backend as bluesky_backend

    BLUESKY_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - depends on the installed extras
    bluesky_context = bluesky_queue = target_banner = bluesky_backend = None  # type: ignore[assignment]
    BLUESKY_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_PATHS = (str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src"))

#: The image under test, and the simulation data both containers serve.
IMAGE = os.environ.get("OSPREY_VA_E2E_IMAGE", "osprey-va-full:latest")
DATA_DIR = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation"

#: Container-name prefixes; ``_serving`` appends the run's own ephemeral port.
CONTAINER_VA = "osprey-va-e2e-switch-va"
CONTAINER_LIVE = "osprey-va-e2e-switch-live"

#: Boot is generous on purpose: the image is pinned ``linux/amd64`` and a local
#: run on Apple Silicon is emulated (see ``conftest.py``).
BOOT_TIMEOUT_S = 180.0

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 18

# -- the namespace both containers serve ------------------------------------

#: What each target's switch reads to prove itself reachable. A pyat-coupled
#: readback, never written by anything in this file.
VA_PROBE = "SR:MAG:HCM:01:CURRENT:RB"
LIVE_PROBE = VA_PROBE

#: The channel that tells the two containers apart. Container L is booted with
#: this BPM's readout error seeded and container V is not, so the two serve
#: different numbers for a machine in the same (quiescent) state.
SEEDED_BPM = "SR:DIAG:BPM:11:POSITION:X"
SEEDED_DEVICE = "BPM11"
SEEDED_OFFSET_X = 50e-6
SEEDED_GAIN_X = 1.05
VA_BPM_ERRORS = f"{SEEDED_DEVICE}:offset_x={SEEDED_OFFSET_X},gain_x={SEEDED_GAIN_X}"

#: The one setpoint this module ever writes -- in the executor scenario, and
#: restored to zero by that test before it returns. Listed in the shipped
#: ``channel_limits.json`` with a +-12 A band, so the values below pass limits.
CORRECTOR_SP = "SR:MAG:HCM:01:CURRENT:SP"
PINNED_WRITE_VALUE = 1.0
REFUSED_WRITE_VALUE = 2.0

#: A name neither container serves. A read of it blocks in the child for the
#: whole connector timeout, which is how the drain scenario gets a request that
#: is genuinely still in flight when the deadline expires.
UNSERVED_CHANNEL = "VA:E2E:SWITCH:NO:SUCH:CHANNEL"

# -- bounds -----------------------------------------------------------------

#: The connector's own timeout, and so the ceiling on a hung read. Far longer
#: than any drain deadline here, so the drain is what ends the read.
CONNECTOR_TIMEOUT_S = 120.0
#: Bound on a read this module expects to answer.
READ_TIMEOUT_S = 20.0
#: Bound on "spawned and answered its init frame" -- a cold pyepics import in an
#: emulated container host is not fast.
SPAWN_TIMEOUT_S = 60.0
#: Bound on the readiness probe a switch runs against a fresh child.
PROBE_TIMEOUT_S = 10.0
#: Bound on polling for something that has already been asked to happen.
SETTLE_TIMEOUT_S = 30.0

#: The drain deadline these tests configure, and the timing criterion built on
#: it: a switch must complete within the deadline plus this slack, however long
#: the request it is draining would have taken.
DRAIN_TIMEOUT_S = 1.0
SWITCH_TIMING_SLACK_S = 10.0
CONSECUTIVE_SWITCHES = 5


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _docker(*args: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _require_image() -> None:
    """Fail loudly unless the image can serve on a port other than 5064.

    A precondition, not a nicety, and the same one ``test_serving_parity.py``
    states: the Channel Access *server* library reads ``EPICS_CAS_SERVER_PORT``
    and does not fall back to the client-side variable, so an image whose entry
    point does not derive one from the other keeps binding its build-time
    default while telling this suite's clients some other port. The symptom
    would be an unexplained boot timeout; this turns it into a sentence naming
    the fix.
    """
    inspected = _docker(
        "image", "inspect", IMAGE, "--format", "{{.Architecture}}|{{.Config.Cmd}}", timeout=60
    )
    if inspected.returncode != 0:
        pytest.fail(
            f"image {IMAGE!r} is not present. Build it with "
            f"scripts/va/build_and_boot_check.sh, or name another with "
            f"OSPREY_VA_E2E_IMAGE."
        )
    architecture, _, command = inspected.stdout.strip().partition("|")
    if "EPICS_CAS_SERVER_PORT" not in command:
        pytest.fail(
            f"image {IMAGE!r} ({architecture}) does not derive EPICS_CAS_SERVER_PORT from "
            f"EPICS_CA_SERVER_PORT, so it cannot serve on any port but its baked default. "
            f"Rebuild it (scripts/va/build_and_boot_check.sh) or point "
            f"OSPREY_VA_E2E_IMAGE at a current build. Its entry point is: {command}"
        )


def _served(port: int) -> bool:
    """Whether a virtual accelerator is answering on *port*, asked out of process.

    In a subprocess for the reason ``conftest.py`` gives: the connector wraps
    synchronous pyepics in a thread-pool executor whose CA context is
    per-thread, so a main-thread pyepics call in *this* process would deadlock
    the children these tests spend their time talking to.
    """
    code = (
        "import sys, epics\n"
        f"v = epics.caget({VA_PROBE!r}, timeout=1.0, connection_timeout=1.0)\n"
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


@contextlib.contextmanager
def _serving(prefix: str, *, seeded: bool):
    """Boot one virtual accelerator container and wait until it serves.

    The published port and the server's own port are the same number by
    construction, and that number also names the container.
    """
    port = _free_port()
    name = f"{prefix}-{port}"
    # Stale-cleanup only. The port is this run's alone, so this can name nothing
    # a concurrent run is using -- which is the point of the suffix.
    _docker("rm", "-f", name, timeout=60)

    arguments = [
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"EPICS_CA_SERVER_PORT={port}",
        "-p",
        f"127.0.0.1:{port}:{port}/tcp",
        "-v",
        f"{DATA_DIR}:/data/simulation:ro",
    ]
    if seeded:
        arguments += ["-e", f"VA_BPM_ERRORS={VA_BPM_ERRORS}"]
    started = _docker(*arguments, IMAGE)
    if started.returncode != 0:
        raise RuntimeError(f"docker run failed: {started.stdout}\n{started.stderr}")

    try:
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            if _served(port):
                break
            time.sleep(1.0)
        else:
            logs = _docker("logs", "--tail", "40", name, timeout=60)
            raise RuntimeError(
                f"{name} never served {VA_PROBE} within {BOOT_TIMEOUT_S}s.\n"
                f"{logs.stdout}\n{logs.stderr}"
            )
        yield port
    finally:
        _docker("rm", "-f", name, timeout=60)


@dataclass(frozen=True)
class Endpoints:
    """The two Channel Access ports this module's deployment is built from."""

    va: int
    live: int


@pytest.fixture(scope="module")
def endpoints():
    """Both machines, up and serving, for the life of this module.

    Container L carries the seeded BPM readout error and container V does not,
    which is what makes ``SEEDED_BPM`` distinguish them on the wire.
    """
    _require_image()
    with _serving(CONTAINER_VA, seeded=False) as va_port:
        with _serving(CONTAINER_LIVE, seeded=True) as live_port:
            yield Endpoints(va=va_port, live=live_port)


# ---------------------------------------------------------------------------
# The deployment
# ---------------------------------------------------------------------------


def raw_config(
    *,
    va_port: int,
    live_port: int,
    drain_timeout_s: float | None = None,
    va_probe: str | None = VA_PROBE,
    live_probe: str | None = LIVE_PROBE,
    project_root: Path | None = None,
) -> dict:
    """A switch-capable deployment with one real CA endpoint per target.

    ``control_system.type`` is ``epics``, so ``live`` is the baseline and
    resolves to the ``epics`` block; ``va`` always resolves to the
    ``virtual_accelerator`` block. Both blocks name an explicit gateway port,
    which wins over ``services.virtual_accelerator.port`` — the containers bind
    ephemeral ports and nothing may guess them.

    The FR-8 posture is set (strict limits against the *shipped* limits
    database, plus the operator acknowledgment naming this harness's own live
    endpoint) so that a switch toward ``live`` is judged on the same terms a
    real deployment would be judged on, rather than on a gate this file quietly
    left open.
    """

    def block(port: int, probe: str | None) -> dict:
        configured = {
            "timeout": CONNECTOR_TIMEOUT_S,
            "gateways": {
                "read_only": {"address": "localhost", "port": port, "use_name_server": True},
                "write_access": {"address": "localhost", "port": port, "use_name_server": True},
            },
        }
        if probe:
            configured["probe_channel"] = probe
        return configured

    target_switch: dict = {"live_gateway_acknowledged": f"localhost:{live_port}"}
    if drain_timeout_s is not None:
        target_switch["drain_timeout_s"] = drain_timeout_s

    config: dict = {
        "control_system": {
            "type": "epics",
            "writes_enabled": True,
            "limits_checking": {
                "enabled": True,
                "allow_unlisted_channels": False,
                "database_path": str(e2e_conftest.LIMITS_DB_PATH),
            },
            "target_switch": target_switch,
            "connector": {
                "epics": block(live_port, live_probe),
                "virtual_accelerator": block(va_port, va_probe),
            },
        },
        # Not the mock: pointing a session at the virtual accelerator while the
        # archiver synthesises history is the pairing eligibility refuses.
        # Nothing in this module builds an archiver connector.
        "archiver": {"type": "mongodb_archiver"},
        "agent_data": {"base_dir": "var/agent_data"},
    }
    if project_root is not None:
        config["project_root"] = str(project_root)
    return config


@pytest.fixture(autouse=True)
def state_root(tmp_path, monkeypatch):
    """Anchor the target-state directory in ``tmp_path``, not a real deployment.

    Returned as the shared data root, which is also what the on-disk config
    below resolves to — the executor scenario runs in another process and has to
    find the same state file this one publishes.
    """
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
    wire, but ``control_system.writes_enabled`` is read by the connector itself
    out of ``config.yml`` — it is a launch-time deployment posture, not a value
    the launch payload may grant. So a deployment that means to select its
    write-capable gateway has to exist as a file, and the manager passes that
    file's path to every child it spawns.
    """
    written: list[Path] = []

    def write(raw: dict) -> Path:
        path = tmp_path / f"config_{len(written)}.yml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        written.append(path)
        return path

    return write


@pytest.fixture
async def make_manager(state_root, endpoints, written_config, tmp_path):
    """Managers whose children are all reaped when the test ends."""
    created: list[ConnectorHostManager] = []

    def factory(raw: dict | None = None, **overrides):
        options = {
            "drain_timeout_s": DRAIN_TIMEOUT_S,
            "probe_timeout_s": PROBE_TIMEOUT_S,
            "spawn_timeout_s": SPAWN_TIMEOUT_S,
            "terminate_grace_s": 2.0,
        }
        options.update(overrides)
        if raw is None:
            raw = raw_config(va_port=endpoints.va, live_port=endpoints.live, project_root=tmp_path)
        manager = ConnectorHostManager(
            MCPServerConfig(raw=raw, config_path=written_config(raw)),
            **options,
        )
        manager.spawned = []  # type: ignore[attr-defined]
        original_spawn = manager._spawn

        async def recording_spawn(target):
            process = await original_spawn(target)
            manager.spawned.append(process)  # type: ignore[attr-defined]
            return process

        manager._spawn = recording_spawn  # type: ignore[method-assign]
        manager.reset_state()
        created.append(manager)
        return manager

    yield factory

    for manager in created:
        with contextlib.suppress(Exception):
            await manager.shutdown()
        for process in manager.spawned:  # type: ignore[attr-defined]
            if process.returncode is None:  # pragma: no cover - teardown safety net
                with contextlib.suppress(ProcessLookupError, OSError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), SETTLE_TIMEOUT_S)


async def started_on(factory, target: str, **overrides) -> ConnectorHostManager:
    """A manager with a live child on *target*."""
    manager = factory(**overrides)
    await manager.start(target)
    assert manager.has_child()
    return manager


@pytest.fixture
def served_context(monkeypatch):
    """Install a context as the controls server's singleton, for the tools."""

    def install(manager: ConnectorHostManager) -> ControlSystemContext:
        context = context_for(manager)
        monkeypatch.setattr(server_context_mod, "_registry", context)
        return context

    yield install
    server_context_mod.reset_server_context()


@pytest.fixture
def quiet_switch_notifications(monkeypatch):
    """Silence the operator-facing switch emit.

    It is fire-and-forget over HTTP to a web terminal that is not running here;
    stubbing it keeps a switch's measured duration a measurement of the switch.
    """
    recorded: list[dict] = []

    async def record(**kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr(control_target, "notify_target_switch_async", record)
    return recorded


async def wait_for(predicate, timeout: float = SETTLE_TIMEOUT_S) -> bool:
    """Poll *predicate* on the event loop until it holds, or give up."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


async def reading(manager: ConnectorHostManager, address: str) -> float:
    """One value off the wire, through whichever child is serving right now."""
    value = await manager.active_proxy().read_channel(address, timeout=READ_TIMEOUT_S)
    assert isinstance(value, ChannelValue)
    return value.value


# ---------------------------------------------------------------------------
# 1. The round trip
# ---------------------------------------------------------------------------


class TestALiveVaRoundTrip:
    """Both directions, and the evidence that the two ends are different machines.

    The values compared are read with no client write anywhere in the module up
    to this point: both containers serve the same lattice at rest, and the only
    thing that separates their readings is the readout error seeded into one of
    them at boot. That is what makes "the wire moved" an observation rather than
    an inference from the switch's own report.
    """

    async def test_a_round_trip_serves_a_different_machine_at_each_end(self, make_manager):
        manager = await started_on(make_manager, "live")

        live_reading = await reading(manager, SEEDED_BPM)
        outbound = await manager.switch("va")
        va_reading = await reading(manager, SEEDED_BPM)
        homebound = await manager.switch("live")
        live_again = await reading(manager, SEEDED_BPM)

        assert va_reading != live_reading, (
            "both ends served the same value for the seeded BPM, so the switch "
            "moved the session between two indistinguishable machines"
        )
        # The unseeded machine at rest has an identically zero closed orbit, and
        # the seeded one puts a fixed offset on that zero. Neither is asserted
        # to a literal here -- what is asserted is that the difference is the
        # seeded offset's order and not float noise.
        assert abs(live_reading - va_reading) > SEEDED_OFFSET_X / 2
        assert live_again == live_reading, (
            "coming home read a different value than leaving did, so something "
            "other than the switch moved the machine"
        )
        assert outbound["target"] == "va"
        assert homebound["target"] == "live"

    async def test_each_leg_moves_the_generation_and_the_endpoint(self, make_manager, endpoints):
        manager = await started_on(make_manager, "live")
        assert manager.active_generation() == 0

        outbound = await manager.switch("va")
        homebound = await manager.switch("live")

        assert outbound["generation"] == 1
        assert homebound["generation"] == 2
        assert outbound["connector_type"] == "virtual_accelerator"
        assert homebound["connector_type"] == "epics"
        # The child really came up on the destination's own port -- verified by
        # the switch itself against the derivation, and reported here.
        assert outbound["endpoint"]["port"] == endpoints.va
        assert homebound["endpoint"]["port"] == endpoints.live
        assert outbound["probe_channel"] == VA_PROBE

    async def test_the_previous_child_dies_on_every_leg(self, make_manager):
        manager = await started_on(make_manager, "live")
        first = manager.spawned[0]

        await manager.switch("va")
        second = manager.spawned[1]
        assert await wait_for(lambda: first.returncode is not None), (
            "the child for the previous target outlived the switch away from it"
        )

        await manager.switch("live")
        assert await wait_for(lambda: second.returncode is not None)
        assert manager.status()["child_pid"] == manager.spawned[2].pid

    async def test_the_state_file_follows_the_session_both_ways(self, make_manager):
        manager = await started_on(make_manager, "live")

        await manager.switch("va")
        assert target_state.read()["target"] == "va"
        assert target_state.read()["generation"] == 1

        result = await manager.switch("live")
        record = target_state.read()
        assert record["target"] == "live"
        assert record["generation"] == 2
        assert record["children"] == [result["child_pid"]]


# ---------------------------------------------------------------------------
# 2. A failed switch (CF-2)
# ---------------------------------------------------------------------------


class TestAFailedSwitchLeavesThePreviousTargetActive:
    """CF-2: a destination that cannot answer never becomes the session's.

    The destination here is a port nothing is listening on — so the candidate
    child spawns, configures exactly the endpoint the derivation named (which is
    why this is a *probe* failure and not a verification one), and then fails to
    read the channel that would have proven the target reachable. What the test
    asserts is not merely that the switch raised, but that the session is still
    working afterwards: same target, same generation, same child, and a real
    read that still comes back off the wire.
    """

    @pytest.fixture
    def wrong_port_deployment(self, endpoints):
        """The ``va`` block pointed at a port no server holds."""
        return raw_config(va_port=_free_port(), live_port=endpoints.live)

    async def test_the_switch_fails_at_the_probe(self, make_manager, wrong_port_deployment):
        manager = await started_on(make_manager, "live", raw=wrong_port_deployment)

        with pytest.raises(SwitchError) as raised:
            await manager.switch("va")

        assert raised.value.stage == "probe"
        assert raised.value.reason == "probe_failed"
        assert raised.value.target == "va"
        # The detail names the target and the step, which is what an operator
        # acts on. It does NOT name the probe channel: the child bounds the
        # probe with its own ``asyncio.wait_for``, which wins the race against
        # the connector's timeout and carries an empty message. Asserted as it
        # is rather than as it might read better, so a change to either bound
        # shows up here.
        assert "'va'" in raised.value.detail
        assert "spawn_probe" in raised.value.detail

    async def test_the_previous_target_is_still_active_and_still_serving(
        self, make_manager, wrong_port_deployment
    ):
        manager = await started_on(make_manager, "live", raw=wrong_port_deployment)
        original_child = manager.status()["child_pid"]

        with pytest.raises(SwitchError):
            await manager.switch("va")

        assert manager.active_target() == "live"
        assert manager.active_generation() == 0
        assert manager.status()["child_pid"] == original_child
        # Still serving, not merely still running: the distinction is the whole
        # point of a failed switch leaving the session usable.
        assert isinstance(await reading(manager, LIVE_PROBE), float)

    async def test_the_candidate_child_is_gone_and_nothing_was_published(
        self, make_manager, wrong_port_deployment
    ):
        manager = await started_on(make_manager, "live", raw=wrong_port_deployment)

        with pytest.raises(SwitchError):
            await manager.switch("va")

        candidate = manager.spawned[1]
        assert await wait_for(lambda: candidate.returncode is not None), (
            "the candidate child outlived the switch that failed to adopt it"
        )
        record = target_state.read()
        assert record["target"] == "live"
        assert record["generation"] == 0

    async def test_the_roster_still_names_live_at_generation_zero(
        self, make_manager, served_context, wrong_port_deployment
    ):
        """CF-2 as the operator meets it: through the tool they would ask.

        Everything above is read off the manager. This reads the ``control_target``
        roster instead — the answer an operator actually gets after a switch has
        failed under them — so a supervisor that kept the right state while the
        reporting tool described a session somewhere else would fail here.

        The ``va`` row still reports ``available_now``, and that is correct
        rather than a contradiction: eligibility is a question about
        configuration, and the destination *is* configured — a gateway, a role,
        a probe channel. What it is not is reachable, which is the part no
        config can answer and which only the switch's own probe found out. The
        roster says so honestly by carrying no measured reachability at all
        here, because no endpoint prober is running in this deployment.
        """
        manager = await started_on(make_manager, "live", raw=wrong_port_deployment)
        served_context(manager)
        unreachable_port = wrong_port_deployment["control_system"]["connector"][
            "virtual_accelerator"
        ]["gateways"]["write_access"]["port"]

        with pytest.raises(SwitchError):
            await manager.switch("va")

        roster = json.loads(await get_tool_fn(control_target.control_target)())

        summary = roster["summary"]
        assert summary["target"] == "live"
        assert summary["generation"] == 0
        assert summary["baseline_target"] == "live"
        assert summary["connector_host_alive"] is True
        rows = roster["access_details"]["targets"]
        assert rows["live"]["active"] is True
        assert rows["live"]["reason"] == "already_active"
        assert rows["va"]["active"] is False
        # The roster still names the endpoint the switch could not reach, which
        # is the one thing an operator needs in order to fix it.
        assert rows["va"]["endpoints"]["write_access"]["port"] == unreachable_port
        assert "endpoint_tcp" not in rows["va"]["endpoints"]["write_access"]
        assert roster["access_details"]["endpoint_probe"]["running"] is False

    async def test_the_session_can_still_switch_to_a_destination_that_answers(
        self, make_manager, written_config, endpoints, wrong_port_deployment
    ):
        """The failure is about the destination, not about the session.

        A session left unable to switch at all after one bad attempt would
        satisfy every assertion above while being broken. The operator's fix —
        correcting the destination's port — is applied here as a config change,
        and the very next switch is expected to work.
        """
        manager = await started_on(make_manager, "live", raw=wrong_port_deployment)
        with pytest.raises(SwitchError):
            await manager.switch("va")

        corrected = raw_config(va_port=endpoints.va, live_port=endpoints.live)
        manager._config = MCPServerConfig(raw=corrected, config_path=written_config(corrected))
        result = await manager.switch("va")

        assert result["target"] == "va"
        assert result["generation"] == 1


# ---------------------------------------------------------------------------
# 3. Fail-closed when the child is killed externally
# ---------------------------------------------------------------------------


class TestFailClosedWhenTheChildIsKilledExternally:
    """FR-1: a session whose connector host died refuses, and says why.

    The child is killed with ``SIGKILL`` from outside — not through a switch,
    which by construction never produces this state. What the session owes its
    operator then is a refusal naming the missing host, not a silent respawn
    onto a target nobody re-selected and not a read answered from somewhere else.

    **The archiver.** FR-1 also says archiver reads keep serving while the
    control system refuses, because the archiver is never routed through the
    connector host. This e2e deployment configures no archiver connector — the
    harness runs two soft-IOCs and no history store, so there is nothing here to
    read a real archive from, and asserting it against a fake would be asserting
    against the fake. The routing claim itself is pinned in process by
    ``tests/mcp_server/test_switch_lifecycle.py::TestServingFromTheChild::
    test_the_archiver_is_never_routed_through_the_connector_host``; what this
    class adds is that the control-system half really does refuse against a real
    child that really died.
    """

    async def test_every_control_system_operation_refuses_with_the_no_child_reason(
        self, make_manager
    ):
        manager = await started_on(make_manager, "va")
        context = context_for(manager)
        proxy = await context.control_system()
        assert isinstance(await proxy.read_channel(VA_PROBE, timeout=READ_TIMEOUT_S), ChannelValue)

        os.kill(manager.status()["child_pid"], signal.SIGKILL)
        assert await wait_for(lambda: not manager.has_child()), (
            "the manager kept reporting a child that had been killed"
        )

        with pytest.raises(NoConnectorHostError) as raised:
            await context.control_system()

        refusal = raised.value
        assert isinstance(refusal, ConnectionError)  # the tools' handler knows this one
        assert refusal.as_dict()["reason"] == "no_connector_host"
        assert refusal.target == "va"
        assert refusal.generation == manager.active_generation()

    async def test_the_session_still_names_the_target_it_was_on(self, make_manager):
        """A dead child is not a switch, and claiming one would invent it."""
        manager = await started_on(make_manager, "va")
        generation = manager.active_generation()

        os.kill(manager.status()["child_pid"], signal.SIGKILL)
        assert await wait_for(lambda: not manager.has_child())

        assert manager.active_target() == "va"
        assert manager.active_generation() == generation
        assert manager.active_proxy() is None
        assert manager.status()["child_alive"] is False
        # Nothing is quietly restarted on the way to a read.
        assert len(manager.spawned) == 1

    async def test_invalidating_brings_a_child_back_on_the_same_target(self, make_manager):
        """Fail-closed, not fail-permanent: the documented recovery really works.

        The refusal is a ``ConnectionError``, which is what the tools' shared
        handler already turns into an invalidation — so this is the path a
        deployment actually takes, driven here against real processes.
        """
        manager = await started_on(make_manager, "va")
        context = context_for(manager)
        await context.control_system()
        generation = manager.active_generation()

        os.kill(manager.status()["child_pid"], signal.SIGKILL)
        assert await wait_for(lambda: not manager.has_child())
        with pytest.raises(NoConnectorHostError):
            await context.control_system()

        await context.invalidate_connector("control_system")

        assert manager.has_child() is True
        assert manager.active_target() == "va"
        assert manager.active_generation() == generation
        assert isinstance(await reading(manager, VA_PROBE), float)


# ---------------------------------------------------------------------------
# 4. The write race
# ---------------------------------------------------------------------------


def write_approval_stamp(operations: list[dict], binding: tuple[str, int]) -> Path:
    """Leave the stamp the approval hook leaves, for *binding*.

    The file name is derived through the tool's own
    ``_approval_stamp_key`` — the hook restates that derivation in stdlib-only
    Python and a separate test pins the two spellings equal, so using the tool's
    is using the contract rather than a copy of it. The record carries this
    process's PID because a stamp another server wrote is deliberately ignored.
    """
    target, generation = binding
    key = channel_write_module._approval_stamp_key(operations, None)
    assert key is not None
    path = (
        target_state.state_dir() / f"{channel_write_module.APPROVAL_STAMP_PREFIX}{key}"
        f"{channel_write_module.APPROVAL_STAMP_SUFFIX}"
    )
    path.write_text(
        json.dumps({"target": target, "generation": generation, "server_pid": os.getpid()}),
        encoding="utf-8",
    )
    return path


class TestAnApprovedWriteIsRefusedAfterASwitch:
    """A value approved for one machine never reaches the other.

    The race is driven with the real parts on both sides: a real approval stamp
    on disk in the shape the hook writes it, and a real switch between two real
    containers in between. The refusal has to land before the connector call —
    which is asserted on the wire, by reading the setpoint back off the machine
    the session ended up on.
    """

    OPERATIONS = [{"channel": CORRECTOR_SP, "value": 0.5}]

    async def test_the_write_is_refused_naming_both_bindings(
        self, make_manager, served_context, quiet_switch_notifications
    ):
        manager = await started_on(make_manager, "va")
        served_context(manager)
        approved = manager.active_binding()
        write_approval_stamp(self.OPERATIONS, approved)

        await manager.switch("live")

        with assert_raises_error(error_type="target_changed") as captured:
            await get_tool_fn(channel_write_module.channel_write)(operations=self.OPERATIONS)

        details = captured["envelope"]["details"]
        assert details["window"] == "approval_prompt_to_entry"
        assert (details["approved_target"], details["approved_generation"]) == approved
        assert (
            details["current_target"],
            details["current_generation"],
        ) == manager.active_binding()
        assert details["approved_target"] != details["current_target"]

    async def test_nothing_reached_either_machine(
        self, make_manager, served_context, quiet_switch_notifications
    ):
        manager = await started_on(make_manager, "va")
        served_context(manager)
        write_approval_stamp(self.OPERATIONS, manager.active_binding())

        await manager.switch("live")
        with assert_raises_error(error_type="target_changed"):
            await get_tool_fn(channel_write_module.channel_write)(operations=self.OPERATIONS)

        assert await reading(manager, CORRECTOR_SP) == pytest.approx(0.0)
        await manager.switch("va")
        assert await reading(manager, CORRECTOR_SP) == pytest.approx(0.0)

    async def test_a_stamp_naming_the_current_binding_is_not_a_refusal(
        self, make_manager, served_context
    ):
        """Anti-vacuous control: the refusal is the switch, not the stamp.

        Same stamp mechanism, same tool, no switch in between — so a
        ``target_changed`` refusal here would mean the test above proved nothing
        about switching. The write itself is left to whatever the machine says;
        only the binding gate is under test.
        """
        manager = await started_on(make_manager, "va")
        served_context(manager)
        write_approval_stamp(self.OPERATIONS, manager.active_binding())

        try:
            result = await get_tool_fn(channel_write_module.channel_write)(
                operations=self.OPERATIONS
            )
            assert json.loads(result)["status"] != "error"
        finally:
            # Every later read in this module is of a machine at rest, so the
            # restore has to happen whether or not the assertion above held --
            # a failing assertion must cost this run one test, not all of them.
            with contextlib.suppress(Exception):
                await manager.active_proxy().write_channel(
                    CORRECTOR_SP, 0.0, timeout=READ_TIMEOUT_S
                )

        assert await reading(manager, CORRECTOR_SP) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. The executor's pinned target
# ---------------------------------------------------------------------------

#: Sandbox-side agent code. Run in a real subprocess carrying the real stamp,
#: because the pin is a contract between two processes and neither can see the
#: other's state directly. Prints one JSON line on stdout and nothing else.
#:
#: ``initialize_registry`` is the sandbox's own setup step, restated here rather
#: than skipped: it is what ``ExecutionWrapper.create_wrapper`` emits ahead of
#: every execution, and it is what populates the connector factory — a bare
#: process importing ``osprey.runtime`` has an empty one and would fail with
#: "Unknown control system type" long before reaching the pin.
#:
#: The verdict is written to a file rather than printed, because registry
#: initialisation is chatty on both streams and a result parsed out of that
#: noise would be a result this test could misread.
#:
#: The exit is abrupt for the reason ``osprey_connectors.ipc.host`` states of
#: its own: a process that has held a Channel Access context can block forever
#: in pyepics' ``finalize_libca`` atexit hook, and a sandbox that will not die
#: is worse than one that skips its hooks. ``conftest._readiness_pv_served``
#: ends the same way. Measured here: the write itself completes in about a
#: second and the teardown then wedges indefinitely.
_PINNED_WRITE = """
import json
import os
import sys
from pathlib import Path

from osprey.registry import initialize_registry

initialize_registry(auto_export=False, config_path=os.environ["CONFIG_FILE"])

import osprey.runtime as runtime

address, value, verdict_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
try:
    runtime.write_channel(address, value, timeout=30.0)
except runtime.ControlTargetChangedError as exc:
    verdict = {"wrote": False, "error": "ControlTargetChangedError", "message": str(exc)}
else:
    verdict = {"wrote": True}
Path(verdict_path).write_text(json.dumps(verdict), encoding="utf-8")
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
"""


@pytest.fixture
def sandbox(tmp_path, endpoints, state_root):
    """Run stamped agent code the way the executor runs it: another process.

    The deployment is written to ``config.yml`` in ``tmp_path`` so that the
    subprocess resolves the *same* project root, and therefore the same shared
    data root, that :func:`state_root` patched in this process — the pin is a
    comparison against a state file, and a subprocess reading a different
    directory would be comparing against nothing.
    """
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        yaml.safe_dump(
            raw_config(va_port=endpoints.va, live_port=endpoints.live, project_root=tmp_path)
        ),
        encoding="utf-8",
    )

    runs = 0

    def run(*, target: str, generation: int, value: float) -> dict:
        nonlocal runs
        runs += 1
        verdict_path = tmp_path / f"sandbox_verdict_{runs}.json"
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(REPO_PATHS),
            "CONFIG_FILE": str(config_path),
            "OSPREY_CONFIG": str(config_path),
            host_executor.ENV_CONTROL_TARGET: target,
            host_executor.ENV_CONTROL_TARGET_GENERATION: str(generation),
            host_executor.ENV_CONTROL_TARGET_STATE_PID: str(os.getpid()),
        }
        for stale in ("EPICS_CA_ADDR_LIST", "EPICS_CA_NAME_SERVERS", "EPICS_CA_SERVER_PORT"):
            environment.pop(stale, None)
        completed = subprocess.run(
            [sys.executable, "-c", _PINNED_WRITE, CORRECTOR_SP, str(value), str(verdict_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(tmp_path),
            env=environment,
        )
        if completed.returncode != 0 or not verdict_path.exists():
            raise RuntimeError(
                f"the stamped sandbox failed:\n{completed.stdout}\n{completed.stderr}"
            )
        return json.loads(verdict_path.read_text(encoding="utf-8"))

    return run


class TestTheExecutorsWriteIsPinnedToItsTarget:
    """A run stamped at one generation stops writing once the session moves.

    Both halves run against real machines: the permitted write really lands on
    the stamped target's soft-IOC, and the refused one really does not. The
    switch between them is a real switch, so the generation the sandbox is
    compared against is one a real spawn-then-swap published.
    """

    async def test_a_stamped_write_lands_and_the_same_stamp_refuses_after_a_switch(
        self, make_manager, sandbox
    ):
        manager = await started_on(make_manager, "va")
        target, generation = manager.active_binding()
        try:
            permitted = sandbox(target=target, generation=generation, value=PINNED_WRITE_VALUE)
            assert permitted == {"wrote": True}
            assert await reading(manager, CORRECTOR_SP) == pytest.approx(PINNED_WRITE_VALUE), (
                "the stamped sandbox reported a write that never reached the machine"
            )

            await manager.switch("live")
            assert manager.active_generation() != generation

            refused = sandbox(target=target, generation=generation, value=REFUSED_WRITE_VALUE)

            assert refused["wrote"] is False
            assert refused["error"] == "ControlTargetChangedError"
            message = refused["message"]
            assert f"{target!r} generation {generation}" in message
            assert f"generation {manager.active_generation()}" in message

            await manager.switch("va")
            assert await reading(manager, CORRECTOR_SP) == pytest.approx(PINNED_WRITE_VALUE), (
                "the refused write moved the machine anyway"
            )
        finally:
            # Every later read in this module is of a machine at rest.
            with contextlib.suppress(Exception):
                if manager.active_target() != "va":
                    await manager.switch("va")
                await manager.active_proxy().write_channel(
                    CORRECTOR_SP, 0.0, timeout=READ_TIMEOUT_S
                )


# ---------------------------------------------------------------------------
# 6. A switch refused while an execution is in flight
# ---------------------------------------------------------------------------


class TestASwitchIsRefusedWhileAnExecutionIsInFlight:
    """The switch tool declines while a run holds the machine.

    The marker is written by the executor's own
    :func:`~osprey.mcp_server.python_executor.executor._in_flight_marker` — the
    production writer, naming this live process — so what is under test is the
    real contract between the two servers and not a hand-rolled file that
    happens to match it.
    """

    async def test_the_refusal_names_the_running_target_and_nothing_moves(
        self, make_manager, served_context, quiet_switch_notifications
    ):
        manager = await started_on(make_manager, "va")
        served_context(manager)
        child_before = manager.status()["child_pid"]

        with host_executor._in_flight_marker("va"):
            assert control_target.in_flight_executions(), (
                "the executor's own marker writer left nothing for the switch to see"
            )
            with assert_raises_error(error_type="target_switch_refused") as captured:
                await get_tool_fn(control_target.control_target_set)(target="live")

        details = captured["envelope"]["details"]
        assert details["reason"] == "execution_in_flight"
        assert details["executing_target"] == "va"
        assert details["executor_pid"] == os.getpid()
        assert manager.active_target() == "va"
        assert manager.active_generation() == 1
        assert manager.status()["child_pid"] == child_before

    async def test_the_same_switch_succeeds_once_the_execution_ends(
        self, make_manager, served_context, quiet_switch_notifications
    ):
        """Anti-vacuous control: the marker is what refused, not the deployment."""
        manager = await started_on(make_manager, "va")
        served_context(manager)

        with host_executor._in_flight_marker("va"):
            with assert_raises_error(error_type="target_switch_refused"):
                await get_tool_fn(control_target.control_target_set)(target="live")

        payload = json.loads(await get_tool_fn(control_target.control_target_set)(target="live"))

        assert payload["summary"]["target"] == "live"
        assert payload["summary"]["previous_target"] == "va"
        assert manager.active_target() == "live"
        assert isinstance(await reading(manager, LIVE_PROBE), float)

    async def test_a_marker_from_a_dead_executor_does_not_wedge_the_switch(
        self, make_manager, served_context, quiet_switch_notifications
    ):
        """Residue from a killed executor is swept, not honoured.

        Without that, one crashed run would make every later switch impossible
        with nothing an operator could stop.
        """
        manager = await started_on(make_manager, "va")
        served_context(manager)
        finished = subprocess.Popen([sys.executable, "-c", ""])
        finished.wait(timeout=SETTLE_TIMEOUT_S)
        stale = (
            target_state.state_dir() / f"{control_target.INFLIGHT_FILE_PREFIX}{finished.pid}_stale"
            f"{control_target.INFLIGHT_FILE_SUFFIX}"
        )
        stale.write_text(
            json.dumps({"pid": finished.pid, "owner_ppid": os.getpid(), "target": "va"}),
            encoding="utf-8",
        )

        payload = json.loads(await get_tool_fn(control_target.control_target_set)(target="live"))

        assert payload["summary"]["target"] == "live"
        assert not stale.exists()


# ---------------------------------------------------------------------------
# 7. The single Bluesky lane, while the session is switched
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    BLUESKY_IMPORT_ERROR is not None,
    reason=f"the Bluesky MCP tools are not importable here ({BLUESKY_IMPORT_ERROR})",
)
class TestTheSingleBlueskyLaneRefusesWhileTheSessionIsSwitched:
    """OC-1, against a state file a real container switch published.

    A deployment renders exactly one plan lane, wired at build time to one
    control target, so a session that has switched away from it must not queue
    or start a PLAN. The refusal is host-side and lands *before* any HTTP call —
    which is what lets this run in an e2e deployment that has no Bluesky stack at
    all: the transport is asserted never to be reached, and that assertion is the
    load-bearing one. What this adds over the in-process pin in
    ``tests/services/test_single_lane_switch_refusal.py`` is that the switch
    driving it is a real spawn-then-swap between two real machines rather than a
    hand-written state record.
    """

    @pytest.fixture
    def lane_deployment(self, monkeypatch, endpoints):
        """Arm the Bluesky tools against this module's deployment."""
        config = raw_config(va_port=endpoints.va, live_port=endpoints.live)
        monkeypatch.setattr(target_banner, "load_osprey_config", lambda: config)
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda key, default=None, config_path=None: (
                True if key == "control_system.writes_enabled" else default
            ),
        )
        monkeypatch.setenv("BLUESKY_LAUNCH_TOKEN", "va-e2e-switch-token")
        bluesky_context.initialize_server_context()
        yield
        bluesky_context.reset_server_context()

    async def test_queue_add_refuses_and_never_reaches_the_bridge(
        self, make_manager, lane_deployment
    ):
        manager = await started_on(make_manager, "live")
        assert manager.baseline == "live"

        await manager.switch("va")

        with patch(f"{bluesky_queue.__name__}._http_post_json") as post:
            with patch(f"{bluesky_queue.__name__}.notify_agent_activity_async"):
                with assert_raises_error(
                    error_type=bluesky_backend.REASON_SESSION_TARGET_MISMATCH
                ) as captured:
                    await get_tool_fn(bluesky_queue.queue_add)(draft_revision=7)

        assert not post.called, "the refusal reached the bridge instead of stopping at the host"
        envelope = captured["envelope"]
        assert envelope["details"]["session_target"] == "va"
        assert "va" in envelope["error_message"] and "live" in envelope["error_message"]

    async def test_the_lane_is_usable_again_once_the_session_comes_home(
        self, make_manager, lane_deployment
    ):
        """Anti-vacuous control: the refusal is the switch, not the arming.

        With the session back on the deployment's baseline the same call reaches
        the transport — so the refusal above was about where the session was
        pointed and not about anything this fixture set up.
        """
        manager = await started_on(make_manager, "live")
        await manager.switch("va")
        await manager.switch("live")

        with patch(
            f"{bluesky_queue.__name__}._http_post_json", return_value=(200, {"run_id": "r1"})
        ) as post:
            with patch(f"{bluesky_queue.__name__}.notify_agent_activity_async"):
                await get_tool_fn(bluesky_queue.queue_add)(draft_revision=7)

        assert post.called
        assert post.call_args.args[0] == "/queue/items"


def test_the_bluesky_lane_scenario_states_which_way_it_went() -> None:
    """Say out loud whether the lane scenario ran or skipped, and why.

    A conditional skip that nobody reads is indistinguishable from coverage. If
    the Bluesky tools are not importable in this checkout the reason is recorded
    here, in the run's own output, rather than only in a skip marker.
    """
    if BLUESKY_IMPORT_ERROR is not None:  # pragma: no cover - environment-dependent
        pytest.skip(
            "the single-lane Bluesky refusal was NOT exercised: this deployment cannot "
            f"import the Bluesky MCP tools ({BLUESKY_IMPORT_ERROR})"
        )
    assert bluesky_backend.REASON_SESSION_TARGET_MISMATCH == "session_target_mismatch"


# ---------------------------------------------------------------------------
# 8. The drain bound
# ---------------------------------------------------------------------------


class TestTheSwitchIsBoundByItsDrainDeadline:
    """A switch completes on the deadline's schedule, not the machine's.

    The request left in flight is a read of a channel neither container serves,
    bounded by the connector's own 120 s timeout — two orders of magnitude past
    the drain deadline, so a switch that waited for it would be unmistakable.
    Five consecutive switches, each with a fresh hung read, because the property
    worth pinning is that the bound holds *repeatedly*: a supervisor that leaked
    one un-drained child per switch would satisfy a single measurement.

    Each hung read must also end by naming the switch. An anonymous
    end-of-pipe would leave an operator with a failed read and no account of
    why, which is the failure this attribution exists to prevent.
    """

    async def test_five_consecutive_switches_stay_within_the_drain_bound(
        self, make_manager, served_context, quiet_switch_notifications, endpoints
    ):
        manager = await started_on(
            make_manager,
            "live",
            raw=raw_config(
                va_port=endpoints.va, live_port=endpoints.live, drain_timeout_s=DRAIN_TIMEOUT_S
            ),
            drain_timeout_s=None,
        )
        served_context(manager)
        bound = DRAIN_TIMEOUT_S + SWITCH_TIMING_SLACK_S
        hung_reads: list[asyncio.Task] = []
        durations: list[float] = []

        try:
            for _ in range(CONSECUTIVE_SWITCHES):
                hung = asyncio.create_task(
                    manager.active_proxy().read_channel(
                        UNSERVED_CHANNEL, timeout=CONNECTOR_TIMEOUT_S
                    )
                )
                hung_reads.append(hung)
                # Let the request reach the child before the switch starts draining.
                await asyncio.sleep(0.5)
                assert not hung.done(), "the hung read finished before the switch began"

                destination = "va" if manager.active_target() == "live" else "live"
                started = time.monotonic()
                payload = json.loads(
                    await get_tool_fn(control_target.control_target_set)(target=destination)
                )
                durations.append(time.monotonic() - started)

                assert payload["summary"]["target"] == destination
                assert payload["access_details"]["drain_timeout_s"] == DRAIN_TIMEOUT_S
                assert payload["access_details"]["previous_drained"] is False, (
                    "the child drained cleanly, so this switch never met the deadline "
                    "the timing criterion is about"
                )

            assert all(duration < bound for duration in durations), (
                f"switch durations {durations} exceeded the {bound}s bound "
                f"(drain deadline {DRAIN_TIMEOUT_S}s + {SWITCH_TIMING_SLACK_S}s)"
            )
            assert manager.active_generation() == CONSECUTIVE_SWITCHES
            assert len(manager.spawned) == CONSECUTIVE_SWITCHES + 1

            for hung in hung_reads:
                with pytest.raises(ConnectionError) as raised:
                    await asyncio.wait_for(hung, SETTLE_TIMEOUT_S)
                message = str(raised.value)
                assert "target switch" in message
                assert "drain deadline" in message
        finally:
            for hung in hung_reads:
                if not hung.done():  # pragma: no cover - teardown safety net
                    hung.cancel()
                with contextlib.suppress(BaseException):
                    await hung

    async def test_an_idle_child_still_drains_cleanly(
        self, make_manager, served_context, quiet_switch_notifications, endpoints
    ):
        """The bound is a ceiling, not a floor.

        With nothing in flight the switch drains rather than killing, which is
        what makes ``previous_drained: False`` above evidence of a deadline
        having actually been met.
        """
        manager = await started_on(
            make_manager,
            "live",
            raw=raw_config(
                va_port=endpoints.va, live_port=endpoints.live, drain_timeout_s=DRAIN_TIMEOUT_S
            ),
            drain_timeout_s=None,
        )
        served_context(manager)

        payload = json.loads(await get_tool_fn(control_target.control_target_set)(target="va"))

        assert payload["access_details"]["previous_drained"] is True
        assert payload["summary"]["target"] == "va"


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_target_switch.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
