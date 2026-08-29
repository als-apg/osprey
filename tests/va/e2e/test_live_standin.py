"""The live stand-in, as an operator meets it: a third machine on the wire.

``virtual_accelerator.live_standin`` stands a **second** virtual accelerator up
and wires it in as the deployment's own ``standin`` control target — its own
connector type (``live_standin``, served by the EPICS connector), its own
``control_system.connector.live_standin`` block, its own name in the roster.
It is not a relabelled ``live``: ``live`` keeps meaning the machine the facility
authored, and a deployment that stands a stand-in up beside it gains a target
rather than losing one.

Three claims make the feature worth having, and this module is the acceptance
gate for all three (SC-10):

* the ``standin`` slot is described as a *real machine* — strict limits,
  ``real_machine: true`` — with nothing but the parenthesis on its label saying
  it is a rehearsal, and standing it up never invents a ``live`` row on a
  deployment that never named its facility's machine;
* the machine behind that label is genuinely a different one from the sandbox,
  which is only decidable by reading the same channel at both ends and getting
  different numbers back;
* and a write on it is judged on hardware's terms: the strict-limits posture
  refuses a channel the shipped limits database does not list, on a target whose
  writes are armed, before anything reaches Channel Access.

None of the three is decidable inside one process and none is decidable against
a connector fake, so this module boots two real ``osprey-va-full`` containers
and reads them over Channel Access through a real connector-host child.

Two containers, and what tells them apart
-----------------------------------------
Both instances run one image over one lattice and one machine description, so
at rest they are indistinguishable — which is exactly why the stand-in ships a
perturbation. Container S (the stand-in, the ``standin`` target) boots with
``VA_BPM_ERRORS`` set to
:data:`~osprey.services.virtual_accelerator.manifest.standin_defaults.STANDIN_BPM_ERRORS_DEFAULT`
and container V (the sandbox, the ``va`` target) boots with none, so the
perturbed BPMs read different numbers at the two ends with no client write
anywhere in this file, from a machine at rest, and reproducibly.

The shipped default is **offsets only**, which is what makes the comparison
below deterministic rather than statistical: with the rest of ``bpm_read``'s
keywords at identity, a seeded reading is exactly ``x - offset`` — no gain, no
roll, no noise. So this module asserts on a difference of the offset's own
order, and separately requires an *unperturbed* BPM to agree between the two
boots to well inside that order. The second assertion is what makes the first
attributable: a global drift between two independently booted containers would
move both channels, and this file would report it as a failed control rather
than as a stand-in that works.

The constants below are **derived from the shipped default**, not written down
beside it. A change to what the stand-in perturbs moves this module's subject
with it, and a change that emptied the table fails the guards in
``TestTheShippedDefaultsAreWhatThisModuleMeasured`` rather than quietly leaving
this file measuring nothing.

Which targets this deployment has, and which it deliberately has not
--------------------------------------------------------------------
The scratch deployment is baselined on the sandbox (``control_system.type:
virtual_accelerator``) and carries exactly two connector blocks:
``virtual_accelerator`` and ``live_standin``. Neither is the facility's own
machine, so ``live`` stays *underivable* here — and that is the point rather
than a gap in the fixture. A stand-in must never be the answer to "where is the
real machine", so the roster below is asserted to offer ``va`` and ``standin``
and no ``live`` row at all, while the display metadata still carries a truthful
``live machine (not configured)`` slot for the readers that render one.

That also makes the switch under test a switch *away* from the baseline toward
``standin``, which is the direction the FR-8 gates guard: the strict limits
posture applies (asserted both ways), and the operator acknowledgment does not
— the stand-in's equivalent was said at build time by the profile line that
stood it up, so this deployment sets no
``control_system.target_switch.live_gateway_acknowledged`` at all and is still
eligible.

The write leg, and why it is the posture that refuses it
--------------------------------------------------------
Writes are armed **on the stand-in's own block** (``control_system.connector.
live_standin.writes_enabled: true``) and nowhere else. That is what makes the
refusal attributable: with writes unarmed the connector's ``writes_enabled``
guard would refuse first, and the test would be measuring the wrong gate. With
them armed, the deployment selects the ``write_access`` gateway for ``standin``
(asserted), the write reaches the limits validator, and
``allow_unlisted_channels: false`` refuses it with ``UNLISTED_CHANNEL`` —
a verdict about the *database*, not about the network. A database that had
failed to load refuses everything with ``LIMITS_DATABASE_UNAVAILABLE`` instead,
so asserting the exact violation type is what separates "the posture read the
shipped database and this channel is not in it" from "the posture blocked
everything because it could not read anything".

Nothing is written to a channel the machine serves. Every channel the packaged
manifest serves is listed in the shipped limits database, so an unlisted address
is necessarily one nothing answers on — and the refusal happens before any
``caput`` is issued, so the stand-in is left exactly as this module found it.

What the raw ``docker run`` here is, and is not
----------------------------------------------
A deployment renders the stand-in through the compose template, which passes
the default as ``VA_BPM_ERRORS: "${VA_STANDIN_BPM_ERRORS-<default>}"`` on the
stand-in service only. There is no compose project here: this module starts two
plain containers and sets ``VA_BPM_ERRORS`` on one of them directly, which is
the same variable the template resolves to. What is *not* covered here is the
rendering — that is ``tests/deployment/``'s subject — and the acceptance gate
for the feature is still a rebuilt demo deployment, by hand.

Ports, names and 5064
---------------------
Each container binds an ephemeral port and publishes it unchanged: a Channel
Access search reply carries the server's own port, so a remap would hand every
client an address nothing answers on. That port also *names* the container, for
the reason ``test_serving_parity.py`` states at length — a fixed name is
mutually destructive between concurrent runs. Nothing here goes near 5064.

The container helpers (``_free_port``, ``_docker``, ``_require_image``,
``_served``, ``_serving``) are copied from ``test_target_switch.py`` rather than
imported from it, as ``test_serving_parity.py`` copies them too: that module
defines fixtures and imports the Bluesky queueserver stack at import time, and a
test module is not an importable helper library. The copies are small, and each
suite's ``_serving`` differs in what it seeds.

Every Channel Access operation in this file happens in **another process**: in
the readiness probe's subprocess, or in a connector-host child. This process
never becomes a CA client, which is the rule ``conftest.py`` states — libca
latches ``EPICS_CA_*`` on initialisation and its contexts are per-thread, so a
main-thread pyepics call here would deadlock the very children under test.

The whole directory is opt-in behind ``OSPREY_VA_E2E_ENABLE=1``; the skip is
applied by ``conftest.pytest_collection_modifyitems`` rather than by a marker
here, so this module collects cleanly and skips cleanly without the flag.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.connector_host_manager import (
    ConnectorHostManager,
    target_display_metadata,
)
from osprey.mcp_server.control_system.server_context import MCPServerConfig
from osprey.mcp_server.control_system.target_eligibility import (
    ACK_LEAF,
    DIRECTION_AWAY,
    REASON_LIMITS_POSTURE,
    REASON_STANDIN_NOT_DEPLOYED,
    evaluate_eligibility,
)
from osprey.mcp_server.control_system.tools.control_target import target_rows
from osprey.services.virtual_accelerator.manifest.standin_defaults import (
    STANDIN_BPM_ERRORS_DEFAULT,
    parse_standin_default,
)
from osprey_connectors.control_system.base import ChannelValue
from osprey_connectors.errors import ChannelLimitsViolationError
from osprey_connectors.types import LIVE_STANDIN, TARGET_LIVE, TARGET_STANDIN, TARGET_VA
from tests.va.e2e import conftest as e2e_conftest

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_PATHS = (str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src"))

#: The image under test, and the simulation data both containers serve.
IMAGE = os.environ.get("OSPREY_VA_E2E_IMAGE", "osprey-va-full:latest")
DATA_DIR = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation"

#: Container-name prefixes; ``_serving`` appends the run's own ephemeral port.
#: Named for the *target* each instance backs, since that is what the deployment
#: below calls them and what every assertion is phrased in.
CONTAINER_SANDBOX = "osprey-va-e2e-standin-va"
CONTAINER_STANDIN = "osprey-va-e2e-standin-standin"

#: Boot is generous on purpose: the image is pinned ``linux/amd64`` and a local
#: run on Apple Silicon is emulated (see ``conftest.py``). Two containers.
BOOT_TIMEOUT_S = 180.0

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 21

# -- the namespace both containers serve ------------------------------------

#: What each target's switch reads to prove itself reachable, and what the
#: readiness probe below waits for. A pyat-coupled corrector readback, served
#: identically by both instances and never written by anything in this file.
PROBE_CHANNEL = "SR:MAG:HCM:01:CURRENT:RB"

#: The shipped perturbation, as numbers. Everything below is derived from it,
#: so this module's subject moves with the default rather than beside it.
STANDIN_OFFSETS = parse_standin_default()


def _bpm(device: str, axis: str) -> str:
    """The served address of one BPM axis, spelled as the physics bridge spells it.

    ``VA_BPM_ERRORS`` names devices by fam_name (``BPM03``) and the namespace
    serves them by id (``SR:DIAG:BPM:03:POSITION:X``), so the two spellings are
    joined here once. Hand-spelling the address beside the device name is how a
    change to one of them leaves this module reading a channel it is no longer
    talking about.
    """
    return f"SR:DIAG:BPM:{device.removeprefix('BPM')}:POSITION:{axis}"


#: The device this module reads, and the axis it reads it on: a large
#: horizontal offset in the shipped table, so the divergence is well clear of
#: float noise.
PERTURBED_DEVICE = "BPM03"
PERTURBED_BPM = _bpm(PERTURBED_DEVICE, "X")
PERTURBED_OFFSET_X = STANDIN_OFFSETS[PERTURBED_DEVICE]["offset_x"]

#: The control: a BPM the shipped table does not name at all, so both instances
#: read it through the identity transform. If the two boots disagree here, they
#: are not in the same machine state and any divergence measured above is not
#: attributable to the perturbation.
CONTROL_DEVICE = "BPM01"
CONTROL_BPM = _bpm(CONTROL_DEVICE, "X")

# -- the write leg's channels -----------------------------------------------

#: The address the write leg attempts, in this deployment's own naming grammar
#: and absent from the shipped limits database — device 99 of a 72-corrector
#: ring. Every channel the packaged manifest serves *is* listed, so an unlisted
#: address is necessarily one nothing answers on; that costs the assertion
#: nothing, because ``allow_unlisted_channels: false`` refuses it in the
#: validator before any ``caput`` is issued. The guard in
#: ``TestTheShippedDefaultsAreWhatThisModuleMeasured`` is what keeps it unlisted.
UNLISTED_CHANNEL = "SR:MAG:HCM:99:CURRENT:SP"

#: A value inside the window the shipped database gives every SR corrector, so
#: the refusal below cannot be a min/max verdict wearing another name.
UNLISTED_WRITE_VALUE = 1.0

#: A channel the shipped database *does* list, read only by the guard that
#: proves this module pointed the posture at a database that really loaded.
LISTED_CHANNEL = "SR:MAG:HCM:01:CURRENT:SP"

#: The violation the strict posture reports for a channel it has no entry for.
#: Distinct from ``LIMITS_DATABASE_UNAVAILABLE``, which is what a database that
#: failed to load reports for *every* channel — the whole reason the assertion
#: below names an exact type rather than merely expecting a refusal.
UNLISTED_VIOLATION = "UNLISTED_CHANNEL"

# -- bounds -----------------------------------------------------------------

#: The connector's own timeout, and so the ceiling on a hung read.
CONNECTOR_TIMEOUT_S = 120.0
#: Bound on a read this module expects to answer.
READ_TIMEOUT_S = 20.0
#: Bound on "spawned and answered its init frame" -- a cold pyepics import in an
#: emulated container host is not fast.
SPAWN_TIMEOUT_S = 60.0
#: Bound on the readiness probe a switch runs against a fresh child.
PROBE_TIMEOUT_S = 10.0
#: Bound on draining the child a switch is leaving behind. Nothing in this file
#: has a read in flight when it switches, so this is a teardown bound.
DRAIN_TIMEOUT_S = 5.0


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

    A precondition, not a nicety: the Channel Access *server* library reads
    ``EPICS_CAS_SERVER_PORT`` and does not fall back to the client-side
    variable, so an image whose entry point does not derive one from the other
    keeps binding its build-time default while telling this suite's clients some
    other port. The symptom would be an unexplained boot timeout; this turns it
    into a sentence naming the fix.
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


@contextlib.contextmanager
def _serving(prefix: str, *, bpm_errors: str | None):
    """Boot one virtual accelerator container and wait until it serves.

    The published port and the server's own port are the same number by
    construction, and that number also names the container.

    ``VA_LATTICE`` is stated on both boots rather than left to the image's
    default, so the two containers differ in exactly one variable — the seeded
    readout error — and so the fault path this module depends on is armed:
    ``VA_BPM_ERRORS`` is a lattice-physics fault and the entry point refuses it
    outright without the built-in lattice.
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
        "-e",
        "VA_LATTICE=builtin",
        "-p",
        f"127.0.0.1:{port}:{port}/tcp",
        "-v",
        f"{DATA_DIR}:/data/simulation:ro",
    ]
    if bpm_errors is not None:
        arguments += ["-e", f"VA_BPM_ERRORS={bpm_errors}"]
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
                f"{name} never served {PROBE_CHANNEL} within {BOOT_TIMEOUT_S}s.\n"
                f"{logs.stdout}\n{logs.stderr}"
            )
        yield port
    finally:
        _docker("rm", "-f", name, timeout=60)


@dataclass(frozen=True)
class Endpoints:
    """The two Channel Access ports this module's deployment is built from."""

    sandbox: int
    standin: int


@pytest.fixture(scope="module")
def endpoints():
    """Both instances, up and serving, for the life of this module.

    The stand-in carries the shipped perturbation and the sandbox carries none,
    which is what makes :data:`PERTURBED_BPM` distinguish them on the wire.
    """
    _require_image()
    with _serving(CONTAINER_SANDBOX, bpm_errors=None) as sandbox_port:
        with _serving(CONTAINER_STANDIN, bpm_errors=STANDIN_BPM_ERRORS_DEFAULT) as standin_port:
            yield Endpoints(sandbox=sandbox_port, standin=standin_port)


# ---------------------------------------------------------------------------
# The deployment
# ---------------------------------------------------------------------------


def raw_config(
    *,
    sandbox_port: int,
    standin_port: int,
    with_standin_service: bool = True,
    strict_limits: bool = True,
    project_root: Path | None = None,
) -> dict:
    """A stand-in deployment: the sandbox as ``va``, the soft IOC as ``standin``.

    ``control_system.type`` is ``virtual_accelerator``, so ``va`` is the
    deployment baseline, and ``standin`` resolves through the target table to
    the ``live_standin`` block — which dials the stand-in container. That is the
    shape the build produces for ``virtual_accelerator.live_standin``: a
    connector block of the stand-in's own, pointed at loopback on the port
    ``services.live_standin.port`` states.

    Neither connector block describes the facility's own machine, so ``live``
    stays underivable here — deliberately, since a stand-in must never be the
    answer to "where is the real machine". The roster assertions below are about
    a deployment with a ``standin`` slot and no ``live`` one.

    The FR-8 posture that applies to the stand-in is set: strict limits against
    the *shipped* limits database, because the whole point of a stand-in is that
    it is judged on the terms the real machine would be judged on. The operator
    acknowledgment is deliberately **absent** — it is the live machine's alone,
    and its absence here is what the eligibility assertion below reads.

    Writes are armed on the stand-in's block and nowhere else, which is what
    makes the write leg a test of the limits posture rather than of the
    ``writes_enabled`` guard that would otherwise refuse first.

    Args:
        sandbox_port: The sandbox instance's Channel Access port.
        standin_port: The stand-in instance's Channel Access port.
        with_standin_service: When false, the ``services.live_standin`` block is
            omitted and *everything else* is left identical — including the
            ``deployed_services`` entry, so the one conjunct the predicate
            actually reads is the only thing that moved. That is the negative
            control for both the label and the switch gate, since that block is
            the whole evidence the deployment stood a stand-in up.
        strict_limits: When false, the limits posture is loosened
            (``allow_unlisted_channels`` true) and nothing else moves — the
            negative control for the FR-8 gate the stand-in shares with the live
            machine.
        project_root: Written through when given, for children that resolve
            deployment-relative paths.
    """

    def block(port: int, *, writes_enabled: bool | None = None) -> dict:
        gateway_block: dict = {
            "timeout": CONNECTOR_TIMEOUT_S,
            "probe_channel": PROBE_CHANNEL,
            "gateways": {
                "read_only": {"address": "localhost", "port": port, "use_name_server": True},
                "write_access": {"address": "localhost", "port": port, "use_name_server": True},
            },
        }
        if writes_enabled is not None:
            gateway_block["writes_enabled"] = writes_enabled
        return gateway_block

    # ``path`` is carried because the build's service injector writes it: the
    # stand-in is a second INSTANCE of the virtual accelerator service, so both
    # keys name the same template directory. Nothing here reads it — the
    # fixture carries it so the scratch config is the shape a render produces.
    services: dict = {
        "virtual_accelerator": {"path": "./services/virtual_accelerator", "port": sandbox_port}
    }
    if with_standin_service:
        services["live_standin"] = {
            "path": "./services/virtual_accelerator",
            "port": standin_port,
        }

    config: dict = {
        "control_system": {
            "type": "virtual_accelerator",
            # The deployment-wide posture stays unarmed. The stand-in's own
            # block arms it, which is the per-type posture doing exactly what it
            # is for: one machine armed, the sandbox beside it left alone.
            "writes_enabled": False,
            "limits_checking": {
                "enabled": True,
                "allow_unlisted_channels": not strict_limits,
                "database_path": str(e2e_conftest.LIMITS_DB_PATH),
            },
            "connector": {
                "live_standin": block(standin_port, writes_enabled=True),
                "virtual_accelerator": block(sandbox_port),
            },
        },
        "services": services,
        # Both instance keys, in both configs. The predicate deliberately reads
        # no ``deployed_services`` conjunct (a persona render carries only the
        # keys its reach contract projects), so leaving this list alone in the
        # negative control keeps ``services.live_standin`` the single variable.
        "deployed_services": ["virtual_accelerator", "live_standin"],
        # Not the mock: pointing a session at a machine the deployment stands up
        # for itself while the archiver synthesises history is the pairing
        # eligibility refuses. Nothing in this module builds an archiver
        # connector.
        "archiver": {"type": "mongodb_archiver"},
        "agent_data": {"base_dir": "var/agent_data"},
    }
    if project_root is not None:
        config["project_root"] = str(project_root)
    return config


@pytest.fixture
def deployment(endpoints) -> dict:
    """The rendered config a stand-in deployment would hand its readers."""
    return raw_config(sandbox_port=endpoints.sandbox, standin_port=endpoints.standin)


@pytest.fixture(scope="module", autouse=True)
def module_environment(tmp_path_factory):
    """One isolated deployment environment for the whole module.

    Module-scoped, and autouse, because the switch below is module-scoped too:
    a function-scoped patch would be torn down and rebuilt around a session that
    outlives it, and the child processes would be left resolving whatever the
    ambient environment says.

    Three things are isolated. The target-state directory is anchored under a
    temporary root rather than a real deployment's ``var/agent_data``.
    ``PYTHONPATH`` is set explicitly rather than inherited: the interpreter
    running these tests belongs to another checkout's virtualenv, and a
    connector-host child that resolved ``osprey`` there would be a child of a
    different repository. And the three ambient config/posture variables are
    dropped, so nothing here reads a config.yml or a run posture this module did
    not state — the last of which is load-bearing for the write leg, since a
    read-only run would refuse the write for a reason that is not the posture
    under test.
    """
    root = tmp_path_factory.mktemp("standin-state") / "var" / "agent_data"
    (root / target_state.STATE_DIR_NAME).mkdir(parents=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(target_state, "resolve_shared_data_root", lambda: root)
        patch.setenv("PYTHONPATH", os.pathsep.join(REPO_PATHS))
        patch.delenv("CONFIG_FILE", raising=False)
        patch.delenv("OSPREY_CONFIG", raising=False)
        patch.delenv("OSPREY_EXECUTION_MODE", raising=False)
        yield root


async def reading(manager: ConnectorHostManager, address: str) -> float:
    """One value off the wire, through whichever child is serving right now."""
    value = await manager.active_proxy().read_channel(address, timeout=READ_TIMEOUT_S)
    assert isinstance(value, ChannelValue)
    return value.value


@dataclass(frozen=True)
class Refusal:
    """A write refusal, flattened to the fields that crossed the IPC boundary.

    Plain strings and plain numbers for the same reason :class:`RoundTrip` is:
    it travels out of a module-scoped async fixture and may carry nothing
    awaitable. The exception itself is not carried either — only what it said,
    which is all any assertion below reads.
    """

    kind: str
    channel_address: str
    attempted_value: object
    violation_type: str
    message: str


async def refused_write(manager: ConnectorHostManager, address: str, value: float) -> Refusal:
    """Attempt a write through the active child and record how it was refused.

    A refusal is the expected outcome, so an accepted write is turned into a
    failure *here* rather than reported as a missing exception three assertions
    later: this is the moment at which the machine would have moved, and the
    message should say so.
    """
    try:
        result = await manager.active_proxy().write_channel(address, value, timeout=READ_TIMEOUT_S)
    except ChannelLimitsViolationError as exc:
        return Refusal(
            kind=type(exc).__name__,
            channel_address=str(getattr(exc, "channel_address", "")),
            attempted_value=getattr(exc, "attempted_value", None),
            violation_type=str(getattr(exc, "violation_type", "")),
            message=str(exc),
        )
    raise AssertionError(
        f"the strict limits posture accepted a write of {value} to {address!r} on the "
        f"stand-in, which the shipped limits database does not list "
        f"(result: {result!r})"
    )


# ---------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundTrip:
    """What this module measured at both ends of a real switch, and the switch itself.

    Plain numbers and plain mappings, and that is a constraint rather than a
    convenience: this crosses an event-loop boundary (see ``round_trip``), so
    nothing awaitable — no proxy, no manager, no live child — may travel out
    through it.
    """

    sandbox: float
    standin: float
    control_sandbox: float
    control_standin: float
    outbound: dict
    homebound: dict
    refusal: Refusal


@pytest.fixture(scope="module")
async def round_trip(tmp_path_factory, endpoints) -> RoundTrip:
    """Read both BPMs on both targets and attempt one write, across a real switch.

    Module-scoped because the switch is the expensive part and every assertion
    below is about the same four numbers and the same refusal: spawning a fresh
    connector-host child per assertion would re-measure the same machine at
    several times the cost. The only write attempted is the one the posture
    refuses before any ``caput`` is issued, so the readings are of a machine at
    rest in both directions — which is what lets the homebound leg double as the
    check that the switch, and not something else, is what moved the value.

    **It runs on its own event loop.** A module-scoped async fixture is driven
    by a module-scoped loop, while the tests that consume it are function-scoped
    and each get their own — so the manager, its children and every awaitable
    they own belong to a loop no test may await on. That is why :class:`RoundTrip`
    carries only numbers, plain mappings and a flattened :class:`Refusal`, why
    the tests below are plain ``def``, and why the manager is shut down inside
    this fixture rather than left for a test to close.

    The deployment is staged on disk because a connector-host child is handed a
    config *path*: the section travels on the wire, but the write posture and
    the limits policy the connector applies are read from the file itself.
    """
    tmp = tmp_path_factory.mktemp("standin-round-trip")
    raw = raw_config(
        sandbox_port=endpoints.sandbox, standin_port=endpoints.standin, project_root=tmp
    )
    config_path = tmp / "config.yml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    manager = ConnectorHostManager(
        MCPServerConfig(raw=raw, config_path=config_path),
        drain_timeout_s=DRAIN_TIMEOUT_S,
        probe_timeout_s=PROBE_TIMEOUT_S,
        spawn_timeout_s=SPAWN_TIMEOUT_S,
        terminate_grace_s=2.0,
    )
    manager.reset_state()
    try:
        await manager.start(TARGET_VA)
        sandbox = await reading(manager, PERTURBED_BPM)
        control_sandbox = await reading(manager, CONTROL_BPM)

        outbound = await manager.switch(TARGET_STANDIN)
        standin = await reading(manager, PERTURBED_BPM)
        control_standin = await reading(manager, CONTROL_BPM)
        refusal = await refused_write(manager, UNLISTED_CHANNEL, UNLISTED_WRITE_VALUE)

        homebound = await manager.switch(TARGET_VA)
        yield RoundTrip(
            sandbox=sandbox,
            standin=standin,
            control_sandbox=control_sandbox,
            control_standin=control_standin,
            outbound=outbound,
            homebound=homebound,
            refusal=refusal,
        )
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(manager.shutdown(), 60)


# ---------------------------------------------------------------------------
# 1. What the operator is told the stand-in target is
# ---------------------------------------------------------------------------


class TestTheRosterNamesTheStandIn:
    """The roster, read off the surface the operator actually asks.

    :func:`~osprey.mcp_server.control_system.tools.control_target.target_rows`
    is the roster's own row builder and is pure — no process, no socket, no
    write — so these assertions are about the deployment's rendered config and
    nothing else. The ports in that config are this run's real container ports,
    which is what makes the endpoint assertion a statement about the machine
    the session would actually dial.
    """

    def test_the_standin_row_is_labelled_as_the_stand_in(self, deployment) -> None:
        rows = target_rows(deployment, session_target=TARGET_VA, baseline=TARGET_VA)

        assert rows[TARGET_STANDIN]["label"] == "LIVE MACHINE (stand-in)"

    def test_the_stand_in_is_still_the_real_machine(self, deployment) -> None:
        """The parenthesis is the *whole* of what is said differently.

        A stand-in that reported ``real_machine: false`` would be a rehearsal of
        the wrong ritual: every strict limit, approval prompt and banner an
        operator meets on a real machine is gated on this flag.
        """
        rows = target_rows(deployment, session_target=TARGET_VA, baseline=TARGET_VA)

        assert rows[TARGET_STANDIN]["real_machine"] is True
        assert rows[TARGET_VA]["real_machine"] is False

    def test_standing_a_stand_in_up_invents_no_live_machine(self, deployment) -> None:
        """The third target is a third target, not a renamed ``live``.

        Neither connector block in this deployment describes the facility's own
        machine, so there is no ``live`` row to offer — and offering one anyway,
        answered by the stand-in, is the failure this whole target exists to
        prevent. The display metadata still carries a ``live`` slot, because its
        readers render one per name; what it carries is the truthful
        "not configured", not the stand-in wearing the facility's name.
        """
        rows = target_rows(deployment, session_target=TARGET_VA, baseline=TARGET_VA)

        assert sorted(rows) == sorted([TARGET_STANDIN, TARGET_VA])
        assert TARGET_LIVE not in rows

        metadata = target_display_metadata(deployment)
        assert metadata[TARGET_LIVE]["label"] == "live machine (not configured)"
        assert metadata[TARGET_LIVE]["endpoint"] == ""
        assert metadata[TARGET_LIVE]["real_machine"] is False

    def test_the_standin_row_names_the_stand_ins_endpoint(self, deployment, endpoints) -> None:
        metadata = target_display_metadata(deployment)

        assert metadata[TARGET_STANDIN]["endpoint"] == f"localhost:{endpoints.standin}"
        assert metadata[TARGET_VA]["endpoint"] == f"localhost:{endpoints.sandbox}"

    def test_the_standin_target_is_available_now(self, deployment) -> None:
        """A stand-in nobody may switch to rehearses nothing."""
        rows = target_rows(deployment, session_target=TARGET_VA, baseline=TARGET_VA)

        assert rows[TARGET_STANDIN]["available_now"] is True
        assert rows[TARGET_STANDIN]["reason"] is None
        assert rows[TARGET_STANDIN]["connector_type"] == LIVE_STANDIN

    def test_the_sandbox_is_the_baseline_the_session_stands_on(self, deployment) -> None:
        rows = target_rows(deployment, session_target=TARGET_VA, baseline=TARGET_VA)

        assert rows[TARGET_VA]["is_baseline"] is True
        assert rows[TARGET_VA]["active"] is True
        assert rows[TARGET_STANDIN]["is_baseline"] is False

    def test_eligibility_refuses_nothing_about_the_stand_in(self, deployment) -> None:
        """The FR-8 gates that apply are met rather than bypassed.

        Switching *toward* the stand-in requires the strict limits posture, and
        the stand-in is not exempt from it — this deployment satisfies it, which
        is why the row above says available. A refusal here would name it.
        """
        verdict = evaluate_eligibility(deployment, TARGET_STANDIN, direction=DIRECTION_AWAY)

        assert verdict.eligible is True
        assert verdict.reason is None

    def test_the_stand_in_asks_for_no_operator_acknowledgment(self, deployment) -> None:
        """The acknowledgment is the live machine's alone.

        It is the operator saying the configured gateways really are this
        facility's, and the stand-in's equivalent was said at build time by the
        profile line that stood it up. This deployment sets the key nowhere —
        asserted, so the eligibility above cannot be passing on one that crept
        into the fixture — and is still eligible for the switch.
        """
        section = deployment["control_system"]
        assert ACK_LEAF not in section.get("target_switch", {})

        assert evaluate_eligibility(deployment, TARGET_STANDIN, direction=DIRECTION_AWAY).eligible

    def test_the_stand_in_does_ask_for_the_strict_limits_posture(self, endpoints) -> None:
        """The negative control for the gate the stand-in *does* share.

        A rehearsal on a permissive posture rehearses the wrong facility, so the
        stand-in is judged on the same limits posture the live machine is. One
        variable moves — ``allow_unlisted_channels`` — and the switch is refused
        by name.
        """
        loose = raw_config(
            sandbox_port=endpoints.sandbox,
            standin_port=endpoints.standin,
            strict_limits=False,
        )

        verdict = evaluate_eligibility(loose, TARGET_STANDIN, direction=DIRECTION_AWAY)

        assert verdict.eligible is False
        assert verdict.reason == REASON_LIMITS_POSTURE

    def test_without_the_services_block_the_same_endpoint_is_just_a_live_machine(
        self, endpoints
    ) -> None:
        """The negative control for the parenthesis, and the SSH-tunnel case.

        Identical gateways, identical loopback host, identical port, identical
        ``deployed_services`` — and no ``services.live_standin`` block, which is
        the single conjunct the predicate reads. A deployment that forwards a
        real gateway to loopback looks exactly like this, and the honest answer
        is ``LIVE MACHINE``: the operator would be one hop from hardware. So the
        label loses its parenthesis *and* the switch is refused, because a block
        that is not the deployment's own stand-in must not be reachable under a
        soft label.
        """
        plain = raw_config(
            sandbox_port=endpoints.sandbox,
            standin_port=endpoints.standin,
            with_standin_service=False,
        )

        rows = target_rows(plain, session_target=TARGET_VA, baseline=TARGET_VA)

        assert rows[TARGET_STANDIN]["label"] == "LIVE MACHINE"
        assert rows[TARGET_STANDIN]["real_machine"] is True
        assert rows[TARGET_STANDIN]["available_now"] is False
        assert rows[TARGET_STANDIN]["reason"] == REASON_STANDIN_NOT_DEPLOYED


# ---------------------------------------------------------------------------
# 2. The two targets are different machines on the wire
# ---------------------------------------------------------------------------


class TestTheStandInIsADifferentMachine:
    """The same channel, read at both ends of a real switch.

    Every number here came back through a connector-host child over Channel
    Access — the session really moved, and the read on the stand-in is a
    post-switch read rather than a direct query of a container this file
    happens to know the port of.

    The tests are plain ``def``: all the awaiting happened in ``round_trip``, on
    its own event loop, and there is nothing here left to await.
    """

    def test_the_perturbed_bpm_reads_differently_on_each_target(
        self, round_trip: RoundTrip
    ) -> None:
        assert round_trip.standin != round_trip.sandbox, (
            "both targets served the same value for the perturbed BPM, so the "
            "stand-in is indistinguishable from the machine it stands in for"
        )
        # Not asserted to a literal: what is asserted is that the difference is
        # the seeded offset's order and not float noise.
        assert abs(round_trip.standin - round_trip.sandbox) >= abs(PERTURBED_OFFSET_X) / 2

    def test_an_unperturbed_bpm_agrees_across_both_targets(self, round_trip: RoundTrip) -> None:
        """The control, and what makes the divergence above attributable.

        The shipped table names no error for this device, so both instances read
        it through the identity transform. Two independently booted containers
        that had drifted apart would move this channel too, and the divergence
        above would be weather rather than the stand-in's perturbation.

        Bounded by the same half-offset the positive assertion uses, rather than
        compared for bit equality. Agreement to the last bit is not what makes
        the comparison sound — agreement well below the offset's own order is —
        and two closed-orbit solves in two containers may legitimately reduce in
        a different order. An exact-equality control would eventually fail for a
        reason that says nothing about the stand-in.
        """
        assert (
            abs(round_trip.control_standin - round_trip.control_sandbox)
            < abs(PERTURBED_OFFSET_X) / 2
        )

    def test_the_switch_landed_on_the_stand_ins_own_port(
        self, round_trip: RoundTrip, endpoints
    ) -> None:
        assert round_trip.outbound["target"] == TARGET_STANDIN
        assert round_trip.outbound["connector_type"] == LIVE_STANDIN
        assert round_trip.outbound["endpoint"]["port"] == endpoints.standin

    def test_the_session_can_come_home_to_the_sandbox(
        self, round_trip: RoundTrip, endpoints
    ) -> None:
        assert round_trip.homebound["target"] == TARGET_VA
        assert round_trip.homebound["connector_type"] == "virtual_accelerator"
        assert round_trip.homebound["endpoint"]["port"] == endpoints.sandbox


# ---------------------------------------------------------------------------
# 3. A write on the stand-in is judged on hardware's terms
# ---------------------------------------------------------------------------


class TestTheStrictPostureRefusesAnUnlistedWrite:
    """The write leg, taken on a session that really is pointed at the stand-in.

    The attempt crossed the same connector-host boundary the reads did, was
    refused inside the child, and the refusal was rebuilt in this process from
    its own fields — so what is asserted below is what the stand-in's connector
    actually said, not a message a test helper composed.
    """

    def test_the_unlisted_write_was_refused_by_the_limits_database(
        self, round_trip: RoundTrip
    ) -> None:
        assert round_trip.refusal.kind == ChannelLimitsViolationError.__name__
        assert round_trip.refusal.violation_type == UNLISTED_VIOLATION
        assert round_trip.refusal.channel_address == UNLISTED_CHANNEL
        assert round_trip.refusal.attempted_value == UNLISTED_WRITE_VALUE

    def test_the_refusal_names_the_database_and_not_the_network(
        self, round_trip: RoundTrip
    ) -> None:
        """What separates this from every other way a write can fail.

        ``UNLISTED_CHANNEL`` is a verdict about the shipped database: it is
        reached only after that database loaded and was searched. A database
        that could not be read refuses every channel with
        ``LIMITS_DATABASE_UNAVAILABLE`` instead, and an unreachable endpoint
        never reaches the validator at all — so the exact type is the assertion,
        and the message is required to name the channel it is about rather than
        the machine.
        """
        assert round_trip.refusal.violation_type != "LIMITS_DATABASE_UNAVAILABLE"
        assert UNLISTED_CHANNEL in round_trip.refusal.message
        assert "limits database" in round_trip.refusal.message.lower()

    def test_the_refusal_was_reached_with_the_write_gateway_armed(
        self, round_trip: RoundTrip, deployment
    ) -> None:
        """The refusal is the posture's, not the ``writes_enabled`` guard's.

        With writes unarmed the connector refuses every write before the limits
        validator ever runs, and this class would be measuring a gate it is not
        about. Two things say it was armed: the switch selected the
        ``write_access`` gateway for the stand-in, and the roster reports the
        stand-in's writes as permitted while the sandbox beside it — which the
        deployment-wide posture leaves unarmed — reports them as not.
        """
        assert round_trip.outbound["selected_role"] == "write_access"

        rows = target_rows(deployment, session_target=TARGET_STANDIN, baseline=TARGET_VA)
        assert rows[TARGET_STANDIN]["writes_permitted"] is True
        assert rows[TARGET_VA]["writes_permitted"] is False


# ---------------------------------------------------------------------------
# 4. The shipped defaults this module measured are the ones that ship
# ---------------------------------------------------------------------------


class TestTheShippedDefaultsAreWhatThisModuleMeasured:
    """Guards on the constants the classes above are built from.

    All three are cheap and none touches a container: they exist so that a
    change to a shipped default cannot leave this module measuring an offset
    nobody deploys, comparing a "control" the default has since started
    perturbing, or attempting a write the limits database has since started
    allowing for a reason that has nothing to do with the posture.
    """

    def test_the_device_this_module_reads_carries_a_horizontal_offset(self) -> None:
        assert PERTURBED_OFFSET_X != 0.0
        assert set(STANDIN_OFFSETS[PERTURBED_DEVICE]) <= {"offset_x", "offset_y"}, (
            "the shipped stand-in default grew a non-offset field, which would "
            "make a seeded reading depend on the unperturbed value and this "
            "module's deterministic comparison unsound"
        )

    def test_the_control_device_is_untouched_by_the_shipped_default(self) -> None:
        assert CONTROL_DEVICE not in STANDIN_OFFSETS

    def test_the_write_leg_attempts_a_channel_the_shipped_database_omits(self) -> None:
        """The write leg's subject, checked against the database it is judged by.

        Read from the same file the deployment points ``database_path`` at. The
        listed sibling is the other half: a database this module could not read,
        or one whose grammar had moved on, would fail here rather than turn the
        refusal above into a verdict about nothing.
        """
        database = json.loads(e2e_conftest.LIMITS_DB_PATH.read_text(encoding="utf-8"))

        assert UNLISTED_CHANNEL not in database
        assert LISTED_CHANNEL in database


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_live_standin.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
