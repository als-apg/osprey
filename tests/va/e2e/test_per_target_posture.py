"""Per-target write posture, enforced against two real machines on the wire.

The header chip's promise is that an operator can take ONE control target away
from a session that is already running — "the stand-in is read-only for me,
leave the virtual accelerator alone" — and have the very next write obey it,
without the session being respawned and without the other target losing
anything. Nothing about that claim is decidable inside one process: the
operator's gesture lands in the web server, the store it writes is read by a
connector living in another process, and the machine that would have moved is a
soft IOC on the far side of Channel Access. So this module boots two real
``osprey-va-full`` containers, stands the real POST routes up over the same
agent-data root the connector-host children read, and drives the whole thing
end to end.

Five claims, and this module is the acceptance gate for all five:

1. **A narrowing lands on a live child.** With the session pointed at the
   stand-in and its writes armed, ``POST /api/terminal/posture`` with
   ``sandbox`` makes ``channel_write`` and a readwrite ``execute`` refuse —
   naming the session posture and the chip, in each connector result's
   ``error`` *and* in the raised all-refused envelope — while the same
   deployment still writes on the virtual accelerator, and while the
   connector-host child that serves the session keeps its pid. Enforcement is
   at write time; the route that used to terminate the child to re-stamp an
   environment variable is gone.
2. **The realignment waits for a run, then moves the gateway.** With an
   execution marker live, the reconciler publishes ``last_posture_realign:
   pending`` and rebuilds nothing. Once the run ends it rebuilds the child, the
   note becomes ``done``, and both the child's own post-connect report and the
   ``control_target`` roster say ``selected_role: read_only`` for the stand-in
   — and a subsequent switch to it still verifies rather than raising.
3. **A switch asked for by the operator is a file, a poll and an outcome.**
   ``POST /api/terminal/target`` writes a request addressed to this server's
   pid; the reconciler consumes it, switches, and publishes ``last_switch``
   with the request's own id. A second request for the target the session is
   already on comes back ``already_active`` with no child respawned.
4. **A write follows the session, not the conversation.** An agent whose
   session is on the virtual accelerator writes; the operator moves it to the
   stand-in from the web surface; the agent's next ``channel_write`` is judged
   against the stand-in and refused, with nothing on either machine having
   moved.
5. **A widening never reaches a run in flight.** A sandbox launched under
   ``OSPREY_LAUNCH_POSTURE=<target>=sandbox`` keeps refusing after the operator
   widens the store, and a sandbox launched after the widen writes.

Three targets, two containers
-----------------------------
The deployment configures the facility's own machine, the virtual accelerator
and the stand-in; the last two have a container behind them and are the only
ones anything here dials. The facility machine is configured and never touched,
because ``switch_capable`` — the predicate deciding whether the controls server
serves its connector from a child at all — is defined over ``live`` and ``va``
together; :func:`raw_config` says the rest.

Container V backs the ``va`` target and container S the ``standin`` target.
They are two so that "the virtual accelerator still writes while the stand-in
is read-only" is a statement about two machines rather than about one machine
addressed twice, and so that every switch below is a switch between real
endpoints — a fresh connector-host child, a fresh generation, a real Channel
Access connection. Neither carries a seeded readout error: telling the two
machines apart by their *readings* is ``test_live_standin.py``'s subject, and
this module tells them apart by which one a write reaches.

One root, stamped
-----------------
``OSPREY_AGENT_DATA_ROOT`` is set once, to a throwaway directory, and every
participant resolves the store and the state file through it: this process (the
controls server), the web server built over the same environment, each
connector-host child (which inherits it), and the executor sandbox subprocess.
That is the production shape — the stamp is what a session child carries — and
it is also the only way a test can be sure the POST wrote the file the
connector reads. ``OSPREY_POSTURE_SESSION`` is stamped beside it, always as a
pair, for the same reason: a child with one anchor and not the other reads a
store nobody writes.

What is *not* covered here is the browser half — the chip, its popover and its
refetch — which is the Playwright lane's subject, and the per-target GET, whose
contract test lives with the routes. This module reads outcomes from the state
file the controls server publishes, which is the same record that GET renders.

Every Channel Access operation in this file happens in **another process**: in
the readiness probe's subprocess, in a connector-host child, or in the executor
sandbox. This process never becomes a CA client, which is the rule
``conftest.py`` states — libca latches ``EPICS_CA_*`` on initialisation and its
contexts are per-thread, so a main-thread pyepics call here would deadlock the
very children under test.

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
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.audit import posture as audit_posture
from osprey.audit import writer as audit_writer
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.routes import websocket as websocket_routes
from osprey.mcp_server.control_system import server_context as server_context_mod
from osprey.mcp_server.control_system import session_control, target_state
from osprey.mcp_server.control_system.connector_host_manager import ConnectorHostManager
from osprey.mcp_server.control_system.server_context import MCPServerConfig
from osprey.mcp_server.control_system.tools import channel_write as channel_write_module
from osprey.mcp_server.control_system.tools import control_target as control_target_module
from osprey.mcp_server.python_executor import executor as host_executor
from osprey.mcp_server.python_executor.tools import _execution_gates as execution_gates
from osprey.mcp_server.python_executor.tools import python_execute as python_execute_module
from osprey_connectors import session_store
from osprey_connectors.control_system.base import ChannelValue
from osprey_connectors.types import TARGET_LIVE, TARGET_STANDIN, TARGET_VA
from tests.fixtures.control_context import context_for
from tests.mcp_server.conftest import get_tool_fn
from tests.va.e2e import conftest as e2e_conftest

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_PATHS = (str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src"))

#: The image under test, and the simulation data both containers serve.
IMAGE = os.environ.get("OSPREY_VA_E2E_IMAGE", "osprey-va-full:latest")
DATA_DIR = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation"

#: Container-name prefixes; ``_serving`` appends the run's own ephemeral port.
#: Named for the *target* each instance backs, since that is what the deployment
#: below calls them and what every assertion is phrased in.
CONTAINER_VA = "osprey-va-e2e-posture-va"
CONTAINER_STANDIN = "osprey-va-e2e-posture-standin"

#: Boot is generous on purpose: the image is pinned ``linux/amd64`` and a local
#: run on Apple Silicon is emulated (see ``conftest.py``). Two containers.
BOOT_TIMEOUT_S = 180.0

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 25

# -- the namespace both containers serve ------------------------------------

#: What each target's switch reads to prove itself reachable, and what the
#: readiness probe below waits for. A pyat-coupled corrector readback, served
#: identically by both instances and never written by anything in this file.
PROBE_CHANNEL = "SR:MAG:HCM:01:CURRENT:RB"

#: The setpoint every write leg in this module attempts. Listed in the shipped
#: limits database (``[-12, 12]``), which is what lets this module keep the
#: strict limits posture the stand-in's switch gate requires and still have a
#: write that reaches the machine: an unlisted channel would be refused by the
#: limits validator, which is a different gate from the one under test.
CORRECTOR_SP = "SR:MAG:HCM:01:CURRENT:SP"

#: The gateway port the facility's own machine is described on. Nothing serves
#: it and nothing in this module dials it — see :func:`raw_config` for why the
#: block exists at all. Deliberately not an ephemeral port: a number nothing
#: answers on is the honest description of a machine no test may reach.
UNSERVED_PORT = 5164

#: The value a permitted write puts on the machine, and the one a refused write
#: must never put there. Distinct so a read-back names which write it saw.
ARMED_VALUE = 1.0
REFUSED_VALUE = 5.0
LAUNCH_REFUSED_VALUE = 3.0
LAUNCH_PERMITTED_VALUE = 4.0
#: What the machine holds when nothing this module attempted has landed.
REST_VALUE = 0.0

# -- the session the operator is acting on ----------------------------------

#: The posture-store key. A bare lowercase UUID, which is the closed grammar
#: both posture routes accept and the shape a PTY pool key really has.
SESSION_ID = "5a17e600-1111-4222-8333-444444444444"

# -- bounds -----------------------------------------------------------------

#: The connector's own timeout, and so the ceiling on a hung read.
CONNECTOR_TIMEOUT_S = 120.0
#: Bound on a read or a write this module expects to answer.
IO_TIMEOUT_S = 20.0
#: Bound on "spawned and answered its init frame" -- a cold pyepics import in an
#: emulated container host is not fast.
SPAWN_TIMEOUT_S = 60.0
#: Bound on the readiness probe a switch runs against a fresh child.
PROBE_TIMEOUT_S = 10.0
#: Bound on draining the child a switch is leaving behind.
DRAIN_TIMEOUT_S = 5.0
#: Bound on the executor sandbox subprocess, which imports the whole registry.
SANDBOX_TIMEOUT_S = 300.0


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
#
# Copied from ``test_live_standin.py`` rather than imported from it, as that
# module copies them from ``test_target_switch.py``: a test module defines
# fixtures and is not an importable helper library. The copies are small, and
# each suite's ``_serving`` differs in what it boots.


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
def _serving(prefix: str):
    """Boot one virtual accelerator container and wait until it serves.

    The published port and the server's own port are the same number by
    construction — a Channel Access search reply carries the server's own port,
    so a remap would hand every client an address nothing answers on — and that
    number also names the container, which is what keeps two concurrent runs
    from destroying each other. Nothing here goes near 5064.

    ``VA_LATTICE`` is stated rather than left to the image's default, so both
    boots differ in nothing at all: this module's subject is which machine a
    write reaches, and two instances that were not identical would leave a
    reader wondering whether something else told them apart.
    """
    port = _free_port()
    name = f"{prefix}-{port}"
    # Stale-cleanup only. The port is this run's alone, so this can name nothing
    # a concurrent run is using -- which is the point of the suffix.
    _docker("rm", "-f", name, timeout=60)

    started = _docker(
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
        IMAGE,
    )
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


@pytest.fixture(scope="module")
def endpoints():
    """Both instances, up and serving, for the life of this module."""
    _require_image()
    with _serving(CONTAINER_VA) as va_port, _serving(CONTAINER_STANDIN) as standin_port:
        yield {TARGET_VA: va_port, TARGET_STANDIN: standin_port}


# ---------------------------------------------------------------------------
# The deployment
# ---------------------------------------------------------------------------


def raw_config(*, va_port: int, standin_port: int, project_root: Path) -> dict:
    """A three-target deployment with writes armed on the two this module drives.

    ``control_system.type`` is ``virtual_accelerator``, so ``va`` is the
    baseline and ``standin`` resolves through the target table to the
    ``live_standin`` block. Those two are the ones with a container behind them
    and the ones every leg below writes to.

    **The facility's own machine is configured and never touched**, and that is
    a requirement rather than scenery: ``switch_capable`` — the predicate that
    decides whether the controls server serves its connector from a child at all,
    and the predicate ``session_posture`` reads before it will answer a ceiling
    per target — is defined over ``live`` and ``va`` together
    (``osprey_connectors.types.switch_capable``). A deployment carrying only
    ``virtual_accelerator`` and ``live_standin`` blocks answers ``False``, its
    tools take the in-process connector, and every per-target ceiling but the
    baseline's reads unarmed. So the ``epics`` block is here to put this module
    in the two-target world the chip is specified over. Its writes stay unarmed
    (it inherits the deployment-wide ``false``) and nothing here ever switches
    to it — its gateways name a port nothing serves, which is the truth about a
    facility machine no test may reach.

    **Both driven blocks arm writes**, which is what makes every refusal below
    attributable to the session posture: with the stand-in unarmed the
    deployment ceiling would refuse first and this module would be measuring the
    wrong gate, and with the virtual accelerator unarmed "the other target still
    writes" would be untestable. The deployment-wide key stays off so that the
    per-type posture is doing the arming, exactly as a rendered profile does it.

    **Strict limits, on purpose.** ``allow_unlisted_channels: false`` is the
    FR-8 posture a switch *toward* the stand-in requires; loosening it would
    make the stand-in ineligible and every switch leg below refuse for a reason
    that has nothing to do with the posture store. Every write here therefore
    names a channel the shipped database lists.

    The operator acknowledgment is deliberately absent — it is the live
    machine's alone, and the stand-in's equivalent was said at build time by the
    profile line that stood it up.
    """

    def gateways(port: int) -> dict:
        row = {"address": "localhost", "port": port, "use_name_server": True}
        return {"read_only": dict(row), "write_access": dict(row)}

    def block(port: int) -> dict:
        return {
            "timeout": CONNECTOR_TIMEOUT_S,
            "probe_channel": PROBE_CHANNEL,
            "writes_enabled": True,
            "gateways": gateways(port),
        }

    return {
        "control_system": {
            "type": "virtual_accelerator",
            # Off deployment-wide: each driven block arms itself, which is the
            # per-type posture doing what it is for -- and it is what leaves the
            # facility's own machine unarmed without a word being said about it.
            "writes_enabled": False,
            "limits_checking": {
                "enabled": True,
                "allow_unlisted_channels": False,
                "database_path": str(e2e_conftest.LIMITS_DB_PATH),
            },
            "connector": {
                "epics": {"gateways": gateways(UNSERVED_PORT)},
                "live_standin": block(standin_port),
                "virtual_accelerator": block(va_port),
            },
        },
        # Both instance keys: ``services.live_standin`` is the single conjunct
        # that makes the ``standin`` slot the deployment's own stand-in rather
        # than an unreachable live machine wearing a soft label.
        "services": {
            "virtual_accelerator": {"path": "./services/virtual_accelerator", "port": va_port},
            "live_standin": {"path": "./services/virtual_accelerator", "port": standin_port},
        },
        "deployed_services": ["virtual_accelerator", "live_standin"],
        # Not the mock: pointing a session at a machine the deployment stands up
        # for itself while the archiver synthesises history is the pairing
        # eligibility refuses. Nothing in this module builds an archiver.
        "archiver": {"type": "mongodb_archiver"},
        "agent_data": {"base_dir": "var/agent_data"},
        "project_root": str(project_root),
    }


@pytest.fixture(scope="module", autouse=True)
def module_environment(tmp_path_factory, endpoints):
    """One isolated, STAMPED deployment environment for the whole module.

    Module-scoped and autouse because everything below outlives a function
    scope: the containers, the connector-host manager, the web app and the
    subprocesses they spawn.

    ``OSPREY_AGENT_DATA_ROOT`` is the load-bearing stamp. It is the single rule
    the store's writer and both its readers resolve by, it is what
    ``target_state.state_dir()`` prefers, and — being an environment variable —
    it is inherited by every child this module spawns without any of them being
    told about it. Patching ``resolve_shared_data_root`` instead would redirect
    one reader and leave the rest writing into the repository's own
    ``var/agent_data``; it is patched here *as well*, so that a resolver reached
    on a path the stamp does not cover still lands in the same place.
    ``OSPREY_POSTURE_SESSION`` is stamped beside it, always as a pair, because
    that is the pairing the feature guarantees.

    ``PYTHONPATH`` is set explicitly rather than inherited: the interpreter
    running these tests belongs to another checkout's virtualenv, and a child
    that resolved ``osprey`` there would be a child of a different repository.
    ``OSPREY_EXECUTION_MODE`` and the launch pin are dropped so nothing here
    reads a run posture this module did not state — load-bearing, since a
    read-only run would refuse every write for a reason that is not the posture
    under test.

    ``CONFIG_FILE`` is **stated rather than dropped**, which is where this
    module parts company with ``test_target_switch.py``. Both MCP servers a
    session runs are launched with the deployment's config path, and two things
    here read the ambient config rather than one handed to them: the executor's
    ``_target_is_resolvable``, which declines to stamp a target it cannot
    resolve — leaving a run unstamped and pinned everywhere, which would make
    the launch-pin leg measure the wrong pin — and the deployment writes gate.
    Pointing them at the deployment this module staged is stating the config,
    not inheriting one.

    The audit writer is stubbed for the module: several gates below file ledger
    records, and a test that has redirected the state root but not the ledger
    would be writing into whichever deployment this checkout happens to resolve.
    """
    root = tmp_path_factory.mktemp("per-target-posture") / "var" / "agent_data"
    (root / target_state.STATE_DIR_NAME).mkdir(parents=True)
    project = tmp_path_factory.mktemp("per-target-posture-project")
    written = project / "config.yml"
    written.write_text(
        yaml.safe_dump(
            raw_config(
                va_port=endpoints[TARGET_VA],
                standin_port=endpoints[TARGET_STANDIN],
                project_root=project,
            )
        ),
        encoding="utf-8",
    )
    records: list[dict] = []

    with pytest.MonkeyPatch.context() as patch_env:
        patch_env.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
        patch_env.setenv(audit_posture.POSTURE_SESSION_ENV_VAR, SESSION_ID)
        patch_env.setenv("PYTHONPATH", os.pathsep.join(REPO_PATHS))
        patch_env.setenv("CONFIG_FILE", str(written))
        patch_env.setenv("OSPREY_CONFIG", str(written))
        patch_env.delenv("OSPREY_EXECUTION_MODE", raising=False)
        patch_env.delenv(session_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
        patch_env.setattr(target_state, "resolve_shared_data_root", lambda: root)
        session_store.invalidate_cache()
        audit_posture.invalidate_session_target_cache()
        websocket_routes._reset_session_record_memo()
        with patch.object(audit_writer, "record", side_effect=lambda **f: records.append(f)):
            yield SimpleNamespace(root=root, config_path=written)
    session_store.invalidate_cache()
    audit_posture.invalidate_session_target_cache()
    websocket_routes._reset_session_record_memo()


@pytest.fixture(scope="module")
def config_path(module_environment) -> Path:
    """The deployment this module staged, as a path.

    A connector-host child is handed a config *path*: the section travels on the
    wire, but the write posture and the limits policy the connector applies are
    read from the file itself. The executor sandbox reads the same file, and so
    does this process (see :func:`module_environment`).
    """
    return module_environment.config_path


@pytest.fixture(scope="module")
def web(tmp_path_factory, config_path, module_environment):
    """The real web server, over the same stamped root, for the whole module.

    Two seams are patched and nothing else. The PTY registry is made to report
    this process's parent as the session's terminal, which is what lets the
    routes resolve the state record this process really publishes: the record's
    ``owner_ppid`` is ``os.getppid()``, and the resolver walks a record's
    ancestors — itself included — looking for the PTY pid. Session discovery is
    made to report the session as started, which on a real deployment is what
    having sent one prompt means. Everything past those two seams — the refusal
    ladder, the store write, the request file — is the shipped code.
    """
    watch_dir = tmp_path_factory.mktemp("per-target-posture-watch")
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(watch_dir)},
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as client:
            client.app.state.config_path = config_path
            registry = client.app.state.pty_registry
            with (
                patch.object(
                    registry,
                    "get_session",
                    side_effect=lambda sid: (
                        SimpleNamespace(pid=os.getppid()) if sid == SESSION_ID else None
                    ),
                ),
                patch(
                    "osprey.interfaces.web_terminal.session_discovery."
                    "SessionDiscovery.snapshot_session_ids",
                    return_value={SESSION_ID},
                ),
            ):
                yield client


def post_posture(client, target: str, posture: str):
    return client.post(
        "/api/terminal/posture",
        json={"session_id": SESSION_ID, "target": target, "posture": posture},
    )


def post_target(client, target: str):
    return client.post(
        "/api/terminal/target",
        json={"session_id": SESSION_ID, "target": target},
    )


# ---------------------------------------------------------------------------
# Talking to the machines
# ---------------------------------------------------------------------------


async def reading(manager: ConnectorHostManager, address: str) -> float:
    """One value off the wire, through whichever child is serving right now."""
    value = await manager.active_proxy().read_channel(address, timeout=IO_TIMEOUT_S)
    assert isinstance(value, ChannelValue)
    return value.value


async def write_rows(manager: ConnectorHostManager, address: str, value: float) -> list[dict]:
    """Write through the connector and flatten what each result reported.

    ``write_multiple_channels`` is the one write path that *returns* refusals
    instead of raising them, so this is where the per-result contract — the
    ``outcome`` word, the ``refusal_reason`` and the ``error_message`` an
    operator reads — can be observed as the connector produced it, in the child,
    on the far side of the IPC boundary. Flattened to plain strings because it
    travels out of a module-scoped fixture (see :func:`session`).
    """
    results = await manager.active_proxy().write_multiple_channels(
        [(address, value)], timeout=IO_TIMEOUT_S
    )
    return [
        {
            "channel": result.channel_address,
            "outcome": str(result.outcome),
            "refusal_reason": result.refusal_reason,
            "error": result.error_message or "",
        }
        for result in results
    ]


async def write_via_tool(address: str, value: float) -> dict:
    """Call ``channel_write`` the way the agent does, and flatten the answer.

    Returns ``{"raised": False, "payload": ...}`` for a call that returned, and
    ``{"raised": True, "envelope": ...}`` for one that raised the standard error
    envelope — the all-refused case, which is what a single refused write
    produces.
    """
    from fastmcp.exceptions import ToolError

    try:
        result = await get_tool_fn(channel_write_module.channel_write)(
            operations=[{"channel": address, "value": value}]
        )
    except ToolError as exc:
        return {"raised": True, "envelope": json.loads(str(exc))}
    return {"raised": False, "payload": json.loads(result)}


async def execute_readwrite() -> dict:
    """Call the ``execute`` tool in readwrite mode and flatten the answer.

    The code is inert on purpose: every gate this leg is about fires *before*
    the subprocess is launched, so a script that did anything would only add a
    way for the leg to fail for a reason that is not the posture.
    """
    from fastmcp.exceptions import ToolError

    try:
        result = await get_tool_fn(python_execute_module.execute)(
            code="value = 1\n",
            description="posture probe",
            execution_mode="readwrite",
            save_output=False,
        )
    except ToolError as exc:
        return {"raised": True, "envelope": json.loads(str(exc))}
    return {"raised": False, "payload": str(result)}


def gates_are_silent() -> bool:
    """Whether the two write gates ``execute`` runs would let a readwrite run by.

    The anti-vacuous control for the ``execute`` leg, asked of the very
    functions the tool calls — the session-posture clamp and the per-target
    writes gate — rather than by launching a sandbox. Running the tool for real
    on the permitted target would spawn the executor subprocess, and a failure
    in *that* (a workspace, a virtualenv, a timeout) would be reported here as a
    posture verdict, which is the one thing this control must not do.
    """
    from fastmcp.exceptions import ToolError

    try:
        execution_gates.enforce_posture_clamp("readwrite", tool="execute")
        execution_gates.enforce_deployment_writes_gate(
            "readwrite", execution_gates.session_control_target()
        )
    except ToolError:
        return False
    return True


# ---------------------------------------------------------------------------
# The executor sandbox
# ---------------------------------------------------------------------------

#: Sandbox-side agent code, run in a real subprocess carrying the real launch
#: stamp, because the launch pin is a contract between two processes and neither
#: can see the other's state directly.
#:
#: ``initialize_registry`` is the sandbox's own setup step, restated here rather
#: than skipped: it is what ``ExecutionWrapper.create_wrapper`` emits ahead of
#: every execution, and it is what populates the connector factory — a bare
#: process importing ``osprey.runtime`` has an empty one and would fail with
#: "Unknown control system type" long before reaching the pin.
#:
#: The verdict is written to a file rather than printed, because registry
#: initialisation is chatty on both streams and a result parsed out of that
#: noise would be a result this test could misread. The exit is abrupt for the
#: reason ``osprey_connectors.ipc.host`` states of its own: a process that has
#: held a Channel Access context can block forever in pyepics'
#: ``finalize_libca`` atexit hook, and a sandbox that will not die is worse than
#: one that skips its hooks.
_SANDBOX_WRITE = """
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
except Exception as exc:
    verdict = {"wrote": False, "error": type(exc).__name__, "message": str(exc)}
else:
    verdict = {"wrote": True, "error": None, "message": ""}
Path(verdict_path).write_text(json.dumps(verdict), encoding="utf-8")
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
"""


def launch_stamps() -> dict:
    """The environment stamps the executor would give a run launched right now.

    ``_apply_target_stamp`` is the production stamper and it *mutates* the
    environment mapping it is handed (returning the target it chose), so the
    dictionary is what carries the answer — including
    :data:`~osprey.mcp_server.python_executor.executor.ENV_LAUNCH_POSTURE`, the
    pin this module is about. Reading it through the real function rather than
    composing the variable by hand is what makes the two runs below a test of
    the pin and not of a string this file wrote.
    """
    stamps: dict[str, str] = {}
    host_executor._apply_target_stamp(stamps)
    return stamps


def run_sandbox(*, stamps: dict, config_path: Path, value: float, tag: str) -> dict:
    """Run one stamped sandbox and report what its write did.

    *stamps* is whatever ``executor._apply_target_stamp`` produced at the moment
    the run was launched — the production stamper, so what is under test is the
    real launch pin and not a hand-written environment variable that happens to
    look like one.

    ``OSPREY_AGENT_DATA_ROOT`` and ``OSPREY_POSTURE_SESSION`` are inherited from
    this process rather than restated, which is exactly how a real sandbox gets
    them.
    """
    verdict_path = config_path.parent / f"sandbox_verdict_{tag}.json"
    environment = {
        **os.environ,
        **stamps,
        "PYTHONPATH": os.pathsep.join(REPO_PATHS),
        "CONFIG_FILE": str(config_path),
        "OSPREY_CONFIG": str(config_path),
    }
    for stale in ("EPICS_CA_ADDR_LIST", "EPICS_CA_NAME_SERVERS", "EPICS_CA_SERVER_PORT"):
        environment.pop(stale, None)
    completed = subprocess.run(
        [sys.executable, "-c", _SANDBOX_WRITE, CORRECTOR_SP, str(value), str(verdict_path)],
        capture_output=True,
        text=True,
        timeout=SANDBOX_TIMEOUT_S,
        cwd=str(config_path.parent),
        env=environment,
    )
    if completed.returncode != 0 or not verdict_path.exists():
        raise RuntimeError(f"the stamped sandbox failed:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(verdict_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


@contextmanager
def quiet_notifications():
    """Silence the operator-facing switch emit on both surfaces.

    It is fire-and-forget over HTTP to a web terminal that is not listening on
    the port the emitter derives, and it is emitted by the tool and by the
    reconciler through two separately bound names.
    """
    recorded: list[dict] = []

    async def record(**kwargs):
        recorded.append(kwargs)

    with (
        patch.object(control_target_module, "notify_target_switch_async", record),
        patch.object(session_control, "notify_target_switch_async", record),
    ):
        yield recorded


def last_switch():
    """This server's published switch outcome, straight off the state file.

    Read from the record rather than through the per-target GET: the GET renders
    exactly this block (with an ``age_s`` it computes on the way out), and the
    state file is the contract the reconciler actually writes.
    """
    record = target_state.read() or {}
    return record.get("last_switch")


def realign_note():
    """This server's published realignment note, straight off the state file."""
    record = target_state.read() or {}
    return record.get("last_posture_realign")


@pytest.fixture(scope="module")
async def session(config_path, web, module_environment):
    """Drive the whole scripted session once, recording what each step saw.

    Module-scoped because every expensive part is shared: two containers, one
    connector-host manager, one web server and a handful of real switches. A
    per-test bring-up would re-measure the same contract at several times the
    cost, and several of the steps below only mean anything in sequence — a
    narrowing has to land on a child that was already serving, and a launch pin
    only pins if the widen happens after the launch.

    **It runs on its own event loop.** A module-scoped async fixture is driven
    by a module-scoped loop, while the tests that consume it are function-scoped
    and each get their own — so the manager, its children and every awaitable
    they own belong to a loop no test may await on. That is why the journal
    below carries only plain data, why the tests are plain ``def``, and why the
    manager is shut down inside this fixture rather than left for a test to
    close.

    The journal is a mapping rather than a frozen dataclass because it holds two
    dozen unrelated observations whose only structure is the order they were
    taken in; a dataclass would buy one more list to keep in step and nothing
    else.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manager = ConnectorHostManager(
        MCPServerConfig(raw=raw, config_path=config_path),
        drain_timeout_s=DRAIN_TIMEOUT_S,
        probe_timeout_s=PROBE_TIMEOUT_S,
        spawn_timeout_s=SPAWN_TIMEOUT_S,
        terminate_grace_s=2.0,
    )
    manager.reset_state()
    journal: dict = {}

    async def restore_machine() -> None:
        """Put the setpoint back to rest, whatever the current posture is."""
        with contextlib.suppress(Exception):
            await manager.active_proxy().write_channel(
                CORRECTOR_SP, REST_VALUE, timeout=IO_TIMEOUT_S
            )

    with quiet_notifications():
        try:
            await manager.start(TARGET_VA)
            server_context_mod._registry = context_for(manager)
            context = server_context_mod._registry
            reconciler = session_control.SessionControlReconciler()

            # -- 1. A narrowing lands on a live child -----------------------
            #
            # The session moves to the stand-in FIRST, while nothing is
            # narrowed, so the child comes up on the write_access gateway and
            # every refusal after the toggle is the reference monitor's rather
            # than a gateway the operator never had.
            outbound = await manager.switch(TARGET_STANDIN)
            journal["standin_role_before"] = outbound["selected_role"]

            journal["armed_write"] = await write_via_tool(CORRECTOR_SP, ARMED_VALUE)
            journal["armed_readback"] = await reading(manager, CORRECTOR_SP)
            await restore_machine()

            # The reconciler baselines here, BEFORE the toggle: its first pass
            # must record what the store already said rather than replay it as a
            # change, and the realignment leg below depends on the flip being
            # the first change it ever sees.
            await reconciler.poll_once()

            journal["child_before_toggle"] = manager.status()["child_pid"]
            narrow = post_posture(web, TARGET_STANDIN, "sandbox")
            journal["narrow_status"] = narrow.status_code
            journal["narrow_body"] = narrow.json()
            journal["child_after_toggle"] = manager.status()["child_pid"]

            journal["refused_rows"] = await write_rows(manager, CORRECTOR_SP, REFUSED_VALUE)
            journal["refused_tool"] = await write_via_tool(CORRECTOR_SP, REFUSED_VALUE)
            journal["refused_execute"] = await execute_readwrite()
            journal["child_after_refusal"] = manager.status()["child_pid"]
            journal["standin_at_rest"] = await reading(manager, CORRECTOR_SP)

            # ... while the other target still writes. The switch is real, so
            # this write lands on the other container.
            await manager.switch(TARGET_VA)
            journal["va_write_while_narrowed"] = await write_via_tool(CORRECTOR_SP, ARMED_VALUE)
            journal["va_readback_while_narrowed"] = await reading(manager, CORRECTOR_SP)
            journal["va_session_target"] = execution_gates.session_control_target()
            journal["va_gates_silent"] = gates_are_silent()
            await restore_machine()

            # -- 2. The realignment waits for a run, then moves the gateway --
            #
            # Back on the narrowed target, with the child still connected on
            # write_access: this is the state the reconciler owes a rebuild.
            await manager.switch(TARGET_STANDIN)
            journal["child_before_realign"] = manager.status()["child_pid"]
            with host_executor._in_flight_marker(TARGET_STANDIN):
                await reconciler.poll_once()
                journal["realign_while_running"] = realign_note()
                journal["child_while_running"] = manager.status()["child_pid"]
            await reconciler.poll_once()
            journal["realign_after_running"] = realign_note()
            journal["child_after_realign"] = manager.status()["child_pid"]
            journal["role_after_realign"] = manager.status()["selected_role"]
            journal["rows_narrowed"] = control_target_module.target_rows(
                context.config.raw,
                session_target=manager.active_target(),
                baseline=manager.baseline,
            )

            # A switch to the narrowed target still verifies: read-only is a
            # usable posture, not an unreachable machine.
            await manager.switch(TARGET_VA)
            journal["reswitch"] = await manager.switch(TARGET_STANDIN)

            # -- 3. An operator-requested switch, end to end ----------------
            request = post_target(web, TARGET_VA)
            journal["request_status"] = request.status_code
            journal["request_id"] = request.json()["request_id"]
            await reconciler.poll_once()
            journal["request_outcome"] = last_switch()
            journal["target_after_request"] = manager.active_target()
            journal["child_after_request"] = manager.status()["child_pid"]

            same = post_target(web, TARGET_VA)
            journal["same_status"] = same.status_code
            journal["same_request_id"] = same.json()["request_id"]
            await reconciler.poll_once()
            journal["same_outcome"] = last_switch()
            journal["child_after_same"] = manager.status()["child_pid"]

            # -- 4. A write follows the session, not the conversation -------
            #
            # The agent is on the virtual accelerator and writing; the operator
            # moves the session from the web surface; the agent's next write is
            # judged against where it now is.
            journal["write_on_a"] = await write_via_tool(CORRECTOR_SP, ARMED_VALUE)
            journal["readback_on_a"] = await reading(manager, CORRECTOR_SP)
            await restore_machine()
            moved = post_target(web, TARGET_STANDIN)
            journal["moved_request_id"] = moved.json()["request_id"]
            await reconciler.poll_once()
            journal["moved_outcome"] = last_switch()
            journal["target_after_move"] = manager.active_target()
            journal["write_on_b"] = await write_via_tool(CORRECTOR_SP, REFUSED_VALUE)
            journal["readback_on_b"] = await reading(manager, CORRECTOR_SP)

            # -- 5. A widening never reaches a run in flight ----------------
            #
            # On the virtual accelerator: see :class:`ARunLaunchedNarrowStays
            # Narrow` for why the pin is measured on that target. The session
            # moves there, the target is narrowed, the stamp is taken from the
            # production stamper, and only then is the store widened.
            await manager.switch(TARGET_VA)
            journal["pin_narrow_status"] = post_posture(web, TARGET_VA, "sandbox").status_code
            journal["narrow_stamps"] = launch_stamps()
            widen = post_posture(web, TARGET_VA, "writes")
            journal["widen_status"] = widen.status_code
            journal["widen_body"] = widen.json()
            journal["wide_stamps"] = launch_stamps()

            journal["launch_refused"] = run_sandbox(
                stamps=journal["narrow_stamps"],
                config_path=config_path,
                value=LAUNCH_REFUSED_VALUE,
                tag="narrow",
            )
            journal["readback_after_pinned_run"] = await reading(manager, CORRECTOR_SP)
            journal["launch_permitted"] = run_sandbox(
                stamps=journal["wide_stamps"],
                config_path=config_path,
                value=LAUNCH_PERMITTED_VALUE,
                tag="wide",
            )
            journal["readback_after_open_run"] = await reading(manager, CORRECTOR_SP)
            await restore_machine()

            yield journal
        finally:
            server_context_mod.reset_server_context()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(manager.shutdown(), 60)


# ---------------------------------------------------------------------------
# 1. A narrowing lands on a live child
# ---------------------------------------------------------------------------


class TestTheNarrowingLandsWithoutARespawn:
    """The toggle, on a session that was already writing to the stand-in.

    Every number here crossed a connector-host boundary to a real soft IOC, and
    the toggle that separates the two halves is a real POST through the shipped
    route into the store the connector reads.
    """

    def test_the_write_was_armed_before_the_toggle(self, session) -> None:
        """The anti-vacuous control: the machine really was writable.

        Without this the refusal below would be indistinguishable from a
        deployment that never armed the stand-in at all — which is the gate this
        module is *not* about.
        """
        assert session["standin_role_before"] == "write_access"
        assert session["armed_write"]["raised"] is False
        assert session["armed_write"]["payload"]["summary"]["outcomes"] == {"confirmed": 1}
        assert session["armed_readback"] == pytest.approx(ARMED_VALUE)

    def test_the_toggle_was_accepted_and_recorded_per_target(self, session) -> None:
        assert session["narrow_status"] == 200
        assert session["narrow_body"]["target"] == TARGET_STANDIN
        assert session["narrow_body"]["posture"] == "sandbox"
        assert session["narrow_body"]["entry"] == {TARGET_STANDIN: "sandbox"}

    def test_no_child_was_respawned_to_apply_it(self, session) -> None:
        """The whole point of enforcing at write time.

        The route this replaced terminated the session's child to re-stamp an
        environment variable, which threw away the conversation for a toggle.
        The pid that served the armed write is the pid that refused the next
        one.
        """
        assert session["child_before_toggle"] == session["child_after_toggle"]
        assert session["child_after_refusal"] == session["child_before_toggle"]

    def test_each_result_names_the_posture_and_the_chip(self, session) -> None:
        """The refusal as the connector produced it, in the child.

        Read off ``write_multiple_channels``, the one write path that returns
        refusals instead of raising them — so this is the per-result ``error``
        an agent would read beside the ``outcome`` word, not a message composed
        by a test helper.
        """
        (row,) = session["refused_rows"]

        assert row["outcome"] == "refused"
        assert row["refusal_reason"] == "WRITES_DISABLED"
        assert f"'{TARGET_STANDIN}' control target" in row["error"]
        assert "writes are off" in row["error"]
        assert "control-target chip in the header" in row["error"]

    def test_the_raised_envelope_carries_that_sentence_too(self, session) -> None:
        """A single-channel refusal raises, so the results are never seen.

        The envelope's headline names the refuser and the channel; without the
        per-result ``error`` folded into it, the one sentence saying WHICH
        posture refused and where to lift it would be dropped on the floor for
        exactly the commonest case.
        """
        assert session["refused_tool"]["raised"] is True
        envelope = session["refused_tool"]["envelope"]

        assert envelope["error_type"] == "write_refused"
        assert envelope["details"]["reason"] == "WRITES_DISABLED"
        assert f"'{TARGET_STANDIN}' control target" in envelope["error_message"]
        assert "control-target chip in the header" in envelope["error_message"]

    def test_a_readwrite_execute_is_refused_by_the_same_posture(self, session) -> None:
        """The executor's gate, on the tool the agent actually calls.

        It fires before anything is launched, so no sandbox is spawned and no
        code runs; what is asserted is that the refusal an agent reads sends it
        to the chip rather than to the deployment config.
        """
        assert session["refused_execute"]["raised"] is True
        envelope = session["refused_execute"]["envelope"]
        spoken = " ".join([envelope["error_message"], *envelope.get("suggestions", [])])

        assert envelope["error_type"] == "safety_error"
        assert "posture" in spoken
        assert "control-target chip in the header" in spoken

    def test_nothing_reached_the_stand_in(self, session) -> None:
        """The refusals happened before Channel Access, not after it."""
        assert session["standin_at_rest"] == pytest.approx(REST_VALUE)

    def test_the_other_target_still_writes(self, session) -> None:
        """A narrowing is per target, which is the entire point of the feature.

        Same session, same store entry, same deployment — and the virtual
        accelerator's write lands on its own container.
        """
        assert session["va_write_while_narrowed"]["raised"] is False
        outcomes = session["va_write_while_narrowed"]["payload"]["summary"]["outcomes"]
        assert outcomes == {"confirmed": 1}
        assert session["va_readback_while_narrowed"] == pytest.approx(ARMED_VALUE)

    def test_the_executors_gates_are_silent_on_the_other_target(self, session) -> None:
        """The control for the ``execute`` refusal above.

        The same two gates the tool runs, asked while the session sits on the
        unnarrowed target: silent. Without this, "the tool refused" would be
        equally consistent with a gate that refuses everything.
        """
        assert session["va_session_target"] == TARGET_VA
        assert session["va_gates_silent"] is True


# ---------------------------------------------------------------------------
# 2. The realignment waits for a run, then moves the gateway
# ---------------------------------------------------------------------------


class TestTheGatewayRealignsOnceNothingIsRunning:
    """The reconciler's half: the child that connected under the old posture.

    Write-time enforcement already refuses (above). This is the follow-through
    that makes the connection itself honest — a child holding the write-access
    gateway of a machine the operator took away is a capability nobody is using
    and nobody should be holding.
    """

    def test_a_running_execution_holds_the_realignment_pending(self, session) -> None:
        """Published before the wait, not after it.

        "Pending" is the answer to "why has my toggle not taken effect", and it
        is only useful while the operator is still asking.
        """
        assert session["realign_while_running"]["state"] == session_control.REALIGN_PENDING
        assert session["child_while_running"] == session["child_before_realign"]

    def test_the_child_is_rebuilt_once_the_run_ends(self, session) -> None:
        assert session["realign_after_running"]["state"] == session_control.REALIGN_DONE
        assert session["child_after_realign"] != session["child_before_realign"]

    def test_the_rebuilt_child_connected_on_the_read_only_gateway(self, session) -> None:
        assert session["role_after_realign"] == "read_only"

    def test_the_roster_reports_the_same_role_the_child_holds(self, session) -> None:
        """The popover and the tool never disagree: one policy function.

        ``target_rows`` derives the row from config and the store; the child
        derived its gateway from the same store through the same predicate. A
        row that named ``write_access`` here would be describing a connection
        that does not exist.
        """
        rows = session["rows_narrowed"]

        assert rows[TARGET_STANDIN]["selected_role"] == "read_only"
        assert rows[TARGET_STANDIN]["writes_permitted"] is False
        assert rows[TARGET_VA]["writes_permitted"] is True

    def test_the_roster_is_the_three_targets_this_deployment_configures(self, session) -> None:
        """The premise every per-target assertion in this module stands on.

        Three rows means the deployment really is in the two-target world the
        chip is specified over — a roster of one would make "per target" an
        empty distinction — and the facility's own machine reports its writes as
        not permitted, which is the deployment ceiling doing its half while the
        store does the other.
        """
        rows = session["rows_narrowed"]

        assert sorted(rows) == sorted([TARGET_LIVE, TARGET_VA, TARGET_STANDIN])
        assert rows[TARGET_LIVE]["writes_permitted"] is False
        assert rows[TARGET_LIVE]["real_machine"] is True

    def test_a_switch_to_the_narrowed_target_still_verifies(self, session) -> None:
        """Read-only is a posture, not an unreachable machine.

        A switch that raised ``SwitchError`` here would mean an operator who
        narrowed a target had also lost the ability to point a session at it —
        which would make the toggle a trap rather than a safety belt.
        """
        landed = session["reswitch"]

        assert landed["target"] == TARGET_STANDIN
        assert landed["selected_role"] == "read_only"


# ---------------------------------------------------------------------------
# 3. An operator-requested switch, end to end
# ---------------------------------------------------------------------------


class TestTheOperatorsSwitchRequest:
    """The web server writes desired state; the controls server decides.

    The route never touches the connector — it cannot; the connector lives in
    another process with no inbound channel — so what is asserted here is the
    whole loop: a request file addressed by pid, a reconcile pass, a real switch
    between two containers, and an outcome published back under the request's
    own id.
    """

    def test_the_request_was_accepted_without_a_verdict(self, session) -> None:
        assert session["request_status"] == 202
        assert session["request_id"]

    def test_the_reconciler_switched_and_published_the_outcome(self, session) -> None:
        outcome = session["request_outcome"]

        assert outcome["request_id"] == session["request_id"]
        assert outcome["target"] == TARGET_VA
        assert outcome["status"] == session_control.STATUS_SUCCESS
        assert outcome["reason"] is None
        assert session["target_after_request"] == TARGET_VA

    def test_a_request_for_the_target_the_session_is_on_is_refused(self, session) -> None:
        """``already_active`` is the roster's own word, arriving through the gate.

        Matched by ``request_id`` so the chip can tell this answer from the
        successful one it was shown a moment earlier.
        """
        outcome = session["same_outcome"]

        assert session["same_status"] == 202
        assert outcome["request_id"] == session["same_request_id"]
        assert outcome["request_id"] != session["request_id"]
        # A refusal names the target it was aimed at too: the popover renders
        # the word on that machine's row, refusals included.
        assert outcome["target"] == TARGET_VA
        assert outcome["status"] == session_control.STATUS_REFUSED
        assert outcome["reason"] == "already_active"

    def test_a_refused_request_respawns_no_child(self, session) -> None:
        """The gate is asked before ``switch()``, so nothing is torn down."""
        assert session["child_after_same"] == session["child_after_request"]


# ---------------------------------------------------------------------------
# 4. A write follows the session, not the conversation
# ---------------------------------------------------------------------------


class TestTheAgentsNextWriteIsJudgedWhereTheSessionNowIs:
    """Mid-conversation, with nobody telling the agent anything.

    The agent writes on the target it was pointed at, an operator moves the
    session from the web surface, and the agent's very next write is evaluated
    against the machine it is now on — which happens to be the narrowed one, so
    it is refused. No respawn, no in-band announcement, no re-read the agent had
    to remember to do.
    """

    def test_the_write_before_the_move_landed_on_the_first_target(self, session) -> None:
        assert session["write_on_a"]["raised"] is False
        assert session["readback_on_a"] == pytest.approx(ARMED_VALUE)

    def test_the_operator_moved_the_session(self, session) -> None:
        outcome = session["moved_outcome"]

        assert outcome["request_id"] == session["moved_request_id"]
        assert outcome["target"] == TARGET_STANDIN
        assert outcome["status"] == session_control.STATUS_SUCCESS
        assert session["target_after_move"] == TARGET_STANDIN

    def test_the_next_write_is_evaluated_against_the_new_target(self, session) -> None:
        assert session["write_on_b"]["raised"] is True
        envelope = session["write_on_b"]["envelope"]

        assert envelope["details"]["reason"] == "WRITES_DISABLED"
        assert f"'{TARGET_STANDIN}' control target" in envelope["error_message"]

    def test_the_second_machine_never_moved(self, session) -> None:
        assert session["readback_on_b"] == pytest.approx(REST_VALUE)


# ---------------------------------------------------------------------------
# 5. A widening never reaches a run in flight
# ---------------------------------------------------------------------------


class TestARunLaunchedNarrowStaysNarrow:
    """The launch pin, across a real process boundary.

    Both stamps come from ``executor._apply_target_stamp`` — the production
    stamper — taken either side of a real widening POST, and both runs are real
    subprocesses that build their own connector from the deployment file. The
    store says ``writes`` for both of them; the only thing that separates them
    is the posture each was LAUNCHED under.

    **Measured on the virtual accelerator**, unlike every other leg in this
    module. The pin is target-agnostic — ``store_permits`` takes the target as
    an argument and the stamp names whichever one the run was placed on — and
    the virtual accelerator is the target a sandbox on this deployment can
    build: ``osprey.registry.builtins`` does not list ``live_standin`` among the
    built-in control systems, so ``initialize_registry`` (which every sandbox
    runs before any agent code) leaves that type unregistered and a run stamped
    for the stand-in fails on ``Unknown control system type`` before reaching
    any posture at all. Pointing this leg at the stand-in would measure that,
    not the pin.
    """

    def test_the_stamps_straddle_the_widening(self, session) -> None:
        """The premise, asserted rather than assumed.

        A stamper that had stopped stamping, or a widen the route refused, would
        leave both runs identical and the comparison below vacuous.
        """
        assert session["pin_narrow_status"] == 200
        assert session["widen_status"] == 200
        assert TARGET_VA not in session["widen_body"]["entry"]
        assert session["narrow_stamps"][host_executor.ENV_LAUNCH_POSTURE] == f"{TARGET_VA}=sandbox"
        assert session["wide_stamps"][host_executor.ENV_LAUNCH_POSTURE] == f"{TARGET_VA}=writes"

    def test_the_run_that_launched_narrow_is_still_refused(self, session) -> None:
        verdict = session["launch_refused"]

        assert verdict["wrote"] is False
        assert "launched while" in verdict["message"]
        assert TARGET_VA in verdict["message"]
        assert "not to one already in flight" in verdict["message"]

    def test_the_refused_run_left_the_machine_alone(self, session) -> None:
        assert session["readback_after_pinned_run"] == pytest.approx(REST_VALUE)

    def test_a_run_launched_after_the_widen_writes(self, session) -> None:
        """The control: the widen really did widen.

        Same store, same deployment, same machine — and a run stamped after the
        toggle puts its value on the setpoint, which is what makes the refusal
        above a statement about the pin rather than about a store that never
        changed.
        """
        assert session["launch_permitted"]["wrote"] is True
        assert session["readback_after_open_run"] == pytest.approx(LAUNCH_PERMITTED_VALUE)


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_per_target_posture.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
