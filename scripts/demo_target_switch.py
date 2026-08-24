#!/usr/bin/env python3
"""Operator demo: prove a runtime control-target switch really moved the session.

What this is
------------
A single runnable script that drives one session across two Channel Access
endpoints and prints an **audit trail** an operator reads afterwards. It is the
operator-facing sibling of ``tests/va/e2e/test_target_switch.py``: the same
claims, driven through the same production functions, but as one linear
narrative with an exit code instead of a pytest report.

Every line it prints names the target the session is on at that moment::

    STEP 4 [target=va] reading SR:DIAG:BPM:11:POSITION:X on the new target
    STEP 4 [target=va] OK value B = 5.250000e-05, distinct from A (delta 5.250e-05)

so the trail can be read top to bottom, and grepped (``grep '^STEP'``) without
knowing anything about this file.

The posture is an INPUT, not a fact about this repository
--------------------------------------------------------
The switch is only meaningful against a deployment that is *configured* to
switch: an archiver that does not invent history, strict limits with the demo's
channels listed, an operator acknowledgment for the live gateway, and one
distinct endpoint per target. None of that is hardcoded here — a real facility's
hostnames belong in that facility's ``config.yml`` and not in a repository. So
this script takes the posture two ways:

* ``--config <path>`` — demo against a deployment the operator already has. The
  endpoints, the acknowledgment and the probe channels are read from that file.
* ``--self-provision`` — stand the posture up locally from the scratch stack:
  two ``osprey-va-full`` containers on two ephemeral Channel Access ports, one
  of them booted with a seeded BPM readout error so the two serve *different*
  numbers for the same channel with nothing ever written to either.

``--dry-run`` prints the plan (``PLAN n …`` lines) and touches nothing, which is
what review and CI can smoke.

What it proves, in order
------------------------
1. ``va`` is eligible **before any switch has happened** — a roster verdict from
   configuration alone.
2. The probe channel reads value **A** on the baseline target.
3. The switch happens, through the same ``control_target_set`` tool an agent
   session would call.
4. The same channel now reads value **B**, and B is not A. Two endpoints that
   answered the same number would be a switch that moved nothing.
5. An ``execute()`` sandbox — a real stamped subprocess, the way the python
   executor launches one — reads **B** and not A, so the routing follows the
   session into another process.
6. A write lands on the active target and is confirmed there.
7. Coming home, the written value is **not** visible on the other target: a
   value approved for one machine never reached the other.
8. The final roster names the target the session ended on.

Scope, stated rather than implied
---------------------------------
* **Name-server mode only.** Both endpoints are reached with
  ``use_name_server: true``. The UDP address-list path (``use_name_server:
  false``) is covered by the unit and integration suites and is deliberately not
  exercised here — a demo that quietly swapped transports would be reporting on
  a deployment other than the one it described.
* **The archiver leg is a config assertion.** The required posture is a
  mongodb-backed archive (the ``va_archiver:`` build block, which renders
  ``archiver.type: mongodb_archiver``). This demo asserts that posture with the
  production predicate — ``osprey_connectors.honesty.pairing_for_target`` — and
  says so in the trail. It boots no ``mongod`` and reads no history: there is no
  archive here to read, and a value read back from a store this script populated
  itself would be evidence of nothing.
* **The only write happens while the session is on the virtual accelerator.**
  Step 6 runs after step 3 and before step 7, by construction. It reads the
  channel first and, on an operator's own deployment, puts *that* value back
  through the same ``channel_write`` tool — same limits, same readback
  verification — on every exit path, including a failure part-way through. Where
  it cannot restore (or where the endpoint is a container about to be removed),
  the trail names the channel and the value left on it. Silence is never an
  outcome.
* Nothing here has touched hardware, and nothing here is a statement about a
  real facility's gateways.

Running it
----------
The repository has no installed console entry point for this script, so it is
run against a checkout with the two source trees on ``PYTHONPATH``::

    PYTHONPATH=src:packages/osprey-connectors/src \\
      python scripts/demo_target_switch.py --self-provision

    PYTHONPATH=src:packages/osprey-connectors/src \\
      python scripts/demo_target_switch.py \\
        --config /path/to/deployment/build/config.yml \\
        --write-channel SR:MAG:HCM:01:CURRENT:SP

``--self-provision`` needs Docker and the ``osprey-va-full`` image (build it
with ``scripts/va/build_and_boot_check.sh``).

Exits 0 only if every step held; otherwise 1, with the failing step named on the
last line.

Why the container recipe is restated here
-----------------------------------------
It duplicates the one in ``tests/va/e2e/test_target_switch.py`` deliberately.
That one lives in pytest fixtures, and an operator script that imported pytest
fixtures to boot a container would drag the test framework into the demo — and
would break the moment the suite reorganises its fixtures. The duplication is
about forty lines and is the cheaper of the two couplings.
"""

from __future__ import annotations

import argparse
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
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_PATHS = (str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src"))

# -- the scratch stack ------------------------------------------------------

#: Image the self-provisioned posture boots, one container per target.
DEFAULT_IMAGE = os.environ.get("OSPREY_VA_DEMO_IMAGE", "osprey-va-full:latest")
#: The simulation model both containers serve.
DATA_DIR = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation"
#: The strict limits database the demo posture points at. Both demo channels are
#: listed in it, which is what ``allow_unlisted_channels: false`` requires.
LIMITS_DB_PATH = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/channel_limits.json"

CONTAINER_PREFIX_VA = "osprey-va-demo-switch-va"
CONTAINER_PREFIX_LIVE = "osprey-va-demo-switch-live"

#: Generous on purpose: the image is pinned ``linux/amd64``, so a local boot on
#: Apple Silicon is emulated.
BOOT_TIMEOUT_S = 180.0

# -- the channels the demo speaks about -------------------------------------

#: The channel that tells the two endpoints apart, and the probe channel this
#: demo's posture configures for both targets. One container is booted with this
#: BPM's readout error seeded and the other is not, so the two serve different
#: numbers for a machine in the same quiescent state — with no client write
#: anywhere, and reproducibly.
DEFAULT_PROBE_CHANNEL = "SR:DIAG:BPM:11:POSITION:X"
SEEDED_DEVICE = "BPM11"
SEEDED_OFFSET_X = 50e-6
SEEDED_GAIN_X = 1.05
VA_BPM_ERRORS = f"{SEEDED_DEVICE}:offset_x={SEEDED_OFFSET_X},gain_x={SEEDED_GAIN_X}"

#: The one setpoint this demo writes. Listed in the shipped limits database with
#: a +-12 A band, so the value below passes the strict limits posture.
DEFAULT_WRITE_CHANNEL = "SR:MAG:HCM:01:CURRENT:SP"
DEFAULT_WRITE_VALUE = 1.0

#: What the plan prints for a channel an operator posture takes from its own
#: config rather than from this file. A default printed here would be a channel
#: name this repository invented for somebody else's facility.
PLACEHOLDER_PROBE = "<from config probe_channel>"
PLACEHOLDER_WRITE = "<required via --write-channel>"

# -- bounds -----------------------------------------------------------------

CONNECTOR_TIMEOUT_S = 60.0
READ_TIMEOUT_S = 30.0
SPAWN_TIMEOUT_S = 60.0
PROBE_TIMEOUT_S = 10.0
DRAIN_TIMEOUT_S = 5.0
SANDBOX_TIMEOUT_S = 300.0

#: How close the sandbox's reading must be to the in-session reading of the same
#: channel on the same target, as a fraction of the distance between the two
#: targets' values. The claim is "the sandbox read THIS machine", so the
#: criterion is stated relative to the only other candidate.
SANDBOX_TOLERANCE_FRACTION = 0.1

#: Below this relative separation, two readings are reported as distinct but
#: NOT as evidence on their own: at that scale a live machine's own jitter would
#: satisfy the comparison. The demo says so in the line rather than passing
#: quietly, and points at step 7 — the write-isolation claim, which no amount of
#: jitter can produce.
MIN_RELATIVE_SEPARATION = 1e-3


# ---------------------------------------------------------------------------
# The audit trail
# ---------------------------------------------------------------------------


#: What the trail says when nothing has established a target yet — a failure
#: before the session exists still has to name something rather than imply a
#: target it never had.
UNKNOWN_TARGET = "unknown"


class StepFailed(RuntimeError):
    """A demo step whose claim did not hold.

    Carries the target the session was on when the claim broke, so the failing
    line reads like every other line in the trail. A verdict that named the step
    but not the target would make the operator go looking for it, and the answer
    ("which machine was this about?") is the first thing they need.
    """

    def __init__(self, number: int, message: str, target: str = UNKNOWN_TARGET) -> None:
        super().__init__(message)
        self.number = number
        self.message = message
        self.target = target


class Audit:
    """The operator-facing trail. One stable, grep-friendly line per event.

    Line shapes, all of them fixed:

    * ``STEP <n> [target=<t>] <what is about to happen>``
    * ``STEP <n> [target=<t>] OK <what held>``
    * ``STEP <n> [target=<t>] FAILED <what did not hold>``
    * ``NOTE <context that is not a step>``
    * ``PLAN <n> <what a real run would do here>`` (``--dry-run`` only)
    * ``RESULT PASS`` / ``RESULT FAIL step <n> [target=<t>]: <why>``
    """

    def __init__(self, stream: Any = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def _emit(self, line: str) -> None:
        print(line, file=self._stream, flush=True)

    def note(self, text: str) -> None:
        self._emit(f"NOTE {text}")

    def plan(self, number: int, text: str) -> None:
        self._emit(f"PLAN {number} {text}")

    def step(self, number: int, target: str, text: str) -> None:
        self._emit(f"STEP {number} [target={target}] {text}")

    def ok(self, number: int, target: str, text: str) -> None:
        self._emit(f"STEP {number} [target={target}] OK {text}")

    def failed(self, number: int, target: str, text: str) -> None:
        self._emit(f"STEP {number} [target={target}] FAILED {text}")

    def result_pass(self) -> None:
        self._emit("RESULT PASS")

    def result_fail(self, number: int, why: str, target: str = UNKNOWN_TARGET) -> None:
        self._emit(f"RESULT FAIL step {number} [target={target}]: {why}")


# ---------------------------------------------------------------------------
# The scratch stack: two containers on two ephemeral Channel Access ports
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _docker(*args: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _require_image(image: str) -> None:
    """Refuse to start unless the image can serve on a port other than 5064.

    The Channel Access *server* library reads ``EPICS_CAS_SERVER_PORT`` and does
    not fall back to the client-side variable, so an image whose entry point
    does not derive one from the other keeps binding its build-time default
    while telling this demo's clients some other port. The symptom would be an
    unexplained boot timeout; this turns it into a sentence naming the fix.
    """
    inspected = _docker(
        "image", "inspect", image, "--format", "{{.Architecture}}|{{.Config.Cmd}}", timeout=60
    )
    if inspected.returncode != 0:
        raise RuntimeError(
            f"image {image!r} is not present. Build it with "
            f"scripts/va/build_and_boot_check.sh, or name another with --image."
        )
    architecture, _, command = inspected.stdout.strip().partition("|")
    if "EPICS_CAS_SERVER_PORT" not in command:
        raise RuntimeError(
            f"image {image!r} ({architecture}) does not derive EPICS_CAS_SERVER_PORT from "
            f"EPICS_CA_SERVER_PORT, so it cannot serve on any port but its baked default. "
            f"Rebuild it with scripts/va/build_and_boot_check.sh. Its entry point is: {command}"
        )


def _served(port: int, address: str) -> bool:
    """Whether a virtual accelerator answers for *address* on *port*.

    Asked in a subprocess, and this is not a style choice: pyepics latches
    ``EPICS_CA_*`` when its library initialises and its contexts are per-thread,
    so a Channel Access call in *this* process would poison the demo's own
    connector-host children. This process never becomes a CA client.
    """
    code = (
        "import sys, epics\n"
        f"v = epics.caget({address!r}, timeout=1.0, connection_timeout=1.0)\n"
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
def _serving(prefix: str, *, image: str, seeded: bool, probe_channel: str, audit: Audit):
    """Boot one virtual accelerator container and wait until it serves.

    The published port and the server's own port are the same number by
    construction — a Channel Access search reply carries the server's own port,
    so a remap would hand every client an address nothing answers on — and that
    number also names the container, so two concurrent runs cannot collide.
    """
    port = _free_port()
    name = f"{prefix}-{port}"
    _docker("rm", "-f", name, timeout=60)  # stale cleanup; the port is this run's alone

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
    started = _docker(*arguments, image)
    if started.returncode != 0:
        raise RuntimeError(f"docker run failed: {started.stdout}\n{started.stderr}")
    audit.note(
        f"booted container {name} on Channel Access port {port} "
        f"({'seeded ' + VA_BPM_ERRORS if seeded else 'unseeded'})"
    )

    try:
        deadline = time.monotonic() + BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            if _served(port, probe_channel):
                break
            time.sleep(1.0)
        else:
            logs = _docker("logs", "--tail", "40", name, timeout=60)
            raise RuntimeError(
                f"{name} never served {probe_channel} within {BOOT_TIMEOUT_S}s.\n"
                f"{logs.stdout}\n{logs.stderr}"
            )
        audit.note(f"container {name} is serving {probe_channel}")
        yield port
    finally:
        _docker("rm", "-f", name, timeout=60)


# ---------------------------------------------------------------------------
# The posture
# ---------------------------------------------------------------------------


@dataclass
class WriteLedger:
    """What step 6 did to the machine, readable from the demo's ``finally``.

    Three states, because they call for three different sentences: nothing was
    attempted (say nothing); a write was attempted and may or may not have
    landed (say so, and restore); a write landed (restore, and name the value it
    replaced). ``original`` is what the channel read *before* the write — the
    value a restore puts back.
    """

    channel: str
    original: float | None = None
    attempted: bool = False
    landed: bool = False


@dataclass
class Posture:
    """The deployment this demo runs against, however it was obtained."""

    config_path: Path
    raw: dict[str, Any]
    probe_channel: str
    write_channel: str
    write_value: float
    self_provisioned: bool
    #: Distance the seeded readout error puts between the two endpoints, when
    #: this demo seeded them itself. ``None`` for an operator-supplied posture,
    #: where nothing here may assume a magnitude.
    expected_separation: float | None = None


def scratch_config(
    *,
    va_port: int,
    live_port: int,
    probe_channel: str,
    acknowledgment: str,
    project_root: Path,
) -> dict[str, Any]:
    """The self-provisioned posture, in rendered-``config.yml`` shape.

    ``control_system.type`` is ``epics``, which makes ``live`` this
    deployment's baseline and ``va`` the other target — the arrangement under
    which ``switch_capable`` holds. The ``live`` end is a simulator wearing the
    live target's connector type, which is what a scratch stack can honestly
    provide: a second real Channel Access endpoint under this script's control.

    Both blocks name an explicit gateway port, because the containers bind
    ephemeral ports and nothing may guess them, and both are reached in
    name-server mode.
    """

    def block(port: int) -> dict[str, Any]:
        return {
            "timeout": CONNECTOR_TIMEOUT_S,
            "probe_channel": probe_channel,
            "gateways": {
                "read_only": {"address": "localhost", "port": port, "use_name_server": True},
                "write_access": {"address": "localhost", "port": port, "use_name_server": True},
            },
        }

    return {
        "project_root": str(project_root),
        "control_system": {
            "type": "epics",
            "writes_enabled": True,
            "limits_checking": {
                "enabled": True,
                "allow_unlisted_channels": False,
                "database_path": str(LIMITS_DB_PATH),
            },
            "target_switch": {
                "live_gateway_acknowledged": acknowledgment,
                "drain_timeout_s": DRAIN_TIMEOUT_S,
            },
            "connector": {
                "epics": block(live_port),
                "virtual_accelerator": block(va_port),
            },
        },
        # The mongodb-backed archive the posture requires. Nothing in this demo
        # builds an archiver connector or reads history; see the module
        # docstring for why this leg is a config assertion.
        "archiver": {"type": "mongodb_archiver"},
        "agent_data": {"base_dir": "var/agent_data"},
    }


def limits_note(*channels: str) -> str:
    """What the strict limits database actually says about *channels*.

    Read rather than asserted. The claim "both demo channels are listed" is only
    true for the demo's own defaults, and a channel named with ``--probe-channel``
    or ``--write-channel`` may well be absent — under
    ``allow_unlisted_channels: false`` a write to an absent channel is refused,
    which is a thing the operator should learn from the first line of the trail
    and not from a step-6 failure.
    """
    try:
        database = json.loads(LIMITS_DB_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return (
            f"strict limits: allow_unlisted_channels=false against {LIMITS_DB_PATH.name}, "
            f"which could not be read here ({type(exc).__name__}); membership unverified"
        )
    listed = [name for name in channels if name in database]
    missing = [name for name in channels if name not in database]
    membership = f"lists {', '.join(listed)}" if listed else "lists neither demo channel"
    if missing:
        membership += (
            f"; does NOT list {', '.join(missing)} — under this posture a write to an "
            f"unlisted channel is refused"
        )
    return (
        f"strict limits: allow_unlisted_channels=false against {LIMITS_DB_PATH.name}, "
        f"which {membership}"
    )


def probe_channel_from_config(raw: dict[str, Any], connector_type: str) -> str:
    """The probe channel an operator's config configures for one connector type."""
    section = raw.get("control_system") if isinstance(raw, dict) else None
    connector = section.get("connector", {}) if isinstance(section, dict) else {}
    block = connector.get(connector_type, {}) if isinstance(connector, dict) else {}
    value = block.get("probe_channel") if isinstance(block, dict) else None
    return str(value or "")


# ---------------------------------------------------------------------------
# The sandbox: a stamped subprocess, the way the python executor launches one
# ---------------------------------------------------------------------------

#: Sandbox-side agent code. Run in a real subprocess carrying the real target
#: stamp, because the routing is a contract between two processes and neither
#: can see the other's state directly.
#:
#: ``initialize_registry`` is the sandbox's own setup step, restated here rather
#: than skipped: it is what the executor's wrapper emits ahead of every
#: execution, and it is what populates the connector factory — a bare process
#: importing ``osprey.runtime`` has an empty one and would fail with "Unknown
#: control system type" long before reaching the read.
#:
#: The verdict is written to a file rather than printed, because registry
#: initialisation is chatty on both streams and a result parsed out of that
#: noise would be a result this demo could misread. The exit is abrupt for the
#: reason ``osprey_connectors.ipc.host`` states of its own: a process that has
#: held a Channel Access context can block forever in pyepics' ``finalize_libca``
#: atexit hook, and a sandbox that will not die is worse than one that skips its
#: hooks.
_SANDBOX_READ = """
import json
import os
import sys
from pathlib import Path

from osprey.registry import initialize_registry

initialize_registry(auto_export=False, config_path=os.environ["CONFIG_FILE"])

import osprey.runtime as runtime

address, verdict_path = sys.argv[1], sys.argv[2]
try:
    value = runtime.read_channel(address, timeout=30.0)
except Exception as exc:
    verdict = {"read": False, "error": type(exc).__name__, "message": str(exc)}
else:
    verdict = {"read": True, "value": float(value)}
Path(verdict_path).write_text(json.dumps(verdict), encoding="utf-8")
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
"""


def sandbox_read(
    *, address: str, target: str, generation: int, config_path: Path, work_dir: Path
) -> dict[str, Any]:
    """Read *address* from a subprocess stamped for *target* at *generation*.

    The three stamp variables are exactly the ones the python executor sets, and
    the state PID is this process's — the demo is the controls server here, and
    a stamp naming any other process would be a stamp about somebody else's
    session.
    """
    verdict_path = work_dir / "sandbox_verdict.json"
    with contextlib.suppress(FileNotFoundError):
        verdict_path.unlink()

    from osprey.mcp_server.python_executor import executor as host_executor

    inherited = os.environ.get("PYTHONPATH", "")
    python_path = (
        os.pathsep.join([*REPO_PATHS, inherited]) if inherited else os.pathsep.join(REPO_PATHS)
    )
    environment = {
        **os.environ,
        "PYTHONPATH": python_path,
        "CONFIG_FILE": str(config_path),
        "OSPREY_CONFIG": str(config_path),
        host_executor.ENV_CONTROL_TARGET: target,
        host_executor.ENV_CONTROL_TARGET_GENERATION: str(generation),
        host_executor.ENV_CONTROL_TARGET_STATE_PID: str(os.getpid()),
    }
    for stale in ("EPICS_CA_ADDR_LIST", "EPICS_CA_NAME_SERVERS", "EPICS_CA_SERVER_PORT"):
        environment.pop(stale, None)

    completed = subprocess.run(
        [sys.executable, "-c", _SANDBOX_READ, address, str(verdict_path)],
        capture_output=True,
        text=True,
        timeout=SANDBOX_TIMEOUT_S,
        cwd=str(work_dir),
        env=environment,
    )
    if completed.returncode != 0 or not verdict_path.exists():
        raise RuntimeError(f"the stamped sandbox failed:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(verdict_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The demo itself
# ---------------------------------------------------------------------------


def _tool(tool_or_fn: Any) -> Any:
    """The raw async function behind a FastMCP tool object.

    The demo calls the tools an agent session calls, not private helpers behind
    them — so a demo that passed while the tool an operator actually reaches was
    broken is not a shape this script can take.
    """
    return getattr(tool_or_fn, "fn", tool_or_fn)


class Claims:
    """Raises :class:`StepFailed` with the target the session is on right now.

    A free function could not do this: the target moves as the demo runs, and
    every failing line has to name where the session actually was — which only
    the manager knows, and only at the moment the claim breaks.
    """

    def __init__(self, current_target: Any) -> None:
        self._current_target = current_target

    def target(self) -> str:
        try:
            return str(self._current_target())
        except Exception:  # pragma: no cover - a trail line must never crash the demo
            return UNKNOWN_TARGET

    def require(self, condition: bool, number: int, message: str) -> None:
        if not condition:
            raise StepFailed(number, message, self.target())


async def _read_one(address: str, number: int, claims: Claims) -> float:
    """One value, through the ``channel_read`` tool an agent session would call."""
    from osprey.mcp_server.control_system.tools import channel_read as channel_read_tool

    payload = json.loads(
        await _tool(channel_read_tool.channel_read)(channels=[address], include_metadata=False)
    )
    readings = payload.get("summary", {}).get("readings", {})
    claims.require(
        address in readings,
        number,
        f"the control system did not answer for {address}: {json.dumps(payload)[:400]}",
    )
    return float(readings[address]["value"])


async def run_demo(posture: Posture, audit: Audit) -> None:
    """Drive the eight steps. Raises :class:`StepFailed` at the first broken claim."""
    from osprey.mcp_server.control_system import server_context as server_context_mod
    from osprey.mcp_server.control_system import target_state
    from osprey.mcp_server.control_system.connector_host_manager import ConnectorHostManager
    from osprey.mcp_server.control_system.server_context import (
        ConnectorEntry,
        ControlSystemContext,
        MCPServerConfig,
    )
    from osprey.mcp_server.control_system.tools import channel_write as channel_write_tool
    from osprey.mcp_server.control_system.tools import control_target as control_target_tools
    from osprey_connectors import honesty

    config = MCPServerConfig(raw=posture.raw, config_path=posture.config_path)
    manager = ConnectorHostManager(
        config,
        probe_timeout_s=PROBE_TIMEOUT_S,
        spawn_timeout_s=SPAWN_TIMEOUT_S,
        terminate_grace_s=2.0,
    )
    manager.reset_state()

    # The server context the tools read through, wired to this manager. Same
    # shape the e2e suite builds: ``initialize()`` would read a deployment's
    # config off disk, and everything the tools below exercise is downstream of
    # that.
    context = ControlSystemContext()
    context._config = config
    context._connector_hosts = manager
    context._connectors["control_system"] = ConnectorEntry(
        config=config.control_system, connector_type="control_system"
    )
    context._connectors["archiver"] = ConnectorEntry(
        config=config.archiver, connector_type="archiver"
    )
    server_context_mod._registry = context

    baseline = manager.baseline
    claims = Claims(manager.active_target)
    #: What step 6 found on the write channel before it wrote, and whether it
    #: got as far as writing. Read by the settle pass below, which runs on every
    #: exit path — including a failure in the middle of step 6.
    ledger = WriteLedger(channel=posture.write_channel)
    audit.note(
        f"deployment baseline target is {baseline!r}; state file under {target_state.state_dir()}"
    )

    # The archiver leg of the posture, asserted with the production predicate
    # rather than by eye — and named as an assertion, not a measurement.
    pairing = honesty.pairing_for_target(posture.raw, "va")
    archiver_type = str((posture.raw.get("archiver") or {}).get("type", "")) or "unset"
    audit.note(
        f"archiver posture ASSERTED FROM CONFIG (not measured): archiver.type={archiver_type!r}; "
        f"honesty.pairing_for_target(config, 'va').is_invented_history="
        f"{pairing.is_invented_history}. No mongod is booted and no history is read by this "
        f"demo — a store this script populated itself would prove nothing."
    )
    claims.require(
        not pairing.is_invented_history,
        1,
        f"the posture pairs a virtual accelerator with an archiver that invents history "
        f"(archiver.type is {pairing.archiver_phrase}); switching to 'va' is refused by design",
    )
    audit.note("transport: name-server mode only; the UDP address-list path is covered by units")

    try:
        # -- 1. eligibility from configuration alone, before any switch -------
        audit.step(1, baseline, "asking the target roster BEFORE anything has been switched")
        roster = json.loads(await _tool(control_target_tools.control_target)())
        rows = roster["access_details"]["targets"]
        va_row = rows["va"]
        claims.require(
            va_row["available_now"],
            1,
            f"'va' is not switchable from this configuration: {va_row['reason']} — "
            f"{va_row['detail']}",
        )
        claims.require(
            roster["summary"]["target"] == baseline and roster["summary"]["generation"] == 0,
            1,
            f"the session did not start on the baseline at generation 0: {roster['summary']}",
        )
        audit.ok(
            1,
            baseline,
            f"'va' is eligible from config alone ({_reason_of(va_row)}, "
            f"connector_type={va_row.get('connector_type')}, probe={va_row.get('probe_channel')}); "
            f"switchable now: {roster['summary']['switchable_targets']}",
        )

        # -- 2. the baseline reading ------------------------------------------
        audit.step(2, baseline, f"starting the connector host and reading {posture.probe_channel}")
        await manager.start(baseline)
        claims.require(manager.has_child(), 2, "the connector host did not come up on the baseline")
        value_a = await _read_one(posture.probe_channel, 2, claims)
        status = manager.status()
        audit.ok(
            2,
            manager.active_target(),
            f"value A = {value_a:.6e} from child pid {status['child_pid']} "
            f"(connector_type={status['connector_type']}, role={status['selected_role']})",
        )

        # -- 3. the switch, through the tool an agent session calls ------------
        audit.step(3, manager.active_target(), "switching to 'va' via control_target_set")
        payload = json.loads(await _tool(control_target_tools.control_target_set)(target="va"))
        claims.require(
            payload.get("status") == "success",
            3,
            f"the switch did not succeed: {json.dumps(payload)[:400]}",
        )
        summary = payload["summary"]
        endpoint = payload["access_details"]["endpoint"]
        claims.require(summary["target"] == "va", 3, f"the session landed on {summary['target']!r}")
        audit.ok(
            3,
            manager.active_target(),
            f"generation {summary['generation']}, previous_target={summary['previous_target']!r}, "
            f"connector_type={summary['connector_type']}, endpoint="
            f"{endpoint.get('host')}:{endpoint.get('port')}, probe={summary['probe_channel']!r}, "
            f"child pid {payload['access_details']['child_pid']}",
        )

        # -- 4. the same channel, a different machine --------------------------
        audit.step(4, manager.active_target(), f"reading {posture.probe_channel} on the new target")
        value_b = await _read_one(posture.probe_channel, 4, claims)
        separation = abs(value_b - value_a)
        claims.require(
            value_b != value_a,
            4,
            f"both targets served {value_a:.6e} for {posture.probe_channel}, so the switch "
            f"moved the session between two indistinguishable machines",
        )
        if posture.expected_separation is not None:
            # This demo seeded the separation itself, so it knows the size to
            # expect and holds the reading to it.
            claims.require(
                separation > posture.expected_separation / 2,
                4,
                f"the two readings differ by {separation:.3e}, which is float noise rather than "
                f"the seeded separation of {posture.expected_separation:.3e}",
            )
        audit.ok(
            4,
            manager.active_target(),
            f"value B = {value_b:.6e}, distinct from A = {value_a:.6e} (delta {separation:.3e})"
            + _separation_caveat(posture, value_a, value_b),
        )

        # -- 5. the same routing, from an execute() sandbox --------------------
        target, generation = manager.active_binding()
        audit.step(
            5,
            manager.active_target(),
            f"reading {posture.probe_channel} from an execute() sandbox stamped "
            f"{target!r} generation {generation}",
        )
        # The sandbox's scratch directory is this demo's own, never the
        # deployment's: an operator's config directory is theirs, and a demo has
        # no business leaving a verdict file in it. The subprocess is pointed at
        # the config through ``OSPREY_CONFIG``, so its working directory is free.
        with _temporary_directory() as sandbox_dir:
            verdict = sandbox_read(
                address=posture.probe_channel,
                target=target,
                generation=generation,
                config_path=posture.config_path,
                work_dir=Path(sandbox_dir),
            )
        claims.require(
            verdict.get("read") is True,
            5,
            f"the sandbox could not read the channel: {verdict}",
        )
        sandboxed = float(verdict["value"])
        claims.require(
            abs(sandboxed - value_b) < abs(sandboxed - value_a),
            5,
            f"the sandbox read {sandboxed:.6e}, which is nearer the other target's "
            f"{value_a:.6e} than this one's {value_b:.6e}",
        )
        claims.require(
            abs(sandboxed - value_b) <= SANDBOX_TOLERANCE_FRACTION * separation,
            5,
            f"the sandbox read {sandboxed:.6e}, further from this target's {value_b:.6e} than "
            f"{SANDBOX_TOLERANCE_FRACTION:.0%} of the distance between the two targets",
        )
        audit.ok(
            5,
            manager.active_target(),
            f"sandbox read {sandboxed:.6e} — this target's value B, not the other target's "
            f"{value_a:.6e}; routing followed the session into another process",
        )

        # -- 6. a write, on the target the session is on -----------------------
        # What the channel reads BEFORE the write is what the settle pass puts
        # back — a demo restores the machine it found, not a value this file
        # decided was normal. Captured before the write for the obvious reason,
        # and recorded on the ledger so a failure between here and the end still
        # gets the value disclosed.
        ledger.original = await _read_one(posture.write_channel, 6, claims)
        audit.step(
            6,
            manager.active_target(),
            f"writing {posture.write_value} to {posture.write_channel} through channel_write "
            f"(it reads {ledger.original:.6e} here now)",
        )
        ledger.attempted = True
        written = json.loads(
            await _tool(channel_write_tool.channel_write)(
                operations=[{"channel": posture.write_channel, "value": posture.write_value}]
            )
        )
        claims.require(
            written.get("status") == "success",
            6,
            f"the write did not succeed: {json.dumps(written)[:400]}",
        )
        result = written["summary"]["results"][0]
        claims.require(
            result["write_state"] not in ("blocked", "write_failed"),
            6,
            f"the write was {result['write_state']}: {json.dumps(result)[:400]}",
        )
        readback = await _read_one(posture.write_channel, 6, claims)
        ledger.landed = True
        claims.require(
            abs(readback - posture.write_value) <= max(1e-9, abs(posture.write_value) * 0.01),
            6,
            f"{posture.write_channel} reads {readback:.6e} on this target after a write of "
            f"{posture.write_value}",
        )
        audit.ok(
            6,
            manager.active_target(),
            f"write_state={result['write_state']!r}; {posture.write_channel} now reads "
            f"{readback:.6e} on this target",
        )

        # -- 7. switch back, and the write is not there -------------------------
        audit.step(
            7,
            manager.active_target(),
            f"switching back to {baseline!r} and reading {posture.write_channel} there",
        )
        payload = json.loads(await _tool(control_target_tools.control_target_set)(target=baseline))
        claims.require(
            payload.get("status") == "success",
            7,
            f"the switch home did not succeed: {json.dumps(payload)[:400]}",
        )
        other = await _read_one(posture.write_channel, 7, claims)
        claims.require(
            abs(other - posture.write_value) > max(1e-9, abs(posture.write_value) * 0.01),
            7,
            f"{posture.write_channel} reads {other:.6e} HERE too — the value written on the "
            f"other target is visible on this one, so the two targets are not isolated",
        )
        audit.ok(
            7,
            manager.active_target(),
            f"generation {payload['summary']['generation']}; {posture.write_channel} reads "
            f"{other:.6e} here against {readback:.6e} on the other target — the write never "
            f"reached this machine",
        )

        # -- 8. the roster the operator is left with ---------------------------
        audit.step(8, manager.active_target(), "asking the target roster after the round trip")
        roster = json.loads(await _tool(control_target_tools.control_target)())
        summary = roster["summary"]
        claims.require(
            summary["target"] == baseline,
            8,
            f"the session ended on {summary['target']!r}, not on {baseline!r}",
        )
        claims.require(
            summary["generation"] == 2,
            8,
            f"two switches left the session at generation {summary['generation']}",
        )
        claims.require(
            summary["connector_host_alive"],
            8,
            "the session has no connector host after the round trip",
        )
        rows = roster["access_details"]["targets"]
        audit.ok(
            8,
            manager.active_target(),
            f"generation {summary['generation']}, baseline {summary['baseline_target']!r}, "
            f"host alive; rows name both targets: "
            + ", ".join(
                f"{name}(active={row['active']}, available_now={row['available_now']}, "
                f"{_reason_of(row)}, endpoint={_endpoint_of(row)})"
                for name, row in sorted(rows.items())
            ),
        )

    finally:
        # Every exit path, including a failure in the middle of step 6: the
        # machine is either put back or the leftover is named. Silence is the
        # one outcome that is never acceptable, which is why this runs here and
        # not at the end of the happy path.
        await _settle(manager, posture, ledger, audit)
        with contextlib.suppress(Exception):
            await manager.shutdown()
        server_context_mod.reset_server_context()


def _reason_of(row: dict[str, Any]) -> str:
    """A roster row's verdict in words an operator can act on.

    An eligible row carries no reason at all — there is nothing standing in the
    way — and printing the absence as ``reason=None`` would read like a missing
    answer rather than like a clear one.
    """
    reason = row.get("reason")
    return f"reason={reason}" if reason else "no blocker"


def _endpoint_of(row: dict[str, Any]) -> str:
    """``host:port`` of the gateway role a row's target would select."""
    endpoints = row.get("endpoints") or {}
    selected = endpoints.get(row.get("selected_role")) or {}
    host, port = selected.get("host"), selected.get("port")
    return f"{host}:{port}" if host else "unconfigured"


def _separation_caveat(posture: Posture, value_a: float, value_b: float) -> str:
    """Say when two distinct readings are too close together to mean much.

    Only for postures this demo did not seed. Two endpoints of a real facility
    can differ in the last digits for reasons that have nothing to do with being
    different machines — noise, a filter, a different update phase — so a tiny
    separation is reported as distinct (it is) without being dressed up as
    evidence (it is not). Step 7 is where the isolation claim is actually made,
    and nothing about jitter can produce that result.
    """
    if posture.expected_separation is not None:
        return ""
    scale = max(abs(value_a), abs(value_b))
    relative = abs(value_b - value_a) / scale if scale else float("inf")
    if relative >= MIN_RELATIVE_SEPARATION:
        return ""
    return (
        f" — CAVEAT: that is {relative:.2e} of the reading, below the "
        f"{MIN_RELATIVE_SEPARATION:.0e} this demo treats as meaningful, so at this magnitude "
        f"jitter could produce it; the isolation claim rests on step 7, not on this line"
    )


#: What the settle pass owes the operator, decided before anything is imported
#: or called. Three outcomes, three sentences — never silence.
SETTLE_NOTHING = "nothing"
SETTLE_DISCLOSE = "disclose"
SETTLE_RESTORE = "restore"


def settle_action(posture: Posture, ledger: WriteLedger) -> str:
    """Whether the machine must be put back, merely disclosed, or left alone.

    Nothing was written — nothing to say. Written to an endpoint this demo
    provisioned and is about to destroy — restoring it would be theatre, so the
    trail discloses the value instead. Written to somebody's own deployment —
    put it back.
    """
    if not ledger.attempted:
        return SETTLE_NOTHING
    return SETTLE_DISCLOSE if posture.self_provisioned else SETTLE_RESTORE


def leftover_sentence(posture: Posture, ledger: WriteLedger) -> str:
    """What is on the machine and what it read before — the disclosure itself.

    Deliberately says "may have reached" for a write whose readback never
    happened: the demo does not know that one landed, and a disclosure that
    overstated its certainty would send an operator looking for the wrong thing.
    """
    state = "landed on" if ledger.landed else "may have reached"
    return (
        f"{ledger.channel} {state} 'va' at {posture.write_value} and was NOT restored; "
        f"it read {_number(ledger.original)} before this demo wrote to it — put it back by hand"
    )


def scratch_disclosure(posture: Posture, ledger: WriteLedger) -> str:
    """The disclosure for an endpoint that is about to be removed."""
    state = "landed on" if ledger.landed else "may have reached"
    return (
        f"{ledger.channel} {state} the 'va' endpoint at {posture.write_value} and is NOT "
        f"restored: this demo provisioned that container and removes it on the way out"
    )


async def _settle(manager: Any, posture: Posture, ledger: WriteLedger, audit: Audit) -> None:
    """Put the machine back, or say exactly what was left on it. Never neither.

    Runs from the demo's ``finally``, so it also covers a failure that lands
    between the write and the end of the run — the case where an operator most
    needs to be told that a setpoint is not where they left it.

    The restore is as governed as the write it undoes: same ``channel_write``
    tool, so the same limits database and the same readback verification apply,
    and the value put back is the one step 6 read off the machine rather than a
    constant this file chose. A raw connector write would be a demo asking for
    an exemption from the posture it just finished demonstrating.

    What to do is decided by :func:`settle_action` before anything is imported,
    which is what lets the decision table be exercised without a control system
    anywhere near it.
    """
    action = settle_action(posture, ledger)
    if action == SETTLE_NOTHING:
        return
    if action == SETTLE_DISCLOSE:
        audit.note(scratch_disclosure(posture, ledger))
        return

    from osprey.mcp_server.control_system.tools import channel_write as channel_write_tool
    from osprey.mcp_server.control_system.tools import control_target as control_target_tools

    unrestored = leftover_sentence(posture, ledger)
    try:
        if manager.active_target() != "va":
            await _tool(control_target_tools.control_target_set)(target="va")
        if manager.active_target() != "va" or not manager.has_child():
            audit.note(f"RESTORE FAILED: could not get a connector host on 'va'. {unrestored}")
            return
        if ledger.original is None:  # pragma: no cover - the pre-read failed
            audit.note(f"RESTORE SKIPPED: nothing read the original value. {unrestored}")
            return
        restored = json.loads(
            await _tool(channel_write_tool.channel_write)(
                operations=[{"channel": ledger.channel, "value": ledger.original}]
            )
        )
        result = restored.get("summary", {}).get("results", [{}])[0]
        if restored.get("status") != "success" or result.get("write_state") in (
            "blocked",
            "write_failed",
        ):
            audit.note(f"RESTORE FAILED: write_state={result.get('write_state')!r}. {unrestored}")
            return
        audit.note(
            f"restored {ledger.channel} to {_number(ledger.original)} on 'va' through "
            f"channel_write (write_state={result.get('write_state')!r})"
        )
        with contextlib.suppress(Exception):
            await _tool(control_target_tools.control_target_set)(target=manager.baseline)
            audit.note(f"session returned to {manager.baseline!r}")
    except Exception as exc:  # noqa: BLE001 - a failed restore must still disclose
        audit.note(f"RESTORE FAILED ({type(exc).__name__}: {exc}). {unrestored}")


def _number(value: float | None) -> str:
    return "an unread value" if value is None else f"{value:.6e}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo_target_switch.py",
        description=(
            "Drive one session across two control targets and print an audit trail. "
            "Name-server mode only; the UDP address-list path is covered by the unit suites."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--self-provision",
        action="store_true",
        help=(
            "stand the demo posture up locally: two virtual-accelerator containers on "
            "ephemeral Channel Access ports, one of them seeded so the two serve distinct "
            "probe values"
        ),
    )
    source.add_argument(
        "--config",
        type=Path,
        help="demo against a deployment's rendered config.yml (endpoints, acknowledgment and "
        "probe channels are read from it)",
    )
    parser.add_argument(
        "--probe-channel",
        default=None,
        help=f"channel compared across the two targets (default: the config's own "
        f"probe_channel, or {DEFAULT_PROBE_CHANNEL} when self-provisioning)",
    )
    parser.add_argument(
        "--write-channel",
        default=None,
        help=f"setpoint written on the active target to prove the two are isolated "
        f"(required with --config; default {DEFAULT_WRITE_CHANNEL} when self-provisioning). "
        f"It must be listed in the deployment's limits database.",
    )
    parser.add_argument(
        "--write-value",
        type=float,
        default=DEFAULT_WRITE_VALUE,
        help="value written on the 'va' target. What it is restored to is not an argument: "
        "the demo reads the channel first and puts THAT back.",
    )
    parser.add_argument(
        "--ack",
        default=os.environ.get("OSPREY_DEMO_LIVE_GATEWAY_ACK"),
        help="operator acknowledgment recorded as control_system.target_switch."
        "live_gateway_acknowledged when self-provisioning (default: this run's own "
        "soft-IOC endpoint)",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="container image to self-provision")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without touching containers or the control system",
    )
    return parser


def planned_channels(arguments: argparse.Namespace) -> tuple[str, str]:
    """The channel names the plan may honestly print.

    A self-provisioned run knows both, because this file chose them. An operator
    posture knows only what was passed: the probe channel comes out of *their*
    config at run time and the write channel is theirs to name, so the plan says
    where each will come from instead of printing a channel from the tutorial
    lattice as though it were theirs.
    """
    if arguments.self_provision:
        return (
            arguments.probe_channel or DEFAULT_PROBE_CHANNEL,
            arguments.write_channel or DEFAULT_WRITE_CHANNEL,
        )
    return (
        arguments.probe_channel or PLACEHOLDER_PROBE,
        arguments.write_channel or PLACEHOLDER_WRITE,
    )


def print_plan(arguments: argparse.Namespace, audit: Audit) -> None:
    """The plan a real run would follow, with nothing started."""
    probe, write = planned_channels(arguments)
    source = (
        "a self-provisioned scratch stack (two containers on ephemeral Channel Access ports, "
        f"image {arguments.image}, one seeded with {VA_BPM_ERRORS})"
        if arguments.self_provision
        else f"the deployment configured at {arguments.config}"
    )
    audit.note("DRY RUN: nothing is started, read, written or switched")
    audit.note(f"posture source: {source}")
    audit.note(
        "archiver posture would be ASSERTED FROM CONFIG (archiver.type must not resolve to the "
        "mock archiver); no mongod is booted and no history is read"
    )
    audit.note("transport: name-server mode only; the UDP address-list path is covered by units")
    audit.plan(1, "ask the target roster before anything is switched; 'va' must be eligible")
    audit.plan(2, f"start the connector host on the baseline and read {probe} -> value A")
    audit.plan(3, "switch to 'va' through control_target_set")
    audit.plan(4, f"read {probe} again -> value B, which must differ from A")
    audit.plan(5, f"read {probe} from an execute() sandbox stamped for 'va' -> must equal B")
    audit.plan(6, f"write {arguments.write_value} to {write} on the active target and read it back")
    audit.plan(7, f"switch back to the baseline; {write} must NOT carry the written value there")
    audit.plan(8, "ask the target roster again; it must name both targets and the ending target")
    audit.result_pass()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Refuse an under-specified invocation, before anything is printed or run.

    Checked ahead of the plan as well as ahead of a real run: a dry run that
    happily printed a plan the real run would refuse would be telling the
    operator their invocation is good when it is not.

    Raises:
        RuntimeError: Naming the missing argument and why this file will not
            supply it.
    """
    if arguments.config is not None and not arguments.write_channel:
        raise RuntimeError(
            "--write-channel is required with --config: the demo writes one setpoint on the "
            "'va' target, and nothing here may guess which one is safe on your deployment"
        )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    audit = Audit()

    try:
        validate_arguments(arguments)
    except RuntimeError as refusal:
        audit.note(str(refusal))
        audit.result_fail(0, str(refusal))
        return 1

    if arguments.dry_run:
        print_plan(arguments, audit)
        return 0

    try:
        with contextlib.ExitStack() as stack:
            posture = _resolve_posture(arguments, audit, stack)
            # Every path that resolves the agent-data root goes through the
            # config, so the demo's own state file lands in the deployment it is
            # demonstrating — and the sandbox subprocess, which reads the same
            # file, resolves the same directory.
            os.environ["OSPREY_CONFIG"] = str(posture.config_path)
            from osprey_connectors.workspace import reset_config_cache

            reset_config_cache()
            asyncio.run(run_demo(posture, audit))
    except StepFailed as failure:
        audit.failed(failure.number, failure.target, failure.message)
        audit.result_fail(failure.number, failure.message, failure.target)
        return 1
    except Exception as exc:  # noqa: BLE001 - the demo reports, it does not traceback at operators
        audit.note(f"the demo could not run to completion: {type(exc).__name__}: {exc}")
        audit.result_fail(0, f"{type(exc).__name__}: {exc}")
        return 1

    audit.result_pass()
    return 0


def _resolve_posture(
    arguments: argparse.Namespace, audit: Audit, stack: contextlib.ExitStack
) -> Posture:
    """Obtain the deployment to demo against, standing one up if asked to."""
    if arguments.config is not None:
        config_path = arguments.config.expanduser().resolve()
        if not config_path.is_file():
            raise RuntimeError(f"no config at {config_path}")
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        probe = arguments.probe_channel or probe_channel_from_config(raw, "virtual_accelerator")
        if not probe:
            raise RuntimeError(
                "this deployment configures no probe_channel for the virtual accelerator; "
                "name the channel to compare with --probe-channel"
            )
        audit.note(f"posture read from {config_path}")
        return Posture(
            config_path=config_path,
            raw=raw,
            probe_channel=probe,
            write_channel=arguments.write_channel,
            write_value=arguments.write_value,
            self_provisioned=False,
        )

    probe = arguments.probe_channel or DEFAULT_PROBE_CHANNEL
    _require_image(arguments.image)
    work_dir = Path(stack.enter_context(_temporary_directory()))
    va_port = stack.enter_context(
        _serving(
            CONTAINER_PREFIX_VA,
            image=arguments.image,
            seeded=False,
            probe_channel=probe,
            audit=audit,
        )
    )
    live_port = stack.enter_context(
        _serving(
            CONTAINER_PREFIX_LIVE,
            image=arguments.image,
            seeded=True,
            probe_channel=probe,
            audit=audit,
        )
    )
    acknowledgment = arguments.ack or f"localhost:{live_port}"
    raw = scratch_config(
        va_port=va_port,
        live_port=live_port,
        probe_channel=probe,
        acknowledgment=acknowledgment,
        project_root=work_dir,
    )
    config_path = work_dir / "config.yml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    audit.note(f"scratch posture written to {config_path}")
    audit.note(
        f"operator acknowledgment control_system.target_switch.live_gateway_acknowledged="
        f"{acknowledgment!r} — this run's own soft-IOC, not a facility gateway"
    )
    write_channel = arguments.write_channel or DEFAULT_WRITE_CHANNEL
    audit.note(limits_note(probe, write_channel))
    return Posture(
        config_path=config_path,
        raw=raw,
        probe_channel=probe,
        write_channel=write_channel,
        write_value=arguments.write_value,
        self_provisioned=True,
        expected_separation=SEEDED_OFFSET_X,
    )


@contextlib.contextmanager
def _temporary_directory():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="osprey-demo-target-switch-") as directory:
        yield directory


if __name__ == "__main__":
    raise SystemExit(main())
