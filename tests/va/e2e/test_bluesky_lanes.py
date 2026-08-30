"""Two Bluesky plan lanes, deployed for real, one per control-system target.

A **plan lane** is a whole Bluesky stack -- bridge, RE Manager, its own Redis --
wired at RENDER time to one control target. Every deployment built before this
feature has exactly one; a build profile that sets ``bluesky.second_lane``
renders two, so a session switched off the deployment baseline still has a lane
to queue plans on. The unit suites (``tests/services/test_lane_queue_binding.py``,
``test_lane_capability.py``, ``test_single_lane_switch_refusal.py``) already pin
the routing algebra against fake bridges, and ``tests/deployment/
test_lane_compose.py`` pins what the template renders. Nothing there runs a
container, and so nothing there can answer the four questions this file exists
for:

* do two lanes actually COME UP side by side out of ``osprey init`` ->
  ``osprey build`` -> ``osprey up``, or does the second one collide with the
  first over a port, a Redis keyspace, a network or a container name;
* does a queued PLAN land in -- and drain on -- the lane ``queue_add`` bound it
  to, with the OTHER lane's manager untouched;
* is each lane's ``Capability`` really static across a session target switch,
  with only the host's composed ``active`` field moving;
* is the per-lane secret material that ``osprey up`` mints genuinely separate,
  so neither lane's bridge holds anything that would authenticate it to the
  other lane's manager.

What "live" means here
----------------------
The second lane serves the ``live`` target, and its addressing is required
rather than defaulted (``EPICS_CA_NAME_SERVERS`` is spelled ``${VAR:?}`` in the
rendered compose). This suite has no facility to point it at, so the live lane's
gateway is a SECOND ``osprey-va-full`` container this module boots -- a real,
independent Channel Access endpoint on its own port, reached from inside the
lane's containers as ``host.docker.internal:<port>``. That is honest about what
is proven: the lane axis is exercised against two genuinely separate control
endpoints, and nothing here claims to have talked to real hardware.

Layers, stated plainly
----------------------
Three different layers are exercised, and each test says which one it is on:

* **Containers.** The two-lane deployment and everything queued through it.
* **Compose interpolation.** The ``${EPICS_CA_NAME_SERVERS:?}`` fail-fast is
  asserted at ``docker compose config`` -- the same interpolation pass
  ``osprey up`` performs before it starts anything. It is NOT a container-start
  assertion, because there is no container to start once interpolation refuses.
* **Render.** The single-lane regression rebuilds the same project with the lane
  axis off and asserts the rendered artifacts grew no second lane. Its behavioral
  half is owned by ``tests/services/test_single_lane_switch_refusal.py`` and is
  deliberately not restated here.

CURVE isolation is asserted in its OBSERVATIONAL form, not by driving raw ZMQ:
this suite proves the minted material differs per lane and that each running
container carries only its own lane's keys and certificate mount. Standing up a
hand-rolled CurveZMQ client against a deployed manager would be a second,
untested implementation of the client the stack already ships, and a green
result from it would say more about the probe than about the deployment.

Cost, and why the fixtures are module-scoped
--------------------------------------------
One ``osprey up`` backs every test here; a per-test deploy would take hours. The
whole module skips cleanly without ``OSPREY_VA_E2E_ENABLE=1`` (the directory
conftest's gate), without docker, and without the ``osprey-va-full`` image.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.bluesky_bridge_connection import LANE_ONE, SECOND_LANE_KEYS
from osprey.mcp_server.bluesky import lanes as lanes_module
from osprey.mcp_server.bluesky.server_context import (
    initialize_server_context,
    reset_server_context,
)
from osprey.mcp_server.bluesky.tools import queue as queue_tools
from osprey.mcp_server.control_system import target_state
from osprey.utils.workspace import reset_config_cache
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn
from tests.va.e2e import conftest as e2e_conftest

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker not available"),
]

# -- the deployment ---------------------------------------------------------

#: The repo directory name IS the deployment name, and every container and
#: locally-built image the compose templates render is prefixed with it.
PROJECT_NAME = "lane-e2e"

#: The lane whose target is the deployment baseline (this module pins
#: ``control_system.type`` to the virtual accelerator, see ``_override_yaml``),
#: and the lane the opt-in renders beside it. Imported rather than spelled, so
#: a rename of the service keys fails here instead of drifting.
LANE_VA = LANE_ONE
LANE_LIVE = SECOND_LANE_KEYS["live"]

#: Host ports. Every one is moved off its preset default, because this suite
#: has to be able to run beside an already-deployed stack on a shared dev
#: machine -- a bound-port collision aborts `osprey up` before it creates a
#: single container, which would read as a lane bug rather than as host state.
#: Lane 2's bridge port is DERIVED (``bluesky.port + 100``), never configured.
BRIDGE_PORT = 18490
SECOND_BRIDGE_PORT = BRIDGE_PORT + 100
TILED_PORT = 18491
PANELS_PORT = 18496
VA_CA_PORT = 15264
POSTGRES_PORT = 25932
OPENOBSERVE_PORT = 25980
QMD_PORT = 28180
MONGODB_PORT = 47917

BRIDGE_URLS = {
    LANE_VA: f"http://127.0.0.1:{BRIDGE_PORT}",
    LANE_LIVE: f"http://127.0.0.1:{SECOND_BRIDGE_PORT}",
}

#: Env-var prefix each lane's own secrets and derived settings are minted under
#: in ``.env`` -- the contract ``container_lifecycle`` mints by, the compose
#: template interpolates by, and ``bluesky_bridge_connection`` resolves by.
LANE_ENV_PREFIX = {LANE_VA: "BLUESKY", LANE_LIVE: "BLUESKY_LIVE"}

#: Container-name prefix of the live lane's Channel Access endpoint. The run's
#: own ephemeral port is appended: a fixed name would be mutually destructive
#: between two concurrent runs (each force-removes its name as stale cleanup),
#: which is the convention the rest of this directory already follows.
LIVE_ENDPOINT_PREFIX = "osprey-va-e2e-lane-live"

#: Every container the two-lane deployment must produce, as compose service
#: names under the project prefix. Six of them are the lane axis itself: two
#: bridges, two managers, two Redis instances.
LANE_CONTAINERS = {
    LANE_VA: (
        f"{PROJECT_NAME}-bluesky-bridge",
        f"{PROJECT_NAME}-bluesky-queueserver",
        f"{PROJECT_NAME}-bluesky-redis",
    ),
    LANE_LIVE: (
        f"{PROJECT_NAME}-bluesky-live-bridge",
        f"{PROJECT_NAME}-bluesky-live-queueserver",
        f"{PROJECT_NAME}-bluesky-live-redis",
    ),
}

# -- bounds -----------------------------------------------------------------

#: `osprey up` on a first deploy builds the bridge and virtual-accelerator
#: images from the local checkout, and the VA image is pinned ``linux/amd64``
#: so a local build on Apple Silicon is emulated (see ``conftest.py``).
BUILD_TIMEOUT_S = 900
DEPLOY_UP_TIMEOUT_S = 3600
DEPLOY_DOWN_TIMEOUT_S = 600

#: The manager waits on Redis and imports the bluesky/ophyd stack before its
#: control socket answers, so a lane is not ready when its bridge binds.
BRIDGE_HEALTH_TIMEOUT_S = 300.0
#: Boot of the live lane's Channel Access endpoint. Generous for the same
#: emulation reason the directory conftest gives.
LIVE_ENDPOINT_BOOT_TIMEOUT_S = 240.0
#: Bound on one queued PLAN draining. The sweep below is deliberately tiny.
QUEUE_DRAIN_TIMEOUT_S = 300.0
HTTP_TIMEOUT_S = 30.0

#: Points in the one PLAN this module queues. Small on purpose: what is under
#: test is WHICH lane ran it, not how it scanned.
PLAN_POINTS = 3


# ---------------------------------------------------------------------------
# Process + HTTP helpers
# ---------------------------------------------------------------------------
def _osprey_bin() -> Path:
    candidate = Path(sys.executable).parent / "osprey"
    if candidate.exists():
        return candidate
    found = shutil.which("osprey")
    if found:
        return Path(found)
    raise RuntimeError("Could not locate the 'osprey' console script.")


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run one ``osprey`` command in *cwd*, with this checkout on the path.

    ``PYTHONPATH`` is made ABSOLUTE before it is handed on, and that is
    load-bearing rather than tidy: every command here runs with ``cwd`` set to a
    scratch directory, so a relative entry (which is how this suite is normally
    invoked -- ``PYTHONPATH=src:packages/...``) would resolve against that
    scratch directory and silently hand the child the INSTALLED osprey from the
    venv instead of this worktree's. The deployment would then be built by
    another checkout's code and every assertion below would be about the wrong
    source tree.
    """
    raw = os.environ.get("PYTHONPATH", "")
    absolute = os.pathsep.join(
        str(Path(entry).resolve()) for entry in raw.split(os.pathsep) if entry
    )
    env = {**os.environ, "CLAUDECODE": ""}
    if absolute:
        env["PYTHONPATH"] = absolute
    return subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env
    )


def _docker(*args: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _request(
    base: str,
    path: str,
    method: str = "GET",
    body: dict | None = None,
    timeout: float | None = None,
) -> tuple[int, Any]:
    """One HTTP call to a lane's bridge, returning ``(status, parsed body)``.

    Refusals carry a JSON body and a non-2xx status, and both halves matter --
    a status-code-only assertion passes while the refusal code drifts.
    """
    payload = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(  # noqa: S310 - loopback only
        f"{base}{path}",
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"} if payload else {},
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - loopback only
            request, timeout=timeout or HTTP_TIMEOUT_S
        ) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw or b"null")
        except ValueError:
            return exc.code, raw.decode(errors="replace")


def _wait_for_bridge(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = "(no response yet)"
    while time.monotonic() < deadline:
        try:
            status, body = _request(base, "/health", timeout=5.0)
        except Exception as exc:  # noqa: BLE001 - any transport failure is "not up yet"
            last = repr(exc)
        else:
            if status == 200:
                return
            last = f"{status} {body}"
        time.sleep(2.0)
    raise RuntimeError(f"{base}/health never answered 200 within {timeout}s (last: {last})")


def _env_values(repo: Path) -> dict[str, str]:
    """Every ``KEY=value`` in the deployment's ``.env``, as a mapping.

    ``.env`` is the deployment's whole secret store and the file every compose
    invocation is pointed at, so it is also where ``osprey up`` leaves the
    per-lane material this module inspects.
    """
    values: dict[str, str] = {}
    for line in (repo / ".env").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


# ---------------------------------------------------------------------------
# The live lane's Channel Access endpoint
# ---------------------------------------------------------------------------
def _served(port: int) -> bool:
    """Whether a virtual accelerator answers on *port*, asked OUT OF PROCESS.

    Never in-process: libca latches ``EPICS_CA_*`` when it initialises and this
    directory's session fixture publishes on a different port, so one process
    cannot be a client of both (the reason the directory conftest's own
    readiness probe is a subprocess).
    """
    code = (
        "import sys, epics\n"
        f"v = epics.caget({e2e_conftest.READINESS_ADDRESS!r}, timeout=1.0, "
        "connection_timeout=1.0)\n"
        "sys.stdout.write('SERVED' if v is not None else 'NONE')\n"
        "sys.stdout.flush()\n"
        "import os; os._exit(0)\n"
    )
    env = {
        **os.environ,
        "EPICS_CA_NAME_SERVERS": f"localhost:{port}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }
    env.pop("EPICS_CA_ADDR_LIST", None)
    env.pop("EPICS_CA_SERVER_PORT", None)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=15, env=env
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.stdout.strip() == "SERVED"


@pytest.fixture(scope="module")
def live_endpoint():
    """A second Channel Access endpoint: the machine the LIVE lane addresses.

    The published port and the server's own port are the same number by
    construction -- a Channel Access search reply carries the server's own port,
    so a remap would hand every client an address nothing answers on. That
    number also names the container.
    """
    if not e2e_conftest.E2E_ENABLED:  # pragma: no cover - collection-time skip path
        pytest.skip("VA e2e disabled")
    inspected = _docker("image", "inspect", e2e_conftest.IMAGE, timeout=60)
    if inspected.returncode != 0:
        pytest.skip(f"{e2e_conftest.IMAGE} is not built on this host")

    port = _free_port()
    name = f"{LIVE_ENDPOINT_PREFIX}-{port}"
    # Stale-cleanup only: the port is this run's alone, so this can name nothing
    # a concurrent run is using.
    _docker("rm", "-f", name, timeout=60)
    started = _docker(
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"EPICS_CA_SERVER_PORT={port}",
        "-p",
        f"127.0.0.1:{port}:{port}/tcp",
        "-v",
        f"{e2e_conftest.PRESET_SIM_DIR}:/data/simulation:ro",
        e2e_conftest.IMAGE,
        timeout=120,
    )
    if started.returncode != 0:
        pytest.fail(f"docker run failed for the live endpoint: {started.stdout}\n{started.stderr}")

    try:
        deadline = time.monotonic() + LIVE_ENDPOINT_BOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            if _served(port):
                break
            time.sleep(2.0)
        else:
            logs = _docker("logs", "--tail", "40", name, timeout=60)
            pytest.fail(
                f"the live lane's endpoint never served "
                f"{e2e_conftest.READINESS_ADDRESS} within "
                f"{LIVE_ENDPOINT_BOOT_TIMEOUT_S}s.\n{logs.stdout}\n{logs.stderr}"
            )
        yield port
    finally:
        _docker("rm", "-f", name, timeout=60)


# ---------------------------------------------------------------------------
# The two-lane deployment
# ---------------------------------------------------------------------------
def _override_yaml() -> str:
    """Host hygiene, CI sizing, and the VA baseline -- never the lane axis.

    ``dispatch: null`` and ``modules.web_terminals.enabled: false`` drop two
    stacks nothing here touches and both slow to build. The port keys move
    services the preset deploys unconditionally, with no profile knob, off
    defaults a locally-running stack routinely holds. ``va_archiver`` is shrunk
    to a CI-sized archive -- a sizing change, not a behavioral one: the store
    and its recorder still deploy and still record.

    ``control_system.type`` is pinned to the virtual accelerator because this
    module's lane pair is the VA/live one: the preset baselines on its live
    stand-in, whose second lane is the VA rather than the live lane, and the
    axis under test here is precisely the lane that demands a facility gateway.
    The pin selects the baseline whose pair carries that lane; it does not
    paper over anything about the preset.

    Flat dotted keys under ``config:`` (the preset's own convention): a ``--set``
    would build a NESTED dict for every dotted segment and replace whole blocks.
    """
    return (
        "config:\n"
        "  control_system.type: virtual_accelerator\n"
        "  claude_code.servers.bluesky.enabled: true\n"
        "  modules.web_terminals.enabled: false\n"
        f"  services.postgresql.port_host: {POSTGRES_PORT}\n"
        f"  services.openobserve.port: {OPENOBSERVE_PORT}\n"
        f"  services.qmd.port: {QMD_PORT}\n"
        "dispatch: null\n"
        "va_archiver:\n"
        "  retention_days: 2\n"
        "  hot_span_hours: 2\n"
        f"  port_host: {MONGODB_PORT}\n"
    )


def _init_and_build(base: Path, name: str, *, second_lane: bool) -> Path:
    """``osprey init`` + ``osprey build`` one project; return its repo path.

    Two steps because the surface has two: ``init`` writes the repo's source
    zone from the preset, ``build`` renders ``build/`` from it. The only
    difference between this module's two projects is ``bluesky.second_lane``,
    which is the whole point -- the single-lane regression must differ from the
    two-lane deployment in exactly one profile key.
    """
    repo = base / name
    override = base / f"override-{name}.yml"
    override.write_text(_override_yaml(), encoding="utf-8")

    argv = [
        str(_osprey_bin()),
        "init",
        str(repo),
        "--preset",
        "control-assistant",
        "--no-git",
        "--override",
        str(override),
        "--set",
        f"virtual_accelerator.port={VA_CA_PORT}",
        "--set",
        f"bluesky.port={BRIDGE_PORT}",
        "--set",
        f"bluesky.tiled_port={TILED_PORT}",
        "--set",
        f"bluesky_web.port={PANELS_PORT}",
        # This module's own thousand-port block (see
        # test_dispatch_deploy.py's 20700 note): everything not pinned
        # explicitly follows it instead of landing on a real deployment's
        # default 10000 block.
        "--set",
        "port_base=22100",
    ]
    if second_lane:
        argv += ["--set", "bluesky.second_lane=true"]

    init = _run(argv, cwd=base, timeout=BUILD_TIMEOUT_S)
    if init.returncode != 0:
        pytest.fail(f"osprey init failed (rc={init.returncode}):\n{init.stdout}\n{init.stderr}")

    build = _run(
        [
            str(_osprey_bin()),
            "build",
            "--repo",
            str(repo),
            "--skip-deps",
            "--skip-lifecycle",
            "--dev",
        ],
        cwd=base,
        timeout=BUILD_TIMEOUT_S,
    )
    if build.returncode != 0:
        pytest.fail(f"osprey build failed (rc={build.returncode}):\n{build.stdout}\n{build.stderr}")
    return repo


@dataclass
class LaneStack:
    """Everything the tests need about the one two-lane deployment."""

    repo: Path
    live_port: int
    env: dict[str, str]
    #: ``{lane: {axis name: (setpoint address, readback address)}}`` -- derived
    #: per lane by ``osprey up`` into ``<PREFIX>_EPICS_SETPOINTS``.
    correctors: dict[str, dict[str, tuple[str, str]]]
    bpms: dict[str, dict[str, str]]
    limits: dict[str, Any]

    @property
    def config_path(self) -> Path:
        return self.repo / "build" / "config.yml"


def _parse_pairs(raw: str) -> dict[str, tuple[str, str]]:
    """``name=SP|RB,...`` back into ``{name: (setpoint, readback)}``."""
    out: dict[str, tuple[str, str]] = {}
    for chunk in raw.split(","):
        if "=" not in chunk:
            continue
        name, _, addresses = chunk.partition("=")
        setpoint, _, readback = addresses.partition("|")
        out[name.strip()] = (setpoint.strip(), readback.strip())
    return out


def _parse_singles(raw: str) -> dict[str, str]:
    """``name=ADDRESS,...`` back into ``{name: address}``."""
    out: dict[str, str] = {}
    for chunk in raw.split(","):
        if "=" not in chunk:
            continue
        name, _, address = chunk.partition("=")
        out[name.strip()] = address.strip()
    return out


def _remove_stale_deployment() -> None:
    """Force-remove any container left behind by a KILLED earlier run.

    The deployment's project name is FIXED (``PROJECT_NAME``) rather than
    per-run, because every container and locally-built image the compose
    templates render is named from it -- so unlike ``live_endpoint``, whose
    ephemeral port makes its container name unique to this run, this name is
    shared by every run of this module.

    That is safe here for a reason worth stating rather than assuming: two
    concurrent runs of this module were never possible in the first place, since
    they would collide on the fixed host ports above and ``osprey up``'s port
    preflight would abort the second one before it created anything. So this can
    only ever name a container whose run has already ended -- normally none,
    because the ``stack`` fixture's ``finally`` runs ``osprey down``, but that
    teardown does not run if a previous pytest process was SIGKILLed. Without
    this, such a leftover holds the bridge ports and wedges every later run.

    Best-effort by construction: it runs before anything is built, and a docker
    that answers nothing here should surface as the real failure later, not as a
    cleanup error now.
    """
    listed = _docker("ps", "-aq", "--filter", f"name=^{PROJECT_NAME}-", timeout=60).stdout.split()
    if listed:
        _docker("rm", "-f", *listed, timeout=180)


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory, live_endpoint: int):
    """Init + build + ``osprey up --dev`` the TWO-LANE deployment; tear it down.

    The live lane's ``EPICS_CA_NAME_SERVERS`` is appended to ``.env`` before
    ``up``: the rendered compose spells it ``${VAR:?}``, so a deployment that
    did not supply it would refuse to start at all -- which is the contract
    ``test_live_lane_refuses_to_start_without_its_addressing`` asserts against
    this very file.
    """
    base = tmp_path_factory.mktemp("bluesky_lane_e2e")
    _remove_stale_deployment()
    repo = _init_and_build(base, PROJECT_NAME, second_lane=True)

    env_path = repo / ".env"
    if not env_path.exists():
        shutil.copy(repo / ".env.example", env_path)
    with env_path.open("a", encoding="utf-8") as handle:
        handle.write(f"EPICS_CA_NAME_SERVERS=host.docker.internal:{live_endpoint}\n")

    try:
        up = _run([str(_osprey_bin()), "up", "-d", "--dev"], cwd=repo, timeout=DEPLOY_UP_TIMEOUT_S)
        if up.returncode != 0:
            pytest.fail(f"osprey up --dev failed (rc={up.returncode}):\n{up.stdout}\n{up.stderr}")

        for lane, base_url in BRIDGE_URLS.items():
            try:
                _wait_for_bridge(base_url, BRIDGE_HEALTH_TIMEOUT_S)
            except RuntimeError as exc:
                logs = _docker("logs", "--tail", "60", LANE_CONTAINERS[lane][0], timeout=60)
                pytest.fail(f"{exc}\n--- {lane} bridge logs ---\n{logs.stdout}\n{logs.stderr}")

        env = _env_values(repo)
        limits = json.loads(
            (repo / "build" / "data" / "channel_limits.json").read_text(encoding="utf-8")
        )
        yield LaneStack(
            repo=repo,
            live_port=live_endpoint,
            env=env,
            correctors={
                lane: _parse_pairs(env.get(f"{prefix}_EPICS_SETPOINTS", ""))
                for lane, prefix in LANE_ENV_PREFIX.items()
            },
            bpms={
                lane: _parse_singles(env.get(f"{prefix}_EPICS_READBACKS", ""))
                for lane, prefix in LANE_ENV_PREFIX.items()
            },
            limits=limits,
        )
    finally:
        down = _run([str(_osprey_bin()), "down"], cwd=repo, timeout=DEPLOY_DOWN_TIMEOUT_S)
        if down.returncode != 0:
            print(  # noqa: T201 - surface teardown issues in the run log
                f"osprey down rc={down.returncode}\n{down.stdout}\n{down.stderr}"
            )
        # `osprey down` keeps volumes by design; drop this project's own, or a
        # rerun inherits the Redis-backed queue and history this one left.
        from osprey.deployment.compose_generator import resolve_project_name
        from tests.e2e._volumes import remove_project_volumes

        remove_project_volumes(resolve_project_name({"project_name": PROJECT_NAME}))


# ---------------------------------------------------------------------------
# Host-side session state and the MCP tools
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def session_state(tmp_path, monkeypatch):
    """Anchor the target-state directory in ``tmp_path``, not a real deployment.

    The session target is host state: it lives in a file the controls MCP server
    writes, outside every bridge container. This module is not that server, so
    it writes the record itself -- with ``owner_ppid`` set to this process's own
    parent, which is what ``target_banner.resolve_session_target`` matches on.
    """
    root = tmp_path / "var" / "agent_data"
    (root / target_state.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)

    def switch_session_to(target: str, *, generation: int = 0) -> None:
        target_state.write_on_start("va", server_pid=os.getpid(), owner_ppid=os.getppid())
        if target != "va":
            target_state.publish_switch(target, generation + 1, server_pid=os.getpid())

    yield switch_session_to
    target_state.delete_on_shutdown(server_pid=os.getpid())


@pytest.fixture
def bluesky_tools(stack: LaneStack, monkeypatch):
    """The real host-side Bluesky MCP tools, pointed at the real deployment.

    Lane discovery reads the rendered ``services.<lane>`` blocks the build
    actually wrote, and the per-lane launch tokens come from the ``.env``
    ``osprey up`` minted -- the same two sources a deployed MCP server reads.

    ``OSPREY_CONFIG`` is the variable that repoints the framework's config
    loader (``osprey_connectors.workspace.resolve_config_path``); ``CONFIG_FILE``
    is deliberately NOT read there, and is set alongside it only because the
    limits database resolves relative to it. Setting one and not the other is
    the trap this comment exists to stop: with only ``CONFIG_FILE`` set, lane
    discovery silently reads whatever config the CWD happens to hold and reports
    a SINGLE lane, which every assertion below would then be making about the
    wrong deployment.
    """
    monkeypatch.setenv("OSPREY_CONFIG", str(stack.config_path))
    monkeypatch.setenv("CONFIG_FILE", str(stack.config_path))
    for lane, prefix in LANE_ENV_PREFIX.items():
        monkeypatch.setenv(f"{prefix}_BRIDGE_URL", BRIDGE_URLS[lane])
        token = stack.env.get(f"{prefix}_LAUNCH_TOKEN")
        if token:
            monkeypatch.setenv(f"{prefix}_LAUNCH_TOKEN", token)

    # The rendered config is cached process-wide; this fixture repoints the
    # loader at the deployment, so the cache has to be dropped on the way in
    # AND on the way out or a later test reads this deployment's config.
    reset_config_cache()
    initialize_server_context()
    try:
        yield queue_tools
    finally:
        reset_server_context()
        reset_config_cache()


def _tool_result(raw: str) -> dict:
    parsed = json.loads(raw)
    assert isinstance(parsed, dict), f"tool returned a non-object: {parsed!r}"
    return parsed


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------
#: Makes each staged draft a genuinely DIFFERENT plan from the last.
#:
#: A draft revision is consumable exactly once (the bridge refuses a re-add with
#: ``draft_revision_already_launched``), and ``PATCH /draft`` with args identical
#: to the ones already staged leaves the draft — and therefore its revision —
#: unchanged. Several tests here stage a plan against one module-scoped
#: deployment, so without this every staging after the first would hand back the
#: SPENT revision and the queue tool would refuse for a reason that has nothing
#: to do with lanes.
_STAGING = itertools.count()


def _plan_args(stack: LaneStack, lane: str) -> dict[str, Any]:
    """Minimal ``grid_scan`` args for one lane: one corrector axis, one readback.

    The sweep band is the middle half of the corrector's OWN
    ``channel_limits.json`` entry, so nothing here hardcodes a facility channel
    or asks for a value outside the band the deployment declares. Each call
    shifts that band by a sub-percent, band-relative nudge (see
    :data:`_STAGING`) — enough to make the draft new, far too little to change
    what any test asserts about it, and bounded so the sweep never leaves the
    middle of the declared band however many times this is called.
    """
    correctors = stack.correctors[lane]
    bpms = stack.bpms[lane]
    assert correctors, f"osprey up wrote no setpoints for lane {lane!r}"
    assert bpms, f"osprey up wrote no readbacks for lane {lane!r}"
    axis_name = next(iter(correctors))
    setpoint_address, _readback = correctors[axis_name]
    entry = stack.limits[setpoint_address]
    low, high = float(entry["min_value"]), float(entry["max_value"])
    span = high - low
    nudge = (next(_STAGING) % 10) * 0.005 * span
    return {
        "readbacks": [next(iter(bpms))],
        "axes": [
            {
                "setpoint": axis_name,
                "start": low + 0.375 * span + nudge,
                "stop": low + 0.625 * span + nudge,
                "num_points": PLAN_POINTS,
            }
        ],
    }


#: The revision each lane's bridge last handed back, so a staging that failed to
#: mint a new one is caught HERE rather than three calls later as an unrelated
#: ``draft_revision_already_launched`` from the queue tool.
_LAST_REVISION: dict[str, int] = {}


def _patch_draft(lane: str, args: dict[str, Any]) -> int:
    """Stage a ``grid_scan`` on ONE lane's bridge; return its fresh draft revision."""
    status, body = _request(
        BRIDGE_URLS[lane],
        "/draft",
        "PATCH",
        {"plan_name": "grid_scan", "plan_args_patch": args, "client_id": "lane-e2e"},
    )
    assert status == 200, f"PATCH /draft on lane {lane!r} failed: {status} {body}"
    revision = body.get("revision")
    assert isinstance(revision, int), f"no integer revision from lane {lane!r}: {body}"
    assert revision != _LAST_REVISION.get(lane), (
        f"lane {lane!r} handed back revision {revision} again; the staged plan was "
        "not new, and a spent revision cannot be queued"
    )
    _LAST_REVISION[lane] = revision
    return revision


def _queue_snapshot(lane: str) -> dict:
    status, body = _request(BRIDGE_URLS[lane], "/queue")
    assert status == 200, f"GET /queue on lane {lane!r} failed: {status} {body}"
    return body


def _queue_item_uids(lane: str) -> list[str]:
    return [item.get("item_uid") for item in _queue_snapshot(lane).get("items", [])]


def _capability(lane: str) -> dict:
    status, body = _request(BRIDGE_URLS[lane], "/health")
    assert status == 200, f"GET /health on lane {lane!r} failed: {status} {body}"
    capability = body.get("capability")
    assert isinstance(capability, dict), f"lane {lane!r} published no capability: {body}"
    return capability


# ---------------------------------------------------------------------------
# 1. The deployment came up as two lanes
# ---------------------------------------------------------------------------
def test_both_lanes_deploy_as_separate_stacks(stack: LaneStack) -> None:
    """Six containers, three per lane, all running. LAYER: containers.

    The lane axis turns four single-set resources into per-lane ones; the
    coarsest way that can go wrong is two lanes fighting over a container name,
    a port or a Redis keyspace, and the symptom is a container that is simply
    not there.
    """
    running = _docker("ps", "--format", "{{.Names}}", timeout=60).stdout.split()
    for lane, containers in LANE_CONTAINERS.items():
        for container in containers:
            assert container in running, (
                f"lane {lane!r} did not deploy {container!r}; running containers "
                f"for this project: {[n for n in running if n.startswith(PROJECT_NAME)]}"
            )


def test_each_lane_publishes_its_own_static_identity(stack: LaneStack) -> None:
    """Each bridge says which lane it IS and which target it serves. LAYER: containers.

    This is the producer half of the split the lane axis is built on: a bridge
    knows its own lane and the target that lane was rendered for, and can know
    nothing else -- the session target lives on the host. Two lanes that
    published the same identity would make every downstream routing decision
    meaningless, so the values are asserted concretely AND against each other.
    """
    va = _capability(LANE_VA)
    live = _capability(LANE_LIVE)

    assert va["lane"] == LANE_VA, f"lane 1 published the wrong lane id: {va}"
    assert va["lane_target"] == "va", f"lane 1 does not serve the va target: {va}"
    assert live["lane"] == LANE_LIVE, f"lane 2 published the wrong lane id: {live}"
    assert live["lane_target"] == "live", f"lane 2 does not serve the live target: {live}"
    assert va["lane"] != live["lane"] and va["lane_target"] != live["lane_target"], (
        "the two lanes published the same identity, so nothing downstream can route: "
        f"{va} vs {live}"
    )


def test_the_live_lane_addresses_the_endpoint_it_was_given(stack: LaneStack) -> None:
    """The live lane's containers were handed the harness's endpoint. LAYER: containers.

    The rendered compose requires ``EPICS_CA_NAME_SERVERS`` rather than
    defaulting it, and this asserts the required value actually reached the
    lane's processes -- distinguishing "the variable was supplied" from "the
    container is addressing the co-deployed simulator like lane 1".
    """
    expected = f"host.docker.internal:{stack.live_port}"
    for container in LANE_CONTAINERS[LANE_LIVE][:2]:
        env = _docker("inspect", "-f", "{{json .Config.Env}}", container, timeout=60).stdout
        assert f"EPICS_CA_NAME_SERVERS={expected}" in env, (
            f"{container} does not address the live endpoint {expected}: {env}"
        )

    for container in LANE_CONTAINERS[LANE_VA][:2]:
        env = _docker("inspect", "-f", "{{json .Config.Env}}", container, timeout=60).stdout
        assert f"EPICS_CA_NAME_SERVERS={expected}" not in env, (
            f"{container} is on the VA lane but addresses the LIVE endpoint: {env}"
        )


# ---------------------------------------------------------------------------
# 2. Capability is static per lane; only the host's composed view moves
# ---------------------------------------------------------------------------
def test_lane_capability_is_static_across_a_session_switch(
    stack: LaneStack, bluesky_tools, session_state
) -> None:
    """A bridge's record does not move when the session does. LAYER: containers.

    The whole producer split rests on this: if a lane's published capability
    tracked the session target, the host would be composing an active/inactive
    view on top of an answer that had already guessed. Both lanes are read with
    the session on ``va`` and again with it on ``live``, and the records must be
    identical -- not merely still truthful.
    """
    session_state("va")
    before = {lane: _capability(lane) for lane in BRIDGE_URLS}

    session_state("live")
    after = {lane: _capability(lane) for lane in BRIDGE_URLS}

    assert before == after, (
        "a lane's capability changed when the session target moved; the bridge "
        f"cannot see the session and must not appear to: {before} -> {after}"
    )
    for lane, capability in after.items():
        assert "active" not in capability, (
            f"lane {lane!r} published an 'active' field, which only the host may "
            f"compose: {capability}"
        )


async def test_the_host_composes_the_active_lane_and_moves_it_on_a_switch(
    stack: LaneStack, bluesky_tools, session_state
) -> None:
    """Exactly one lane is active, and the switch moves it. LAYER: containers + host.

    ``queue_status`` is the surface an agent reads the board from, so the
    composition is asserted through it rather than through the resolver
    underneath: what has to be right is what a caller is told.
    """
    status_fn = get_tool_fn(bluesky_tools.queue_status)

    session_state("va")
    on_va = _tool_result(await status_fn())
    assert on_va["active_lane"] == LANE_VA, f"session on va did not activate lane 1: {on_va}"
    assert on_va["session_target"] == "va" and on_va["baseline_target"] == "va", on_va

    session_state("live")
    on_live = _tool_result(await status_fn())
    assert on_live["active_lane"] == LANE_LIVE, (
        f"a session switched to live did not activate the live lane: {on_live}"
    )

    for view in (on_va, on_live):
        actives = [entry for entry in view["lanes"] if entry.get("active")]
        assert len(actives) == 1, f"exactly one lane must be active: {view}"
        assert actives[0]["capability"]["active"] is True, (
            f"the active lane's composed capability is not marked active: {actives[0]}"
        )
        inactives = [entry for entry in view["lanes"] if not entry.get("active")]
        assert all(entry["capability"]["active"] is False for entry in inactives), (
            f"an inactive lane's composed capability is not marked inactive: {view}"
        )


# ---------------------------------------------------------------------------
# 3. A queued PLAN belongs to the lane it was bound to
# ---------------------------------------------------------------------------
async def test_queue_add_binds_the_item_to_the_active_lane_only(
    stack: LaneStack, bluesky_tools, session_state
) -> None:
    """The item lands on the bound lane and NOWHERE else. LAYER: containers.

    Asserted on both managers, because "it reached the right lane" and "it did
    not also reach the other one" are different claims and only the second one
    catches a fan-out.

    The queued item is withdrawn on both paths (see :func:`_withdrawn_after`), so
    a failure here cannot leak an item into the drain test's queue.
    """
    session_state("va")
    before_live = _queue_item_uids(LANE_LIVE)

    revision = _patch_draft(LANE_VA, _plan_args(stack, LANE_VA))
    result = _tool_result(await get_tool_fn(bluesky_tools.queue_add)(revision))
    item_uid = result["item"]["item_uid"]

    with _withdrawn_after(LANE_VA, item_uid):
        assert result["lane"] == LANE_VA, f"queue_add bound the item to the wrong lane: {result}"
        assert item_uid in _queue_item_uids(LANE_VA), (
            f"the item queue_add reported is not in lane {LANE_VA!r}'s queue: {result}"
        )
        assert _queue_item_uids(LANE_LIVE) == before_live, (
            "queuing on the va lane changed the live lane's queue; a PLAN composed "
            "for one machine reached the other's manager"
        )


async def test_queue_start_refuses_the_lane_the_session_left(
    stack: LaneStack, bluesky_tools, session_state
) -> None:
    """A start naming a lane the session has left is refused. LAYER: containers.

    The mid-queue switch, which is the case the whole binding exists for: the
    item is bound to the lane it was queued on, and starting it after a switch
    would arm the machine the session is no longer pointed at. The refusal is
    asserted on its machine-readable code, and on both managers staying idle --
    a refusal that had already started something would be worthless.

    The item is withdrawn on both paths for the same reason as the test above
    (see :func:`_withdrawn_after`): it is queued precisely so it can be left
    unstarted, and on a failure it would otherwise survive into the drain test.
    """
    session_state("va")
    revision = _patch_draft(LANE_VA, _plan_args(stack, LANE_VA))
    added = _tool_result(await get_tool_fn(bluesky_tools.queue_add)(revision))

    with _withdrawn_after(LANE_VA, added["item"]["item_uid"]):
        session_state("live")
        with assert_raises_error(error_type=lanes_module.REASON_LANE_MISMATCH) as refusal:
            await get_tool_fn(bluesky_tools.queue_start)(LANE_VA)

        details = refusal["envelope"].get("details", {})
        assert details.get("active_lane") == LANE_LIVE, (
            f"the refusal does not name the lane the session moved to: {refusal['envelope']}"
        )
        for lane in BRIDGE_URLS:
            state = _queue_snapshot(lane)["status"].get("manager_state")
            assert state in {"idle", "closed"}, (
                f"lane {lane!r}'s manager is {state!r} after a refused start; the "
                "refusal armed something"
            )


async def test_the_bound_lane_is_the_lane_that_executes(
    stack: LaneStack, bluesky_tools, session_state
) -> None:
    """The PLAN drains on its own lane, and the other lane runs nothing.

    LAYER: containers -- the acceptance claim of the whole axis. A start on the
    bound lane is followed to completion and the run is looked for on BOTH
    lanes, because a plan that executed on the wrong machine would still look
    like a success from the lane that was asked.
    """
    session_state("va")
    runs_before = {lane: set(_run_records(lane)) for lane in BRIDGE_URLS}

    revision = _patch_draft(LANE_VA, _plan_args(stack, LANE_VA))
    added = _tool_result(await get_tool_fn(bluesky_tools.queue_add)(revision))
    assert added["lane"] == LANE_VA, added
    item_uid = added["item"]["item_uid"]

    started = _tool_result(await get_tool_fn(bluesky_tools.queue_start)(LANE_VA))
    assert started.get("lane") == LANE_VA, f"the start did not report its lane: {started}"

    _drain(LANE_VA, QUEUE_DRAIN_TIMEOUT_S)

    on_va = _run_records(LANE_VA)
    new_on_va = set(on_va) - runs_before[LANE_VA]
    new_on_live = set(_run_records(LANE_LIVE)) - runs_before[LANE_LIVE]
    assert len(new_on_va) == 1, f"the bound lane did not record exactly one run: {new_on_va}"
    assert not new_on_live, (
        f"the LIVE lane recorded a run for a PLAN queued on the va lane: {new_on_live}"
    )

    # The run is the queued ITEM, it finished, and it swept the points the plan
    # declared -- a record that merely appeared would not distinguish a plan
    # that executed from one the manager rejected on arrival.
    record = on_va[next(iter(new_on_va))]
    assert record["item_uid"] == item_uid, (
        f"the bound lane's run is not the item queue_add bound: {record}"
    )
    assert record["status"] == "completed", f"the PLAN did not complete: {record}"
    assert record["progress"]["rows_seen"] == PLAN_POINTS, (
        f"the PLAN did not sweep the {PLAN_POINTS} points it declared: {record}"
    )


def _run_records(lane: str) -> dict[str, dict]:
    """One lane's runs, keyed by OSPREY's run id (the ``id`` field of ``/runs``)."""
    status, body = _request(BRIDGE_URLS[lane], "/runs")
    assert status == 200, f"GET /runs on lane {lane!r} failed: {status} {body}"
    assert isinstance(body, list), f"GET /runs on lane {lane!r} is not a list: {body!r}"
    return {
        record["id"]: record for record in body if isinstance(record, dict) and record.get("id")
    }


def _withdraw(lane: str, item_uid: str) -> None:
    """Remove one pending item from *lane*'s queue, through the bridge's own route.

    Never by reaching into Redis, which would be a different (and untested) path
    from the one an operator has. Called by the tests that deliberately leave an
    item queued, so the next test starts from a queue it fully accounts for --
    the queue is Redis-backed and outlives everything short of a volume removal.
    """
    status, body = _request(BRIDGE_URLS[lane], f"/queue/items/{item_uid}", "DELETE")
    assert status == 200, f"could not withdraw {item_uid} from lane {lane!r}: {status} {body}"


@contextlib.contextmanager
def _withdrawn_after(lane: str, item_uid: str):
    """Run the block, then take *item_uid* back out of *lane*'s queue -- either way.

    Cleanup has to happen on the FAILURE path too: the deployment is
    module-scoped and its queue is Redis-backed, so an item left behind by a
    failing assertion here is still sitting in that lane's queue when the drain
    test starts it, turning one real failure into a second, misleading one.

    Not a bare ``finally``, and the difference is the point. On the success path
    the withdraw is asserted, because a cleanup that silently did not happen is
    the same leak. While an exception is already propagating it is best-effort
    and merely reported -- a queue the refusal left in an unexpected state would
    otherwise raise from the cleanup and REPLACE the failure that is the actual
    news with a confusing one about deleting a queue item.
    """
    try:
        yield
    except BaseException:
        try:
            _withdraw(lane, item_uid)
        except Exception as cleanup:  # noqa: BLE001 - must never mask the real failure
            print(  # noqa: T201 - surface it in the run log, next to the failure
                f"[cleanup] could not withdraw {item_uid} from lane {lane!r}: {cleanup}"
            )
        raise
    _withdraw(lane, item_uid)


def _drain(lane: str, timeout: float) -> None:
    """Wait until *lane*'s queue is empty and its manager is idle again."""
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        snapshot = _queue_snapshot(lane)
        last = snapshot.get("status", {})
        if last.get("items_in_queue") == 0 and last.get("manager_state") in {"idle", "closed"}:
            return
        time.sleep(2.0)
    raise AssertionError(f"lane {lane!r} never drained within {timeout}s (last status: {last})")


# ---------------------------------------------------------------------------
# 4. The live lane's addressing is required, not defaulted
# ---------------------------------------------------------------------------
def test_live_lane_refuses_to_start_without_its_addressing(stack: LaneStack) -> None:
    """No ``EPICS_CA_NAME_SERVERS`` => compose refuses, naming the variable.

    LAYER: compose interpolation -- the pass ``osprey up`` performs before it
    starts anything, and deliberately not a container-start assertion: once
    interpolation refuses there is no container to start, which is the entire
    point of spelling the variable ``${VAR:?}`` instead of letting it default to
    the empty string. An empty value would leave the lane looking healthy while
    searching for channels at nowhere.

    The deployment's own ``.env`` is reused minus that one line, so every OTHER
    required variable (the per-lane control-plane keys, the launch tokens) is
    present and the refusal can only be about this one.

    One simplification worth naming: ``osprey up`` composes its environment from
    a chain (``.env.shared`` then ``.env``, plus the process env), while this
    hands compose a single ``--env-file``. That is deliberate and does not
    weaken the claim -- interpolation is what is under test, and it sees one
    merged mapping either way -- but it does mean this asserts the variable is
    REQUIRED, not that the deploy path assembles it from the layer an operator
    happened to set it in.
    """
    env_path = stack.repo / ".env"
    stripped = "\n".join(
        line
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if not line.strip().startswith("EPICS_CA_NAME_SERVERS=")
    )
    without = stack.repo / ".env.no-live-addressing"
    without.write_text(stripped + "\n", encoding="utf-8")

    child_env = {key: value for key, value in os.environ.items() if key != "EPICS_CA_NAME_SERVERS"}
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(without),
            "-f",
            str(stack.repo / "build" / "services" / "bluesky" / "docker-compose.yml"),
            "config",
            "-q",
        ],
        cwd=str(stack.repo),
        capture_output=True,
        text=True,
        timeout=120,
        env=child_env,
    )

    assert result.returncode != 0, (
        "compose accepted a live lane with no EPICS_CA_NAME_SERVERS; the lane "
        "would come up searching for channels at nowhere"
    )
    combined = f"{result.stdout}\n{result.stderr}"
    assert "EPICS_CA_NAME_SERVERS" in combined, (
        f"the refusal does not name the variable an operator has to set: {combined}"
    )


# ---------------------------------------------------------------------------
# 5. Per-lane CURVE material
# ---------------------------------------------------------------------------
def test_each_lane_holds_only_its_own_curve_material(stack: LaneStack) -> None:
    """No credential is shared between the lanes. LAYER: containers.

    Asserted in its OBSERVATIONAL form, and this is the honest statement of what
    that proves: the minted material differs per lane, and each running
    container carries only its own lane's keys and certificate mount. It does
    NOT drive lane 1's bridge credentials against lane 2's manager over raw ZMQ
    -- that would be a hand-rolled second implementation of the CurveZMQ client
    the stack already ships, and a green result from it would say more about the
    probe than about the deployment. What is ruled out here is the failure that
    makes such a drive possible at all: shared key material.
    """
    control_plane = {
        lane: (
            stack.env.get(f"{prefix}_QSERVER_ZMQ_PRIVATE_KEY"),
            stack.env.get(f"{prefix}_QSERVER_ZMQ_PUBLIC_KEY"),
        )
        for lane, prefix in LANE_ENV_PREFIX.items()
    }
    for lane, (private, public) in control_plane.items():
        assert private and public, f"osprey up minted no control-plane keypair for lane {lane!r}"
    assert control_plane[LANE_VA] != control_plane[LANE_LIVE], (
        "both lanes were minted the SAME control-socket keypair, so either lane's "
        "bridge could drive the other lane's queue manager"
    )

    tokens = {
        lane: stack.env.get(f"{prefix}_LAUNCH_TOKEN") for lane, prefix in LANE_ENV_PREFIX.items()
    }
    for lane, token in tokens.items():
        assert token, f"osprey up minted no launch token for lane {lane!r}"
    assert tokens[LANE_VA] != tokens[LANE_LIVE], (
        "one launch token across both lanes would let a launch a human approved "
        "against one machine be replayed against the other"
    )

    certificate_dirs = {
        LANE_VA: stack.repo / "data" / ".runtime" / "bluesky_curve",
        LANE_LIVE: stack.repo / "data" / ".runtime" / "bluesky_live_curve",
    }
    for lane, directory in certificate_dirs.items():
        assert directory.is_dir(), f"lane {lane!r} got no document-plane certificate directory"
    va_secrets = _secret_bytes(certificate_dirs[LANE_VA])
    live_secrets = _secret_bytes(certificate_dirs[LANE_LIVE])
    assert va_secrets and live_secrets, "a lane's certificate directory holds no secret key"
    assert not (va_secrets & live_secrets), (
        "the two lanes share document-plane secret key material, so either lane's "
        "publisher would authenticate to the other lane's proxy"
    )

    # And the running containers carry only their own lane's material.
    for lane, containers in LANE_CONTAINERS.items():
        other = LANE_LIVE if lane == LANE_VA else LANE_VA
        foreign_private, foreign_public = control_plane[other]
        for container in containers[:2]:
            env = _docker("inspect", "-f", "{{json .Config.Env}}", container, timeout=60).stdout
            assert foreign_private not in env and foreign_public not in env, (
                f"{container} (lane {lane!r}) carries lane {other!r}'s control-plane key"
            )
            mounts = _docker("inspect", "-f", "{{json .Mounts}}", container, timeout=60).stdout
            assert str(certificate_dirs[other]) not in mounts, (
                f"{container} (lane {lane!r}) mounts lane {other!r}'s certificate directory"
            )


def _secret_bytes(directory: Path) -> set[bytes]:
    """Every ``*.key_secret`` file's contents under *directory*."""
    return {path.read_bytes() for path in directory.rglob("*.key_secret")}


# ---------------------------------------------------------------------------
# 6. Single-lane regression
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def single_lane_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same project with ``bluesky.second_lane`` off, built but not deployed.

    Built rather than deployed on purpose: what a lanes-off deployment DOES is
    already owned by the existing suites (``tests/e2e/test_bluesky_queue_e2e.py``
    for the queue stack, ``tests/services/test_single_lane_switch_refusal.py``
    for the refusal contract, ``tests/deployment/test_lane_compose.py`` for
    byte-identity against the pre-lane goldens). What no container-level test
    covers is that a project built from the SAME inputs as the two-lane
    deployment above, differing in exactly one profile key, grows no second lane
    anywhere in what would be deployed.
    """
    if not e2e_conftest.E2E_ENABLED:  # pragma: no cover - collection-time skip path
        pytest.skip("VA e2e disabled")
    base = tmp_path_factory.mktemp("bluesky_single_lane_e2e")
    return _init_and_build(base, "single-lane-e2e", second_lane=False)


def test_lanes_off_renders_exactly_one_lane(single_lane_repo: Path) -> None:
    """LAYER: render. One service block, one deployed service, no lane keys.

    The keys a single-lane block must NOT grow are as load-bearing as the block
    that must not appear: ``target`` and ``ca_name_servers`` are what a lane
    declares when there is another lane to be told apart from, and a single-lane
    deployment has never had either.
    """
    config = yaml.safe_load((single_lane_repo / "build" / "config.yml").read_text(encoding="utf-8"))
    services = config["services"]

    assert LANE_VA in services, f"the single lane is missing entirely: {sorted(services)}"
    for key in SECOND_LANE_KEYS.values():
        assert key not in services, f"a lanes-off build rendered a {key!r} service block"
        assert key not in config.get("deployed_services", []), (
            f"a lanes-off build registered {key!r} as a deployed service"
        )
    assert "target" not in services[LANE_VA], (
        f"the single lane declared a target it has no sibling to be told apart from: "
        f"{services[LANE_VA]}"
    )
    assert "ca_name_servers" not in services[LANE_VA], (
        f"the single lane grew live-lane addressing: {services[LANE_VA]}"
    )


def test_lanes_off_renders_no_second_lane_containers(single_lane_repo: Path) -> None:
    """LAYER: render. Nothing in the compose file would deploy a second stack.

    Read off the rendered compose rather than the config, because the compose
    file is what ``osprey up`` hands to the container runtime -- a second lane
    that survived only there would deploy regardless of what the config said.
    """
    compose = yaml.safe_load(
        (single_lane_repo / "build" / "services" / "bluesky" / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
    )
    rendered = set(compose.get("services", {}))
    assert "bluesky-bridge" in rendered, f"the single lane's bridge is missing: {sorted(rendered)}"
    strays = {name for name in rendered if "live" in name or "-va-" in name}
    assert not strays, f"a lanes-off build rendered second-lane services: {sorted(strays)}"

    body = (single_lane_repo / "build" / "services" / "bluesky" / "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    assert "EPICS_CA_NAME_SERVERS:?" not in body, (
        "a lanes-off build made the operator supply live-lane addressing it has "
        "no live lane to use it for"
    )
