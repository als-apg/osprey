"""Shared fixtures for the Virtual Accelerator live-container e2e suite.

Every test in this directory needs a real ``osprey-va-full`` container
actually serving Channel Access -- these are integration tests against a
live soft-IOC, not unit tests, and are opt-in via ``OSPREY_VA_E2E_ENABLE=1``
(unset, the whole directory collects cleanly and every test skips).

Container lifecycle: ONE session-scoped container per pytest process, named
``osprey-va-e2e-<pid>`` and never anything else (the containment rules in the
run notes apply to that exact name), shared by every test in this directory;
they run serially in one pytest process, so there's no port contention. The
pid suffix is what keeps two concurrent runs from destroying each other: this
fixture force-removes its container by name on the way in, as stale-cleanup
from a crashed prior run, and against a *live* peer sharing one fixed name
that is not cleanup -- it kills the peer mid-test, which then fails with a
boot timeout or a dropped connection that reads as environmental rather than
as a collision. The ``osprey-va-e2e`` prefix is kept so a stray container is
still recognisable by eye. It's bind-mounted against a *scratch* data root
rendered from the Control Assistant preset (never the repo's own copy --
``osprey sim apply`` mutates ``active_scenarios`` and this suite adds its own
synthetic scenario), so the fixture is free to write into it. What gets
rendered is the layout ``osprey build`` writes for a project: the simulator
view under ``data/simulator/``, from the preset's ``data/facility``. The
container serves as the instance ``VA_INSTANCE`` names: the entrypoint refuses
to boot without one. See ``stage_demo_data_dir``.

Process-boundary note: this conftest and every test module in this directory
may import ``epics`` (pyepics, a CA *client*), but must NEVER import a Channel
Access server -- the server extension exports the ca_* client symbols too, and
importing it in-process breaks this process's ability to act as a CA client.
The IOC itself only ever runs inside the container, in its own process.

``sweep_check`` (below) is loaded here, once, from its file path -- it's a
script under ``scripts/va/``, not an importable dotted package -- so
test_full_sweep.py and test_finder_live_reads.py can both import it from this
conftest instead of each re-doing the ``importlib.util`` load.
"""

from __future__ import annotations

import atexit
import gc
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

# import-time required because scripts/ is not a package: sweep_check.py is
# loaded by path and registered in sys.modules before exec so the fixtures
# below can reference it at collection time.
_SWEEP_SCRIPT = REPO_ROOT / "scripts" / "va" / "sweep_check.py"
_spec = importlib.util.spec_from_file_location("va_sweep_check", _SWEEP_SCRIPT)
assert _spec is not None and _spec.loader is not None
sweep_check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sweep_check
_spec.loader.exec_module(sweep_check)

ENV_FLAG = "OSPREY_VA_E2E_ENABLE"
E2E_ENABLED = os.environ.get(ENV_FLAG) == "1"

IMAGE = "osprey-va-full:latest"
# Per-process, so two concurrent runs cannot force-remove each other's live
# container -- see the module docstring. Everything that creates, inspects or
# removes the container uses this one name.
CONTAINER_NAME = f"osprey-va-e2e-{os.getpid()}"
# The CONTAINER serves Channel Access on the protocol default, which the image
# fixes; the HOST side of the publish is an ephemeral free port, because 5064
# on this host belongs to whatever real deployment the operator is running
# (`port_layout.CA_DEFAULT_PORT` keeps VA instance 1 there on purpose, so a
# dev machine routinely holds it). Name-server mode (`use_name_server: true`,
# EPICS_CA_NAME_SERVERS) carries the remap: the client dials the mapped host
# port over TCP and never needs the container's own number. The shipped 5064
# default itself is covered at render level — tests/cli/test_va_default_config.py
# and tests/cli/test_rendered_va_block.py pin it — so nothing here has to
# squat the live port to keep that contract tested.
CONTAINER_CA_PORT = 5064


def _reserve_free_port() -> int:
    """A host port that was free at reservation time.

    The bind is released before docker publishes the port, so a racing process
    could take it in between; per-process module state keeps every user of
    ``CA_PORT`` in this run on the one reserved number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# import-time required because CA_PORT is per-process module state: every
# fixture and helper in this package must agree on the one reserved number,
# and a fixture-scoped reservation would hand xdist workers different ports
# for constants already bound at collection.
CA_PORT = _reserve_free_port()
# Generous on purpose. Boot-to-first-served-answer measured 9-15 s across six
# container boots on this host -- the VA images are pinned ``linux/amd64`` and
# the host is Apple Silicon, so every local boot is emulated. A 30 s ceiling is
# ~2x a 15 s boot, thin enough to flake; a slow boot that still succeeds costs
# nothing here, while a real failure gets its container logs into the error
# either way.
CONTAINER_BOOT_TIMEOUT_S = 120.0

PRESET_FACILITY_DIR = REPO_ROOT / "src/osprey/templates/facilities/example"

#: The config a demo view is rendered with: the preset's control-system type
#: and limits posture, every model served.
VIEW_CONFIG: dict[str, Any] = {
    "control_system": {
        "type": "virtual_accelerator",
        "limits_checking": {"enabled": True, "mode": "optional"},
    }
}


def _render_limits_view() -> Path:
    """The limits view of the preset's ``data/facility``, rendered once per session.

    The view is what a build writes as ``channel_limits.json``, so the suite's
    lanes read exactly the bands a built project carries. It lands in a
    per-process temp directory removed at exit.
    """
    from osprey.facility.build import build_facility
    from osprey.facility.views.limits import LIMITS_FILE, limits_document

    root = Path(tempfile.mkdtemp(prefix="osprey-va-e2e-limits-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    document = limits_document(
        build_facility(PRESET_FACILITY_DIR, project_name="control_assistant")
    )
    target = root / LIMITS_FILE
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return target


# import-time required because lanes read the bands at import
# (``LIMITS_OVERRIDES`` in test_limits_enforcement), before any fixture runs.
LIMITS_DB_PATH = _render_limits_view()
OSPREY_CLI = REPO_ROOT / ".venv" / "bin" / "osprey"

#: The instance every container of this suite serves as, and the ``docker
#: run`` arguments naming it. The entrypoint refuses a boot without one.
INSTANCE = "virtual_accelerator"
DEMO_NAMESPACE_RUN_ARGS = ("-e", f"VA_INSTANCE={INSTANCE}")


def stage_demo_data_dir(root: Path, *, still_monitors: bool = True) -> Path:
    """Render, under *root*, the data root a demo container mounts; return *root*.

    The layout the IOC reads is the one ``osprey build`` writes for a project:
    the simulator view under ``<root>/simulator/``. It is rendered from a copy
    of the preset's ``data/facility`` that also carries this suite's synthetic
    additions, so each is one the container's composite and ``osprey sim
    apply`` both know:

    * the burst scenario (:data:`BURST_SCENARIO_NAME`), which moves one gauge;
    * the unstable scenario (:data:`UNSTABLE_SCENARIO_NAME`), under which the
      physics model's closed-orbit solve fails;
    * the kick scenario (:data:`KICK_SCENARIO_NAME`), which puts a nonzero
      closed orbit at :data:`KICK_MONITOR`;
    * the seeded-readout scenario (:data:`SEEDED_READOUT_SCENARIO_NAME`),
      which gives :data:`SEEDED_READOUT_BPM` a readout offset and gain;
    * a string channel (:data:`STRING_CHANNEL`) seeded with
      :data:`STRING_NOMINAL`, so the served view holds a string channel the
      texture owns beside the physics model's status channel.

    The shared container serves monitors without their declared motion, so a
    suite whose oracle is the noiseless model reads the solved orbit; a caller
    that needs the declared motion renders with ``still_monitors=False``.

    Rendered rather than layered on with extra bind mounts because a bind
    mount INTO a read-only mount cannot create its own mountpoint: the runtime
    refuses with EROFS.
    """
    from osprey.facility.build import build_facility
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.simulator import write_simulator_view
    from osprey_connectors.simulation.view import VIEW_RELPATH
    from tests.e2e._monitor_motion import still_monitor_motion

    with tempfile.TemporaryDirectory(prefix="osprey-va-e2e-facility-") as scratch:
        facility = Path(scratch) / "facility"
        shutil.copytree(PRESET_FACILITY_DIR, facility)
        for name, description, overrides in (
            (
                BURST_SCENARIO_NAME,
                "e2e-only synthetic scenario: overrides one VAC gauge to a "
                "value unambiguously distinct from its nominal baseline.",
                {BURST_CHANNEL: BURST_VALUE},
            ),
            (
                UNSTABLE_SCENARIO_NAME,
                "e2e-only synthetic scenario: a focusing quadrupole at a current "
                "with no stable closed orbit, so the physics model fails.",
                {UNSTABLE_QUADRUPOLE: UNSTABLE_CURRENT},
            ),
            (
                KICK_SCENARIO_NAME,
                "e2e-only synthetic scenario: one vertical corrector excited, so "
                "the vertical closed orbit is nonzero at the monitors.",
                {KICK_CORRECTOR: KICK_CURRENT},
            ),
        ):
            (facility / "scenarios" / f"{name}.yaml").write_text(
                yaml.safe_dump({"description": description, "overrides": overrides}),
                encoding="utf-8",
            )
        (facility / "scenarios" / f"{SEEDED_READOUT_SCENARIO_NAME}.yaml").write_text(
            yaml.safe_dump(
                {
                    "description": "e2e-only synthetic scenario: one monitor's readout offset "
                    "and gain, so a container booted with it reads that monitor differently "
                    "from one booted without it.",
                    "faults": {
                        "SR": {
                            SEEDED_READOUT_BPM: {
                                "offset": SEEDED_READOUT_OFFSET,
                                "gain": SEEDED_READOUT_GAIN,
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        with (facility / "records" / "channels.yaml").open("a", encoding="utf-8") as records:
            records.write(
                yaml.safe_dump(
                    [
                        {
                            "id": STRING_CHANNEL,
                            "value_type": "string",
                            "description": "e2e-only synthetic string channel",
                        }
                    ]
                )
            )
        with (facility / "seeds.yaml").open("a", encoding="utf-8") as seeds:
            seeds.write(yaml.safe_dump({STRING_CHANNEL: {"nominal": STRING_NOMINAL}}))
        if still_monitors:
            still_monitor_motion(Path(scratch))
        doc = build_facility(facility, project_name="control_assistant")
        write_simulator_view(
            root / VIEW_RELPATH.name,
            ViewInputs(
                doc=doc,
                rendered_config=VIEW_CONFIG,
                facility_dir=facility,
                served=resolve_served(VIEW_CONFIG, doc),
                reported=None,
            ),
        )
    return root


def data_root_run_args(data_root: Path) -> tuple[str, ...]:
    """The ``docker run`` arguments that mount *data_root* as the container's ``/data``.

    The entrypoint serves ``/data/simulator/``, its default data root, so the
    mount is all a container needs to find the view.
    """
    return ("-v", f"{data_root}:/data:ro")


def demo_data_run_args() -> tuple[str, ...]:
    """:func:`data_root_run_args` for this process's rendered demo data root."""
    return data_root_run_args(demo_data_dir())


_DEMO_DATA_DIR: Path | None = None


def demo_data_dir() -> Path:
    """The rendered demo data root, one per pytest process.

    A plain function rather than a fixture because most of this suite's
    containers are booted from context-manager helpers, not from fixtures, and
    every one of them wants the same directory. Rendered on first use and
    removed at process exit, so a run that skips the whole directory renders
    nothing.
    """
    global _DEMO_DATA_DIR
    if _DEMO_DATA_DIR is None:
        root = Path(tempfile.mkdtemp(prefix="osprey-va-demo-data-"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        _DEMO_DATA_DIR = stage_demo_data_dir(root)
    return _DEMO_DATA_DIR


# A channel harmless to read at boot time: pyat-coupled, never written by the
# readiness probe itself.
READINESS_ADDRESS = "SR:MAG:HCM:01:CURRENT:RB"

# Synthetic scenario this suite adds to the rendered preset view, so
# test_scenario_reload.py has a scenario with a real ``overrides`` entry to
# apply (the shipped nominal/rf-thermal/vacuum-burst scenarios only carry
# archiver history events, not live-telemetry overrides -- see that test's
# module docstring for why).
BURST_SCENARIO_NAME = "va-e2e-burst"
BURST_CHANNEL = "SR:VAC:GAUGE:SR07:PRESSURE:RB"
BURST_VALUE = 3.0e-6  # nominal machine.json baseline is 5e-8 Torr (3% noise) -- unambiguous jump

#: Synthetic scenario under which the physics model's closed-orbit solve
#: fails: the quadrupole at twice its demo current has no stable orbit.
UNSTABLE_SCENARIO_NAME = "va-e2e-unstable"
UNSTABLE_QUADRUPOLE = "SR:MAG:QF:01:CURRENT:SP"
UNSTABLE_CURRENT = 712.2

#: Synthetic scenario that excites one vertical corrector, so the vertical
#: closed orbit at :data:`KICK_MONITOR` is nonzero: at rest the demo's closed
#: orbit is exactly zero at every monitor.
KICK_SCENARIO_NAME = "va-e2e-kick"
KICK_CORRECTOR = "SR:MAG:VCM:05:CURRENT:SP"
KICK_CURRENT = 1.0
KICK_MONITOR = "SR:DIAG:BPM:17:POSITION:Y"

#: Synthetic scenario that gives one monitor a readout offset and gain, so a
#: container booted with it active reads that monitor as
#: ``(x - SEEDED_READOUT_OFFSET) * SEEDED_READOUT_GAIN`` of the position ``x`` a
#: container booted without it reads.
SEEDED_READOUT_SCENARIO_NAME = "va-e2e-seeded-readout"
SEEDED_READOUT_BPM = "SR:DIAG:BPM:11:POSITION:X"
SEEDED_READOUT_OFFSET = 50e-6
SEEDED_READOUT_GAIN = 1.05

#: Synthetic string channel, owned by the texture, and the text it is seeded with.
STRING_CHANNEL = "SR:DIAG:E2E:TEXT"
STRING_NOMINAL = "synthetic e2e text"

# CA gateway config: read_only and write_access both point at the container's
# single published port (matches the preset's config.yml.j2 virtual_accelerator
# block), so gateway selection is inert here -- writes_enabled is gated purely
# by the base-class guard tested by test_approval_smoke.py.
VA_GATEWAY_CONFIG: dict[str, Any] = {
    "timeout_s": 5.0,
    "gateways": {
        "read_only": {"address": "localhost", "port": CA_PORT, "use_name_server": True},
        "write_access": {"address": "localhost", "port": CA_PORT, "use_name_server": True},
    },
}
CONNECTOR_CONFIG: dict[str, Any] = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": VA_GATEWAY_CONFIG},
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001 - pytest resolves a hook's arguments by name
    """Skip every test under this directory unless the e2e flag is set.

    Applied at collection time (not via a per-file ``pytestmark``) so the
    guard can never be accidentally dropped by a new test module -- the
    directory must always collect cleanly and skip cleanly with the flag
    unset.
    """
    if E2E_ENABLED:
        return
    skip = pytest.mark.skip(reason=f"set {ENV_FLAG}=1 to run VA live-container e2e tests")
    this_dir = Path(__file__).resolve().parent
    for item in items:
        if Path(str(item.fspath)).resolve().is_relative_to(this_dir):
            item.add_marker(skip)


@contextmanager
def patched_config(**overrides: Any) -> Iterator[None]:
    """Patch ``osprey.utils.config.get_config_value`` for the duration of the block.

    Every connector config lookup in this suite goes through this instead of
    relying on an ambient ``config.yml`` -- explicit and immune to whatever
    CONFIG_FILE/cwd state another test in the same pytest process left behind
    (see ``tests/connectors/test_simulation_integration.py`` for the same
    pattern). Must stay active for the entire connect()+read/write sequence:
    ``_writes_enabled`` is re-evaluated by the base-class guard on every
    single write call, not cached at connect time.
    """

    def _get_config_value(key: str, default: Any = None) -> Any:
        # Answer a SECTION read the way the real loader does: a nested mapping
        # assembled from every override under that prefix. The per-type write
        # posture reads ``control_system`` whole and indexes the connector
        # block by name (a custom type is a dotted module path, so the leaf
        # cannot be looked up by dotted path), and a flat map that only
        # answered exact keys would leave that read empty and writes unarmed.
        if key in overrides:
            return overrides[key]
        prefix = key + "."
        section: dict[str, Any] = {}
        for dotted, value in overrides.items():
            if not dotted.startswith(prefix):
                continue
            node = section
            *parents, leaf = dotted[len(prefix) :].split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
        return section or default

    with patch("osprey.utils.config.get_config_value", side_effect=_get_config_value):
        yield


# Connectors handed out by connect_va(), pending deterministic teardown. A
# pyepics PV whose subscription is still live segfaults libca if it is
# finalized by a garbage-collection cycle at an arbitrary point (observed
# reproducibly: GC triggered inside a later test's json.load collects a prior
# test's PVs -> PV.__del__ -> ca.clear_subscription -> SIGSEGV). Tests must
# never leave connector PVs to die by GC.
_LIVE_CONNECTORS: list[Any] = []


async def connect_va(**config_overrides: Any):
    """Create + connect a VirtualAcceleratorConnector via the real ConnectorFactory.

    Must be called from inside a ``with patched_config(...):`` block that
    stays open for as long as the returned connector is used to write.

    The connector is registered for automatic disconnect after the test (see
    ``_disconnect_va_connectors``); callers need no try/finally of their own.
    """
    from osprey.connectors.factory import ConnectorFactory, register_builtin_connectors

    register_builtin_connectors()
    connector = await ConnectorFactory.create_control_system_connector(CONNECTOR_CONFIG)
    _LIVE_CONNECTORS.append(connector)
    return connector


@pytest.fixture(autouse=True)
async def _disconnect_va_connectors() -> Any:
    """Disconnect every connector a test created, then collect finalizers.

    Runs in the test's own event loop while the CA context is healthy, so PV
    subscriptions are cleared on the supported path; the explicit
    ``gc.collect()`` then runs any remaining ``PV.__del__`` on
    already-disconnected PVs (a no-op for subscriptions) at a controlled
    moment instead of mid-test.
    """
    yield
    while _LIVE_CONNECTORS:
        connector = _LIVE_CONNECTORS.pop()
        try:
            await connector.disconnect()
        except Exception:
            pass  # best-effort: teardown must never mask a test result
    gc.collect()


@dataclass
class VaProject:
    """A scratch deployment repo: a ``profile.yml`` root, a render under
    ``build/`` holding its ``config.yml`` and the simulator view under
    ``build/data/simulator/`` (``data_dir`` is that render's data root), and the
    mutable state dir under ``var/agent_data/`` -- the three zones ``osprey sim
    apply`` resolves, and nothing more."""

    project_dir: Path
    data_dir: Path
    state_dir: Path

    def sim_apply(self, *scenario_names: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(OSPREY_CLI), "sim", "apply", *scenario_names, "--no-seed"],
            cwd=self.project_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def stage_va_project(root: Path) -> VaProject:
    """Materialize, under *root*, the deployment repo this suite's container serves.

    The exemplar repo supplies the root ``profile.yml`` that ``osprey sim
    apply`` discovers by walking up from its working directory, and the state
    zone it writes ``active_scenarios`` into; a stubbed render supplies the
    ``build/config.yml`` every repo-scoped verb reads, and the preset's
    simulator view is rendered beside it (see ``stage_demo_data_dir``), so
    ``osprey sim apply`` and the container read the same view.

    A plain function rather than the fixture body so a caller can stage the
    repo and drive ``osprey sim apply`` at it without a pytest session.
    """
    from osprey_connectors.workspace import resolve_simulation_state_dir
    from tests.cli._lifecycle_build import stub_build
    from tests.fixtures.lifecycle_repo import build_exemplar_repo

    project_dir = build_exemplar_repo(root / "va-e2e")

    config = {
        "control_system": {
            "type": "virtual_accelerator",
            "writes_enabled": True,
        },
    }
    build = stub_build(project_dir, config=yaml.safe_dump(config))
    data_dir = stage_demo_data_dir(build / "data")

    state_dir = resolve_simulation_state_dir(config, project_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "active_scenarios").write_text("nominal\n")

    return VaProject(project_dir=project_dir, data_dir=data_dir, state_dir=state_dir)


@pytest.fixture(scope="session")
def va_project(tmp_path_factory: pytest.TempPathFactory) -> VaProject:
    """The scratch deployment repo this session's container serves."""
    return stage_va_project(tmp_path_factory.mktemp("va_e2e_project"))


def _docker_rm(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


#: How many ports a boot tries before giving up on publishing one.
PUBLISH_ATTEMPTS = 5


def run_on_free_port(
    container_for: Callable[[int], tuple[str, list[str]]],
) -> tuple[int, str]:
    """``docker run`` a container published on a free port, retrying a lost race.

    A reserved port is free only until the probe socket closes, and anything on
    the host can take it before ``docker run`` binds it. Docker then refuses
    the publish with "address already in use", and the answer is another port
    rather than a failed boot.

    Args:
        container_for: Given a port, the container's name and the arguments
            after ``docker``.

    Returns:
        The port the container is published on, and its name.
    """
    for _ in range(PUBLISH_ATTEMPTS):
        port = _reserve_free_port()
        name, arguments = container_for(port)
        # Stale-cleanup only, of a container an earlier run left under this name.
        _docker_rm(name)
        started = subprocess.run(
            ["docker", *arguments], capture_output=True, text=True, timeout=120
        )
        if started.returncode == 0:
            return port, name
        # A refused publish still leaves the created container behind.
        _docker_rm(name)
        if "address already in use" not in started.stderr:
            raise RuntimeError(f"docker run failed: {started.stdout}\n{started.stderr}")
    raise RuntimeError(f"no free port could be published in {PUBLISH_ATTEMPTS} attempts")


def _listening_ports(proc_net: str, *, tcp: bool) -> list[int]:
    """Local ports in a ``/proc/net/{tcp,tcp6,udp,udp6}`` dump; TCP in LISTEN only."""
    ports: set[int] = set()
    for line in proc_net.splitlines():
        fields = line.split()
        if len(fields) < 4 or ":" not in fields[1] or fields[0] == "sl":
            continue
        if tcp and fields[3] != "0A":
            continue
        ports.add(int(fields[1].rsplit(":", 1)[1], 16))
    return sorted(ports)


def boot_report(container: str, host_port: int) -> str:
    """What the host and the container say about a port a client could not reach.

    For a boot that logged it was serving yet never answered: which ports the
    container publishes, which its server listens on inside, whether a plain
    TCP connect to the published port lands, and who holds that port on the
    host.
    """

    def run(*command: str) -> str:
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"<{type(exc).__name__}: {exc}>"
        return (done.stdout + done.stderr).strip() or "<nothing>"

    lines = [f"docker port: {run('docker', 'port', container)}"]
    for family, tcp in (("tcp", True), ("tcp6", True), ("udp", False), ("udp6", False)):
        dump = run("docker", "exec", container, "cat", f"/proc/net/{family}")
        lines.append(f"container {family} ports: {_listening_ports(dump, tcp=tcp)}")
    try:
        with socket.create_connection(("127.0.0.1", host_port), timeout=5):
            lines.append(f"host TCP connect to 127.0.0.1:{host_port}: accepted")
    except OSError as exc:
        lines.append(f"host TCP connect to 127.0.0.1:{host_port}: {exc!r}")
    if shutil.which("ss"):
        lines.append(f"host holders of {host_port}: {run('ss', '-Htanp', f'sport = :{host_port}')}")
    return "\n".join(lines)


def _readiness_pv_served() -> bool:
    """Probe container readiness in a SUBPROCESS.

    The async connector wraps *sync* pyepics in a thread-pool executor, and
    libca CA contexts are per-thread: a main-thread pyepics CA operation in
    this process deadlocks the connector's executor-thread caget/caput calls.
    So the readiness check must never touch pyepics in-process -- run it
    out-of-process, exactly as the probe's caget check does. Returns True once
    the readiness PV is served.

    The probe leaves through ``os._exit`` rather than returning. This child
    calls ``epics.caget`` directly and builds no connector, so nothing takes
    pyepics' ``finalize_libca`` off its exit hooks the way
    ``EPICSConnector.connect`` does for the processes that go through it. That
    finalizer's recorded hang follows Channel Access use on a worker thread --
    what the connector's executor does -- rather than the one main-thread
    ``caget`` this child makes, which has not been seen to hang. The forced
    exit is kept as a bound that costs nothing: a probe that will not die is
    read here as a container that is not serving, and the word is written and
    flushed before the exit.
    """
    code = (
        "import sys, epics\n"
        f"v = epics.caget({READINESS_ADDRESS!r}, timeout=1.0, connection_timeout=1.0)\n"
        "sys.stdout.write('SERVED' if v is not None else 'NONE')\n"
        "sys.stdout.flush()\n"
        "import os; os._exit(0)\n"
    )
    env = {
        **os.environ,
        "EPICS_CA_NAME_SERVERS": f"localhost:{CA_PORT}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }
    env.pop("EPICS_CA_ADDR_LIST", None)
    env.pop("EPICS_CA_SERVER_PORT", None)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False
    return proc.stdout.strip() == "SERVED"


@pytest.fixture(scope="session")
def va_container(va_project: VaProject) -> Iterator[VaProject]:
    """Boot the session-shared VA container and wait for it to serve PVs.

    Exact container name ``CONTAINER_NAME`` -- ``osprey-va-e2e-<pid>``
    (containment rule); created, logged and torn down by that same name
    regardless of how this fixture exits, and by no other.
    """
    # Stale-cleanup only: with the pid suffix this can name nothing but a
    # container left behind by an earlier process that has since exited and
    # whose pid was reused. It can no longer reach a concurrent peer's.
    _docker_rm(CONTAINER_NAME)

    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"127.0.0.1:{CA_PORT}:{CONTAINER_CA_PORT}/tcp",
            *data_root_run_args(va_project.data_dir),
            # Scenario state is a SEPARATE mount: the host writes it at run
            # time (`osprey sim apply`) while data/ is build-owned.
            "-v",
            f"{va_project.state_dir}:/state/simulation:ro",
            "-e",
            "VA_STATE_DIR=/state/simulation",
            # The instance, named. Without it the IOC refuses to boot.
            *DEMO_NAMESPACE_RUN_ARGS,
            IMAGE,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker run failed: {result.stdout}\n{result.stderr}")

    os.environ["EPICS_CA_NAME_SERVERS"] = f"localhost:{CA_PORT}"
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    os.environ.pop("EPICS_CA_ADDR_LIST", None)
    os.environ.pop("EPICS_CA_SERVER_PORT", None)

    deadline = time.monotonic() + CONTAINER_BOOT_TIMEOUT_S
    served = False
    while time.monotonic() < deadline:
        if _readiness_pv_served():
            served = True
            break
        time.sleep(0.5)

    if not served:
        logs = subprocess.run(
            ["docker", "logs", CONTAINER_NAME], capture_output=True, text=True, timeout=10
        )
        _docker_rm(CONTAINER_NAME)
        raise RuntimeError(
            f"VA container never came up (no value for {READINESS_ADDRESS} after "
            f"{CONTAINER_BOOT_TIMEOUT_S}s). Container logs:\n{logs.stdout}\n{logs.stderr}"
        )

    try:
        yield va_project
    finally:
        _docker_rm(CONTAINER_NAME)


#: Fast enough that a switch tool's wait is never the reconciler's clock.
RECONCILE_INTERVAL_S = 0.05


@pytest.fixture
async def reconciling():
    """``await reconciling()`` runs this server's reconcile loop, as its lifespan does.

    ``control_target_set`` writes the control-context record and then waits for
    THIS server's connector host to reach the generation it minted; the loop is
    what moves it. Started on the call rather than on the fixture, because a
    pass that ran before the server context was installed would claim the
    record at the deployment baseline rather than at the target the suite
    started its manager on.
    """
    from osprey.mcp_server.control_system.session_control import SessionControlReconciler

    started: list[Any] = []

    async def start() -> Any:
        loop = SessionControlReconciler(interval_s=RECONCILE_INTERVAL_S)
        await loop.start()
        started.append(loop)
        return loop

    yield start
    for loop in started:
        await loop.stop()


# ---------------------------------------------------------------------------
# The corrector a lane drives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Corrector:
    """One corrector of the served view: the address a lane writes and the one it reads back."""

    setpoint_address: str
    readback_address: str


def corrector_at_slot(slot: int) -> Corrector:
    """The ``slot``-th corrector of the view this suite's containers serve.

    A lane names the corrector it drives by SLOT rather than by address: which
    channels kick the beam is the served view's wiring to answer, and where
    each of them reads its own field back is its channel's ``pair``, rather
    than a device name written into a test. Wiring order is the facility
    file's record order, so one slot names one magnet on every run against a
    given tree.

    A slot is owned by one lane for the life of the session container -- two
    lanes driving one corrector would read each other's writes -- so each lane
    takes a slot of its own.

    Called from a lane's fixture rather than at import: a served view that is
    absent, unreadable or unusable then fails the lanes that drive a
    corrector, instead of failing collection for every lane in this directory.

    Raises:
        AssertionError: If the view wires fewer correctors than ``slot``
            requires, or if the corrector at ``slot`` is served with no
            readback of its own.
    """
    from osprey_connectors.simulation.view import VIEW_RELPATH, SimulatorView

    view = SimulatorView.open(demo_data_dir() / VIEW_RELPATH.name)
    kicks = [
        binding.address for binding in view.bindings(role="setpoint") if binding.plane is not None
    ]
    assert len(kicks) > slot, (
        f"the served view wires {len(kicks)} correctors, too few for a lane's slot {slot}"
    )
    setpoint = kicks[slot]
    readback = view.channel(setpoint).pair
    assert readback is not None and readback != setpoint, (
        f"{setpoint} is served with no readback of its own, so a "
        f"lane cannot tell a magnet's reading from the demand written to it"
    )
    return Corrector(setpoint_address=setpoint, readback_address=str(readback))
