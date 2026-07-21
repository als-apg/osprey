"""Shared fixtures for the Virtual Accelerator live-container e2e suite.

Every test in this directory needs a real ``osprey-va-full`` container
actually serving Channel Access -- these are integration tests against a
live soft-IOC, not unit tests, and are opt-in via ``OSPREY_VA_E2E_ENABLE=1``
(unset, the whole directory collects cleanly and every test skips).

Container lifecycle: ONE session-scoped container (name ``osprey-va-e2e``,
never anything else -- see the containment rules in the run notes) shared by
every test in this directory; they run serially in one pytest process, so
there's no port contention. It's bind-mounted against a *scratch* copy of the
Control Assistant preset's ``data/simulation`` directory (never the repo's
own copy -- ``osprey sim apply`` mutates ``active_scenarios`` and this suite
adds its own synthetic scenario), so the fixture is free to write into it.

Process-boundary note (mirrors ``tests/va/test_record_factory.py``): this
conftest and every test module in this directory may import ``epics``
(pyepics, a CA *client*) and the softioc-free
``osprey.services.virtual_accelerator.manifest`` package, but must NEVER
import ``ioc.records`` or ``softioc.builder`` -- doing so in-process
permanently breaks this process's ability to act as a CA client (see that
module's docstring for the empirical finding). The IOC itself only ever runs
inside the container, in its own process.

``sweep_check`` (below) is loaded here, once, from its file path -- it's a
script under ``scripts/va/``, not an importable dotted package -- so
test_full_sweep.py and test_finder_live_reads.py can both import it from this
conftest instead of each re-doing the ``importlib.util`` load.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_SWEEP_SCRIPT = REPO_ROOT / "scripts" / "va" / "sweep_check.py"
_spec = importlib.util.spec_from_file_location("va_sweep_check", _SWEEP_SCRIPT)
assert _spec is not None and _spec.loader is not None
sweep_check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sweep_check
_spec.loader.exec_module(sweep_check)

ENV_FLAG = "OSPREY_VA_E2E_ENABLE"
E2E_ENABLED = os.environ.get(ENV_FLAG) == "1"

IMAGE = "osprey-va-full:latest"
CONTAINER_NAME = "osprey-va-e2e"
CA_PORT = 5064
CONTAINER_BOOT_TIMEOUT_S = 30.0

PRESET_SIM_DIR = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/simulation"
LIMITS_DB_PATH = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/channel_limits.json"
OSPREY_CLI = REPO_ROOT / ".venv" / "bin" / "osprey"

# A channel harmless to read at boot time: pyat-coupled, never written by the
# readiness probe itself.
READINESS_ADDRESS = "SR:MAG:HCM:01:CURRENT:RB"

# Synthetic scenario this suite adds on top of the copied preset data, so
# test_scenario_reload.py has a scenario with a real ``overrides`` entry to
# apply (the shipped nominal/rf-thermal/vacuum-burst scenarios only carry
# archiver history events, not live-telemetry overrides -- see that test's
# module docstring for why).
BURST_SCENARIO_NAME = "va-e2e-burst"
BURST_CHANNEL = "SR:VAC:GAUGE:SR07:PRESSURE:RB"
BURST_VALUE = 3.0e-6  # nominal machine.json baseline is 5e-8 Torr (3% noise) -- unambiguous jump

# CA gateway config: read_only and write_access both point at the container's
# single published port (matches the preset's config.yml.j2 virtual_accelerator
# block), so gateway selection is inert here -- writes_enabled is gated purely
# by the base-class guard tested by test_approval_smoke.py.
VA_GATEWAY_CONFIG: dict[str, Any] = {
    "timeout": 5.0,
    "gateways": {
        "read_only": {"address": "localhost", "port": CA_PORT, "use_name_server": True},
        "write_access": {"address": "localhost", "port": CA_PORT, "use_name_server": True},
    },
}
CONNECTOR_CONFIG: dict[str, Any] = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": VA_GATEWAY_CONFIG},
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
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
        return overrides.get(key, default)

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
    """A scratch project directory: ``config.yml`` + ``data/simulation/`` --
    just enough for ``osprey sim apply`` to run against it."""

    project_dir: Path
    data_dir: Path

    def sim_apply(self, *scenario_names: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(OSPREY_CLI), "sim", "apply", *scenario_names, "--no-seed"],
            cwd=self.project_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


@pytest.fixture(scope="session")
def va_project(tmp_path_factory: pytest.TempPathFactory) -> VaProject:
    """Build the scratch project this session's container serves.

    A copy of the Control Assistant preset's ``data/simulation`` (never the
    repo's own copy -- this fixture writes into it via ``osprey sim apply``
    and adds a synthetic scenario), plus a minimal ``config.yml`` so the
    ``sim`` CLI's type-aware lookup resolves ``control_system.type:
    virtual_accelerator`` to this directory's ``machine.json``.
    """
    project_dir = tmp_path_factory.mktemp("va_e2e_project")
    data_dir = project_dir / "data" / "simulation"
    shutil.copytree(PRESET_SIM_DIR, data_dir)

    burst_dir = data_dir / "scenarios" / BURST_SCENARIO_NAME
    burst_dir.mkdir(parents=True)
    (burst_dir / "scenario.json").write_text(
        json.dumps(
            {
                "description": (
                    "e2e-only synthetic scenario: overrides one VAC gauge to a "
                    "value unambiguously distinct from its nominal baseline."
                ),
                "overrides": {BURST_CHANNEL: BURST_VALUE},
            }
        )
    )
    (data_dir / "active_scenarios").write_text("nominal\n")

    (project_dir / "config.yml").write_text(
        "control_system:\n"
        "  type: virtual_accelerator\n"
        "  writes_enabled: true\n"
        "  connector:\n"
        "    virtual_accelerator:\n"
        "      simulation_file: data/simulation/machine.json\n"
    )

    return VaProject(project_dir=project_dir, data_dir=data_dir)


def _docker_rm(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


def _readiness_pv_served() -> bool:
    """Probe container readiness in a SUBPROCESS.

    The async connector wraps *sync* pyepics in a thread-pool executor, and
    libca CA contexts are per-thread: a main-thread pyepics CA operation in
    this process deadlocks the connector's executor-thread caget/caput calls.
    So the readiness check must never touch pyepics in-process -- run it
    out-of-process, exactly as the probe's caget check does. Returns True once
    the readiness PV is served.
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

    Exact container name ``osprey-va-e2e`` (containment rule); torn down by
    that exact name regardless of how this fixture exits.
    """
    _docker_rm(CONTAINER_NAME)  # clear any stale container from a prior crashed run

    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"127.0.0.1:{CA_PORT}:{CA_PORT}/tcp",
            "-v",
            f"{va_project.data_dir}:/data/simulation:ro",
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
