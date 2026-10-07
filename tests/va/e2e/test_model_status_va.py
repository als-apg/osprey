"""``osprey sim status`` on a virtual accelerator target, against both instances.

A target served by a separate process reports each served physics model's
status on its status channel, ``<code>:SIM:<model>:STATUS``, and ``sim status``
reads it through the target's connector. This module boots two real
``osprey-va-full`` containers over one simulator view, the sandbox (the ``va``
target) and the stand-in (the ``standin`` target), points a scratch deployment
at them, and runs ``sim status --target`` for each: both print ``SR: ok``.

The container helpers are copied from ``test_live_standin.py`` rather than
imported from it, as that module's own docstring explains: a test module is not
an importable helper library.

Every Channel Access operation happens in another process: the readiness probe
and each ``sim status`` run are subprocesses, which leave through ``os._exit``
for the reason ``test_live_standin.py`` gives for its probe. This process never
becomes a CA client.

The whole directory is opt-in behind ``OSPREY_VA_E2E_ENABLE=1``; the skip is
applied by ``conftest.pytest_collection_modifyitems``.
"""

from __future__ import annotations

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

from osprey_connectors.types import TARGET_STANDIN, TARGET_VA
from tests.va.e2e import conftest as e2e_conftest

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_PATHS = (str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src"))

#: The image under test.
IMAGE = os.environ.get("OSPREY_VA_E2E_IMAGE", "osprey-va-full:latest")

#: Container-name prefixes; ``_serving`` appends the run's own ephemeral port.
CONTAINER_SANDBOX = "osprey-va-e2e-status-va"
CONTAINER_STANDIN = "osprey-va-e2e-status-standin"

#: The instance each container serves as.
SANDBOX_INSTANCE = "virtual_accelerator"
STANDIN_INSTANCE = "live_standin"

#: Two containers, and a local run on Apple Silicon is emulated.
BOOT_TIMEOUT_S = 180.0

#: What the readiness probe waits for: a readback both instances serve.
PROBE_CHANNEL = "SR:MAG:HCM:01:CURRENT:RB"

#: The connector's own timeout.
CONNECTOR_TIMEOUT_S = 120.0

#: Bound on one ``sim status`` run.
STATUS_TIMEOUT_S = 120.0

#: The demo's physics model.
DEMO_PHYSICS_MODEL = "SR"

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 2


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _docker(*args: str, timeout: float = 180.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _require_image() -> None:
    """Fail loudly unless the image is present."""
    inspected = _docker("image", "inspect", IMAGE, "--format", "{{.Config.Cmd}}", timeout=60)
    if inspected.returncode != 0:
        pytest.fail(
            f"image {IMAGE!r} is not present. Build it with "
            f"scripts/va/build_and_boot_check.sh, or name another with OSPREY_VA_E2E_IMAGE."
        )


def _ca_environment(port: int) -> dict[str, str]:
    environment = {
        **os.environ,
        "EPICS_CA_NAME_SERVERS": f"localhost:{port}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
    }
    for stale in ("EPICS_CA_ADDR_LIST", "EPICS_CA_SERVER_PORT", "EPICS_CAS_SERVER_PORT"):
        environment.pop(stale, None)
    return environment


def _served(port: int) -> bool:
    """Whether a virtual accelerator is answering on *port*, asked out of process."""
    code = (
        "import sys, epics\n"
        f"v = epics.caget({PROBE_CHANNEL!r}, timeout=1.0, connection_timeout=1.0)\n"
        "sys.stdout.write('SERVED' if v is not None else 'NONE')\n"
        "sys.stdout.flush()\n"
        "import os; os._exit(0)\n"
    )
    try:
        probe = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            env=_ca_environment(port),
        )
    except subprocess.TimeoutExpired:
        return False
    return probe.stdout.strip() == "SERVED"


@contextlib.contextmanager
def _serving(prefix: str, *, instance: str):
    """Boot one virtual accelerator container as *instance*; yield its CA port."""
    port = _free_port()
    pva_port = _free_port()
    name = f"{prefix}-{port}"
    _docker("rm", "-f", name, timeout=60)
    started = _docker(
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"EPICS_CA_SERVER_PORT={port}",
        "-e",
        f"EPICS_PVAS_SERVER_PORT={pva_port}",
        "-e",
        f"VA_INSTANCE={instance}",
        "-p",
        f"127.0.0.1:{port}:{port}/tcp",
        "-p",
        f"127.0.0.1:{pva_port}:{pva_port}/tcp",
        *e2e_conftest.demo_data_run_args(),
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


@dataclass(frozen=True)
class Endpoints:
    """The Channel Access port of each instance."""

    sandbox: int
    standin: int


@pytest.fixture(scope="module")
def endpoints():
    """Both instances, up and serving, for the life of this module."""
    _require_image()
    with _serving(CONTAINER_SANDBOX, instance=SANDBOX_INSTANCE) as sandbox:
        with _serving(CONTAINER_STANDIN, instance=STANDIN_INSTANCE) as standin:
            yield Endpoints(sandbox=sandbox, standin=standin)


def _block(port: int) -> dict:
    return {
        "timeout_s": CONNECTOR_TIMEOUT_S,
        "probe_channel": PROBE_CHANNEL,
        "gateways": {
            "read_only": {"address": "localhost", "port": port, "use_name_server": True},
            "write_access": {"address": "localhost", "port": port, "use_name_server": True},
        },
    }


@pytest.fixture(scope="module")
def repo(endpoints: Endpoints, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A scratch deployment baselined on the sandbox, with the stand-in beside it."""
    project = e2e_conftest.stage_va_project(tmp_path_factory.mktemp("va_model_status"))
    config = {
        "control_system": {
            "type": "virtual_accelerator",
            "writes_enabled": False,
            "connector": {
                "virtual_accelerator": _block(endpoints.sandbox),
                "live_standin": _block(endpoints.standin),
            },
        },
        "services": {
            "virtual_accelerator": {"port": endpoints.sandbox},
            "live_standin": {"port": endpoints.standin},
        },
        "deployed_services": ["virtual_accelerator", "live_standin"],
    }
    rendered = project.project_dir / "build" / "config.yml"
    rendered.write_text(yaml.safe_dump(config), encoding="utf-8")
    return project.project_dir


def _sim_status(repo: Path, target: str) -> tuple[int, str]:
    """``osprey sim status --target <target>``, run in its own process."""
    code = (
        "import json, os, sys\n"
        "from click.testing import CliRunner\n"
        "from osprey.cli.sim import sim_group\n"
        f"args = ['status', '--repo', {str(repo)!r}, '--target', {target!r}]\n"
        "result = CliRunner().invoke(sim_group, args)\n"
        "sys.stdout.write(json.dumps({'code': result.exit_code, 'output': result.output}))\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
    )
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(REPO_PATHS)}
    for ambient in ("CONFIG_FILE", "OSPREY_CONFIG", "OSPREY_EXECUTION_MODE"):
        environment.pop(ambient, None)
    run = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=STATUS_TIMEOUT_S,
        env=environment,
        cwd=repo,
    )
    assert run.stdout, run.stderr
    reply = json.loads(run.stdout)
    return int(reply["code"]), str(reply["output"])


@pytest.mark.parametrize("target", [TARGET_VA, TARGET_STANDIN])
def test_each_instance_reports_its_physics_model_ok(repo: Path, target: str) -> None:
    code, output = _sim_status(repo, target)

    assert code == 0, output
    assert f"{DEMO_PHYSICS_MODEL}: ok" in output.splitlines()


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_model_status_va.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
