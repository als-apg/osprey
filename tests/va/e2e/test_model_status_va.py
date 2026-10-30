"""``osprey sim status`` on a virtual accelerator target, against both instances.

A target served by a separate process reports each served physics model's
status on its status channel, ``<code>:SIM:<model>:STATUS``, and ``sim status``
reads it through the target's connector. This module boots two real
``osprey-va-full`` containers over one simulator view, the sandbox (the ``va``
target) and the stand-in (the ``standin`` target), points a scratch deployment
at them, and runs ``sim status --target`` for each: both print ``SR: ok``.

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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from osprey_connectors.types import TARGET_STANDIN, TARGET_VA
from osprey_connectors_tests._harness._ports import reserved_ports
from tests.va._collection import assert_module_collected
from tests.va.e2e import conftest as e2e_conftest

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_PATHS = (str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src"))

#: Container-name prefixes; ``_serving`` appends the run's own reserved port.
CONTAINER_SANDBOX = "osprey-va-e2e-status-va"
CONTAINER_STANDIN = "osprey-va-e2e-status-standin"

#: The instance each container serves as.
SANDBOX_INSTANCE = "virtual_accelerator"
STANDIN_INSTANCE = "live_standin"

#: Two containers, and a local run on Apple Silicon is emulated.
BOOT_TIMEOUT_S = 180.0

#: What the readiness probe waits for: a readback both instances serve.
PROBE_CHANNEL = e2e_conftest.READINESS_ADDRESS

#: The connector's own timeout.
CONNECTOR_TIMEOUT_S = 120.0

#: Bound on one ``sim status`` run.
STATUS_TIMEOUT_S = 120.0

#: The demo's physics model.
DEMO_PHYSICS_MODEL = "SR"

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 2


@contextlib.contextmanager
def _serving(prefix: str, *, instance: str):
    """Boot one virtual accelerator container as *instance*; yield its CA port."""
    with reserved_ports(2) as (port, pva_port):
        with e2e_conftest.serving_container(
            f"{prefix}-{port}",
            port,
            "-e",
            f"EPICS_PVAS_SERVER_PORT={pva_port}",
            "-e",
            f"VA_INSTANCE={instance}",
            "-p",
            f"127.0.0.1:{pva_port}:{pva_port}/tcp",
            *e2e_conftest.demo_data_run_args(),
            boot_timeout=BOOT_TIMEOUT_S,
        ):
            yield port


@dataclass(frozen=True)
class Endpoints:
    """The Channel Access port of each instance."""

    sandbox: int
    standin: int


@pytest.fixture(scope="module")
def endpoints():
    """Both instances, up and serving, for the life of this module."""
    e2e_conftest.require_image()
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
            "writes": "never",
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
    assert_module_collected(request, __file__, MIN_COLLECTED_TESTS)
