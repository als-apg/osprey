"""Two simulator instances of one render, each on a scenario set of its own.

A deployment that runs the live stand-in serves one simulator view from two
containers. Each mounts its own scenario state directory at the one container
path, so the two machines can run different sets: here the first boots on the
clean model and the stand-in with a corrector kick. This module is the
acceptance gate for that, on the wire:

* **Each instance boots in its own start set.** The render's
  ``simulation.default_scenarios`` names a set per instance, the deploy path
  activates each, and one monitor reads zero on the first instance and the
  kicked orbit on the stand-in.
* **A set applied to one instance moves that instance alone.**
  ``osprey sim apply --target va`` puts the kick on the first instance and the
  stand-in's state file is untouched; ``--target standin`` takes the kick off
  the stand-in, says it changed telemetry only, and leaves the first instance
  kicked.

Both containers serve their monitors without the declared motion
(:data:`~tests.va.e2e.conftest.SUITE_SCENARIOS`), so a monitor reading is the
solved orbit and the two instances, on one model and one kick, read the same
number.

Every Channel Access read happens in a subprocess pointed at one container's
port, for the reason ``test_live_standin`` gives. The containers are plain
``docker run`` over one data root; rendering the two compose blocks is
``tests/deployment/``'s subject.

The whole directory is opt-in behind ``OSPREY_VA_E2E_ENABLE=1``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from osprey.deployment import container_lifecycle
from osprey.simulation.targets import target_state_dir
from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME
from osprey_connectors.types import LIVE_STANDIN, TARGET_STANDIN, TARGET_VA, VIRTUAL_ACCELERATOR
from osprey_connectors_tests._harness._ports import reserved_ports
from tests._live import wait_until
from tests.va._collection import assert_module_collected
from tests.va.e2e import _guarded_measure as gm
from tests.va.e2e import conftest as e2e_conftest

#: Floor for this module's own test count -- a guard against a refactor that
#: leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 6

#: Container-name prefixes; the run's own reserved port is appended.
CONTAINER_PREFIX = {
    TARGET_VA: "osprey-va-e2e-sets-va",
    TARGET_STANDIN: "osprey-va-e2e-sets-standin",
}

#: The instance each target's container serves as.
INSTANCE = {TARGET_VA: VIRTUAL_ACCELERATOR, TARGET_STANDIN: LIVE_STANDIN}

#: Two emulated containers on a laptop boot slowly.
BOOT_TIMEOUT_S = 180.0
#: How long a set may take to show on the wire: the runtime's file sharing,
#: then the composite's rebuild at the new set.
SETTLE_BOUND_S = 60.0

MONITOR = e2e_conftest.KICK_MONITOR
KICK = e2e_conftest.KICK_SCENARIO_NAME
SUITE = list(e2e_conftest.SUITE_SCENARIOS)

#: The render's config: both instances deployed, the first the deployment's
#: own target, and a start set each. Only the stand-in starts kicked.
CONFIG = {
    "control_system": {"type": VIRTUAL_ACCELERATOR},
    "deployed_services": [VIRTUAL_ACCELERATOR, LIVE_STANDIN],
    "simulation": {
        "default_scenarios": {TARGET_VA: SUITE, TARGET_STANDIN: [*SUITE, KICK]},
    },
}


def _caget(port: int, address: str) -> float:
    """Read *address* from the container serving *port*, out of process."""
    code = (
        "import asyncio, os, sys, aioca\n"
        f"v = asyncio.run(aioca.caget({address!r}, timeout=10.0))\n"
        "sys.stdout.write(repr(float(v)))\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
    )
    read = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=e2e_conftest.ca_client_environment(port),
    )
    if read.returncode != 0 or not read.stdout.strip():
        raise RuntimeError(f"caget {address} on port {port} failed: {read.stdout}\n{read.stderr}")
    return float(read.stdout)


@dataclass(frozen=True)
class Deployment:
    """The staged repo and the Channel Access port of each instance's container."""

    root: Path
    ports: dict[str, int]

    def state_file(self, target: str) -> Path:
        return target_state_dir(CONFIG, self.root, target) / ACTIVE_SCENARIOS_FILENAME

    def read(self, target: str) -> float:
        return _caget(self.ports[target], MONITOR)

    def settle(self, target: str, predicate) -> float:
        """Poll the monitor on *target* until *predicate* holds; return the reading."""
        seen: list[float] = []

        def probe() -> bool:
            seen.append(self.read(target))
            return bool(predicate(seen[-1]))

        wait_until(
            probe,
            timeout=SETTLE_BOUND_S,
            interval=0.5,
            what=f"{MONITOR} on {target} settling (last read {seen[-1:]})",
        )
        return seen[-1]

    def sim_apply(self, target: str, *names: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(e2e_conftest.find_osprey_console_script()),
                "sim",
                "apply",
                "--target",
                target,
                "--yes",
                *names,
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=60,
        )


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Deployment]:
    """One render, its start sets activated as a deploy does, and a container per instance."""
    from tests.cli._lifecycle_build import stub_build
    from tests.fixtures.lifecycle_repo import build_exemplar_repo

    e2e_conftest.require_image_serving_any_port()
    root = build_exemplar_repo(tmp_path_factory.mktemp("instance-sets") / "repo")
    data_dir = stub_build(root, config=yaml.safe_dump(CONFIG)) / "data"
    shutil.copytree(e2e_conftest.demo_data_dir(), data_dir, dirs_exist_ok=True)
    gm.readable_by_anyone(data_dir)

    container_lifecycle._activate_default_scenarios(CONFIG, root)
    state_dirs = {t: target_state_dir(CONFIG, root, t) for t in (TARGET_VA, TARGET_STANDIN)}
    gm.readable_by_anyone(state_dirs[TARGET_VA])

    with reserved_ports(2) as (va_port, standin_port):
        ports = {TARGET_VA: va_port, TARGET_STANDIN: standin_port}

        def serving(target: str):
            return e2e_conftest.serving_container(
                f"{CONTAINER_PREFIX[target]}-{ports[target]}",
                ports[target],
                "-e",
                f"VA_INSTANCE={INSTANCE[target]}",
                *e2e_conftest.data_root_run_args(data_dir),
                "-v",
                f"{state_dirs[target]}:/state/simulation:ro",
                "-e",
                "VA_STATE_DIR=/state/simulation",
                boot_timeout=BOOT_TIMEOUT_S,
            )

        with serving(TARGET_VA), serving(TARGET_STANDIN):
            yield Deployment(root=root, ports=ports)


@dataclass(frozen=True)
class Readings:
    """The monitor on each instance at each step, and what the commands printed."""

    booted: dict[str, float]
    after_va_apply: dict[str, float]
    standin_state_kept: bool
    after_standin_apply: dict[str, float]
    va_state_kept: bool
    standin_apply_output: str


@pytest.fixture(scope="module")
def readings(deployment: Deployment) -> Readings:
    """Boot, kick the first instance, then take the kick off the stand-in."""
    booted = {
        TARGET_STANDIN: deployment.settle(TARGET_STANDIN, lambda value: value != 0.0),
        TARGET_VA: deployment.read(TARGET_VA),
    }

    standin_state = deployment.state_file(TARGET_STANDIN).read_bytes()
    applied = deployment.sim_apply(TARGET_VA, *SUITE, KICK, "--no-seed")
    assert applied.returncode == 0, applied.stdout + applied.stderr
    after_va_apply = {
        TARGET_VA: deployment.settle(TARGET_VA, lambda value: value != 0.0),
        TARGET_STANDIN: deployment.read(TARGET_STANDIN),
    }
    standin_state_kept = deployment.state_file(TARGET_STANDIN).read_bytes() == standin_state

    va_state = deployment.state_file(TARGET_VA).read_bytes()
    # No seeding flag: on the stand-in the command seeds nothing by itself.
    cleared = deployment.sim_apply(TARGET_STANDIN, *SUITE)
    assert cleared.returncode == 0, cleared.stdout + cleared.stderr
    after_standin_apply = {
        TARGET_STANDIN: deployment.settle(TARGET_STANDIN, lambda value: value == 0.0),
        TARGET_VA: deployment.read(TARGET_VA),
    }
    return Readings(
        booted=booted,
        after_va_apply=after_va_apply,
        standin_state_kept=standin_state_kept,
        after_standin_apply=after_standin_apply,
        va_state_kept=deployment.state_file(TARGET_VA).read_bytes() == va_state,
        standin_apply_output=cleared.stdout + cleared.stderr,
    )


def test_each_instance_mounts_a_state_file_of_its_own(deployment: Deployment) -> None:
    assert deployment.state_file(TARGET_VA) != deployment.state_file(TARGET_STANDIN)
    assert deployment.state_file(TARGET_VA).is_file()
    assert deployment.state_file(TARGET_STANDIN).is_file()


def test_each_instance_boots_in_its_own_start_set(readings: Readings) -> None:
    assert readings.booted[TARGET_VA] == 0.0
    assert readings.booted[TARGET_STANDIN] != 0.0


def test_a_set_applied_to_va_moves_va_alone(readings: Readings) -> None:
    kicked = readings.booted[TARGET_STANDIN]

    assert readings.after_va_apply[TARGET_VA] == pytest.approx(kicked)
    assert readings.after_va_apply[TARGET_STANDIN] == kicked
    assert readings.standin_state_kept


def test_a_set_applied_to_the_standin_moves_the_standin_alone(readings: Readings) -> None:
    assert readings.after_standin_apply[TARGET_STANDIN] == 0.0
    assert readings.after_standin_apply[TARGET_VA] == readings.after_va_apply[TARGET_VA]
    assert readings.va_state_kept


def test_a_set_on_the_standin_says_it_changed_telemetry_only(readings: Readings) -> None:
    assert "Active scenarios on standin" in readings.standin_apply_output
    assert "telemetry only" in readings.standin_apply_output


# ---------------------------------------------------------------------------


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    assert_module_collected(request, __file__, MIN_COLLECTED_TESTS)
