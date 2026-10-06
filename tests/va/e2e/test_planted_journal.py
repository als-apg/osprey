"""A session-writes journal planted in the state mount moves no served setpoint.

The mock connector keeps the writes it took in ``<state dir>/mock/writes.json``
and replays them in every mock connector on the same view; that journal is the
mock's alone. A virtual accelerator container mounts the same state directory
for its active scenarios, so a journal found there must leave it untouched:
the container serves its wiring, never a write some other process recorded.

This module plants a well-formed journal -- the shape the mock connector
writes, under the active set the containers serve, naming writable setpoints
at values inside their bands and away from their defaults -- before booting
two containers over the same view and state directory, the sandbox
(``VA_INSTANCE=virtual_accelerator``) and the stand-in
(``VA_INSTANCE=live_standin``). A wired setpoint's start value is its wiring
default and a texture setpoint's is its seed's nominal, and the journal names
one of each kind. Three ticks after each container serves, every such setpoint
reads its start value over Channel Access.

A last leg is the control: a mock connector on the same view and state
directory reads the planted values back, so the journal the containers ignored
is one its own reader replays.

Every Channel Access operation happens in a subprocess that leaves through
``os._exit``, for the reasons ``conftest.py`` gives; this process never becomes
a Channel Access client.

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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.va.e2e import conftest as e2e_conftest

#: The image under test.
IMAGE = os.environ.get("OSPREY_VA_E2E_IMAGE", "osprey-va-full:latest")

#: Container-name prefix; ``_serving`` appends the run's own ephemeral port.
CONTAINER_PREFIX = "osprey-va-e2e-journal"

#: The instance each container serves as.
SANDBOX_INSTANCE = "virtual_accelerator"
STANDIN_INSTANCE = "live_standin"
INSTANCES = (SANDBOX_INSTANCE, STANDIN_INSTANCE)

#: A local run on Apple Silicon is emulated.
BOOT_TIMEOUT_S = 180.0

#: What the readiness probe waits for.
PROBE_CHANNEL = "SR:MAG:HCM:01:CURRENT:RB"

#: The containers' tick, stated rather than left to the default, and how many
#: of them pass before the setpoints are read.
TICK_S = 1.0
TICKS = 3

#: The prefix of the one line a Channel Access child reports on; the client
#: library writes notices of its own to the same stream.
REPORT_MARK = "REPORT "

#: Bound on one Channel Access subprocess reading every wired setpoint.
CA_CHILD_TIMEOUT_S = 120.0

#: The planted writes: wired setpoints and one texture setpoint, each at a value
#: inside its band and away from its start value.
PLANTED: dict[str, float] = {
    "SR:MAG:HCM:01:CURRENT:SP": 3.0,
    "SR:MAG:HCM:02:CURRENT:SP": 2.5,
    "SR:VAC:ION-PUMP:02:VOLTAGE:SP": 4000.0,
}

#: The mock connector's journal, under the state directory.
JOURNAL = ("mock", "writes.json")


# ---------------------------------------------------------------------------
# The served tree, its wiring defaults and the planted journal
# ---------------------------------------------------------------------------


def _wiring_defaults(view: Path) -> dict[str, float]:
    """Every writable setpoint the view's wiring gives a default, with that default."""
    document = json.loads((view / "variables.json").read_text(encoding="utf-8"))
    writable = {
        str(channel["address"])
        for channel in document["channels"]
        if channel.get("role") == "setpoint" and channel.get("writable") is True
    }
    return {
        str(entry["address"]): float(entry["default"])
        for model in document["models"]
        for entry in model.get("wiring") or []
        if entry.get("direction") == "write"
        and entry.get("default") is not None
        and str(entry["address"]) in writable
    }


def _seed_nominals(view: Path) -> dict[str, float]:
    """Every writable texture setpoint whose seed states a nominal, with that nominal."""
    document = json.loads((view / "variables.json").read_text(encoding="utf-8"))
    seeds = json.loads((view / "seeds.json").read_text(encoding="utf-8"))["seeds"]
    nominals: dict[str, float] = {}
    for channel in document["channels"]:
        address = str(channel["address"])
        if (
            channel.get("role") == "setpoint"
            and channel.get("writable") is True
            and channel.get("owner") == "texture"
            and (seeds.get(address) or {}).get("nominal") is not None
        ):
            nominals[address] = float(seeds[address]["nominal"])
    return nominals


def _plant_journal(state_dir: Path) -> Path:
    """Write the journal the mock connector would have written; return its path."""
    from osprey_connectors.control_system.mock_connector import active_set_sha256

    active = (state_dir / "active_scenarios").read_text(encoding="utf-8").split()
    path = state_dir.joinpath(*JOURNAL)
    path.parent.mkdir(parents=True, exist_ok=True)
    writes = [[seq, address, value] for seq, (address, value) in enumerate(PLANTED.items(), 1)]
    document = {
        "active_set_sha256": active_set_sha256(active),
        "seq": len(writes),
        "writes": writes,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _readable_by_anyone(root: Path) -> None:
    """Open *root*'s tree to the container's user, which is not this process's."""
    for directory, _subdirs, files in os.walk(root):
        Path(directory).chmod(0o755)
        for name in files:
            (Path(directory) / name).chmod(0o644)


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> e2e_conftest.VaProject:
    """A scratch deployment whose state directory holds the planted journal."""
    staged = e2e_conftest.stage_va_project(tmp_path_factory.mktemp("va_planted_journal"))
    _plant_journal(staged.state_dir)
    _readable_by_anyone(staged.state_dir)
    _readable_by_anyone(staged.data_dir)
    return staged


@pytest.fixture(scope="module")
def defaults(project: e2e_conftest.VaProject) -> dict[str, float]:
    """The start value of every writable setpoint the containers serve with one."""
    view = project.data_dir / "simulator"
    wired = _wiring_defaults(view)
    textured = _seed_nominals(view)
    assert not wired.keys() & textured.keys(), "a setpoint is both wired and texture-owned"
    found = {**wired, **textured}
    for address, value in PLANTED.items():
        assert address in found, f"{address} has no start value in the served view"
        assert value != found[address], f"{address} is planted at its own start value"
    return found


# ---------------------------------------------------------------------------
# The containers
# ---------------------------------------------------------------------------


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


def _caget_many(port: int, addresses: list[str], *, timeout: float) -> dict[str, Any]:
    """Each address's value on the container serving *port*, read out of process.

    An address that does not answer reads ``None``.
    """
    code = (
        "import json, os, sys, epics\n"
        f"names = {addresses!r}\n"
        f"values = epics.caget_many(names, connection_timeout={timeout}, timeout={timeout})\n"
        "out = {n: (None if v is None else float(v)) for n, v in zip(names, values)}\n"
        f"sys.stdout.write('\\n' + {REPORT_MARK!r} + json.dumps(out) + '\\n')\n"
        "sys.stdout.flush()\n"
        "os._exit(0)\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=CA_CHILD_TIMEOUT_S,
        env=_ca_environment(port),
    )
    lines = [line for line in run.stdout.splitlines() if line.startswith(REPORT_MARK)]
    assert lines, f"the Channel Access child reported nothing:\n{run.stdout}\n{run.stderr}"
    report: dict[str, Any] = json.loads(lines[-1][len(REPORT_MARK) :])
    return report


def _served(port: int) -> bool:
    """Whether the container on *port* answers the probe channel, asked out of process."""
    try:
        return _caget_many(port, [PROBE_CHANNEL], timeout=1.0)[PROBE_CHANNEL] is not None
    except (subprocess.TimeoutExpired, AssertionError):
        return False


@contextlib.contextmanager
def _serving(project: e2e_conftest.VaProject, *, instance: str) -> Iterator[int]:
    """Boot one container as *instance* over the project's view and state; yield its port."""
    port = _free_port()
    name = f"{CONTAINER_PREFIX}-{port}"
    _docker("rm", "-f", name, timeout=60)
    started = _docker(
        "run",
        "-d",
        "--name",
        name,
        "-e",
        f"EPICS_CA_SERVER_PORT={port}",
        "-e",
        f"VA_INSTANCE={instance}",
        "-e",
        f"VA_POLL_INTERVAL_S={TICK_S}",
        "-e",
        "VA_STATE_DIR=/state/simulation",
        "-p",
        f"127.0.0.1:{port}:{port}/tcp",
        *e2e_conftest.data_root_run_args(project.data_dir),
        "-v",
        f"{project.state_dir}:/state/simulation:ro",
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
def ports(project: e2e_conftest.VaProject) -> Iterator[dict[str, int]]:
    """Both instances, booted after the journal was planted, by instance name."""
    _require_image()
    with _serving(project, instance=SANDBOX_INSTANCE) as sandbox:
        with _serving(project, instance=STANDIN_INSTANCE) as standin:
            yield {SANDBOX_INSTANCE: sandbox, STANDIN_INSTANCE: standin}


# ---------------------------------------------------------------------------
# The legs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("instance", INSTANCES)
def test_every_setpoint_holds_its_wiring_default_after_three_ticks(
    ports: dict[str, int], defaults: dict[str, float], instance: str
) -> None:
    time.sleep(TICKS * TICK_S)

    served = _caget_many(ports[instance], sorted(defaults), timeout=10.0)

    unanswered = sorted(address for address, value in served.items() if value is None)
    moved = {
        address: (served[address], default)
        for address, default in defaults.items()
        if served[address] is not None
        and served[address] != pytest.approx(default, rel=1e-9, abs=1e-12)
    }
    assert not unanswered, f"{instance} does not answer {unanswered}"
    assert not moved, f"{instance} serves setpoints away from their wiring default: {moved}"


@pytest.mark.asyncio
async def test_the_mock_connector_replays_the_planted_journal(
    project: e2e_conftest.VaProject, ports: dict[str, int]
) -> None:
    from osprey.connectors.control_system.mock_connector import MockConnector

    del ports  # the containers read the journal first, and the mock may rewrite it
    connector = MockConnector()
    with e2e_conftest.patched_config():
        await connector.connect(
            {"simulator_view": str(project.data_dir / "simulator"), "response_delay_ms": 0}
        )
        try:
            replayed = {
                address: (await connector.read_channel(address)).value for address in PLANTED
            }
        finally:
            await connector.disconnect()

    assert replayed == pytest.approx(PLANTED)
