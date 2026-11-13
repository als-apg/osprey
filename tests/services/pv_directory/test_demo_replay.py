"""Tests for the alarm replay packaged with OSPREY as ``demo:example-alarms``.

The replay names only addresses the example facility's virtual accelerator
serves, folds through the real replay backend into one latched, one acknowledged
and one cleared alarm, each configured in the alarm tree, and reads fresh when
loaded. The resolver finds it where an installed wheel puts it.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("osprey_connectors.service_answers")

from osprey_connectors.alarm_state import AlarmStateRecord
from osprey_connectors.service_answers import BackendSpec

_replay = pytest.importorskip("osprey_connectors.alarm_state._replay")

from osprey.services.pv_directory import demo  # noqa: E402
from osprey_connectors.simulation.view import SimulatorView  # noqa: E402
from tests._facility_file import demo_facility_document  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
NAME = "demo:example-alarms"
PACKAGED = Path(demo.__file__).parent / "example-alarms.ndjson"
EXAMPLE_FACILITY = REPO_ROOT / "src/osprey/templates/facilities/example"

LATCHED = "SR:RF:CAVITY:01:TEMPERATURE:RB"
ACKNOWLEDGED = "SR:VAC:GAUGE:SR03:PRESSURE:RB"
CLEARED = "ca:SIM:SR:HEARTBEAT"


def _messages() -> list[dict[str, Any]]:
    lines = PACKAGED.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _replay_pvs() -> set[str]:
    """The PV each message's tree path ends in."""
    pvs = set()
    for message in _messages():
        _, _, path = message["key"].partition(":")
        pvs.add(path.rsplit("/", 1)[-1])
    return pvs


@pytest.fixture(scope="module")
def example_view(tmp_path_factory: pytest.TempPathFactory) -> SimulatorView:
    """The example facility's simulator view, as a build renders it."""
    from osprey.facility.records import FacilityRecords
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.simulator import write_simulator_view

    config = {"control_system": {"type": "virtual_accelerator"}}
    records = FacilityRecords(demo_facility_document())
    target = tmp_path_factory.mktemp("example-view") / "simulator"
    write_simulator_view(
        target,
        ViewInputs(
            records=records,
            rendered_config=config,
            facility_dir=EXAMPLE_FACILITY,
            served=resolve_served(config, records),
            reported=None,
        ),
    )
    return SimulatorView.open(target)


def _fold() -> tuple[dict[str, Any], dict[str, Any], str]:
    """Every replay PV's state and configuration, and the probe, from one load."""
    spec = BackendSpec(
        kind="alarm_state",
        type="alarm_replay",
        target="va",
        settings={"replay_file": str(demo.resolve_replay_file(NAME, None))},
        rehearsal=False,
    )
    backend = _replay.ReplayAlarmState(spec)

    async def run() -> tuple[dict[str, Any], dict[str, Any], str]:
        await backend.start()
        try:
            pvs = sorted(_replay_pvs())
            states = {pv: await backend.state_for(pv, 1.0) for pv in pvs}
            configs = {pv: await backend.config_for(pv, 1.0) for pv in pvs}
            return states, configs, await backend.probe(1.0)
        finally:
            await backend.aclose()

    return asyncio.run(run())


class TestAddresses:
    def test_every_replay_pv_is_served_by_the_example_va(self, example_view: SimulatorView):
        served = set(example_view.channels())
        served.update(example_view.status_addresses().values())
        for signals in example_view.ioc_health_by_model().values():
            served.update(signals.values())

        assert _replay_pvs() - served == set()

    def test_the_replay_covers_an_ioc_health_address(self, example_view: SimulatorView):
        health = {
            address
            for signals in example_view.ioc_health_by_model().values()
            for address in signals.values()
        }

        assert CLEARED in health


class TestFold:
    def test_the_file_folds_and_is_not_stale_on_load(self):
        states, _, probe = _fold()

        assert all(isinstance(state, AlarmStateRecord) for state in states.values())
        assert not any(state.stale for state in states.values())
        assert "stopped" not in probe

    def test_one_latched_one_acknowledged_one_cleared(self):
        states, _, _ = _fold()

        latched, acked, cleared = states[LATCHED], states[ACKNOWLEDGED], states[CLEARED]
        assert (latched.latched, latched.acknowledged, latched.severity) == (True, False, "MAJOR")
        assert (acked.latched, acked.acknowledged, acked.severity) == (False, True, "MINOR")
        assert (cleared.latched, cleared.acknowledged, cleared.severity) == (False, False, "OK")

    def test_each_alarm_is_configured_in_the_tree(self):
        _, configs, _ = _fold()

        for pv in (LATCHED, ACKNOWLEDGED, CLEARED):
            config = configs[pv]
            assert config.in_tree == "yes", pv
            assert config.path is not None and config.path.endswith(f"/{pv}"), pv
            assert config.guidance, pv
            assert config.displays, pv


class TestInstalledLayout:
    def test_the_wheel_carries_the_replay(self):
        """Hatchling packages ``src/osprey`` minus git-ignored and excluded paths."""
        relative = PACKAGED.relative_to(REPO_ROOT)
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(relative)], cwd=REPO_ROOT, check=False
        )
        build = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        hatch = build["tool"]["hatch"]["build"]
        excludes = [*hatch.get("exclude", ()), *hatch["targets"]["wheel"].get("exclude", ())]

        assert ignored.returncode == 1
        assert not any(fnmatch.fnmatch(str(relative), pattern) for pattern in excludes)

    def test_the_resolver_finds_it_in_site_packages(self, tmp_path: Path):
        """Only the packaged directory and empty parents, outside the source tree."""
        site = tmp_path / "site-packages"
        package = site / "osprey" / "services" / "pv_directory"
        shutil.copytree(PACKAGED.parent, package / "demo", ignore=shutil.ignore_patterns("*.pyc"))
        for parent in (site / "osprey", site / "osprey/services", package):
            (parent / "__init__.py").write_text("", encoding="utf-8")
        probe = (
            "import sys; sys.path.insert(0, sys.argv[1])\n"
            "from osprey.services.pv_directory.demo import resolve_replay_file\n"
            f"path = resolve_replay_file({NAME!r}, None)\n"
            "print(path.resolve(), path.is_file())\n"
        )

        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", probe, str(site)],
            capture_output=True,
            text=True,
            check=True,
        )

        resolved = package / "demo" / "example-alarms.ndjson"
        assert result.stdout.split() == [str(resolved.resolve()), "True"]
