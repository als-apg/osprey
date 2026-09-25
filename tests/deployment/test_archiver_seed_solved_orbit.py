"""The seeded past of a lattice-served monitor is centred on the orbit it serves.

A monitor the served tree binds to the lattice reads the closed orbit the model
solves, not the level its ``machine.json`` entry declares. The deploy-time seed
therefore synthesizes that monitor's history around the solved orbit, so a
seeded sample and a recorded one describe the same beam: at one instant, with
no write having moved the orbit, they are the same number.

This runs on the shipped tree (a real lattice under the readings) mounted into
a throwaway project, and checks the seed against the bridge the virtual
accelerator serves through -- end to end, not against a restatement of either.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from osprey.deployment import container_lifecycle
from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge
from osprey.services.virtual_accelerator.lattice.errors import resolve_device_seeds
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS
from osprey.services.virtual_accelerator.manifest.standin_defaults import parse_standin_default
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel
from osprey.simulation.archiver_seed import synthesize_documents
from osprey.simulation.engine import SimulationEngine

#: Whole seconds, so the instants the seed stores and the ones the bridge is
#: clocked at are the same float.
_INSTANTS = np.asarray([1_764_000_000.0 + 3_600.0 * k for k in range(6)])

_STANDIN_CONFIG = {
    "services": {"live_standin": {"port": 5164}},
    "deployed_services": ["mongodb", "archiver_recorder", "live_standin"],
}


class _Record:
    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(scope="module")
def channels() -> list[dict]:
    return build_manifest()["channels"]


@pytest.fixture(scope="module")
def monitors() -> dict[str, str]:
    """Every monitor reading the shipped tree publishes -> the element it sits at."""
    document = load_bindings(PACKAGE_PATHS.va_bindings)
    return {
        binding.setpoint_address: str(binding.element)
        for binding in document.bindings
        if binding.kind == "monitor" and binding.element is not None
    }


def _project(tmp_path: Path, channels: list[dict], *, lattice: str | None) -> Path:
    """A deployment whose render mounts the shipped tree.

    The tree is linked, never copied or written: the manifest the seed reads is
    placed beside the project and named by absolute path.
    """
    build = tmp_path / "build"
    build.mkdir()
    (build / "data").symlink_to(PACKAGE_PATHS.data_root, target_is_directory=True)
    manifest = tmp_path / "channel_manifest.json"
    manifest.write_text(json.dumps({"channels": channels}))
    lattice_line = "" if lattice is None else f"VA_LATTICE={lattice}\n"
    (tmp_path / ".env").write_text(f"VA_CHANNELS_FILE={manifest}\n{lattice_line}")
    return tmp_path


def _config(extra: dict | None = None) -> dict:
    machine = PACKAGE_PATHS.data_root / "simulation" / "machine.json"
    config = {"control_system": {"connector": {"mock": {"simulation_file": str(machine)}}}}
    return {**config, **(extra or {})}


def _served_truth(channels: list[dict]) -> dict[str, float]:
    return PhysicsBridge(PyATRingModel(PACKAGE_PATHS.data_root, channels)).bpm_positions()


def _seeded(project: Path, config: dict, addresses: list[str]) -> dict[str, list[float]]:
    """Each address's seeded values at :data:`_INSTANTS`, as the deploy seeds them."""
    channels, engine, boot_values, transform, _ = container_lifecycle._archiver_seed_inputs(
        config, project
    )
    wanted = [channel for channel in channels if channel["address"] in addresses]
    documents = synthesize_documents(
        wanted, _INSTANTS, engine=engine, boot_values=boot_values, value_transform=transform
    )
    return {address: [document[address] for document in documents] for address in addresses}


def _recorded(
    channels: list[dict], addresses: list[str], bpm_errors: dict | None = None
) -> dict[str, list[float]]:
    """Each address's live reading at :data:`_INSTANTS`, as the bridge serves it."""
    machine = PACKAGE_PATHS.data_root / "simulation" / "machine.json"
    engine = SimulationEngine(json.loads(machine.read_text()), machine, state_dir=machine.parent)
    clock = _Clock()
    clock.now = float(_INSTANTS[0])
    bridge = PhysicsBridge(
        PyATRingModel(PACKAGE_PATHS.data_root, channels, bpm_errors=bpm_errors),
        motion=engine,
        clock=clock,
    )
    records = {address: _Record() for address in addresses}
    bridge.bind(records)
    served: dict[str, list[float]] = {address: [] for address in addresses}
    for t in _INSTANTS:
        clock.now = float(t)
        bridge.tick()
        for address in addresses:
            served[address].append(records[address].value)
    return served


class TestTheSeamIsGone:
    def test_a_seeded_sample_and_a_recorded_one_are_the_same_number(
        self, tmp_path: Path, channels: list[dict], monitors: dict[str, str]
    ) -> None:
        project = _project(tmp_path, channels, lattice="lattice.json")
        addresses = sorted(monitors)

        seeded = _seeded(project, _config(), addresses)
        recorded = _recorded(channels, addresses)

        assert seeded == recorded
        # Not trivially equal: the orbit is off zero somewhere, and it moves.
        truth = _served_truth(channels)
        assert any(value != 0.0 for value in truth.values())
        assert len({v for series in seeded.values() for v in series}) > len(addresses)

    def test_the_standins_offsets_still_apply_on_top(
        self, tmp_path: Path, channels: list[dict], monitors: dict[str, str]
    ) -> None:
        project = _project(tmp_path, channels, lattice="lattice.json")
        addresses = sorted(monitors)
        offsets = resolve_device_seeds(
            parse_standin_default(), monitors, device="monitor", seeds="readout errors"
        )

        seeded = _seeded(project, _config(_STANDIN_CONFIG), addresses)
        recorded = _recorded(channels, addresses, bpm_errors=offsets)
        clean = _seeded(project, _config(), addresses)

        assert seeded == recorded
        displaced = [address for address in addresses if seeded[address] != clean[address]]
        assert displaced


class TestTheBaselinesAreTheServedOrbit:
    def test_every_served_monitor_reading_is_rebased_on_its_solved_value(
        self, tmp_path: Path, channels: list[dict], monitors: dict[str, str]
    ) -> None:
        project = _project(tmp_path, channels, lattice="lattice.json")
        baselines = container_lifecycle._solved_monitor_baselines(project, channels)
        assert baselines == _served_truth(channels)
        assert set(baselines) == set(monitors)

    def test_the_machine_file_is_left_as_authored(
        self, tmp_path: Path, channels: list[dict]
    ) -> None:
        machine = PACKAGE_PATHS.data_root / "simulation" / "machine.json"
        before = hashlib.sha256(machine.read_bytes()).hexdigest()
        _seeded(_project(tmp_path, channels, lattice="lattice.json"), _config(), [])
        assert hashlib.sha256(machine.read_bytes()).hexdigest() == before


class TestTheFingerprintTracksTheBaselines:
    def _fingerprint(self, project: Path, config: dict):
        return container_lifecycle._archiver_seed_inputs(config, project)[4]

    def test_a_lattice_served_seed_says_so(self, tmp_path: Path, channels: list[dict]) -> None:
        fingerprint = self._fingerprint(
            _project(tmp_path, channels, lattice="lattice.json"), _config()
        )
        assert fingerprint is not None
        assert fingerprint["kind"] == "solved_monitor_baselines"

    def test_a_moved_orbit_is_a_different_store(
        self, tmp_path: Path, channels: list[dict], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = _project(tmp_path, channels, lattice="lattice.json")
        before = self._fingerprint(project, _config())
        solved = container_lifecycle._solved_monitor_baselines

        def shifted(project_dir: Path, served: list[dict]) -> dict[str, float]:
            return {address: value + 1e-9 for address, value in solved(project_dir, served).items()}

        monkeypatch.setattr(container_lifecycle, "_solved_monitor_baselines", shifted)
        assert self._fingerprint(project, _config()) != before

    def test_the_standins_offsets_are_still_in_it(
        self, tmp_path: Path, channels: list[dict]
    ) -> None:
        project = _project(tmp_path, channels, lattice="lattice.json")
        fingerprint = self._fingerprint(project, _config(_STANDIN_CONFIG))
        assert fingerprint["readout"]["kind"] == "bpm_offsets"
        assert fingerprint["readout"]["offsets"]
        assert self._fingerprint(project, _config())["readout"] is None


class TestADeploymentWithoutALatticeIsUnaffected:
    @pytest.mark.parametrize("lattice", [None, "none"])
    def test_no_baselines_and_the_fingerprint_it_always_had(
        self, tmp_path: Path, channels: list[dict], monitors: dict[str, str], lattice: str | None
    ) -> None:
        project = _project(tmp_path, channels, lattice=lattice)
        assert container_lifecycle._solved_monitor_baselines(project, channels) == {}
        _channels, engine, boot_values, transform, fingerprint = (
            container_lifecycle._archiver_seed_inputs(_config(), project)
        )
        assert transform is None
        assert fingerprint is None
        address = sorted(monitors)[0]
        assert boot_values[address] == 0.0
        at = datetime.fromtimestamp(float(_INSTANTS[0]), UTC)
        machine = PACKAGE_PATHS.data_root / "simulation" / "machine.json"
        plain = SimulationEngine(json.loads(machine.read_text()), machine, state_dir=tmp_path)
        assert engine.synthesize_series(address, [at]) == plain.synthesize_series(address, [at])
