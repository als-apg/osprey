"""The ORM and bump e2e stacks serve their monitors without declared motion.

Those lanes compare what the deployed stack measures with the noiseless model,
so ``tests/e2e/_orm_stack.still_monitor_motion`` removes the drift and noise the
deployment's ``machine.json`` gives its monitor readings before the build
stages it. These tests run it on the shipped preset's own data tree: every
monitor the bindings claim is left without motion, and every other channel is
left exactly as the file declares it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS, ManifestPaths
from osprey.simulation.engine import SimulationEngine


def _repo_with_preset_data(tmp_path: Path) -> Path:
    """A deployment repo whose ``data/simulation`` is the shipped preset's."""
    repo = tmp_path / "repo"
    simulation = ManifestPaths(repo / "data").machine_json.parent
    simulation.mkdir(parents=True)
    shutil.copy2(PACKAGE_PATHS.machine_json, simulation / "machine.json")
    shutil.copy2(PACKAGE_PATHS.va_bindings, simulation / "va_bindings.json")
    return repo


def _monitors() -> set[str]:
    return {
        address
        for binding in load_bindings(PACKAGE_PATHS.va_bindings).bindings
        if binding.kind == "monitor"
        for address in (binding.setpoint_address, binding.readback_address)
        if address is not None
    }


def test_every_monitor_is_stilled_and_nothing_else_changes(tmp_path: Path) -> None:
    from tests.e2e._orm_stack import still_monitor_motion

    repo = _repo_with_preset_data(tmp_path)
    machine_json = ManifestPaths(repo / "data").machine_json
    declared = json.loads(PACKAGE_PATHS.machine_json.read_text(encoding="utf-8"))["channels"]
    monitors = _monitors()
    preset = SimulationEngine.from_file(PACKAGE_PATHS.machine_json, state_dir=tmp_path)
    moving = {address for address in monitors if preset.has_motion(address)}
    assert moving, "the preset declares no monitor motion, so there is nothing to still"

    stilled = still_monitor_motion(repo)

    assert stilled == moving
    engine = SimulationEngine.from_file(machine_json, state_dir=tmp_path)
    assert not [address for address in monitors if engine.has_motion(address)]
    written = json.loads(machine_json.read_text(encoding="utf-8"))["channels"]
    assert written.keys() == declared.keys()
    assert {address: entry for address, entry in written.items() if address not in monitors} == {
        address: entry for address, entry in declared.items() if address not in monitors
    }
