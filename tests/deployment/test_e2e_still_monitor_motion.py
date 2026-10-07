"""The noiseless-oracle e2e stacks serve their readings without declared motion.

The live VA suites compare what the stack serves with the noiseless model, so
``tests/e2e/_monitor_motion`` removes the noise and drift the facility's
``seeds.yaml`` gives every monitor reading -- an address whose wiring in
``models.yaml`` reads an ``axis`` -- before the tree is built or mounted. The
ORM and bump round trips also settle on and read back the corrector readbacks,
so they still every reading the model wires instead. These tests run both on
the shipped example facility: every moving reading in scope is left without
motion, and every other seed and key is left exactly as the facility declares it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

#: The example facility the control-assistant preset ships, the tree the live
#: VA suites render their container's view from.
PRESET_FACILITY_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates" / "facilities" / "example"
)
#: The seed keys that move a reading on its own.
MOTION_KEYS = ("noise", "drift")


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _repo_with_preset_facility(tmp_path: Path) -> Path:
    """A deployment repo whose ``data/facility`` is the shipped example facility."""
    repo = tmp_path / "repo"
    shutil.copytree(PRESET_FACILITY_DIR, repo / "data" / "facility")
    return repo


def _monitors(models: list[dict[str, Any]]) -> set[str]:
    return {
        str(record["address"])
        for model in models
        for record in model.get("wiring") or []
        if "axis" in (record.get("engine") or {})
    }


def test_every_moving_monitor_is_stilled_and_nothing_else_changes(tmp_path: Path) -> None:
    from tests.e2e._monitor_motion import still_monitor_motion

    repo = _repo_with_preset_facility(tmp_path)
    seeds_yaml = repo / "data" / "facility" / "seeds.yaml"
    declared = _load(PRESET_FACILITY_DIR / "seeds.yaml")
    monitors = _monitors(_load(PRESET_FACILITY_DIR / "models.yaml"))
    moving = {
        address
        for address in monitors
        if any(key in (declared.get(address) or {}) for key in MOTION_KEYS)
    }
    assert moving, "the example facility declares no monitor motion, so there is nothing to still"

    stilled = still_monitor_motion(repo / "data")

    assert stilled == moving
    written = _load(seeds_yaml)
    assert written.keys() == declared.keys()
    assert [
        address
        for address in monitors
        if any(key in (written.get(address) or {}) for key in MOTION_KEYS)
    ] == []
    assert {address: seed for address, seed in written.items() if address not in monitors} == {
        address: seed for address, seed in declared.items() if address not in monitors
    }


def _wired(models: list[dict[str, Any]]) -> set[str]:
    return {str(record["address"]) for model in models for record in model.get("wiring") or []}


def test_every_wired_reading_is_stilled_and_nothing_else_changes(tmp_path: Path) -> None:
    from tests.e2e._monitor_motion import still_model_motion

    repo = _repo_with_preset_facility(tmp_path)
    seeds_yaml = repo / "data" / "facility" / "seeds.yaml"
    declared = _load(PRESET_FACILITY_DIR / "seeds.yaml")
    models = _load(PRESET_FACILITY_DIR / "models.yaml")
    wired = _wired(models)
    moving = {
        address
        for address in wired
        if any(key in (declared.get(address) or {}) for key in MOTION_KEYS)
    }
    moving_non_monitors = moving - _monitors(models)
    assert moving_non_monitors, (
        "the example facility declares no motion on a wired non-monitor reading, so this "
        "cannot tell stilling every wired reading from stilling the monitors alone"
    )

    stilled = still_model_motion(repo / "data")

    assert stilled == moving
    assert stilled & moving_non_monitors == moving_non_monitors
    written = _load(seeds_yaml)
    assert written.keys() == declared.keys()
    assert [
        address
        for address in wired
        if any(key in (written.get(address) or {}) for key in MOTION_KEYS)
    ] == []
    assert {address: seed for address, seed in written.items() if address not in wired} == {
        address: seed for address, seed in declared.items() if address not in wired
    }
    assert {
        address: {key: value for key, value in (seed or {}).items() if key not in MOTION_KEYS}
        for address, seed in written.items()
        if address in wired
    } == {
        address: {key: value for key, value in (seed or {}).items() if key not in MOTION_KEYS}
        for address, seed in declared.items()
        if address in wired
    }
