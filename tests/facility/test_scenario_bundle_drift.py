"""Each simulation bundle and its translation in ``data/facility/scenarios/`` agree.

The bundles under ``data/simulation/scenarios/`` and their translations ship
side by side; these checks hold the two to one content: the same scenario
names, override addresses, archiver scripts, logbook entries, description and
shared-driver blocks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data"
BUNDLES = DATA / "simulation" / "scenarios"
SCENARIOS = DATA / "facility" / "scenarios"

NAMES = sorted(path.name for path in BUNDLES.iterdir() if path.is_dir())


def _bundle(name: str) -> dict[str, Any]:
    return json.loads((BUNDLES / name / "scenario.json").read_text(encoding="utf-8"))


def _logbook(name: str) -> list[Any] | None:
    path = BUNDLES / name / "logbook.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _translation(name: str) -> dict[str, Any]:
    return yaml.safe_load((SCENARIOS / f"{name}.yaml").read_text(encoding="utf-8"))


def test_the_scenario_names_are_the_bundle_names() -> None:
    assert sorted(path.stem for path in SCENARIOS.glob("*.yaml")) == NAMES
    assert len(NAMES) == 6


@pytest.mark.parametrize("name", NAMES)
def test_the_override_addresses_agree(name: str) -> None:
    assert sorted(_translation(name).get("overrides") or {}) == sorted(
        _bundle(name).get("overrides") or {}
    )


@pytest.mark.parametrize("name", NAMES)
def test_the_archiver_scripts_agree(name: str) -> None:
    assert _translation(name).get("archiver") == _bundle(name).get("archiver")


@pytest.mark.parametrize("name", NAMES)
def test_the_logbook_entries_agree(name: str) -> None:
    assert _translation(name).get("logbook") == _logbook(name)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("slot", ["description", "drivers", "couple", "noise"])
def test_the_carried_slots_agree(name: str, slot: str) -> None:
    assert _translation(name).get(slot) == _bundle(name).get(slot)
