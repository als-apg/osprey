"""The narrative half of a simulated world reaches ARIEL on its own.

``osprey up`` composes a machine out of two halves: the archiver's stored history
(seeded by the deploy) and the logbook that documents it (until now seeded only
by an explicit ``osprey sim apply``). A deployment that shipped one without the
other came up with an ARIEL panel whose "Browse Entries" tab was empty for a
machine whose archive was full.

:func:`seed_active_logbook` closes that: it writes the entries the ALREADY-active
scenarios narrate, at the anchor the running world is already on, and only into a
logbook that has none — an operator's own entries are never overwritten by a
deploy.

DB-free: the two ARIEL calls are stubbed, so the contract is pinned without a
Postgres dependency and runs in the fast suite.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from osprey.simulation.apply import (
    active_logbook_entries,
    apply_scenarios,
    seed_active_logbook,
)
from tests._simulator_view import facility_scenarios, write_scenarios_view

TEMPLATE_DATA = (
    Path(__file__).resolve().parents[2] / "src/osprey/templates/apps/control_assistant/data"
)
TEMPLATE_SIM = TEMPLATE_DATA / "simulation"

ARIEL_CONFIG = {"database": {"uri": "postgresql://unused-mocked/none"}}


def _make_project(tmp_path: Path) -> Path:
    """Stage a sim-backed project and its simulator view."""
    sim_dst = tmp_path / "data" / "simulation"
    sim_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_SIM, sim_dst)
    write_scenarios_view(tmp_path, facility_scenarios(TEMPLATE_DATA / "facility" / "scenarios"))
    config = {
        "control_system": {
            "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}}
        },
        "ariel": ARIEL_CONFIG,
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    return tmp_path


def _activate(project: Path, monkeypatch, names: list[str]) -> None:
    """Activate scenarios the way an operator would, without touching a database."""

    async def _no_seed(_ariel_config, _entries):
        return 0, False

    monkeypatch.setattr("osprey.simulation.apply._seed_logbook", _no_seed)
    apply_scenarios(project, names, seed_archive=False)


def _stub_ariel(monkeypatch, *, existing: int) -> dict:
    """Stub ARIEL's database calls; return what the seeder was asked to write."""
    seen: dict = {"seeded": None, "counted": 0, "mirrored": 0}

    async def _count(_config_dict):
        seen["counted"] += 1
        return existing

    async def _seed(_config_dict, entries, _progress=None):
        seen["seeded"] = entries
        return len(entries)

    async def _resync(config_dict, rebuild=False, page_size=None, progress=None):  # noqa: ARG001 - stands in for run_qmd_resync, whose caller names rebuild and progress
        seen["mirrored"] += 1
        return None

    import osprey.services.ariel_search.cli_operations as ops

    monkeypatch.setattr(ops, "logbook_entry_count", _count, raising=False)
    monkeypatch.setattr(ops, "seed_logbook_entries", _seed)
    monkeypatch.setattr(ops, "run_qmd_resync", _resync)
    return seen


def test_active_logbook_entries_follow_the_active_scenarios(tmp_path, monkeypatch):
    """The entries are the ones the currently-active set narrates -- read back from
    the project's own state, not re-activated from an argument."""
    # Arrange
    project = _make_project(tmp_path)
    _activate(project, monkeypatch, ["rf-thermal"])
    config = yaml.safe_load((project / "config.yml").read_text())

    # Act
    entries = active_logbook_entries(config, project)

    # Assert
    assert entries, "the active scenario narrates entries, so some must be built"
    assert all(entry["timestamp"].tzinfo is not None for entry in entries)
    ids = [entry["entry_id"] for entry in entries]
    assert "DEMO-026" in ids, "rf-thermal's own narrative is part of the active set's"


def test_active_logbook_entries_are_the_view_s_logbook_blocks(tmp_path, monkeypatch):
    """The entries come from the simulator view: a narrative only the view states
    is the one that is seeded."""
    # Arrange
    project = _make_project(tmp_path)
    _activate(project, monkeypatch, ["rf-thermal"])
    config = yaml.safe_load((project / "config.yml").read_text())
    entry = {
        "entry_id": "VIEW-1",
        "when": {"days_ago": 1, "time": "02:00:00"},
        "author": "View",
        "title": "Stated in the view",
        "text": "Only the simulator view carries this entry.",
    }
    write_scenarios_view(project, {"nominal": {}, "rf-thermal": {"logbook": [entry]}})

    # Act
    entries = active_logbook_entries(config, project)

    # Assert
    assert [e["entry_id"] for e in entries] == ["VIEW-1"]


def test_seed_active_logbook_writes_into_an_empty_logbook(tmp_path, monkeypatch):
    """The deploy's own seed: a first bring-up finds no entries and writes the
    narrative that matches the archive it just seeded."""
    # Arrange
    project = _make_project(tmp_path)
    _activate(project, monkeypatch, ["rf-thermal"])
    config = yaml.safe_load((project / "config.yml").read_text())
    seen = _stub_ariel(monkeypatch, existing=0)

    # Act
    seeded = seed_active_logbook(config, project, ARIEL_CONFIG)

    # Assert
    assert seeded == len(seen["seeded"])
    assert seeded > 0
    # Seeding skips the enhancement passes, so nothing else would write the
    # markdown mirror the qmd sidecar indexes -- hybrid search would answer an
    # empty index while keyword search worked.
    assert seen["mirrored"] == 1


def test_seed_active_logbook_never_overwrites_an_existing_logbook(tmp_path, monkeypatch):
    """A logbook with entries in it is history this deploy was not asked to touch --
    an operator's own entries, or a narrative already seeded and since edited."""
    # Arrange
    project = _make_project(tmp_path)
    _activate(project, monkeypatch, ["rf-thermal"])
    config = yaml.safe_load((project / "config.yml").read_text())
    seen = _stub_ariel(monkeypatch, existing=7)

    # Act
    seeded = seed_active_logbook(config, project, ARIEL_CONFIG)

    # Assert
    assert seeded == 0
    assert seen["seeded"] is None
    # Nothing was written, so there is nothing to mirror either.
    assert seen["mirrored"] == 0


def test_seed_active_logbook_is_a_no_op_without_a_simulator_view(tmp_path, monkeypatch):
    """A project with no simulated scenarios has no narrative to seed, which is a
    normal configuration and not a fault."""
    # Arrange
    (tmp_path / "config.yml").write_text(yaml.safe_dump({"ariel": ARIEL_CONFIG}))
    config = yaml.safe_load((tmp_path / "config.yml").read_text())
    seen = _stub_ariel(monkeypatch, existing=0)

    # Act
    seeded = seed_active_logbook(config, tmp_path, ARIEL_CONFIG)

    # Assert
    assert seeded == 0
    assert seen["seeded"] is None
