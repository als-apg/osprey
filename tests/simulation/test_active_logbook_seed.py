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

TEMPLATE_SIM = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/simulation"
)

ARIEL_CONFIG = {"database": {"uri": "postgresql://unused-mocked/none"}}


def _make_project(tmp_path: Path) -> Path:
    """Stage a sim-backed project with `rf-thermal` already active."""
    sim_dst = tmp_path / "data" / "simulation"
    sim_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_SIM, sim_dst)
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

    async def _no_seed(_ariel_config, _entries, _pictures):
        return 0, False

    monkeypatch.setattr("osprey.simulation.apply._seed_logbook", _no_seed)
    apply_scenarios(project, names, seed_archive=False)


def _stub_ariel(monkeypatch, *, existing: int) -> dict:
    """Stub ARIEL's database calls; return what the seeder was asked to write."""
    seen: dict = {"seeded": None, "pictures": None, "counted": 0, "mirrored": 0}

    async def _count(_config_dict):
        seen["counted"] += 1
        return existing

    async def _seed(_config_dict, entries, _progress=None, *, pictures=None):
        seen["seeded"] = entries
        seen["pictures"] = pictures
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


def test_seed_active_logbook_hands_over_each_entrys_pictures(tmp_path, monkeypatch):
    """The pictures a bundle entry names travel with it into the seed, keyed by the
    entry they belong to, as files inside the project's own scenario tree."""
    # Arrange
    project = _make_project(tmp_path)
    _activate(project, monkeypatch, ["rf-thermal"])
    config = yaml.safe_load((project / "config.yml").read_text())
    seen = _stub_ariel(monkeypatch, existing=0)

    # Act
    seed_active_logbook(config, project, ARIEL_CONFIG)

    # Assert
    pictures = seen["pictures"]
    assert {entry_id: [p.name for p in paths] for entry_id, paths in pictures.items()} == {
        "DEMO-011": ["orbit_rms_week.png"],
        "DEMO-027": ["cavity_temperatures_week.png"],
    }
    scenarios = (project / "data" / "simulation" / "scenarios").resolve()
    assert all(p.is_file() and p.is_relative_to(scenarios) for ps in pictures.values() for p in ps)
    # The row itself is written bare; the seeder links the stored pictures.
    assert all(entry["attachments"] == [] for entry in seen["seeded"])


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


def test_seed_active_logbook_is_a_no_op_without_a_machine_model(tmp_path, monkeypatch):
    """A project with no simulation has no narrative to seed, which is a normal
    configuration and not a fault."""
    # Arrange
    (tmp_path / "config.yml").write_text(yaml.safe_dump({"ariel": ARIEL_CONFIG}))
    config = yaml.safe_load((tmp_path / "config.yml").read_text())
    seen = _stub_ariel(monkeypatch, existing=0)

    # Act
    seeded = seed_active_logbook(config, tmp_path, ARIEL_CONFIG)

    # Assert
    assert seeded == 0
    assert seen["seeded"] is None


def _narrative_project(tmp_path: Path) -> tuple[dict, Path]:
    """A project with no simulation whose ``ariel.demo_narrative`` names the bundles."""
    shutil.copytree(TEMPLATE_SIM / "scenarios", tmp_path / "data" / "narratives")
    config = {"ariel": {**ARIEL_CONFIG, "demo_narrative": "data/narratives"}}
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    return config, tmp_path


def test_a_project_without_a_simulation_seeds_its_demo_narrative(tmp_path, monkeypatch):
    """Every narrative in the named directory, nominal first, pictures handed over."""
    config, project = _narrative_project(tmp_path)
    seen = _stub_ariel(monkeypatch, existing=0)

    seeded = seed_active_logbook(config, project, config["ariel"])

    ids = [entry["entry_id"] for entry in seen["seeded"]]
    assert seeded == len(ids) == 25 + 3 + 1
    assert ids[:25] == [f"DEMO-{i:03d}" for i in range(1, 26)]
    assert ids[25:] == ["DEMO-031", "DEMO-026", "DEMO-027", "DEMO-028"]  # bpm-polarity, rf-thermal
    assert sorted(seen["pictures"]) == ["DEMO-011", "DEMO-027", "DEMO-031"]
    assert seen["mirrored"] == 1


def test_a_demo_narrative_is_never_seeded_over_existing_entries(tmp_path, monkeypatch):
    config, project = _narrative_project(tmp_path)
    seen = _stub_ariel(monkeypatch, existing=4)

    assert seed_active_logbook(config, project, config["ariel"]) == 0
    assert seen["seeded"] is None


def test_a_missing_demo_narrative_directory_is_named(tmp_path):
    import pytest

    from osprey.simulation.apply import demo_narrative_logbook

    with pytest.raises(ValueError, match="not a directory"):
        demo_narrative_logbook({"demo_narrative": "data/nowhere"}, tmp_path)


def _state_file(project: Path) -> Path:
    from osprey.simulation.engine import ACTIVE_SCENARIOS_FILENAME, resolve_state_dir

    config = yaml.safe_load((project / "config.yml").read_text())
    return resolve_state_dir(config, project) / ACTIVE_SCENARIOS_FILENAME


def test_a_deployment_that_never_chose_starts_in_the_machines_default_set(tmp_path):
    """The shipped machine names rf-thermal; a deploy with no scenario state
    activates it, anchored, and the deploy-time seed then narrates it."""
    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    assert not _state_file(project).exists()

    config = yaml.safe_load((project / "config.yml").read_text())
    active = activate_default_scenarios(config, project)

    assert active == ("nominal", "rf-thermal")
    assert "anchor=" in _state_file(project).read_text()
    config = yaml.safe_load((project / "config.yml").read_text())
    ids = {entry["entry_id"] for entry in active_logbook_entries(config, project)}
    assert {"DEMO-001", "DEMO-026", "DEMO-027", "DEMO-028"} <= ids
    assert "DEMO-031" not in ids


def test_a_chosen_set_is_never_replaced_by_the_default(tmp_path, monkeypatch):
    """`osprey sim apply` means exactly the set it names, nominal alone included."""
    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    _activate(project, monkeypatch, ["nominal"])
    before = _state_file(project).read_text()
    config = yaml.safe_load((project / "config.yml").read_text())

    assert activate_default_scenarios(config, project) == ()
    assert _state_file(project).read_text() == before


def test_a_machine_without_defaults_activates_nothing(tmp_path):
    import json

    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    machine_path = project / "data" / "simulation" / "machine.json"
    machine = json.loads(machine_path.read_text())
    del machine["default_scenarios"]
    machine_path.write_text(json.dumps(machine))
    config = yaml.safe_load((project / "config.yml").read_text())

    assert activate_default_scenarios(config, project) == ()
    assert not _state_file(project).exists()


def test_a_default_naming_an_unknown_scenario_is_refused_like_sim_apply(tmp_path):
    import json

    import pytest

    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    machine_path = project / "data" / "simulation" / "machine.json"
    machine = json.loads(machine_path.read_text())
    machine["default_scenarios"] = ["ghost"]
    machine_path.write_text(json.dumps(machine))
    config = yaml.safe_load((project / "config.yml").read_text())

    with pytest.raises(ValueError, match="ghost"):
        activate_default_scenarios(config, project)
    assert not _state_file(project).exists()


def test_a_project_without_a_machine_model_activates_nothing(tmp_path):
    from osprey.simulation.apply import activate_default_scenarios

    assert activate_default_scenarios({"ariel": ARIEL_CONFIG}, tmp_path) == ()
