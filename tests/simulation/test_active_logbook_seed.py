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

import json
from pathlib import Path

import pytest
import yaml

from osprey.simulation.apply import (
    active_logbook_entries,
    apply_scenarios,
    seed_active_logbook,
)
from tests._simulator_view import facility_scenarios, write_scenarios_view

TEMPLATE_FACILITY = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"

ARIEL_CONFIG = {"database": {"uri": "postgresql://unused-mocked/none"}}


def _make_project(tmp_path: Path) -> Path:
    """Stage a project whose render holds the example facility's simulator view."""
    scenarios = TEMPLATE_FACILITY / "scenarios"
    write_scenarios_view(tmp_path, facility_scenarios(scenarios), scenarios)
    config = {"ariel": ARIEL_CONFIG}
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
    seen: dict = {"seeded": None, "pictures": None, "bytes": None, "counted": 0, "mirrored": 0}

    async def _count(_config_dict):
        seen["counted"] += 1
        return existing

    async def _seed(_config_dict, entries, _progress=None, *, pictures=None):
        seen["seeded"] = entries
        seen["pictures"] = pictures
        # Read while seeding runs: a drawn picture's file lives only that long.
        seen["bytes"] = {
            entry_id: [path.read_bytes() for path in paths]
            for entry_id, paths in (pictures or {}).items()
        }
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


def _entry(entry_id: str) -> dict:
    return {
        "entry_id": entry_id,
        "when": {"days_ago": 1, "time": "02:00:00"},
        "author": "View",
        "title": entry_id,
        "text": f"{entry_id} narrates its scenario.",
    }


def test_a_persisted_set_that_does_not_compose_narrates_only_nominal(tmp_path):
    """Two active scenarios writing one channel are served as nominal alone, and
    the narrative says what is served."""
    # Arrange
    project = _make_project(tmp_path)
    burst = {"channel": "SR:VAC:PRESSURE", "events": []}
    write_scenarios_view(
        project,
        {
            "nominal": {"logbook": [_entry("NOMINAL-1")]},
            "burst": {"archiver": [burst], "logbook": [_entry("BURST-1")]},
            "leak": {"archiver": [burst], "logbook": [_entry("LEAK-1")]},
        },
    )
    _state_file(project).parent.mkdir(parents=True, exist_ok=True)
    _state_file(project).write_text("burst\nleak\n", encoding="utf-8")
    config = yaml.safe_load((project / "config.yml").read_text())

    # Act
    entries = active_logbook_entries(config, project)

    # Assert
    assert [entry["entry_id"] for entry in entries] == ["NOMINAL-1"]


def _apply_seeding(project: Path, monkeypatch, names: list[str]) -> list[dict]:
    """``osprey sim apply`` with its logbook seed captured instead of written."""
    seeded: list[dict] = []

    async def _capture(_ariel_config, entries, _pictures):
        seeded.extend(entries)
        return len(entries), True

    monkeypatch.setattr("osprey.simulation.apply._seed_logbook", _capture)
    apply_scenarios(project, names, seed_archive=False)
    return seeded


def test_a_scenario_only_the_facility_states_is_applied_and_narrates(tmp_path, monkeypatch):
    """A scenario the simulator view lists, and no machine model does, is one
    ``osprey sim apply`` activates and seeds."""
    # Arrange
    project = _make_project(tmp_path)
    write_scenarios_view(project, {"nominal": {}, "facility-only": {"logbook": [_entry("FAC-1")]}})

    # Act
    seeded = _apply_seeding(project, monkeypatch, ["facility-only"])

    # Assert
    assert [entry["entry_id"] for entry in seeded] == ["FAC-1"]
    assert _state_file(project).read_text().splitlines()[1:] == ["facility-only"]


def test_sim_apply_seeds_a_facility_scenario_s_edited_logbook_text(tmp_path, monkeypatch):
    """The narrative ``osprey sim apply`` seeds is the view's, so an edit to a
    scenario's story is what the next apply writes."""
    # Arrange
    project = _make_project(tmp_path)
    edited = {**_entry("DEMO-026"), "text": "Edited: the cavity warmed after the RF trip."}
    scenarios = TEMPLATE_FACILITY / "scenarios"
    view = facility_scenarios(scenarios)
    view["rf-thermal"] = {**view["rf-thermal"], "logbook": [edited]}
    write_scenarios_view(project, view)

    # Act
    seeded = _apply_seeding(project, monkeypatch, ["rf-thermal"])

    # Assert
    texts = {entry["entry_id"]: entry["raw_text"] for entry in seeded}
    assert texts["DEMO-026"].endswith("Edited: the cavity warmed after the RF trip.")


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
    """The pictures a view entry names travel with it into the seed, keyed by the
    entry they belong to: each drawn from its plot spec at the entry's timestamp."""
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
        "DEMO-011": ["orbit_rms.png"],
        "DEMO-027": ["cavity_temperatures.png"],
    }
    assert all(data.startswith(b"\x89PNG") for ds in seen["bytes"].values() for data in ds)
    # The row itself is written bare; the seeder links the stored pictures.
    assert all(entry["attachments"] == [] for entry in seen["seeded"])


def test_seed_active_logbook_hands_over_a_shipped_picture_from_the_view(tmp_path, monkeypatch):
    """A shipped picture travels as the view's copy of the scenario's file, byte for byte."""
    # Arrange
    project = _make_project(tmp_path)
    _activate(project, monkeypatch, ["bpm-polarity"])
    config = yaml.safe_load((project / "config.yml").read_text())
    seen = _stub_ariel(monkeypatch, existing=0)

    # Act
    seed_active_logbook(config, project, ARIEL_CONFIG)

    # Assert
    (path,) = seen["pictures"]["DEMO-031"]
    view = project / "data" / "simulator" / "scenarios" / "bpm-polarity"
    assert path == (view / "plots" / "corrector_bump_test.png").resolve()
    source = TEMPLATE_FACILITY / "scenarios" / "bpm-polarity"
    assert seen["bytes"]["DEMO-031"] == [
        (source / "plots" / "corrector_bump_test.png").read_bytes()
    ]


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


def _narrative_project(tmp_path: Path, narrative: object = "all") -> tuple[dict, Path]:
    """A project with no simulation whose simulator view lists the example scenarios."""
    scenarios = TEMPLATE_FACILITY / "scenarios"
    write_scenarios_view(tmp_path, facility_scenarios(scenarios), files=scenarios)
    config = {"ariel": {**ARIEL_CONFIG, "demo_narrative": narrative}}
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    return config, tmp_path


def test_a_project_without_a_simulation_seeds_its_demo_narrative(tmp_path, monkeypatch):
    """Every scenario the view lists, nominal first, pictures handed over."""
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


def test_a_demo_narrative_of_named_scenarios_seeds_those_nominal_first(tmp_path, monkeypatch):
    config, project = _narrative_project(tmp_path, ["rf-thermal", "nominal"])
    seen = _stub_ariel(monkeypatch, existing=0)

    assert seed_active_logbook(config, project, config["ariel"]) == 28

    ids = [entry["entry_id"] for entry in seen["seeded"]]
    assert ids == [f"DEMO-{i:03d}" for i in range(1, 29)]


def test_a_demo_narrative_beats_an_active_nominal(tmp_path, monkeypatch):
    """A set demo narrative is what the deploy seeds, whatever set is active."""
    project = _make_project(tmp_path)
    _activate(project, monkeypatch, ["nominal", "rf-thermal"])
    config = yaml.safe_load((project / "config.yml").read_text())
    ariel = {**ARIEL_CONFIG, "demo_narrative": ["bpm-polarity"]}
    seen = _stub_ariel(monkeypatch, existing=0)

    assert seed_active_logbook(config, project, ariel) == 1
    assert [entry["entry_id"] for entry in seen["seeded"]] == ["DEMO-031"]


def test_a_demo_narrative_naming_an_unknown_scenario_is_refused(tmp_path):
    from osprey.simulation.apply import demo_narrative_logbook

    config, project = _narrative_project(tmp_path, ["nominal", "no-such-story"])

    with pytest.raises(ValueError, match="'no-such-story'"):
        demo_narrative_logbook(config["ariel"], project)


def test_a_demo_narrative_naming_a_directory_is_refused_naming_the_form(tmp_path):
    from osprey.simulation.apply import demo_narrative_logbook

    config, project = _narrative_project(tmp_path, "data/logbook_seed")

    with pytest.raises(ValueError, match="'all' or a list of scenario names"):
        demo_narrative_logbook(config["ariel"], project)


def test_a_demo_narrative_without_a_project_is_refused_naming_the_view() -> None:
    from osprey.simulation.apply import demo_narrative_logbook

    with pytest.raises(ValueError, match="simulator view"):
        demo_narrative_logbook({"demo_narrative": "all"}, None)


def _state_file(project: Path) -> Path:
    from osprey.simulation.engine import ACTIVE_SCENARIOS_FILENAME, resolve_state_dir

    config = yaml.safe_load((project / "config.yml").read_text())
    return resolve_state_dir(config, project) / ACTIVE_SCENARIOS_FILENAME


def _start_in(project: Path, names) -> dict:
    """State ``simulation.default_scenarios`` in the project's config; return the config."""
    config = yaml.safe_load((project / "config.yml").read_text())
    config["simulation"] = {"default_scenarios": names}
    (project / "config.yml").write_text(yaml.safe_dump(config))
    return config


def test_a_deployment_that_never_chose_starts_in_the_configured_default_set(tmp_path):
    """The config names rf-thermal; a deploy with no scenario state activates it,
    anchored, and the deploy-time seed then narrates it."""
    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    assert not _state_file(project).exists()

    config = _start_in(project, ["rf-thermal"])
    active = activate_default_scenarios(config, project)

    assert active == ("nominal", "rf-thermal")
    assert "anchor=" in _state_file(project).read_text()
    config = yaml.safe_load((project / "config.yml").read_text())
    ids = {entry["entry_id"] for entry in active_logbook_entries(config, project)}
    assert {"DEMO-001", "DEMO-026", "DEMO-027", "DEMO-028"} <= ids
    assert "DEMO-031" not in ids


def test_a_default_naming_a_scenario_only_the_facility_states_is_activated_and_narrates(
    tmp_path,
):
    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    write_scenarios_view(project, {"nominal": {}, "facility-only": {"logbook": [_entry("FAC-1")]}})
    config = _start_in(project, ["facility-only"])

    active = activate_default_scenarios(config, project)

    assert active == ("nominal", "facility-only")
    assert _state_file(project).read_text().splitlines()[1:] == ["facility-only"]
    ids = [entry["entry_id"] for entry in active_logbook_entries(config, project)]
    assert ids == ["FAC-1"]


def test_a_chosen_set_is_never_replaced_by_the_default(tmp_path, monkeypatch):
    """`osprey sim apply` means exactly the set it names, nominal alone included."""
    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    _activate(project, monkeypatch, ["nominal"])
    before = _state_file(project).read_text()
    config = _start_in(project, ["rf-thermal"])

    assert activate_default_scenarios(config, project) == ()
    assert _state_file(project).read_text() == before


@pytest.mark.parametrize("names", [None, []])
def test_a_config_without_defaults_activates_nothing(tmp_path, names):
    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    config = _start_in(project, names)

    assert activate_default_scenarios(config, project) == ()
    assert not _state_file(project).exists()


def test_a_config_without_a_simulation_section_activates_nothing(tmp_path):
    """The start set is the profile's to state."""
    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    config = yaml.safe_load((project / "config.yml").read_text())

    assert activate_default_scenarios(config, project) == ()
    assert not _state_file(project).exists()


def test_a_default_naming_an_unknown_scenario_is_refused_like_sim_apply(tmp_path):
    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    config = _start_in(project, ["ghost"])

    with pytest.raises(ValueError, match="ghost"):
        activate_default_scenarios(config, project)
    assert not _state_file(project).exists()


@pytest.mark.parametrize("names", ["rf-thermal", [1], [""]])
def test_a_malformed_default_list_is_refused_by_its_key(tmp_path, names):
    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    config = _start_in(project, names)

    with pytest.raises(ValueError, match="simulation.default_scenarios must be a list"):
        activate_default_scenarios(config, project)
    assert not _state_file(project).exists()


def test_each_default_is_activated_once_in_order(tmp_path):
    from osprey.simulation.apply import activate_default_scenarios

    project = _make_project(tmp_path)
    config = _start_in(project, ["rf-thermal", "rf-thermal"])

    assert activate_default_scenarios(config, project) == ("nominal", "rf-thermal")


def test_a_project_without_a_simulator_view_activates_nothing(tmp_path):
    from osprey.simulation.apply import activate_default_scenarios

    config = {"ariel": ARIEL_CONFIG, "simulation": {"default_scenarios": ["rf-thermal"]}}
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))

    assert activate_default_scenarios(config, tmp_path) == ()
    assert not _state_file(tmp_path).exists()


_PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00"
    b":~\x9bU\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND"
    b"\xaeB`\x82"
)


def _plot_spec_project(tmp_path: Path) -> tuple[dict, Path]:
    """A demo narrative of one entry carrying a plot spec and a shipped picture."""
    files = tmp_path / "facility-scenarios"
    bundle = files / "drift"
    (bundle / "plots").mkdir(parents=True)
    (bundle / "plots" / "shipped.png").write_bytes(_PNG_1X1)
    spec = {
        "filename": "orbit_rms.png",
        "title": "SR orbit RMS",
        "ylabel": "µm",
        "hours_before": [48.0, 24.0, 0.0],
        "series": [{"label": "X", "values": [10.0, 11.0, 10.5]}],
    }
    (bundle / "plots" / "orbit_rms.json").write_text(json.dumps(spec))
    entry = {
        "entry_id": "E1",
        "when": {"days_ago": 3, "time": "10:00:00"},
        "author": "ops",
        "title": "Orbit check",
        "text": "Attached: orbit RMS X, past two days.",
        "attachments": [{"plot": "plots/orbit_rms.json"}, {"path": "plots/shipped.png"}],
    }
    write_scenarios_view(tmp_path, {"drift": {"logbook": [entry]}}, files=files)
    config = {"ariel": {**ARIEL_CONFIG, "demo_narrative": "all"}}
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    return config, tmp_path


def test_a_plot_spec_is_drawn_against_its_entrys_own_timestamp(tmp_path, monkeypatch):
    """The drawn picture shows the dates of the entry it is attached to: the spec
    drawn at the seeded row's timestamp, handed over in the entry's own order."""
    from osprey.facility.scenarios import scenario_logbook
    from osprey.facility.views.simulator import SCENARIOS_DIR
    from osprey.simulation.plots import render_plot_spec

    config, project = _plot_spec_project(tmp_path)
    seen = _stub_ariel(monkeypatch, existing=0)

    assert seed_active_logbook(config, project, config["ariel"]) == 1

    (row,) = seen["seeded"]
    view = json.loads((project / "data" / "simulator" / "scenarios.json").read_text())
    (scenario,) = view["scenarios"]
    (entry,) = scenario_logbook(scenario, project / "data" / "simulator" / SCENARIOS_DIR / "drift")
    spec, shipped = entry.attachments
    drawn_path, shipped_path = seen["pictures"]["E1"]
    assert (drawn_path.name, shipped_path) == ("orbit_rms.png", shipped)
    drawn_bytes, shipped_bytes = seen["bytes"]["E1"]
    assert row["timestamp"].tzinfo is not None
    assert drawn_bytes == render_plot_spec(spec, row["timestamp"])
    assert shipped_bytes == _PNG_1X1
    # The drawn file was scratch for the seed only; nothing is left behind.
    assert not drawn_path.exists()
    assert not list((project / "data").rglob("orbit_rms.png"))
