"""The scenario-sessions seed file a deploy's default activation writes.

``activate_default_scenarios`` is the one writer: when the set it activates holds
``sessions``, every identity that gets a sidebar history receives a
``scenario-sessions.json`` in its control-context state dir, anchored on the
clock the state file records. Every other activation path -- ``apply_scenarios``,
``osprey sim apply``, benchmark activation -- writes nothing.
"""

from __future__ import annotations

import json
import shutil
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.records import FacilityRecords
from osprey.simulation.apply import (
    SCENARIO_SESSIONS_FILENAME,
    activate_default_scenarios,
    apply_scenarios,
    history_identities,
    write_scenario_sessions,
)
from osprey_connectors.simulation.state import read_active_state
from osprey_connectors.workspace import resolve_simulation_state_dir
from tests._simulator_view import write_scenarios_view

TEMPLATE_FACILITY = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"
PRESET = Path(__file__).resolve().parents[2] / "src/osprey/profiles/presets/control-assistant.yml"
HOST = "deployer"


def _preset_web_terminals() -> dict[str, Any]:
    """The shipped preset's roster, exactly as authored."""
    preset = yaml.safe_load(PRESET.read_text())
    web_terminals = preset["config"]["modules.web_terminals"]
    return {**web_terminals, "enabled": True}


def _config(*, web_terminals: dict[str, Any] | None = None, defaults=None) -> dict[str, Any]:
    config: dict[str, Any] = {"deployment": {"host_identity": HOST}}
    if web_terminals is not None:
        config["modules"] = {"web_terminals": web_terminals}
    if defaults is not None:
        config["simulation"] = {"default_scenarios": defaults}
    return config


#: The config the example's view is rendered with: every model served.
VIEW_CONFIG: dict[str, Any] = {"control_system": {"type": "virtual_accelerator"}}


@pytest.fixture(scope="module")
def built_view(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The example facility's REAL simulator view, as a build writes it.

    Built through the view writer rather than staged from the YAML, so a writer
    that dropped ``sessions`` (or the archived channels a plot reads) would
    leave every seed test here without a history.
    """
    from osprey.facility.build import build_facility
    from osprey.facility.served import resolve_served
    from osprey.facility.views import ViewInputs
    from osprey.facility.views.simulator import write_simulator_view

    root = tmp_path_factory.mktemp("view")
    doc = build_facility(TEMPLATE_FACILITY, project_name="example")
    write_simulator_view(
        root,
        ViewInputs(
            records=FacilityRecords(doc),
            rendered_config=VIEW_CONFIG,
            facility_dir=TEMPLATE_FACILITY,
            served=resolve_served(VIEW_CONFIG, FacilityRecords(doc)),
            reported=None,
        ),
    )
    return root


@pytest.fixture
def make_project(built_view: Path):
    """Stage a project holding the real view and *config*."""

    def _make(root: Path, config: dict[str, Any]) -> Path:
        shutil.copytree(built_view, root / "data" / "simulator")
        (root / "config.yml").write_text(yaml.safe_dump({**VIEW_CONFIG, **config}))
        return root

    return _make


def _seed(project: Path, identity: str) -> Path:
    return project / "var/agent_data/control_target" / identity / SCENARIO_SESSIONS_FILENAME


def _seeds(project: Path) -> list[Path]:
    return sorted((project / "var/agent_data").rglob(SCENARIO_SESSIONS_FILENAME))


def _roster(*users, default_tier: str = "readonly") -> dict[str, Any]:
    return {"enabled": True, "default_tier": default_tier, "users": list(users)}


# ---------------------------------------------------------------------------
# history_identities
# ---------------------------------------------------------------------------


def test_apply_session_identities_of_the_preset_roster_are_alice_and_the_host():
    """bob is readonly and carol admin: only alice, the first readwrite user, and the host."""
    assert history_identities(_config(web_terminals=_preset_web_terminals())) == ["alice", HOST]


def test_apply_session_identities_take_only_the_first_readwrite_user():
    roster = _roster(
        {"name": "ann", "tier": "readwrite"},
        {"name": "ben", "tier": "readwrite"},
    )
    assert history_identities(_config(web_terminals=roster)) == ["ann", HOST]


def test_apply_session_identities_without_a_readwrite_user_are_the_host_alone():
    roster = _roster({"name": "ann", "tier": "readonly"}, {"name": "ben"})
    assert history_identities(_config(web_terminals=roster)) == [HOST]


def test_apply_session_identities_read_the_effective_tier_of_an_untiered_user():
    """A user with no ``tier`` takes ``default_tier``: readwrite counts, readonly does not."""
    assert history_identities(
        _config(web_terminals=_roster({"name": "ann"}, default_tier="readwrite"))
    ) == ["ann", HOST]
    assert history_identities(
        _config(web_terminals=_roster({"name": "ann"}, default_tier="readonly"))
    ) == [HOST]


def test_apply_session_identities_without_web_terminals_are_the_host_alone():
    roster = {**_roster({"name": "ann", "tier": "readwrite"}), "enabled": False}
    assert history_identities(_config(web_terminals=roster)) == [HOST]
    assert history_identities(_config()) == [HOST]


# ---------------------------------------------------------------------------
# The seed file
# ---------------------------------------------------------------------------


def _activate_demo_history(
    make_project, tmp_path: Path, **config_kw
) -> tuple[Path, dict[str, Any]]:
    config = {**VIEW_CONFIG, **_config(defaults=["rf-thermal", "demo-history"], **config_kw)}
    project = make_project(tmp_path, config)
    active = activate_default_scenarios(config, project)
    assert "demo-history" in active
    return project, config


def test_apply_session_default_activation_seeds_alice_and_the_host_only(make_project, tmp_path):
    project, _ = _activate_demo_history(
        make_project, tmp_path, web_terminals=_preset_web_terminals()
    )

    assert _seeds(project) == sorted([_seed(project, "alice"), _seed(project, HOST)])
    for identity in ("bob", "carol"):
        assert not _seed(project, identity).exists()


def test_apply_session_seed_file_carries_the_state_files_anchor(make_project, tmp_path):
    project, config = _activate_demo_history(make_project, tmp_path)

    seed = json.loads(_seed(project, HOST).read_text())
    _, epoch = read_active_state(resolve_simulation_state_dir(config, project))
    assert seed["version"] == 1
    assert datetime.fromisoformat(seed["anchor"]).timestamp() == epoch
    assert seed["sessions"], "a seed file never carries an empty sessions list"


def test_apply_session_seed_file_has_the_contract_shape(make_project, tmp_path):
    project, _ = _activate_demo_history(make_project, tmp_path)

    seed = json.loads(_seed(project, HOST).read_text())
    slugs = {s["slug"] for s in seed["sessions"]}
    assert {"sector-7-beam-loss", "orbit-table-and-stats"} <= slugs
    for session in seed["sessions"]:
        assert session["scenario"] == "demo-history"
        assert session["key"] == session["parts"][0]["transcript_id"]
        transcripts = {part["transcript_id"] for part in session["parts"]}
        for part in session["parts"]:
            assert part["title"] and part["turns"]
            assert all("at" in turn and "steps" in turn for turn in part["turns"])
        for artifact in session["artifacts"]:
            assert set(artifact) == {
                "entry_id",
                "transcript_id",
                "at",
                "filename",
                "title",
                "artifact_type",
                "mime_type",
                "kind",
                "tool_source",
                "content_b64",
            }
            assert len(artifact["entry_id"]) == 12
            int(artifact["entry_id"], 16)
            assert artifact["transcript_id"] in transcripts
    multi = next(s for s in seed["sessions"] if s["slug"] == "orbit-table-and-stats")
    assert len(multi["parts"]) == 2
    beam = next(s for s in seed["sessions"] if s["slug"] == "sector-7-beam-loss")
    assert {a["title"] for a in beam["artifacts"]} and len(beam["artifacts"]) == 2


def _ids(seed: dict) -> set[str]:
    ids = set()
    for session in seed["sessions"]:
        ids.add(session["key"])
        ids.update(part["transcript_id"] for part in session["parts"])
        ids.update(a["entry_id"] for a in session["artifacts"])
    return ids


def test_apply_session_identities_get_distinct_ids(make_project, tmp_path):
    project, _ = _activate_demo_history(
        make_project, tmp_path, web_terminals=_preset_web_terminals()
    )

    alice = json.loads(_seed(project, "alice").read_text())
    host = json.loads(_seed(project, HOST).read_text())
    assert _ids(alice) and not _ids(alice) & _ids(host)


def test_apply_session_ids_are_stable_across_calls(make_project, tmp_path):
    project, config = _activate_demo_history(make_project, tmp_path)
    first = json.loads(_seed(project, HOST).read_text())

    anchor = datetime.fromisoformat(first["anchor"])
    write_scenario_sessions(config, project, ("nominal", "rf-thermal", "demo-history"), anchor)
    second = json.loads(_seed(project, HOST).read_text())
    assert _ids(first) == _ids(second)
    assert first == second


def test_apply_session_seed_file_is_group_shared_in_a_setgid_dir(make_project, tmp_path):
    project, _ = _activate_demo_history(make_project, tmp_path)

    seed = _seed(project, HOST)
    assert stat.S_IMODE(seed.stat().st_mode) == 0o660
    assert stat.S_IMODE(seed.parent.stat().st_mode) == 0o2770


def test_apply_session_seed_lands_under_the_passed_project_from_anywhere(
    make_project, tmp_path, monkeypatch
):
    """Neither the cwd nor an ambient agent-data stamp redirects the file."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    for key in ("OSPREY_AGENT_DATA_DIR", "OSPREY_STATE_DIR", "OSPREY_PROJECT_DIR"):
        monkeypatch.setenv(key, str(elsewhere))
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    project, _ = _activate_demo_history(make_project, project_dir)

    assert _seed(project, HOST).is_file()
    assert not list(elsewhere.rglob(SCENARIO_SESSIONS_FILENAME))


# ---------------------------------------------------------------------------
# What writes nothing
# ---------------------------------------------------------------------------


def test_apply_session_free_default_set_writes_no_seed(make_project, tmp_path):
    config = _config(defaults=["rf-thermal"])
    project = make_project(tmp_path, config)

    assert activate_default_scenarios(config, project) == ("nominal", "rf-thermal")
    assert _seeds(project) == []


def test_apply_session_free_facility_writes_no_seed(tmp_path):
    config = _config(defaults=["fault"])
    project = tmp_path
    write_scenarios_view(project, {"nominal": {}, "fault": {"logbook": []}})
    (project / "config.yml").write_text(yaml.safe_dump(config))

    assert activate_default_scenarios(config, project) == ("nominal", "fault")
    assert _seeds(project) == []


def _no_logbook_seed(monkeypatch) -> None:
    async def _no_seed(_ariel_config, _entries, _pictures):
        return 0, False

    monkeypatch.setattr("osprey.simulation.apply._seed_logbook", _no_seed)


def test_apply_session_apply_scenarios_writes_no_seed_and_leaves_one_untouched(
    make_project, tmp_path, monkeypatch
):
    """``osprey sim apply`` and benchmark activation go through ``apply_scenarios``."""
    project, config = _activate_demo_history(make_project, tmp_path)
    seed = _seed(project, HOST)
    before = seed.read_bytes()
    seed.unlink()
    _no_logbook_seed(monkeypatch)

    apply_scenarios(project, ["demo-history"], seed_archive=False)
    assert _seeds(project) == []

    seed.write_bytes(before)
    apply_scenarios(project, ["demo-history"], seed_logbook=True, seed_archive=False)
    assert seed.read_bytes() == before


@pytest.mark.parametrize("names", [("rf-thermal",), ("rf-thermal", "demo-history")])
def test_apply_session_unaffected_benchmark_activation_writes_no_seed(
    make_project, tmp_path, monkeypatch, names
):
    """The e2e lanes activate through ``sdk_helpers.activate_scenarios``: no seed.

    That helper is the benchmark path -- ``apply_scenarios`` with the logbook
    seeded -- so it never lays a history down, not even when the set it is
    handed holds ``demo-history``. A lane that wants the sessions writes them
    itself.
    """
    from tests.e2e.sdk_helpers import activate_scenarios

    config = _config(defaults=["rf-thermal"])
    project = make_project(tmp_path, config)
    _no_logbook_seed(monkeypatch)

    result = activate_scenarios(project, *names)

    assert set(names) <= set(result.active)
    assert _seeds(project) == []


def test_apply_session_second_activation_writes_nothing(make_project, tmp_path):
    project, config = _activate_demo_history(make_project, tmp_path)
    _seed(project, HOST).unlink()

    assert activate_default_scenarios(config, project) == ()
    assert _seeds(project) == []


def test_apply_session_failed_seed_still_returns_the_activated_set(
    make_project, tmp_path, monkeypatch
):
    def _boom(*_a, **_kw):
        raise RuntimeError("seed writer exploded")

    monkeypatch.setattr("osprey.simulation.apply.write_scenario_sessions", _boom)
    config = _config(defaults=["rf-thermal", "demo-history"])
    project = make_project(tmp_path, config)

    assert activate_default_scenarios(config, project) == ("nominal", "rf-thermal", "demo-history")
    assert _seeds(project) == []


def test_apply_session_naive_anchor_is_refused(make_project, tmp_path):
    config = _config()
    project = make_project(tmp_path, config)
    with pytest.raises(ValueError, match="timezone-aware"):
        write_scenario_sessions(config, project, ("demo-history",), datetime(2026, 1, 1))


# ---------------------------------------------------------------------------
# Through the example facility's real simulator view
# ---------------------------------------------------------------------------


def test_apply_session_real_view_carries_sessions_into_the_seed(make_project, tmp_path):
    """The real view's ``sessions`` reach the seed: a view that drops them seeds nothing."""
    project, _ = _activate_demo_history(make_project, tmp_path)

    seed = json.loads(_seed(project, HOST).read_text())
    assert {s["slug"] for s in seed["sessions"]} == {
        "sector-7-beam-loss",
        "sector-3-corrector-nudge",
        "orbit-table-and-stats",
    }
    plot = next(
        a
        for s in seed["sessions"]
        for a in s["artifacts"]
        if s["slug"] == "sector-7-beam-loss" and a["artifact_type"] == "plot_html"
    )
    assert plot["content_b64"]
