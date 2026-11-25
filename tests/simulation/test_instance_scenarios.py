"""Each simulator instance of a deployment has an active scenario set of its own.

One build serves both instances, so both hold the same channels; what differs
is the set each reads. ``va`` keeps its state where a deployment with one
instance has always kept it, ``standin`` in a subdirectory of it, and a set
applied to one target leaves the other's alone. The stores a deployment holds
once follow ``va``, so a set applied to ``standin`` changes telemetry only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.simulation import DEFAULT_SCENARIOS_KEY
from osprey.simulation.apply import (
    activate_default_scenarios,
    apply_scenarios,
    resolve_default_scenarios,
    started_default_scenarios,
)
from osprey.simulation.targets import (
    HISTORY_TARGET,
    SIMULATOR_TARGETS,
    active_scenarios,
    default_simulator_target,
    deployed_simulator_targets,
    require_simulator_target,
    target_state_dir,
)
from osprey_connectors.simulation.composite import Composite
from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME
from osprey_connectors.workspace import resolve_simulation_state_dir
from tests._simulator_view import write_texture_view

T0 = 1_760_000_000.0
READING = "T:READING"

TWO_INSTANCES: dict[str, Any] = {
    "deployed_services": ["virtual_accelerator", "live_standin"],
}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A render whose view serves one reading and a scenario that cuts it."""
    write_texture_view(
        tmp_path,
        {READING: {"nominal": 10.0}},
        {
            "cut": {"channel_faults": {READING: "disconnected"}},
            "other": {"description": "Changes nothing a reading shows."},
        },
    )
    (tmp_path / "config.yml").write_text(yaml.safe_dump(TWO_INSTANCES), encoding="utf-8")
    return tmp_path


def _instance(project: Path, target: str) -> Composite:
    """The simulator one instance runs: the shared view on the target's own state."""
    return Composite(
        project / "data" / "simulator",
        state_dir=target_state_dir(TWO_INSTANCES, project, target),
        clock=lambda: T0,
        model_log=False,
    )


def _apply(project: Path, names: list[str], target: str | None = None):
    return apply_scenarios(project, names, target=target, seed_logbook=False, seed_archive=False)


# --- the layout -------------------------------------------------------------------


def test_va_keeps_its_state_where_one_instance_always_has(tmp_path: Path) -> None:
    assert target_state_dir({}, tmp_path, "va") == resolve_simulation_state_dir({}, tmp_path)


def test_the_standin_state_is_a_directory_of_its_own_under_it(tmp_path: Path) -> None:
    base = resolve_simulation_state_dir({}, tmp_path)

    assert target_state_dir({}, tmp_path, "standin") == base / "standin"


def test_a_relocated_state_dir_moves_every_target(tmp_path: Path) -> None:
    config = {"simulation": {"state_dir": str(tmp_path / "elsewhere")}}

    assert target_state_dir(config, tmp_path, "va") == tmp_path / "elsewhere"
    assert target_state_dir(config, tmp_path, "standin") == tmp_path / "elsewhere" / "standin"


@pytest.mark.parametrize("target", ["live", "both", "", None])
def test_only_a_simulator_target_holds_a_scenario_set(tmp_path: Path, target: object) -> None:
    with pytest.raises(ValueError, match="va or standin"):
        require_simulator_target(target)
    with pytest.raises(ValueError, match="va or standin"):
        target_state_dir({}, tmp_path, target)  # type: ignore[arg-type]


def test_the_stored_data_follows_va() -> None:
    assert HISTORY_TARGET == "va"
    assert SIMULATOR_TARGETS == ("va", "standin")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, "va"),
        ({"control_system": {"type": "virtual_accelerator"}}, "va"),
        ({"control_system": {"type": "live_standin"}}, "standin"),
        ({"control_system": {"type": "epics"}}, "va"),
    ],
)
def test_the_default_target_is_the_deployments_own(config: dict[str, Any], expected: str) -> None:
    assert default_simulator_target(config) == expected


@pytest.mark.parametrize(
    ("services", "expected"),
    [
        ([], ()),
        (["virtual_accelerator"], ("va",)),
        (["live_standin", "virtual_accelerator"], ("va", "standin")),
        (["live_standin"], ("standin",)),
    ],
)
def test_the_deployed_targets_are_the_instances_run_as_services(
    services: list[str], expected: tuple[str, ...]
) -> None:
    assert deployed_simulator_targets({"deployed_services": services}) == expected


# --- one view, two sets -------------------------------------------------------------


def test_a_set_applied_to_the_standin_shows_on_the_standin_alone(project: Path) -> None:
    result = _apply(project, ["cut"], target="standin")

    assert result.target == "standin"
    assert result.active == ("nominal", "cut")
    assert active_scenarios(TWO_INSTANCES, project, "standin") == ["cut"]
    assert active_scenarios(TWO_INSTANCES, project, "va") == []
    assert math.isnan(_instance(project, "standin").get(READING))
    assert math.isfinite(_instance(project, "va").get(READING))


def test_a_set_applied_to_va_leaves_the_standin_alone(project: Path) -> None:
    _apply(project, ["cut"], target="standin")
    before = (
        target_state_dir(TWO_INSTANCES, project, "standin") / ACTIVE_SCENARIOS_FILENAME
    ).read_bytes()

    result = _apply(project, ["other"], target="va")

    assert result.target == "va"
    assert active_scenarios(TWO_INSTANCES, project, "va") == ["other"]
    after = (
        target_state_dir(TWO_INSTANCES, project, "standin") / ACTIVE_SCENARIOS_FILENAME
    ).read_bytes()
    assert after == before
    assert math.isnan(_instance(project, "standin").get(READING))
    assert math.isfinite(_instance(project, "va").get(READING))


def test_a_running_instance_follows_its_own_set(project: Path) -> None:
    standin, baseline = _instance(project, "standin"), _instance(project, "va")
    assert math.isfinite(standin.get(READING))

    _apply(project, ["cut"], target="standin")

    assert math.isnan(standin.get(READING))
    assert math.isfinite(baseline.get(READING))


def test_an_apply_that_names_no_target_is_vas(project: Path) -> None:
    result = _apply(project, ["cut"])

    assert result.target == "va"
    assert not result.telemetry_only
    assert active_scenarios(TWO_INSTANCES, project, "standin") == []


def test_a_set_on_the_standin_changes_telemetry_only(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seeding asked for on the stand-in is not done: the stores follow va."""
    config = {**TWO_INSTANCES, "ariel": {"database": {"uri": "unreachable"}}}
    (project / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")

    def _refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a store was touched for the stand-in's set")

    monkeypatch.setattr("osprey.simulation.apply._seed_logbook", _refuse)
    monkeypatch.setattr("osprey.simulation.apply.seed_archiver", _refuse)

    result = apply_scenarios(project, ["cut"], target="standin")

    assert result.telemetry_only
    assert result.logbook_seeded == 0
    assert result.archiver is None


def test_an_unknown_target_is_refused_before_anything_is_written(project: Path) -> None:
    with pytest.raises(ValueError, match="va or standin"):
        _apply(project, ["cut"], target="live")

    assert all(active_scenarios(TWO_INSTANCES, project, t) == [] for t in SIMULATOR_TARGETS)


# --- the start set of each ----------------------------------------------------------


def test_a_list_is_the_start_set_of_every_instance() -> None:
    config = {"simulation": {"default_scenarios": ["cut", "other", "cut"]}}

    assert resolve_default_scenarios(config) == ("cut", "other")
    assert resolve_default_scenarios(config, "va") == ("cut", "other")
    assert resolve_default_scenarios(config, "standin") == ("cut", "other")


def test_a_mapping_is_the_start_set_of_each_instance() -> None:
    config = {"simulation": {"default_scenarios": {"va": ["other"], "standin": ["cut", "other"]}}}

    assert resolve_default_scenarios(config, "va") == ("other",)
    assert resolve_default_scenarios(config, "standin") == ("cut", "other")
    assert resolve_default_scenarios(config) == ("other", "cut")


def test_a_target_the_mapping_leaves_out_starts_none() -> None:
    config = {"simulation": {"default_scenarios": {"standin": ["cut"]}}}

    assert resolve_default_scenarios(config, "va") == ()
    assert resolve_default_scenarios(config, "standin") == ("cut",)


@pytest.mark.parametrize(
    "defaults", [None, ["cut"], {"standin": ["cut"]}], ids=["absent", "list", "mapping"]
)
def test_a_start_set_is_asked_of_a_simulator_target_alone(defaults: object) -> None:
    config = {"simulation": {"default_scenarios": defaults}}

    with pytest.raises(ValueError, match="va or standin"):
        resolve_default_scenarios(config, "live")


@pytest.mark.parametrize(
    ("services", "expected"),
    [
        ([], ("other",)),
        (["virtual_accelerator"], ("other",)),
        (["virtual_accelerator", "live_standin"], ("other", "cut")),
    ],
)
def test_the_sets_a_deployment_starts_in_are_those_of_the_instances_it_runs(
    services: list[str], expected: tuple[str, ...]
) -> None:
    config = {
        "deployed_services": services,
        "simulation": {"default_scenarios": {"va": ["other"], "standin": ["cut", "other"]}},
    }

    assert started_default_scenarios(config) == expected


@pytest.mark.parametrize(
    "value",
    [{"live": ["cut"]}, {"standin": "cut"}, {"va": [""]}, {"va": None}, {"standin": [3]}],
)
def test_a_malformed_mapping_is_refused(value: object) -> None:
    with pytest.raises(ValueError, match=f"{DEFAULT_SCENARIOS_KEY} must be a list of scenario"):
        resolve_default_scenarios({"simulation": {"default_scenarios": value}})


def test_each_instance_starts_in_its_own_set(project: Path) -> None:
    config = {**TWO_INSTANCES, "simulation": {"default_scenarios": {"standin": ["cut"]}}}

    assert activate_default_scenarios(config, project) == ()
    assert activate_default_scenarios(config, project, target="standin") == ("nominal", "cut")

    assert active_scenarios(config, project, "va") == []
    assert active_scenarios(config, project, "standin") == ["cut"]


def test_a_list_starts_both_instances_in_the_same_set(project: Path) -> None:
    config = {**TWO_INSTANCES, "simulation": {"default_scenarios": ["cut"]}}

    for target in SIMULATOR_TARGETS:
        assert activate_default_scenarios(config, project, target=target) == ("nominal", "cut")
        assert active_scenarios(config, project, target) == ["cut"]


def test_a_set_chosen_on_one_instance_is_kept_and_the_other_still_starts(project: Path) -> None:
    config = {**TWO_INSTANCES, "simulation": {"default_scenarios": ["cut"]}}
    _apply(project, ["other"], target="va")

    assert activate_default_scenarios(config, project, target="va") == ()
    assert activate_default_scenarios(config, project, target="standin") == ("nominal", "cut")

    assert active_scenarios(config, project, "va") == ["other"]
