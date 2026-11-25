"""The deploy path gives each simulator instance its own start set.

``_activate_default_scenarios`` activates the start set of the instance the
stored data follows, then of a second instance the deployment runs. A second
instance on a deployment that was already running read the first one's set
until that deploy: it is given a set of its own, nothing is copied, and one
line, printed by that deploy alone, says which set it runs now and which it
followed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment import container_lifecycle
from osprey.simulation.apply import active_scenarios, apply_scenarios
from osprey.simulation.targets import target_state_dir
from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME
from tests._simulator_view import write_scenarios_view

SCENARIOS = {"nominal": {}, "warm": {}, "cold": {}}


def _config(defaults: Any, *services: str) -> dict[str, Any]:
    return {
        "deployed_services": list(services),
        "simulation": {"default_scenarios": defaults},
    }


TWO = ("virtual_accelerator", "live_standin")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    write_scenarios_view(tmp_path, SCENARIOS)
    (tmp_path / "config.yml").write_text(yaml.safe_dump({}), encoding="utf-8")
    return tmp_path


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    lines: list[str] = []
    monkeypatch.setattr(container_lifecycle, "_report_step", lines.append)
    return lines


def _has_state(config: dict[str, Any], project: Path, target: str) -> bool:
    return (target_state_dir(config, project, target) / ACTIVE_SCENARIOS_FILENAME).is_file()


def test_a_fresh_deployment_starts_each_instance_in_its_own_set(
    project: Path, steps: list[str]
) -> None:
    config = _config({"va": ["warm"], "standin": ["cold"]}, *TWO)

    assert container_lifecycle._activate_default_scenarios(config, project) == ("nominal", "warm")

    assert active_scenarios(config, project, "va") == ["warm"]
    assert active_scenarios(config, project, "standin") == ["cold"]
    assert steps == [
        "scenarios active by default: nominal, warm",
        "scenarios active by default on standin: nominal, cold",
    ]


def test_a_list_starts_both_instances_in_the_same_set(project: Path, steps: list[str]) -> None:
    config = _config(["warm"], *TWO)

    container_lifecycle._activate_default_scenarios(config, project)

    assert active_scenarios(config, project, "va") == ["warm"]
    assert active_scenarios(config, project, "standin") == ["warm"]
    assert not any("followed" in line for line in steps)


def test_a_standin_set_alone_leaves_va_on_nominal(project: Path, steps: list[str]) -> None:
    config = _config({"standin": ["cold"]}, *TWO)

    assert container_lifecycle._activate_default_scenarios(config, project) == ()

    assert not _has_state(config, project, "va")
    assert active_scenarios(config, project, "standin") == ["cold"]
    assert steps == ["scenarios active by default on standin: nominal, cold"]


def test_a_deployment_with_one_instance_writes_no_standin_state(
    project: Path, steps: list[str]
) -> None:
    config = _config({"va": ["warm"], "standin": ["cold"]}, "virtual_accelerator")

    container_lifecycle._activate_default_scenarios(config, project)

    assert active_scenarios(config, project, "va") == ["warm"]
    assert not _has_state(config, project, "standin")
    assert steps == ["scenarios active by default: nominal, warm"]


def test_a_running_standin_is_told_once_that_it_stopped_following(
    project: Path, steps: list[str]
) -> None:
    """Before instances had sets of their own, the stand-in read the one state file."""
    config = _config({"va": ["warm"], "standin": ["cold"]}, *TWO)
    apply_scenarios(project, ["warm", "cold"], seed_logbook=False, seed_archive=False)
    assert not _has_state(config, project, "standin")

    container_lifecycle._activate_default_scenarios(config, project)

    assert active_scenarios(config, project, "va") == ["warm", "cold"]
    assert active_scenarios(config, project, "standin") == ["cold"]
    assert steps == [
        "standin now holds a scenario set of its own: nominal, cold. Until this deploy it "
        "followed the va set (nominal, warm, cold); "
        "`osprey sim apply --target standin warm cold` puts that set back"
    ]

    steps.clear()
    container_lifecycle._activate_default_scenarios(config, project)

    assert steps == []
    assert active_scenarios(config, project, "standin") == ["cold"]


def test_a_running_standin_with_no_start_set_is_given_nominal_and_told(
    project: Path, steps: list[str]
) -> None:
    config = _config(None, *TWO)
    apply_scenarios(project, ["warm"], seed_logbook=False, seed_archive=False)

    container_lifecycle._activate_default_scenarios(config, project)

    assert _has_state(config, project, "standin")
    assert active_scenarios(config, project, "standin") == ["nominal"]
    assert len(steps) == 1
    assert steps[0].startswith("standin now holds a scenario set of its own: nominal. ")
    assert "`osprey sim apply --target standin warm`" in steps[0]

    steps.clear()
    container_lifecycle._activate_default_scenarios(config, project)
    assert steps == []


def test_a_fresh_standin_with_no_start_set_is_given_its_state_silently(
    project: Path, steps: list[str]
) -> None:
    """Its state file exists from the first deploy, so no later one reads it as following."""
    config = _config(None, *TWO)

    container_lifecycle._activate_default_scenarios(config, project)

    assert _has_state(config, project, "standin")
    assert steps == []

    apply_scenarios(project, ["warm"], seed_logbook=False, seed_archive=False)
    container_lifecycle._activate_default_scenarios(config, project)

    assert steps == []


def test_a_set_chosen_on_the_standin_is_never_replaced(project: Path, steps: list[str]) -> None:
    config = _config({"standin": ["cold"]}, *TWO)
    apply_scenarios(project, ["warm"], target="standin")

    container_lifecycle._activate_default_scenarios(config, project)

    assert active_scenarios(config, project, "standin") == ["warm"]
    assert steps == []


def test_a_standin_start_set_that_cannot_be_activated_is_reported_not_fatal(
    project: Path, steps: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    config = _config({"va": ["warm"], "standin": ["no-such-scenario"]}, *TWO)

    assert container_lifecycle._activate_default_scenarios(config, project) == ("nominal", "warm")

    assert not _has_state(config, project, "standin")
    assert steps == ["scenarios active by default: nominal, warm"]
    assert "osprey sim apply --target standin" in caplog.text
