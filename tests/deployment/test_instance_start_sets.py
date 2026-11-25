"""The deploy path gives each simulator instance its own start set.

``_activate_default_scenarios`` activates the start set of the instance the
stored data follows, then of a second instance the deployment runs. A second
instance that served before read the first one's set until that deploy: it is
given a set of its own, nothing is copied, and one line, printed by that deploy
alone, says which set it runs now and which it followed. An instance that is
new to the deployment followed nothing and is told nothing.

Nothing is handed over while the render on disk still mounts one scenario
state into both instances: the compose file is written by ``osprey build``
alone, and until then both containers read the first instance's set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import (
    render_shares_scenario_state,
    simulator_log_mount_sources,
)
from osprey.simulation.apply import apply_scenarios
from osprey.simulation.targets import active_scenarios, target_state_dir
from osprey_connectors.simulation.state import ACTIVE_SCENARIOS_FILENAME
from tests._simulator_view import write_scenarios_view

SCENARIOS = {"nominal": {}, "warm": {}, "cold": {}}


def _config(defaults: Any, *services: str) -> dict[str, Any]:
    return {
        "deployed_services": list(services),
        "simulation": {"default_scenarios": defaults},
    }


TWO = ("virtual_accelerator", "live_standin")

#: Where the simulator's compose file is rendered, and the config that says so.
SERVICES = {"virtual_accelerator": {"path": "services/virtual_accelerator"}}
COMPOSE = Path("build/services/virtual_accelerator/docker-compose.yml")

STATE = "./var/agent_data/simulation"


def _standin_has_served(project: Path) -> None:
    """Leave what a stand-in container leaves behind: a model log in its own directory."""
    logs = project / simulator_log_mount_sources()["live_standin"]
    logs.mkdir(parents=True)
    (logs / "model.log").write_text("served\n", encoding="utf-8")


def _render(project: Path, standin_state: str) -> None:
    """Write the simulator's compose file with the stand-in on *standin_state*."""
    document = {
        "services": {
            "virtual-accelerator": {
                "environment": {"VA_INSTANCE": "virtual_accelerator"},
                "volumes": ["./build/data:/data:ro", f"{STATE}:/state/simulation:ro"],
            },
            "live-standin": {
                "environment": {"VA_INSTANCE": "live_standin"},
                "volumes": ["./build/data:/data:ro", f"{standin_state}:/state/simulation:ro"],
            },
        }
    }
    (project / COMPOSE).parent.mkdir(parents=True, exist_ok=True)
    (project / COMPOSE).write_text(yaml.safe_dump(document), encoding="utf-8")


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
    _standin_has_served(project)
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
    _standin_has_served(project)

    container_lifecycle._activate_default_scenarios(config, project)

    assert _has_state(config, project, "standin")
    assert active_scenarios(config, project, "standin") == ["nominal"]
    assert len(steps) == 1
    assert steps[0].startswith("standin now holds a scenario set of its own: nominal. ")
    assert "`osprey sim apply --target standin warm`" in steps[0]

    steps.clear()
    container_lifecycle._activate_default_scenarios(config, project)
    assert steps == []


def test_a_standin_new_to_a_running_deployment_followed_nothing(
    project: Path, steps: list[str]
) -> None:
    """The first instance holds a set, and the stand-in has never served: no hand-over line.

    The same state is left by a first deploy that stopped after the first
    instance's set was written.
    """
    config = _config({"va": ["warm"], "standin": ["cold"]}, *TWO)
    apply_scenarios(project, ["warm"], seed_logbook=False, seed_archive=False)

    container_lifecycle._activate_default_scenarios(config, project)

    assert active_scenarios(config, project, "standin") == ["cold"]
    assert steps == ["scenarios active by default on standin: nominal, cold"]


def test_a_render_with_a_state_mount_per_instance_is_not_shared(project: Path) -> None:
    config = {**_config(None, *TWO), "services": SERVICES}
    assert not render_shares_scenario_state(config, project)

    _render(project, f"{STATE}/standin")

    assert not render_shares_scenario_state(config, project)


@pytest.mark.parametrize(
    "services",
    [
        None,
        {},
        {"virtual_accelerator": {"port": 5064}, "live_standin": {"port": 5066}},
        {"virtual_accelerator": None},
        ["virtual_accelerator"],
    ],
    ids=["absent", "empty", "no-path", "null-block", "not-a-mapping"],
)
def test_a_config_that_declares_no_rendered_simulator_shares_nothing(
    project: Path, services: object
) -> None:
    """Instances named without a template path have no compose file here to read."""
    config = {**_config(None, *TWO), "services": services}
    _render(project, STATE)

    assert not render_shares_scenario_state(config, project)


def test_a_render_that_mounts_one_state_into_both_instances_is_shared(project: Path) -> None:
    config = {**_config(None, *TWO), "services": SERVICES}
    _render(project, STATE)

    assert render_shares_scenario_state(config, project)


def test_nothing_is_handed_over_on_a_render_that_shares_the_state(
    project: Path, steps: list[str]
) -> None:
    """Deployed again without a build: both containers still read the first instance's set."""
    config = {**_config({"va": ["warm"], "standin": ["cold"]}, *TWO), "services": SERVICES}
    apply_scenarios(project, ["warm"], seed_logbook=False, seed_archive=False)
    _standin_has_served(project)
    _render(project, STATE)

    container_lifecycle._activate_default_scenarios(config, project)

    assert not _has_state(config, project, "standin")
    assert steps == [
        "standin follows the va scenario set: this render mounts one scenario state into "
        "both simulator instances. Run `osprey build`, then `osprey up`, to give each "
        "instance a set of its own"
    ]

    # Built again: the hand-over happens at the next deploy, and says so once.
    steps.clear()
    _render(project, f"{STATE}/standin")
    container_lifecycle._activate_default_scenarios(config, project)

    assert active_scenarios(config, project, "standin") == ["cold"]
    assert len(steps) == 1
    assert steps[0].startswith("standin now holds a scenario set of its own: nominal, cold. ")


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
