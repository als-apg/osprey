"""``osprey sim`` names the simulator instance whose scenario set is meant.

``sim list`` and ``sim apply`` take ``--target va|standin``. A command that
names none acts on the deployment's own target, and on a deployment with one
instance prints what it always did. A set applied to the stand-in changes
telemetry only, and the stand-in's set is refused on a deployment that runs
no stand-in.

On a render that still mounts one scenario state into both instances, made
before instances had sets of their own, a command acts on the set both follow
and says so, and the stand-in's set is refused with the fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.sim import sim_group
from osprey.simulation.targets import active_scenarios
from osprey_connectors.control_system._va_in_process_connector import simulation_state_dir
from tests._simulator_view import write_scenarios_view
from tests.cli._lifecycle_build import STUB_CONFIG, stub_build

ONE_INSTANCE = STUB_CONFIG + "deployed_services:\n  - virtual_accelerator\n"
TWO_INSTANCES = ONE_INSTANCE + "  - live_standin\n"
STANDIN_IS_OWN = (
    TWO_INSTANCES + "control_system:\n  type: live_standin\n  connector:\n    live_standin: {}\n"
)

SCENARIOS = {"nominal": {}, "warm": {"description": "Warm."}, "cold": {"description": "Cold."}}


def _deployment(repo: Path, config: str) -> dict:
    write_scenarios_view(stub_build(repo, config=config), SCENARIOS)
    return yaml.safe_load(config)


def _sim(repo: Path, *args: str):
    return CliRunner().invoke(sim_group, [*args, "--repo", str(repo)])


def _apply(repo: Path, *args: str):
    return _sim(repo, "apply", "--yes", "--no-seed", *args)


# --- which set a command means ------------------------------------------------------


def test_apply_without_a_target_is_vas_and_prints_what_it_always_did(
    lifecycle_repo: Path,
) -> None:
    config = _deployment(lifecycle_repo, ONE_INSTANCE)

    result = _apply(lifecycle_repo, "warm")

    assert result.exit_code == 0, result.output
    assert "✓ Active scenarios: nominal, warm" in result.output.splitlines()
    assert "telemetry only" not in result.output
    assert active_scenarios(config, lifecycle_repo, "va") == ["warm"]
    assert active_scenarios(config, lifecycle_repo, "standin") == []


def test_apply_names_the_instance_when_two_are_deployed(lifecycle_repo: Path) -> None:
    config = _deployment(lifecycle_repo, TWO_INSTANCES)

    result = _apply(lifecycle_repo, "warm")

    assert result.exit_code == 0, result.output
    assert "✓ Active scenarios on va: nominal, warm" in result.output.splitlines()
    assert active_scenarios(config, lifecycle_repo, "standin") == []


def test_apply_to_the_standin_moves_the_standin_alone_and_says_telemetry_only(
    lifecycle_repo: Path,
) -> None:
    config = _deployment(lifecycle_repo, TWO_INSTANCES)
    assert _apply(lifecycle_repo, "warm").exit_code == 0

    result = CliRunner().invoke(
        sim_group, ["apply", "--yes", "--target", "standin", "cold", "--repo", str(lifecycle_repo)]
    )

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert "✓ Active scenarios on standin: nominal, cold" in lines
    assert any(
        "telemetry only" in line and "follow va" in line and "--target va" in line for line in lines
    )
    assert active_scenarios(config, lifecycle_repo, "va") == ["warm"]
    assert active_scenarios(config, lifecycle_repo, "standin") == ["cold"]


def test_the_default_target_is_the_deployments_own(lifecycle_repo: Path) -> None:
    config = _deployment(lifecycle_repo, STANDIN_IS_OWN)

    result = _apply(lifecycle_repo, "cold")

    assert result.exit_code == 0, result.output
    assert "✓ Active scenarios on standin: nominal, cold" in result.output.splitlines()
    assert active_scenarios(config, lifecycle_repo, "va") == []
    assert active_scenarios(config, lifecycle_repo, "standin") == ["cold"]


def test_a_standin_baseline_that_runs_no_standin_here_acts_on_va(lifecycle_repo: Path) -> None:
    """The deployment's own target is the stand-in, served elsewhere: the one set here is va's."""
    config = _deployment(
        lifecycle_repo,
        ONE_INSTANCE
        + "control_system:\n  type: live_standin\n  connector:\n    live_standin: {}\n",
    )

    result = _apply(lifecycle_repo, "warm")

    assert result.exit_code == 0, result.output
    assert "✓ Active scenarios: nominal, warm" in result.output.splitlines()
    assert "telemetry only" not in result.output
    assert active_scenarios(config, lifecycle_repo, "va") == ["warm"]


@pytest.mark.parametrize("verb", [("apply", "--yes", "--no-seed", "warm"), ("list",)])
def test_the_standin_set_is_refused_where_no_standin_runs(
    lifecycle_repo: Path, verb: tuple[str, ...]
) -> None:
    config = _deployment(lifecycle_repo, ONE_INSTANCE)

    result = _sim(lifecycle_repo, *verb, "--target", "standin")

    assert result.exit_code == 1, result.output
    assert "runs no live stand-in" in result.output
    assert "--target va" in result.output
    assert active_scenarios(config, lifecycle_repo, "standin") == []


def test_an_unknown_target_is_a_usage_error(lifecycle_repo: Path) -> None:
    _deployment(lifecycle_repo, TWO_INSTANCES)

    result = _apply(lifecycle_repo, "warm", "--target", "live")

    assert result.exit_code == 2, result.output


# --- list ---------------------------------------------------------------------------


def test_list_marks_the_set_of_the_instance_asked_for(lifecycle_repo: Path) -> None:
    _deployment(lifecycle_repo, TWO_INSTANCES)
    assert _apply(lifecycle_repo, "warm").exit_code == 0
    assert _apply(lifecycle_repo, "cold", "--target", "standin").exit_code == 0

    baseline = _sim(lifecycle_repo, "list").output.splitlines()
    standin = _sim(lifecycle_repo, "list", "--target", "standin").output.splitlines()

    assert baseline[0] == "active set of va:"
    assert [line for line in baseline if line.startswith("*")] == [
        "* nominal  (logbook: no)",
        "* warm  (logbook: no)",
    ]
    assert standin[0] == "active set of standin:"
    assert [line for line in standin if line.startswith("*")] == [
        "* cold  (logbook: no)",
        "* nominal  (logbook: no)",
    ]


def test_list_on_one_instance_prints_no_target_line(lifecycle_repo: Path) -> None:
    _deployment(lifecycle_repo, ONE_INSTANCE)

    result = _sim(lifecycle_repo, "list")

    assert result.exit_code == 0, result.output
    assert "active set of" not in result.output


# --- a render that still shares the state ---------------------------------------------

SHARED_RENDER = (
    TWO_INSTANCES + "services:\n  virtual_accelerator:\n    path: services/virtual_accelerator\n"
)
STATE = "./var/agent_data/simulation"


def _render_compose(repo: Path, standin_state: str) -> None:
    """Write the simulator's compose file with the stand-in on *standin_state*."""
    document = {
        "services": {
            "live-standin": {
                "environment": {"VA_INSTANCE": "live_standin"},
                "volumes": [f"{standin_state}:/state/simulation:ro"],
            }
        }
    }
    compose = repo / "build" / "services" / "virtual_accelerator" / "docker-compose.yml"
    compose.parent.mkdir(parents=True, exist_ok=True)
    compose.write_text(yaml.safe_dump(document), encoding="utf-8")


def test_on_a_shared_render_apply_changes_the_set_both_follow_and_says_so(
    lifecycle_repo: Path,
) -> None:
    config = _deployment(lifecycle_repo, SHARED_RENDER)
    _render_compose(lifecycle_repo, STATE)

    result = _apply(lifecycle_repo, "warm")

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert "✓ Active scenarios: nominal, warm" in lines
    assert any("both simulator instances follow this set" in line for line in lines)
    assert any("'osprey build'" in line and "'osprey up'" in line for line in lines)
    assert active_scenarios(config, lifecycle_repo, "va") == ["warm"]
    assert active_scenarios(config, lifecycle_repo, "standin") == []
    assert _sim(lifecycle_repo, "list").output.splitlines()[0] == (
        "active set of both simulator instances:"
    )


@pytest.mark.parametrize("verb", [("apply", "--yes", "--no-seed", "warm"), ("list",)])
def test_on_a_shared_render_the_standin_set_is_refused_with_the_fix(
    lifecycle_repo: Path, verb: tuple[str, ...]
) -> None:
    """A set written for the stand-in would be read by nothing until the next build."""
    config = _deployment(lifecycle_repo, SHARED_RENDER)
    _render_compose(lifecycle_repo, STATE)

    result = _sim(lifecycle_repo, *verb, "--target", "standin")

    assert result.exit_code == 1, result.output
    assert "no scenario set of its own yet" in result.output
    assert "osprey build" in result.output
    assert active_scenarios(config, lifecycle_repo, "standin") == []


def test_once_built_again_the_standin_set_is_its_own(lifecycle_repo: Path) -> None:
    config = _deployment(lifecycle_repo, SHARED_RENDER)
    _render_compose(lifecycle_repo, f"{STATE}/standin")

    result = _apply(lifecycle_repo, "cold", "--target", "standin")

    assert result.exit_code == 0, result.output
    assert "✓ Active scenarios on standin: nominal, cold" in result.output.splitlines()
    assert active_scenarios(config, lifecycle_repo, "standin") == ["cold"]
    assert active_scenarios(config, lifecycle_repo, "va") == []


# --- the simulator in process ---------------------------------------------------------


def test_the_simulator_in_process_reads_vas_set(lifecycle_repo: Path) -> None:
    """The in-process simulator is the va instance: it reads va's state, never the stand-in's."""
    config = _deployment(lifecycle_repo, TWO_INSTANCES)
    assert _apply(lifecycle_repo, "warm").exit_code == 0
    assert _apply(lifecycle_repo, "cold", "--target", "standin").exit_code == 0
    view = lifecycle_repo / "build" / "data" / "simulator"

    from osprey.simulation.targets import target_state_dir

    assert simulation_state_dir(view) == target_state_dir(config, lifecycle_repo, "va")
    assert simulation_state_dir(view) != target_state_dir(config, lifecycle_repo, "standin")
