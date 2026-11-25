"""``osprey sim`` names the simulator instance whose scenario set is meant.

``sim list`` and ``sim apply`` take ``--target va|standin``. A command that
names none acts on the deployment's own target, and on a deployment with one
instance prints what it always did. A set applied to the stand-in changes
telemetry only, and the stand-in's set is refused on a deployment that runs
no stand-in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.sim import sim_group
from osprey.simulation.apply import active_scenarios
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
    assert any("telemetry only" in line and "follow va" in line for line in lines)
    assert active_scenarios(config, lifecycle_repo, "va") == ["warm"]
    assert active_scenarios(config, lifecycle_repo, "standin") == ["cold"]


def test_the_default_target_is_the_deployments_own(lifecycle_repo: Path) -> None:
    config = _deployment(lifecycle_repo, STANDIN_IS_OWN)

    result = _apply(lifecycle_repo, "cold")

    assert result.exit_code == 0, result.output
    assert "✓ Active scenarios on standin: nominal, cold" in result.output.splitlines()
    assert active_scenarios(config, lifecycle_repo, "va") == []
    assert active_scenarios(config, lifecycle_repo, "standin") == ["cold"]


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
