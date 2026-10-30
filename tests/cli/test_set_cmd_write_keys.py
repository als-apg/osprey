"""`osprey set` and the write ceiling: it may narrow, never raise.

The profile states the write ceiling as ``control_system.writes`` (``allowed``
or ``never``). No CLI path turns writes on, so ``osprey set`` accepts
``config.control_system.writes=never`` and refuses everything else that would
touch the posture: ``=allowed`` (edit the profile or tier file instead), any
other value (naming the two words), and every retired ``writes_enabled`` leaf
(pointing at ``control_system.writes``). A refusal rejects the whole command
line before anything is written.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.set_cmd import set as set_command


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _invoke(runner: CliRunner, repo: Path, *args: str):
    return runner.invoke(set_command, ["--repo", str(repo), *args])


def _profile_bytes(repo: Path) -> bytes:
    return (repo / "profile.yml").read_bytes()


def test_never_is_accepted_and_written(runner, lifecycle_repo):
    result = _invoke(runner, lifecycle_repo, "config.control_system.writes=never")

    assert result.exit_code == 0, result.output
    assert "control_system.writes: never" in (lifecycle_repo / "profile.yml").read_text()


def test_allowed_is_refused_and_points_at_the_profile(runner, lifecycle_repo):
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, "config.control_system.writes=allowed")

    assert result.exit_code != 0
    assert "profile" in result.output and "tier file" in result.output, result.output
    assert _profile_bytes(lifecycle_repo) == before


@pytest.mark.parametrize(
    "value",
    ["yes", "true", "Allowed", "NEVER", "", "[allowed]", "{x: 1}"],
    ids=["yes", "bool", "capitalized", "upper", "empty", "list", "mapping"],
)
def test_any_other_value_is_refused_naming_the_two_words(runner, lifecycle_repo, value):
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, f"config.control_system.writes={value}")

    assert result.exit_code != 0
    assert "allowed" in result.output and "never" in result.output, result.output
    assert _profile_bytes(lifecycle_repo) == before


@pytest.mark.parametrize(
    "pair",
    [
        "config.control_system.writes_enabled=false",
        "config.control_system.writes_enabled=true",
        "config.control_system.connector.virtual_accelerator.writes_enabled=true",
        "config.control_system.connector.epics.writes_enabled=false",
    ],
    ids=["global-false", "global-true", "per-type-true", "per-type-false"],
)
def test_retired_key_is_refused_and_points_at_the_new_key(runner, lifecycle_repo, pair):
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, pair)

    assert result.exit_code != 0
    assert "control_system.writes" in result.output, result.output
    assert "retired" in result.output, result.output
    assert _profile_bytes(lifecycle_repo) == before


@pytest.mark.parametrize(
    "pair",
    [
        "config.control_system={writes: allowed}",
        "config.control_system={writes_enabled: true}",
        "config.control_system.connector={epics: {writes_enabled: true}}",
        "config={control_system: {writes: allowed}}",
    ],
    ids=["block-new-allowed", "block-retired", "connector-block-retired", "config-block"],
)
def test_a_mapping_value_cannot_smuggle_the_posture(runner, lifecycle_repo, pair):
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(runner, lifecycle_repo, pair)

    assert result.exit_code != 0
    assert "control_system.writes" in result.output, result.output
    assert _profile_bytes(lifecycle_repo) == before


def test_a_mapping_that_only_narrows_is_accepted(runner, lifecycle_repo):
    result = _invoke(runner, lifecycle_repo, "config.control_system={writes: never}")

    assert result.exit_code == 0, result.output


def test_one_refused_pair_rejects_the_whole_command_line(runner, lifecycle_repo):
    before = _profile_bytes(lifecycle_repo)

    result = _invoke(
        runner,
        lifecycle_repo,
        "config.system.timezone=UTC",
        "config.control_system.writes=allowed",
    )

    assert result.exit_code != 0
    assert _profile_bytes(lifecycle_repo) == before


def test_help_example_names_the_new_key(runner):
    result = runner.invoke(set_command, ["--help"])

    assert result.exit_code == 0
    assert "config.control_system.writes=never" in result.output
    assert "writes_enabled" not in result.output
