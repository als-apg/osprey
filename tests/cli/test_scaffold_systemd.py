"""``osprey scaffold systemd`` — the verb, not the unit it renders.

The rendered bytes are pinned elsewhere: ``tests/deployment/
test_scaffold_systemd_unit.py`` holds the template to a golden, directive by
directive. What is tested here is everything around the render — where the file
lands, which repo the verb decides it is acting on, what happens when something
is already at that path, and what an operator is told to do with the file once
it is written.

Three of those are worth naming.

*The host, not the profile.* The unit is rendered from a repo with its
``deploy:`` block commented out, which is what ``osprey init`` emits. A verb
that reached for deployment coordinates it does not need would fail here, and
would fail on every repo that is deployed by hand.

*The two absolute paths.* A unit starts with no working directory and a short
PATH, so both the repo and the ``osprey`` program have to be written into it in
full. The executable is resolved through a helper that reads the machine, so
every test here freezes it: a unit whose contents depend on the developer's
PATH is a unit nobody can pin.

*The instruction.* The file is written to the repo root and takes effect from
``~/.config/systemd/user/``, so the verb's output is the only thing standing
between "emitted" and "installed".
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.deploy_scaffold import SYSTEMD_OUTPUT_NAME, scaffold_systemd_unit
from osprey.cli.deploy_scaffold_templates import SYSTEMD_MARKER
from osprey.cli.main import cli
from osprey.cli.scaffold_cmd import scaffold
from osprey.errors import ConfigurationError
from tests.fixtures.lifecycle_repo import build_exemplar_repo

#: The ``osprey`` the emitted unit names. Frozen, because the real resolver
#: reads PATH and the user-local bin directories of whoever runs the tests.
FROZEN_BIN = "/opt/osprey/bin/osprey"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def frozen_osprey_bin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the executable the unit is rendered against.

    Patched where it is used rather than where it is defined: the templates
    module imported the name, so patching the helper's own module would leave
    this binding pointing at the real one.
    """
    monkeypatch.setattr(
        "osprey.cli.deploy_scaffold_templates.resolve_shell_command",
        lambda command: FROZEN_BIN,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The exemplar repo as ``osprey init`` leaves it: no deploy coordinates."""
    return build_exemplar_repo(tmp_path / "als-exemplar")


def emit(runner: CliRunner, repo: Path, *args: str):
    """Run the verb against *repo* from outside it."""
    return runner.invoke(scaffold, ["systemd", "--repo", str(repo), *args])


def unit_of(repo: Path) -> Path:
    return repo / SYSTEMD_OUTPUT_NAME


# ── What it writes, and where ────────────────────────────────────────────────


def test_the_unit_lands_at_the_repo_root(runner: CliRunner, repo: Path) -> None:
    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    unit = unit_of(repo)
    assert unit.is_file()
    assert unit.name == "osprey.service"
    assert f"osprey-scaffold: {SYSTEMD_MARKER}" in unit.read_text(encoding="utf-8")


def test_a_repo_with_no_deploy_block_still_gets_a_unit(runner: CliRunner, repo: Path) -> None:
    """The unit runs the deployment in place, so it reads no coordinates."""
    profile = (repo / "profile.yml").read_text(encoding="utf-8")
    assert "\ndeploy:" not in profile, "the fixture is supposed to have no deploy: block"

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert unit_of(repo).is_file()


def test_both_host_paths_are_written_in_full(runner: CliRunner, repo: Path) -> None:
    """A unit inherits no working directory and a short PATH."""
    emit(runner, repo)

    text = unit_of(repo).read_text(encoding="utf-8")
    assert f"WorkingDirectory={repo}" in text
    assert f"ExecStart={FROZEN_BIN} up -d" in text
    assert f"ExecStop={FROZEN_BIN} down" in text


def test_the_description_comes_from_the_profile(runner: CliRunner, repo: Path) -> None:
    emit(runner, repo)

    assert "Description=Als Exemplar OSPREY deployment" in unit_of(repo).read_text(encoding="utf-8")


def test_no_temporary_file_is_left_behind(runner: CliRunner, repo: Path) -> None:
    """The write goes through a temp file in the destination directory."""
    emit(runner, repo)

    assert not list(repo.glob("*.tmp"))
    assert not list(repo.glob(".osprey.service.*"))


# ── Which repo it acts on ────────────────────────────────────────────────────


def test_it_finds_the_repo_from_a_subdirectory(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No --repo means the repo the operator is standing in, at any depth."""
    inside = repo / "data" / "raw"
    inside.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(inside)

    result = runner.invoke(scaffold, ["systemd"])

    assert result.exit_code == 0, result.output
    assert unit_of(repo).is_file()


def test_outside_a_repo_it_reports_the_missing_repo(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "not-a-repo"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = runner.invoke(scaffold, ["systemd"])

    assert result.exit_code == 1
    assert "profile.yml" in result.output


def test_a_profileless_directory_is_told_to_re_run_this_verb(tmp_path: Path) -> None:
    """The engine names its caller, since two verbs render from one profile.

    Reached directly rather than through the command: repo discovery is what
    finds the profile, so a CLI run that got past it has one by construction.
    The engine is called by ``osprey init`` too, and the message it raises is
    the one an operator sees.
    """
    with pytest.raises(ConfigurationError) as raised:
        scaffold_systemd_unit(tmp_path)

    assert "osprey scaffold systemd" in str(raised.value)


# ── Re-running ───────────────────────────────────────────────────────────────


def test_a_second_run_writes_nothing(runner: CliRunner, repo: Path) -> None:
    emit(runner, repo)
    unit = unit_of(repo)
    first = unit.read_bytes()

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert unit.read_bytes() == first
    assert "unchanged" in result.output


def test_a_stale_version_stamp_alone_is_not_a_change(runner: CliRunner, repo: Path) -> None:
    """An OSPREY upgrade on its own must not produce a diff."""
    emit(runner, repo)
    unit = unit_of(repo)
    aged = unit.read_text(encoding="utf-8").replace("osprey-version:", "osprey-version: 0.0.1 #")
    unit.write_text(aged, encoding="utf-8")

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert unit.read_text(encoding="utf-8") == aged
    assert "unchanged" in result.output


def test_a_moved_repo_rewrites_the_unit(runner: CliRunner, repo: Path, tmp_path: Path) -> None:
    """The paths in the unit are the point, so a real change is written."""
    emit(runner, repo)
    unit = unit_of(repo)
    moved = unit.read_text(encoding="utf-8").replace(
        f"WorkingDirectory={repo}", "WorkingDirectory=/old"
    )
    unit.write_text(moved, encoding="utf-8")

    result = emit(runner, repo)

    assert result.exit_code == 0, result.output
    assert f"WorkingDirectory={repo}" in unit.read_text(encoding="utf-8")
    assert "updated" in result.output


# ── A file we did not write ──────────────────────────────────────────────────


def test_a_hand_written_unit_is_left_alone(runner: CliRunner, repo: Path) -> None:
    unit = unit_of(repo)
    unit.write_text("[Unit]\nDescription=mine\n", encoding="utf-8")

    result = emit(runner, repo)

    assert result.exit_code == 1
    assert unit.read_text(encoding="utf-8") == "[Unit]\nDescription=mine\n"
    assert "osprey scaffold systemd --force" in result.output


def test_force_replaces_it(runner: CliRunner, repo: Path) -> None:
    unit = unit_of(repo)
    unit.write_text("[Unit]\nDescription=mine\n", encoding="utf-8")

    result = emit(runner, repo, "--force")

    assert result.exit_code == 0, result.output
    assert f"osprey-scaffold: {SYSTEMD_MARKER}" in unit.read_text(encoding="utf-8")


# ── No osprey to name ────────────────────────────────────────────────────────


def test_an_unresolvable_osprey_writes_nothing(
    runner: CliRunner, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A unit naming a program that is not there fails at boot, unwatched."""

    def missing(command: str) -> str:
        raise FileNotFoundError(f"{command!r} was not found on PATH.")

    monkeypatch.setattr("osprey.cli.deploy_scaffold_templates.resolve_shell_command", missing)

    result = emit(runner, repo)

    assert result.exit_code == 1
    assert not unit_of(repo).exists()
    assert "command -v osprey" in result.output


# ── What the operator is told ────────────────────────────────────────────────


def test_the_output_says_how_to_install_it(runner: CliRunner, repo: Path) -> None:
    result = emit(runner, repo)

    # The copy names the file in full: an operator who passed --repo, or who is
    # standing in a subdirectory, cannot copy a repo-relative path.
    assert f"cp {unit_of(repo)} ~/.config/systemd/user/" in " ".join(result.output.split())
    assert "systemctl --user daemon-reload" in result.output
    assert "systemctl --user enable --now osprey.service" in result.output


def test_the_output_says_what_boot_also_needs(runner: CliRunner, repo: Path) -> None:
    """A user unit that nobody is logged in for does not start on its own."""
    result = emit(runner, repo)

    assert "loginctl enable-linger" in result.output


def test_a_refusal_prints_no_install_instruction(runner: CliRunner, repo: Path) -> None:
    """Nothing was written, so there is nothing to install."""
    unit_of(repo).write_text("[Unit]\nDescription=mine\n", encoding="utf-8")

    result = emit(runner, repo)

    assert "systemctl --user enable" not in result.output


# ── Reachable as an osprey verb ──────────────────────────────────────────────


def test_the_verb_is_reachable_through_the_top_level_group(runner: CliRunner) -> None:
    """The group is lazily loaded, so the help path is worth exercising."""
    result = runner.invoke(cli, ["scaffold", "systemd", "--help"])

    assert result.exit_code == 0, result.output
    assert "--repo" in result.output
    assert "--force" in result.output


def test_the_group_help_lists_the_verb(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["scaffold", "--help"])

    assert result.exit_code == 0, result.output
    assert "systemd" in result.output
