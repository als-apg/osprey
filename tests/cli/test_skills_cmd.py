"""Regression tests for the ``osprey skills`` CLI surface.

Covers the install command's happy path, backup-on-conflict behavior,
allowlist enforcement, and that the bundled skill resource resolves under
the editable install layout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.skills_cmd import skills


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` and ``$HOME`` to a tmp dir for the test.

    ``skills_cmd.install`` calls ``Path.home()`` directly, so patching the
    method on the ``Path`` class is required; the ``HOME`` env var is patched
    too as a belt-and-braces measure for any subprocess-style code paths.

    Args:
        tmp_path: Pytest-provided temporary directory.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The tmp directory standing in for ``$HOME``.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def test_install_fresh_succeeds(fake_home: Path) -> None:
    """Install copies the bundled skill into a previously empty target."""
    runner = CliRunner()
    result = runner.invoke(skills, ["install", "osprey-build-interview"])

    assert result.exit_code == 0, result.output
    target = fake_home / ".claude" / "skills" / "osprey-build-interview"
    assert (target / "SKILL.md").is_file()
    assert (target / "references" / "osprey-map.md").is_file()


def test_install_backs_up_existing(fake_home: Path) -> None:
    """Existing non-empty target is moved to a timestamped backup dir."""
    target = fake_home / ".claude" / "skills" / "osprey-build-interview"
    target.mkdir(parents=True)
    (target / "sentinel.txt").write_text("preserve me")

    runner = CliRunner()
    result = runner.invoke(skills, ["install", "osprey-build-interview"])

    assert result.exit_code == 0, (result.output, result.stderr)
    assert (target / "SKILL.md").is_file()  # new content installed
    # The backup notice is a warning, so it carries the ⚠ mark and goes to
    # stderr: a caller redirecting stdout still learns the old copy moved.
    assert "⚠" in result.stderr
    assert "osprey-build-interview.bak." in result.stderr
    assert "⚠" not in result.stdout

    backups = list((fake_home / ".claude" / "skills").glob("osprey-build-interview.bak.*"))
    assert len(backups) == 1
    assert (backups[0] / "sentinel.txt").read_text() == "preserve me"


def test_install_unknown_name_errors(fake_home: Path) -> None:
    """Unknown skill name exits nonzero and names the allowlist members."""
    runner = CliRunner()
    result = runner.invoke(skills, ["install", "nonexistent-skill"])

    assert result.exit_code != 0
    # A refusal, so it is on stderr with the ✗ mark and nothing lands on stdout.
    assert "nonexistent-skill" in result.stderr
    assert "osprey-build-interview" in result.stderr
    assert "osprey-contribute" in result.stderr
    assert result.stdout == ""


def test_resource_path_resolves() -> None:
    """The bundled skill resource is reachable via importlib.resources."""
    from importlib.resources import files

    skill_md = files("osprey").joinpath("templates/skills/osprey-build-interview/SKILL.md")
    assert skill_md.is_file()


def test_resource_path_for_design_philosophy_skill_resolves() -> None:
    """The design-philosophy skill is discoverable under templates/skills/."""
    from importlib.resources import files

    skill_md = files("osprey").joinpath("templates/skills/osprey-design-philosophy/SKILL.md")
    assert skill_md.is_file()


def test_install_design_philosophy_skill(fake_home: Path) -> None:
    """The design-philosophy skill installs into the default global target."""
    runner = CliRunner()
    result = runner.invoke(skills, ["install", "osprey-design-philosophy"])

    assert result.exit_code == 0, result.output
    target = fake_home / ".claude" / "skills" / "osprey-design-philosophy"
    assert (target / "SKILL.md").is_file()


def test_no_bundled_skill_survives_the_retired_deploy_surface() -> None:
    """No bundled skill may be an operate-time runbook of retired verbs.

    Asserted as an absence rather than left out, because the failure it guards
    is a re-add: a skill shipped from a branch written against
    ``osprey deploy`` would install cleanly and instruct its reader to run
    commands the CLI does not have.
    """
    from importlib.resources import files

    from osprey.cli.skills_cmd import _SKILL_SOURCES

    assert "osprey-deploy-ops" not in _SKILL_SOURCES
    assert not files("osprey").joinpath("templates/skills/osprey-deploy-ops").is_dir()


def test_install_help_lists_every_bundled_skill() -> None:
    """Every allowlist entry is described in the install command's help text.

    The help text is the only place an operator learns a skill exists, so a
    skill registered in ``_SKILL_SOURCES`` but absent from the docstring is
    shipped and invisible.
    """
    from osprey.cli.skills_cmd import _SKILL_SOURCES

    runner = CliRunner()
    result = runner.invoke(skills, ["install", "--help"])

    assert result.exit_code == 0, result.output
    for name in _SKILL_SOURCES:
        assert name in result.output, f"{name} missing from 'skills install --help'"


def test_resource_path_for_creating_panel_skill_resolves() -> None:
    """The creating-an-osprey-panel skill is discoverable under templates/skills/."""
    from importlib.resources import files

    skill_md = files("osprey").joinpath("templates/skills/creating-an-osprey-panel/SKILL.md")
    assert skill_md.is_file()


def test_install_creating_panel_skill(fake_home: Path) -> None:
    """The creating-an-osprey-panel dev skill installs into the default target."""
    runner = CliRunner()
    result = runner.invoke(skills, ["install", "creating-an-osprey-panel"])

    assert result.exit_code == 0, result.output
    target = fake_home / ".claude" / "skills" / "creating-an-osprey-panel"
    assert (target / "SKILL.md").is_file()


def test_install_skill_to_custom_target(tmp_path: Path) -> None:
    """``--target`` installs a skill into a project-local .claude/skills/.

    This is how a skill gets scoped to one repo instead of the user's home.
    A multi-file skill is used deliberately: the copy must bring the whole
    subtree, not just the SKILL.md at its root.
    """
    target = tmp_path / "build-profile" / ".claude" / "skills"
    runner = CliRunner()
    result = runner.invoke(
        skills,
        ["install", "osprey-build-interview", "--target", str(target)],
    )

    assert result.exit_code == 0, result.output
    installed = target / "osprey-build-interview"
    assert (installed / "SKILL.md").is_file()
    assert (installed / "references" / "osprey-map.md").is_file()


def test_install_target_backs_up_existing(tmp_path: Path) -> None:
    """``--target`` honors the same backup-on-conflict behavior as the default."""
    target = tmp_path / ".claude" / "skills"
    existing = target / "osprey-build-interview"
    existing.mkdir(parents=True)
    (existing / "sentinel.txt").write_text("preserve me")

    runner = CliRunner()
    result = runner.invoke(
        skills,
        ["install", "osprey-build-interview", "--target", str(target)],
    )

    assert result.exit_code == 0, (result.output, result.stderr)
    assert (existing / "SKILL.md").is_file()
    backups = list(target.glob("osprey-build-interview.bak.*"))
    assert len(backups) == 1
    assert (backups[0] / "sentinel.txt").read_text() == "preserve me"


def test_install_target_expands_tilde(fake_home: Path) -> None:
    """``--target`` with a literal ``~`` expands against the user's home dir.

    ``click.Path(path_type=Path)`` does not expand tildes; programmatic
    callers or quoted paths that bypass shell expansion would otherwise
    create a literal ``~`` directory in CWD. The install path should
    expanduser() before resolving.
    """
    runner = CliRunner()
    result = runner.invoke(
        skills,
        ["install", "osprey-build-interview", "--target", "~/my-skills"],
    )

    assert result.exit_code == 0, (result.output, result.stderr)
    installed = fake_home / "my-skills" / "osprey-build-interview"
    assert (installed / "SKILL.md").is_file()
    assert (installed / "references" / "osprey-map.md").is_file()
    assert not (Path.cwd() / "~").exists(), "literal ~ dir leaked into CWD"
