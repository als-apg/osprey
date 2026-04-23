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
    result = runner.invoke(skills, ["install", "build-interview"])

    assert result.exit_code == 0, result.output
    target = fake_home / ".claude" / "skills" / "build-interview"
    assert (target / "SKILL.md").is_file()
    assert (target / "references" / "migration-guide.md").is_file()


def test_install_backs_up_existing(fake_home: Path) -> None:
    """Existing non-empty target is moved to a timestamped backup dir."""
    target = fake_home / ".claude" / "skills" / "build-interview"
    target.mkdir(parents=True)
    (target / "sentinel.txt").write_text("preserve me")

    runner = CliRunner()
    result = runner.invoke(skills, ["install", "build-interview"])

    assert result.exit_code == 0, (result.output, result.stderr)
    assert (target / "SKILL.md").is_file()  # new content installed
    assert "Warning" in result.stderr  # backup notice goes to stderr

    backups = list((fake_home / ".claude" / "skills").glob("build-interview.bak.*"))
    assert len(backups) == 1
    assert (backups[0] / "sentinel.txt").read_text() == "preserve me"


def test_install_unknown_name_errors(fake_home: Path) -> None:
    """Unknown skill name exits nonzero and names the allowlist members."""
    runner = CliRunner()
    result = runner.invoke(skills, ["install", "nonexistent-skill"])

    assert result.exit_code != 0
    combined = (result.output or "") + (result.stderr or "")
    assert "nonexistent-skill" in combined
    assert "build-interview" in combined


def test_resource_path_resolves() -> None:
    """The bundled skill resource is reachable via importlib.resources."""
    from importlib.resources import files

    skill_md = files("osprey").joinpath("templates/skills/build-interview/SKILL.md")
    assert skill_md.is_file()
