"""Tests for the ``osprey profile`` command group.

The group holds the verbs that act on a build profile as editable *source*
(``presets``, ``validate``, ``new``), as opposed to ``osprey build``, which
consumes a profile and derives a project from it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_profile import list_presets
from osprey.cli.main import LazyGroup
from osprey.cli.profile_cmd import _materialize_profile_directory, profile

MINIMAL_PROFILE = "name: Minimal\ndata_bundle: hello_world\n"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_profile(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "profile.yml"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_profile_group_is_registered_on_the_cli() -> None:
    """``profile`` is both listed and resolvable through the lazy group."""
    group = LazyGroup(name="osprey")
    assert "profile" in group.list_commands(None)
    assert group.get_command(None, "profile") is profile


def test_profile_help_lists_the_three_subcommands(runner: CliRunner) -> None:
    result = runner.invoke(profile, ["--help"])
    assert result.exit_code == 0, result.output
    for name in ("presets", "validate", "new"):
        assert name in result.output


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------


def test_presets_prints_every_bundled_preset(runner: CliRunner) -> None:
    result = runner.invoke(profile, ["presets"])
    assert result.exit_code == 0, result.output
    assert result.output.split() == list_presets()


def test_presets_matches_build_list_presets(runner: CliRunner) -> None:
    """``osprey profile presets`` is the noun-group spelling of the build flag."""
    from osprey.cli.build_cmd import build

    group_output = runner.invoke(profile, ["presets"]).output
    flag_output = runner.invoke(build, ["--list-presets"]).output
    assert group_output == flag_output


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------


def test_validate_accepts_a_profile_file(runner: CliRunner, tmp_path: Path) -> None:
    path = _write_profile(tmp_path / "p", MINIMAL_PROFILE)
    result = runner.invoke(profile, ["validate", str(path)])
    assert result.exit_code == 0, result.output
    assert "Minimal" in result.output
    assert "Next steps:" in result.output


def test_validate_accepts_a_profile_directory(runner: CliRunner, tmp_path: Path) -> None:
    """A directory resolves to the ``profile.yml`` inside it."""
    _write_profile(tmp_path / "p", MINIMAL_PROFILE)
    result = runner.invoke(profile, ["validate", str(tmp_path / "p")])
    assert result.exit_code == 0, result.output


def test_validate_accepts_an_emitted_profile(runner: CliRunner, tmp_path: Path) -> None:
    """A freshly materialized profile directory validates clean (SC7)."""
    target = tmp_path / "my-profile"
    _materialize_profile_directory(target, "hello-world")

    result = runner.invoke(profile, ["validate", str(target)])
    assert result.exit_code == 0, result.output


def test_validate_reports_all_errors_and_exits_2(runner: CliRunner, tmp_path: Path) -> None:
    """Every problem is reported at once, not just the first (SC7)."""
    _write_profile(
        tmp_path / "p",
        "name: Broken\ndata_bundle: hello_world\ndata: ./nowhere\ntier: 5\n",
    )
    result = runner.invoke(profile, ["validate", str(tmp_path / "p")])

    assert result.exit_code == 2, result.output
    assert "data directory not found" in result.output
    assert "tier must be 1 or 3" in result.output


def test_validate_rejects_a_directory_without_a_profile(runner: CliRunner, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(profile, ["validate", str(empty)])

    assert result.exit_code == 2, result.output
    assert "No profile.yml" in result.output
    assert "profile new" in result.output


def test_validate_rejects_a_missing_path(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(profile, ["validate", str(tmp_path / "absent.yml")])
    assert result.exit_code == 2, result.output


def test_validate_does_not_build_a_project(runner: CliRunner, tmp_path: Path) -> None:
    """Validation is read-only: the profile directory is untouched."""
    target = tmp_path / "p"
    _write_profile(target, MINIMAL_PROFILE)
    before = sorted(p.name for p in tmp_path.iterdir())

    result = runner.invoke(profile, ["validate", str(target)])

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert [p.name for p in target.iterdir()] == ["profile.yml"]


# --------------------------------------------------------------------------
# new
#
# The command's own contract — tree shape, data materialization, baked
# overrides, the negative/atomicity matrix — lives in test_profile_new.py.
# What stays here is the group-level wiring.
# --------------------------------------------------------------------------


def test_new_requires_a_preset(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(profile, ["new", str(tmp_path / "my-profile")])
    assert result.exit_code == 2, result.output
    assert "--preset" in result.output


# --------------------------------------------------------------------------
# Materialization helper
# --------------------------------------------------------------------------


def test_materialize_helper_writes_a_profile_directory(tmp_path: Path) -> None:
    """The helper writes the profile, its data tree, and the overlay seed."""
    target = tmp_path / "my-profile"
    _materialize_profile_directory(target, "hello-world")

    assert (target / "profile.yml").is_file()
    assert (target / "README.md").is_file()
    assert (target / "data").is_dir()
    for kind in ("rules", "skills", "agents", "web-terminal-context"):
        assert (target / "overlays" / kind / ".gitkeep").is_file()


def test_emit_helper_leaves_nothing_behind_on_an_invalid_override(tmp_path: Path) -> None:
    """Fail-before-mutating: a bad ``--set`` scaffolds nothing."""
    import click

    target = tmp_path / "my-profile"
    with pytest.raises(click.UsageError):
        _materialize_profile_directory(target, "hello-world", set_pairs=("tier=5",))
    assert not target.exists()


# --------------------------------------------------------------------------
# Lazy-import budget
# --------------------------------------------------------------------------


def test_module_import_stays_lazy() -> None:
    """Importing the group must not drag in the template machinery.

    ``osprey --help`` renders short help for every registered noun, which
    imports each command module. Keeping the heavy template/profile pipeline
    behind function-local imports is what keeps that path cheap.
    """
    probe = (
        "import sys; import osprey.cli.profile_cmd; "
        "heavy = [m for m in ("
        "'osprey.cli.templates.manager', 'osprey.cli.build_profile', "
        "'osprey.cli.build_profile_emit') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "", f"eagerly imported: {out.stdout.strip()}"
