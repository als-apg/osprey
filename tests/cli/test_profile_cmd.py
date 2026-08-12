"""Tests for the ``osprey profile`` command group.

The group holds the two read-only verbs that act on a build profile as
editable *source* — ``presets`` lists the bundled names, ``validate`` checks
one — plus the materialization helper the group's writer, ``osprey init``,
calls. Creating a deployment repo and one-shot try/build/deploy chains are
``osprey init``'s and ``osprey up``'s, not this group's; their contracts live
in ``test_init_verb.py``.
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


def test_profile_help_lists_the_two_subcommands(runner: CliRunner) -> None:
    result = runner.invoke(profile, ["--help"])
    assert result.exit_code == 0, result.output
    for name in ("presets", "validate"):
        assert name in result.output


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------


def test_presets_prints_every_bundled_preset(runner: CliRunner) -> None:
    result = runner.invoke(profile, ["presets"])
    assert result.exit_code == 0, result.output
    assert result.output.split() == list_presets()


def test_presets_matches_the_list_init_offers(runner: CliRunner) -> None:
    """The two spellings of "what can I materialize?" must not drift apart.

    ``osprey init --list-presets`` is what an operator reaches for first;
    ``osprey profile presets`` is the same question asked of the noun group.
    Two independent readers of the preset directory would eventually disagree.
    """
    from osprey.cli.init_cmd import init

    group_output = runner.invoke(profile, ["presets"]).output
    flag_output = runner.invoke(init, ["--list-presets"]).output
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
    assert "osprey init" in result.output


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
# Materialization helper
#
# The helper `osprey init` calls. Its full contract — tree shape, data
# materialization, baked overrides, the negative/atomicity matrix, per preset —
# lives in test_source_zone_materialization.py. What stays here is that the
# helper is reachable and fails before mutating.
# --------------------------------------------------------------------------


def test_materialize_helper_writes_the_source_zone(tmp_path: Path) -> None:
    """The helper writes the profile and the data tree it names.

    Not the README: a deployment repo's README describes its four zones, which
    is the caller's document to write — this helper owns the source zone only.
    """
    target = tmp_path / "my-deployment"
    _materialize_profile_directory(target, "hello-world")

    assert (target / "profile.yml").is_file()
    assert (target / "data").is_dir()


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
        "'osprey.cli.build_profile_emit', "
        # `try` chains into build and deploy; both must stay behind its
        # function-local import or `osprey --help` pays for the whole chain.
        "'osprey.cli.build_cmd', 'osprey.cli.deploy_cmd') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "", f"eagerly imported: {out.stdout.strip()}"
