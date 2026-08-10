"""Tests for the ``osprey profile`` command group.

The group holds the verbs that act on a build profile as editable *source*
(``presets``, ``validate``, ``new``), as opposed to ``osprey build``, which
consumes a profile and derives a project from it — plus ``try``, the one-shot
that chains profile → build → deploy up into a single command.
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


def test_profile_help_lists_the_four_subcommands(runner: CliRunner) -> None:
    result = runner.invoke(profile, ["--help"])
    assert result.exit_code == 0, result.output
    for name in ("presets", "validate", "new", "try"):
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
    """The helper writes the profile, its data tree, and its documentation."""
    target = tmp_path / "my-profile"
    _materialize_profile_directory(target, "hello-world")

    assert (target / "profile.yml").is_file()
    assert (target / "README.md").is_file()
    assert (target / "data").is_dir()


def test_emit_helper_leaves_nothing_behind_on_an_invalid_override(tmp_path: Path) -> None:
    """Fail-before-mutating: a bad ``--set`` scaffolds nothing."""
    import click

    target = tmp_path / "my-profile"
    with pytest.raises(click.UsageError):
        _materialize_profile_directory(target, "hello-world", set_pairs=("tier=5",))
    assert not target.exists()


# --------------------------------------------------------------------------
# try — the one-shot: build (which settles the profile), then deploy up.
#
# The chained commands are patched out with plain callables: ctx.invoke on a
# non-Command passes through exactly the kwargs `try` supplies, so these tests
# pin the contract between `try` and the commands it chains — everything past
# that seam is build's and deploy's own test coverage.
# --------------------------------------------------------------------------


def test_try_chains_build_then_deploy_up(runner: CliRunner, monkeypatch, tmp_path) -> None:
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "osprey.cli.build_cmd.build", lambda **kw: calls.append(("build", kw)) and None
    )
    monkeypatch.setattr(
        "osprey.cli.deploy_cmd.up", lambda **kw: calls.append(("deploy", kw)) and None
    )

    result = runner.invoke(
        profile,
        [
            "try",
            "my-assistant",
            "--preset",
            "control-assistant",
            "--set",
            "provider=als-apg",
            "--set",
            "model=haiku",
            "--output-dir",
            str(tmp_path),
            "--dev",
            "-d",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [name for name, _ in calls] == ["build", "deploy"]

    build_kw = calls[0][1]
    assert build_kw["project_name"] == "my-assistant"
    assert build_kw["preset"] == "control-assistant"
    assert build_kw["set_pairs"] == ("provider=als-apg", "model=haiku")
    # Rerunnability is the point of a one-shot test loop: the re-render is
    # forced, and (per build --force) the profile itself is never touched.
    assert build_kw["force"] is True

    # The `up` SUBCOMMAND, not the `deploy` group -- the group takes no
    # arguments of its own, so invoking it would run nothing at all.
    deploy_kw = calls[1][1]
    assert deploy_kw["project"] == str(tmp_path / "my-assistant")
    assert deploy_kw["detached"] is True
    assert deploy_kw["dev"] is True
    assert deploy_kw["expose"] is False


def test_try_accepts_a_profile_path_instead_of_a_preset(
    runner: CliRunner, monkeypatch, tmp_path
) -> None:
    """The positional profile passes through; preset stays None (build enforces XOR)."""
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "osprey.cli.build_cmd.build", lambda **kw: calls.append(("build", kw)) and None
    )
    monkeypatch.setattr(
        "osprey.cli.deploy_cmd.up", lambda **kw: calls.append(("deploy", kw)) and None
    )

    result = runner.invoke(
        profile,
        ["try", "my-assistant", "my-profile/profile.yml", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    build_kw = calls[0][1]
    assert build_kw["profile"] == "my-profile/profile.yml"
    assert build_kw["preset"] is None


def test_try_prints_a_separator_before_each_phase(runner: CliRunner, monkeypatch, tmp_path) -> None:
    """The seam between the chained commands is the one thing try narrates."""
    monkeypatch.setattr("osprey.cli.build_cmd.build", lambda **kw: None)
    monkeypatch.setattr("osprey.cli.deploy_cmd.up", lambda **kw: None)

    result = runner.invoke(
        profile, ["try", "my-assistant", "--preset", "hello-world", "-o", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    assert "1/2 build: my-assistant" in result.output
    assert "2/2 deploy up: my-assistant" in result.output


def test_try_skips_deploy_when_the_build_fails(runner: CliRunner, monkeypatch) -> None:
    """A build that did not complete leaves nothing worth deploying."""
    import click as _click

    def _failing_build(**kw):
        raise _click.ClickException("build failed")

    deployed: list[dict] = []
    monkeypatch.setattr("osprey.cli.build_cmd.build", _failing_build)
    monkeypatch.setattr("osprey.cli.deploy_cmd.up", lambda **kw: deployed.append(kw))

    result = runner.invoke(profile, ["try", "my-assistant", "--preset", "hello-world"])

    assert result.exit_code != 0
    assert deployed == []


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
