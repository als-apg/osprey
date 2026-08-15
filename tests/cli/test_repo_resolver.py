"""Tests for the one deployment-repo discovery rule."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from osprey.cli.repo_resolver import (
    EXPLICIT_TARGET_COMMANDS,
    HELD_SOURCE_ZONE_DIRNAME,
    MODE_EXEMPT_FLAGS,
    PROFILE_FILENAME,
    REPO_FREE_COMMANDS,
    SELF_DISCOVERING_COMMANDS,
    RepoNotFoundError,
    find_repo_root,
    is_repo_free,
    repo_option,
)


def make_repo(root: Path) -> Path:
    """Materialize the one marker that makes *root* a deployment repo."""
    root.mkdir(parents=True, exist_ok=True)
    (root / PROFILE_FILENAME).write_text("facility:\n  name: test\n")
    return root


# ---------------------------------------------------------------------------
# find_repo_root
# ---------------------------------------------------------------------------


def test_finds_root_when_standing_in_it(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "als-assistant")

    assert find_repo_root(repo) == repo.resolve()


def test_finds_root_from_nested_subdirectory(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "als-assistant")
    nested = repo / "data" / "channels" / "storage-ring"
    nested.mkdir(parents=True)

    assert find_repo_root(nested) == repo.resolve()


def test_nested_repos_resolve_to_the_nearest_one(tmp_path: Path) -> None:
    outer = make_repo(tmp_path / "outer")
    inner = make_repo(outer / "sub" / "inner")
    deep = inner / "personas"
    deep.mkdir()

    assert find_repo_root(deep) == inner.resolve()


def test_defaults_to_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path / "als-assistant")
    nested = repo / "scripts"
    nested.mkdir()
    monkeypatch.chdir(nested)

    assert find_repo_root() == repo.resolve()


def test_returns_absolute_resolved_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path / "als-assistant")
    monkeypatch.chdir(repo)

    resolved = find_repo_root(Path("."))

    assert resolved.is_absolute()
    assert resolved == repo.resolve()


def test_walks_through_a_symlinked_start(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "als-assistant")
    (repo / "data").mkdir()
    link = tmp_path / "shortcut"
    link.symlink_to(repo / "data", target_is_directory=True)

    assert find_repo_root(link) == repo.resolve()


def test_profile_directory_is_not_a_marker(tmp_path: Path) -> None:
    """A directory named profile.yml is not a manifest, so it marks nothing."""
    decoy = tmp_path / "decoy"
    (decoy / PROFILE_FILENAME).mkdir(parents=True)

    with pytest.raises(RepoNotFoundError):
        find_repo_root(decoy)


def test_missing_repo_raises_with_guidance(tmp_path: Path) -> None:
    outside = tmp_path / "somewhere" / "else"
    outside.mkdir(parents=True)

    with pytest.raises(RepoNotFoundError) as excinfo:
        find_repo_root(outside)

    message = str(excinfo.value)
    assert PROFILE_FILENAME in message
    assert str(outside.resolve()) in message
    assert "--repo" in message
    assert "osprey init" in message
    assert excinfo.value.searched_from == outside.resolve()


def test_a_held_aside_source_zone_is_named_as_such(tmp_path: Path) -> None:
    """A repo mid-replacement is not a missing repo, and must not be told it is.

    `osprey init --force` holds the source zone aside while the new one renders,
    so a run killed in that window leaves a deployment with no ``profile.yml`` at
    its root. Every repo-scoped verb reaches this walk, and the ordinary refusal
    would tell the owner of a whole deployment to create one — advice that reads
    as "yours is gone" when in fact it is one directory down.
    """
    repo = tmp_path / "als-assistant"
    (repo / HELD_SOURCE_ZONE_DIRNAME).mkdir(parents=True)
    (repo / HELD_SOURCE_ZONE_DIRNAME / PROFILE_FILENAME).write_text("name: Ours\n")

    with pytest.raises(RepoNotFoundError) as excinfo:
        find_repo_root(repo)

    message = str(excinfo.value)
    assert HELD_SOURCE_ZONE_DIRNAME in message
    assert str(repo.resolve()) in message
    # The repair is the interrupted command, named as something to type.
    assert "osprey init" in message
    assert "--preset" in message
    # And it must not read as data loss, because there is none.
    assert "Nothing has been lost" in message


def test_a_held_aside_zone_is_found_from_a_subdirectory(tmp_path: Path) -> None:
    """The hint follows the same upward walk the marker does."""
    repo = tmp_path / "als-assistant"
    (repo / HELD_SOURCE_ZONE_DIRNAME).mkdir(parents=True)
    nested = repo / "var" / "audit"
    nested.mkdir(parents=True)

    with pytest.raises(RepoNotFoundError) as excinfo:
        find_repo_root(nested)

    assert str(repo.resolve()) in str(excinfo.value)


def test_an_enclosing_repo_still_wins_over_a_held_aside_one(tmp_path: Path) -> None:
    """The hint never displaces a real answer: a marker above still resolves."""
    outer = make_repo(tmp_path / "outer")
    inner = outer / "nested"
    (inner / HELD_SOURCE_ZONE_DIRNAME).mkdir(parents=True)

    assert find_repo_root(inner) == outer.resolve()


def test_nonexistent_start_names_the_path(tmp_path: Path) -> None:
    ghost = tmp_path / "not-there"

    with pytest.raises(RepoNotFoundError) as excinfo:
        find_repo_root(ghost)

    assert str(ghost.resolve()) in str(excinfo.value)


def test_file_start_is_rejected_as_not_a_directory(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "als-assistant")

    with pytest.raises(RepoNotFoundError) as excinfo:
        find_repo_root(repo / PROFILE_FILENAME)

    assert "Not a directory" in str(excinfo.value)


def test_error_is_a_click_exception_with_exit_code_one(tmp_path: Path) -> None:
    """CLI callers get the message and exit 1, never a traceback."""
    assert issubclass(RepoNotFoundError, click.ClickException)

    with pytest.raises(RepoNotFoundError) as excinfo:
        find_repo_root(tmp_path)

    assert excinfo.value.exit_code == 1


# ---------------------------------------------------------------------------
# --repo option
# ---------------------------------------------------------------------------


@click.command()
@repo_option
def _probe(repo: Path | None) -> None:
    """Report the resolved repo."""
    click.echo(str(find_repo_root(repo)))


def test_repo_option_overrides_the_starting_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path / "als-assistant")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = CliRunner().invoke(_probe, ["--repo", str(repo)])

    assert result.exit_code == 0
    assert result.output.strip() == str(repo.resolve())


def test_repo_option_subdirectory_still_walks_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path / "als-assistant")
    nested = repo / "data"
    nested.mkdir()
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(_probe, ["--repo", str(nested)])

    assert result.exit_code == 0
    assert result.output.strip() == str(repo.resolve())


def test_repo_option_omitted_falls_back_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path / "als-assistant")
    monkeypatch.chdir(repo)

    result = CliRunner().invoke(_probe, [])

    assert result.exit_code == 0
    assert result.output.strip() == str(repo.resolve())


def test_repo_option_rejects_a_file_as_usage_error(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "als-assistant")

    result = CliRunner().invoke(_probe, ["--repo", str(repo / PROFILE_FILENAME)])

    assert result.exit_code == 2


def test_repo_option_rejects_a_missing_path_as_usage_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(_probe, ["--repo", str(tmp_path / "not-there")])

    assert result.exit_code == 2


def test_missing_repo_surfaces_as_a_clean_cli_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(_probe, [])

    assert result.exit_code == 1
    assert "No OSPREY deployment repo found" in result.output


def test_repo_option_help_text_is_shared(tmp_path: Path) -> None:
    result = CliRunner().invoke(_probe, ["--help"])

    assert "--repo" in result.output
    assert "nearest profile.yml" in result.output


# ---------------------------------------------------------------------------
# exemption predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    sorted(REPO_FREE_COMMANDS | SELF_DISCOVERING_COMMANDS | EXPLICIT_TARGET_COMMANDS),
)
def test_exempt_commands_are_repo_free(command: str) -> None:
    assert is_repo_free(command)


@pytest.mark.parametrize(
    "command",
    ["build", "up", "down", "restart", "status", "logs", "reset", "set", "validate", "chat"],
)
def test_lifecycle_verbs_need_a_repo(command: str) -> None:
    assert not is_repo_free(command)


@pytest.mark.parametrize(
    "command",
    ["users seed", "users env", "scaffold ci", "web stop", "config"],
)
def test_repo_scoped_subcommands_need_a_repo(command: str) -> None:
    assert not is_repo_free(command)


def test_subcommands_inherit_an_exempt_group() -> None:
    assert is_repo_free("ariel search")
    assert is_repo_free("channel-finder validate")


def test_osprey_prefix_is_accepted() -> None:
    assert is_repo_free("osprey init")
    assert not is_repo_free("osprey users seed")


def test_unknown_commands_are_not_exempt() -> None:
    assert not is_repo_free("")
    assert not is_repo_free("nonesuch")


def test_mode_exempt_commands_still_take_repo() -> None:
    """A repo-free mode is a flag on a repo-scoped command, not an exemption."""
    for command, flags in MODE_EXEMPT_FLAGS.items():
        assert not is_repo_free(command)
        assert flags


# ---------------------------------------------------------------------------
# import budget (TR-2)
# ---------------------------------------------------------------------------


def test_import_stays_off_heavy_machinery() -> None:
    """Discovery runs before anything is loaded, so it must load nothing."""
    heavy = ["yaml", "jinja2", "osprey.registry", "osprey.deployment"]
    probe = (
        "import sys; import osprey.cli.repo_resolver; "
        f"print([m for m in {heavy!r} if m in sys.modules])"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
