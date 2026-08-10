"""Where ``osprey build`` renders, and why it does not depend on the cwd.

A facility repo keeps its profile in ``profile/`` and its rendered projects in
``build/``. The build finds that repo by walking up from the profile it
resolved, so the same command run from the repo root, from inside ``profile/``,
or from any nested subdirectory lands in one place — a rebuild refreshes the
project instead of leaving copies wherever the operator happened to be
standing. Everything else keeps rendering under the current directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_resolve import (
    FACILITY_BUILD_DIRNAME,
    FACILITY_PROFILE_DIRNAME,
    facility_repo_root,
    resolve_build_output_dir,
)

PROFILE_TEXT = "name: Facility\ndata_bundle: hello_world\nprovider: anthropic\nmodel: haiku\n"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def facility_repo(tmp_path: Path) -> Path:
    """A facility repo: ``profile/profile.yml`` plus a nested working dir."""
    repo = tmp_path / "facility-profiles"
    (repo / FACILITY_PROFILE_DIRNAME).mkdir(parents=True)
    (repo / FACILITY_PROFILE_DIRNAME / "profile.yml").write_text(PROFILE_TEXT, encoding="utf-8")
    (repo / "docs" / "notes").mkdir(parents=True)
    return repo


def _build(runner: CliRunner, repo: Path, *extra: str) -> object:
    profile = repo / FACILITY_PROFILE_DIRNAME / "profile.yml"
    return runner.invoke(
        build,
        ["assistant", str(profile), "--skip-deps", "--skip-lifecycle", *extra],
    )


# ---------------------------------------------------------------------------
# facility_repo_root — the walk itself
# ---------------------------------------------------------------------------


def test_the_repo_root_is_the_parent_of_the_profile_dir(facility_repo: Path) -> None:
    assert facility_repo_root(facility_repo / FACILITY_PROFILE_DIRNAME) == facility_repo


def test_the_walk_finds_the_repo_from_below_the_profile_dir(facility_repo: Path) -> None:
    """Resolution already reports ``profile/`` as the root for a persona delta,
    but the walk does not lean on that: any path under it finds the repo, so a
    persona and its host profile can never resolve to different ones."""
    personas = facility_repo / FACILITY_PROFILE_DIRNAME / "personas"
    personas.mkdir()

    assert facility_repo_root(personas) == facility_repo


def test_a_standalone_profile_directory_belongs_to_no_repo(tmp_path: Path) -> None:
    standalone = tmp_path / "profiles" / "als-dev"
    standalone.mkdir(parents=True)

    assert facility_repo_root(standalone) is None


def test_a_sibling_profile_directory_is_not_a_repo_marker(tmp_path: Path) -> None:
    """Only the ``profile/`` the profile actually lives in counts — an
    unrelated one next door says nothing about where this profile belongs."""
    (tmp_path / FACILITY_PROFILE_DIRNAME).mkdir()
    elsewhere = tmp_path / "scratch"
    elsewhere.mkdir()

    assert facility_repo_root(elsewhere) is None


def test_a_materialized_preset_profile_belongs_to_no_repo(tmp_path: Path) -> None:
    """``<name>-profile/`` sits beside its project, not inside a facility
    repo — which is why a --preset build keeps rendering where it always did."""
    materialized = tmp_path / "assistant-profile"
    materialized.mkdir()

    assert facility_repo_root(materialized) is None


# ---------------------------------------------------------------------------
# resolve_build_output_dir — the three answers, in priority order
# ---------------------------------------------------------------------------


def test_a_repo_profile_resolves_to_the_repo_build_dir(facility_repo: Path) -> None:
    resolved = resolve_build_output_dir(None, facility_repo / FACILITY_PROFILE_DIRNAME)

    assert resolved == facility_repo / FACILITY_BUILD_DIRNAME


def test_an_explicit_output_dir_wins_over_the_repo(facility_repo: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    resolved = resolve_build_output_dir(str(elsewhere), facility_repo / FACILITY_PROFILE_DIRNAME)

    assert resolved == elsewhere.resolve()


def test_a_profile_outside_a_repo_falls_back_to_the_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standalone = tmp_path / "profiles" / "als-dev"
    standalone.mkdir(parents=True)
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)

    assert resolve_build_output_dir(None, standalone) == working.resolve()


# ---------------------------------------------------------------------------
# osprey build — the same destination from anywhere in the repo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "from_subdir",
    ["", FACILITY_PROFILE_DIRNAME, "docs/notes"],
    ids=["repo-root", "inside-profile", "nested-subdir"],
)
def test_a_repo_build_lands_in_the_repo_build_dir_from_anywhere(
    runner: CliRunner,
    facility_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    from_subdir: str,
) -> None:
    monkeypatch.chdir(facility_repo / from_subdir if from_subdir else facility_repo)

    result = _build(runner, facility_repo)

    assert result.exit_code == 0, result.output
    assert (facility_repo / FACILITY_BUILD_DIRNAME / "assistant" / "config.yml").is_file()


def test_the_repo_build_dir_is_created_when_it_does_not_exist(
    runner: CliRunner, facility_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(facility_repo)
    assert not (facility_repo / FACILITY_BUILD_DIRNAME).exists()

    assert _build(runner, facility_repo).exit_code == 0
    assert (facility_repo / FACILITY_BUILD_DIRNAME).is_dir()


def test_nothing_is_rendered_into_the_directory_the_build_ran_from(
    runner: CliRunner, facility_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression this replaces: a build from inside ``profile/`` used to
    render the project into the profile directory itself."""
    profile_dir = facility_repo / FACILITY_PROFILE_DIRNAME
    monkeypatch.chdir(profile_dir)

    assert _build(runner, facility_repo).exit_code == 0
    assert not (profile_dir / "assistant").exists()


def test_output_dir_overrides_the_repo_build_dir(
    runner: CliRunner, facility_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(facility_repo)
    elsewhere = tmp_path / "elsewhere"

    result = _build(runner, facility_repo, "--output-dir", str(elsewhere))

    assert result.exit_code == 0, result.output
    assert (elsewhere / "assistant" / "config.yml").is_file()
    assert not (facility_repo / FACILITY_BUILD_DIRNAME).exists()


def test_a_bare_profile_still_renders_under_the_cwd(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Today's behavior for everything that is not a facility repo."""
    profile = tmp_path / "als-dev.yml"
    profile.write_text(PROFILE_TEXT, encoding="utf-8")
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)

    result = runner.invoke(build, ["assistant", str(profile), "--skip-deps", "--skip-lifecycle"])

    assert result.exit_code == 0, result.output
    assert (working / "assistant" / "config.yml").is_file()


def test_a_preset_build_renders_beside_its_materialized_profile(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --preset build materializes ``<name>-profile/`` in the output
    directory, so the two must stay in the same place the cwd puts them."""
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)

    result = runner.invoke(
        build,
        ["assistant", "--preset", "hello-world", "--skip-deps", "--skip-lifecycle"],
    )

    assert result.exit_code == 0, result.output
    assert (working / "assistant" / "config.yml").is_file()
    assert (working / "assistant-profile" / "profile.yml").is_file()


# ---------------------------------------------------------------------------
# The operator is told where it went
# ---------------------------------------------------------------------------


def test_the_resolved_destination_is_logged(
    runner: CliRunner,
    facility_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A destination the operator did not name is one they should still be
    told — read from the log record, since CliRunner rich-wraps the output."""
    monkeypatch.chdir(facility_repo / "docs" / "notes")

    with caplog.at_level("INFO"):
        assert _build(runner, facility_repo).exit_code == 0

    expected = str(facility_repo / FACILITY_BUILD_DIRNAME / "assistant")
    assert any(expected in record.getMessage() for record in caplog.records), (
        f"destination {expected} never logged; got "
        f"{[record.getMessage()[:80] for record in caplog.records]}"
    )
