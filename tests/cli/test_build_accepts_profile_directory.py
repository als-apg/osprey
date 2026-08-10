"""``osprey build`` takes a profile directory, the way ``profile validate`` does.

A profile directory is the unit a facility works with — ``profile/`` in a
facility repo, ``<name>-profile/`` beside a project — so both verbs that read a
profile accept either the directory or an explicit file. They resolve it through
the same function: one of them refusing what the other accepts is a trap, and
``osprey build my-facility profile/`` is the natural thing to type in a repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.profile_cmd import profile as profile_group

PROFILE_TEXT = "name: Facility\ndata_bundle: hello_world\nprovider: anthropic\nmodel: haiku\n"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def facility_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "facility-profiles"
    (repo / "profile").mkdir(parents=True)
    (repo / "profile" / "profile.yml").write_text(PROFILE_TEXT, encoding="utf-8")
    return repo


def _build(runner: CliRunner, target: Path, out: Path, name: str = "assistant") -> object:
    return runner.invoke(
        build,
        [name, str(target), "--skip-deps", "--skip-lifecycle", "--output-dir", str(out)],
    )


# ---------------------------------------------------------------------------
# Both spellings build
# ---------------------------------------------------------------------------


def test_a_profile_directory_is_accepted(runner: CliRunner, facility_repo: Path) -> None:
    out = facility_repo.parent / "out"

    result = _build(runner, facility_repo / "profile", out)

    assert result.exit_code == 0, result.output
    assert (out / "assistant" / "config.yml").is_file()


def test_an_explicit_profile_file_still_works(runner: CliRunner, facility_repo: Path) -> None:
    out = facility_repo.parent / "out"

    result = _build(runner, facility_repo / "profile" / "profile.yml", out)

    assert result.exit_code == 0, result.output
    assert (out / "assistant" / "config.yml").is_file()


def test_the_two_spellings_render_the_same_project(runner: CliRunner, facility_repo: Path) -> None:
    """The directory spelling resolves to the file, so nothing downstream can
    tell which one was typed."""
    out = facility_repo.parent / "out"

    assert _build(runner, facility_repo / "profile", out, "from-dir").exit_code == 0
    assert (
        _build(runner, facility_repo / "profile" / "profile.yml", out, "from-file").exit_code == 0
    )

    from_dir = yaml.safe_load((out / "from-dir" / "config.yml").read_text(encoding="utf-8"))
    from_file = yaml.safe_load((out / "from-file" / "config.yml").read_text(encoding="utf-8"))
    del from_dir["project_name"], from_file["project_name"]
    del from_dir["project_root"], from_file["project_root"]

    assert from_dir == from_file


def test_the_directory_spelling_still_lands_in_the_repo_build_dir(
    runner: CliRunner, facility_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It composes with output resolution: the resolved file anchors the walk
    to the facility repo exactly as an explicitly-typed one does."""
    monkeypatch.chdir(facility_repo)

    result = runner.invoke(build, ["assistant", "profile/", "--skip-deps", "--skip-lifecycle"])

    assert result.exit_code == 0, result.output
    assert (facility_repo / "build" / "assistant" / "config.yml").is_file()


# ---------------------------------------------------------------------------
# A directory that holds no profile
# ---------------------------------------------------------------------------


def test_a_directory_without_a_profile_is_refused_with_the_way_out(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    empty = tmp_path / "not-a-profile"
    empty.mkdir()

    with caplog.at_level("ERROR"):
        result = runner.invoke(build, ["assistant", str(empty), "--skip-deps", "--skip-lifecycle"])

    assert result.exit_code != 0
    reported = caplog.text + result.output
    assert "No profile.yml" in reported
    assert "osprey profile new" in reported


# ---------------------------------------------------------------------------
# The two verbs agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["profile", "profile/profile.yml"])
def test_validate_and_build_accept_the_same_spellings(
    runner: CliRunner, facility_repo: Path, spelling: str
) -> None:
    target = facility_repo / spelling

    validated = runner.invoke(profile_group, ["validate", str(target)])
    built = _build(runner, target, facility_repo.parent / f"out-{spelling.count('/')}")

    assert validated.exit_code == 0, validated.output
    assert built.exit_code == 0, built.output
