"""`osprey build` and `osprey validate` judge `system.timezone` on the profile."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_timezone import system_timezone_errors
from osprey.cli.init_cmd import init
from osprey.cli.validate_cmd import validate

PROFILE = "profile.yml"
UTC_LINE = "  system.timezone: UTC"

# --------------------------------------------------------------------------- #
# Unit half
# --------------------------------------------------------------------------- #


def test_a_real_zone_draws_no_error():
    assert system_timezone_errors({"system.timezone": "Europe/Berlin"}) == []


def test_utc_draws_no_error():
    assert system_timezone_errors({"system.timezone": "UTC"}) == []


def test_an_absent_key_draws_no_error():
    assert system_timezone_errors({}) == []


def test_a_misspelt_zone_is_refused_by_key_and_value():
    errors = system_timezone_errors({"system.timezone": "Amerika/Los_Angeles"})
    assert len(errors) == 1
    assert "system.timezone" in errors[0]
    assert "'Amerika/Los_Angeles'" in errors[0]
    assert "'America/Los_Angeles'" in errors[0]


def test_a_wrongly_cased_zone_is_refused():
    errors = system_timezone_errors({"system.timezone": "america/los_angeles"})
    assert len(errors) == 1
    assert "'America/Los_Angeles'" in errors[0]


def test_a_nested_spelling_is_judged():
    errors = system_timezone_errors({"system": {"timezone": "Nowhere/City"}})
    assert len(errors) == 1
    assert "system: timezone" in errors[0]


def test_an_environment_reference_is_left_to_health(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FACILITY_TZ", raising=False)
    assert system_timezone_errors({"system.timezone": "${FACILITY_TZ}"}) == []


@pytest.mark.parametrize("value", ["", None])
def test_an_empty_or_null_value_is_refused(value):
    errors = system_timezone_errors({"system.timezone": value})
    assert len(errors) == 1
    assert "system.timezone" in errors[0]


# --------------------------------------------------------------------------- #
# CLI half
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def materialized(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real hello-world materialization to edit copies of."""
    root = tmp_path_factory.mktemp("timezone") / "demo"
    result = CliRunner().invoke(init, [str(root), "--preset", "hello-world", "--no-git"])
    assert result.exit_code == 0, result.output
    return root


@pytest.fixture
def repo(materialized: Path, tmp_path: Path) -> Path:
    target = tmp_path / "demo"
    shutil.copytree(materialized, target)
    return target


def _set_zone(repo: Path, text: str) -> None:
    profile = repo / PROFILE
    body = profile.read_text()
    assert body.count(UTC_LINE + "\n") == 1
    profile.write_text(body.replace(UTC_LINE + "\n", f"  system.timezone: {text}\n"))


def _build(repo: Path):
    return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def _validate(repo: Path):
    return CliRunner().invoke(validate, ["--repo", str(repo)])


def test_build_refuses_a_misspelt_zone(repo: Path):
    _set_zone(repo, "Amerika/Los_Angeles")
    result = _build(repo)
    assert result.exit_code == 2, result.output
    assert "Profile validation failed" in result.output
    assert "system.timezone" in result.output
    assert "Amerika/Los_Angeles" in result.output
    assert not (repo / "build" / "config.yml").exists()


def test_validate_refuses_a_misspelt_zone(repo: Path):
    _set_zone(repo, "Amerika/Los_Angeles")
    result = _validate(repo)
    assert result.exit_code == 2, result.output
    assert "system.timezone" in result.output
    assert "Amerika/Los_Angeles" in result.output


def test_an_unresolved_reference_builds(repo: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FACILITY_TZ", raising=False)
    _set_zone(repo, "${FACILITY_TZ}")
    result = _build(repo)
    assert result.exit_code == 0, result.output
