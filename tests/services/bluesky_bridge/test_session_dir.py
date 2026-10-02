"""Where the bridge's writable plan directory resolves."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from osprey.services.bluesky_bridge import session_dir
from osprey.services.bluesky_bridge.session_dir import resolve_session_plan_dir


@pytest.fixture(autouse=True, scope="module")
def no_per_test_value_outlives_its_test():
    """Hold the variable at its pre-module value once every test here is torn down.

    Several tests below patch the variable on the shared ``monkeypatch``, which
    records this test's suite value as the one to put back. If that undo ran
    after the suite fixture had already restored the environment, it would
    write a dead test's ``tmp_path`` back into ``os.environ``, and anything
    resolving between tests would land there instead of the in-package default
    the session guard watches.
    """
    before = os.environ.get("BLUESKY_SESSION_PLAN_DIR")
    yield
    assert os.environ.get("BLUESKY_SESSION_PLAN_DIR") == before


def test_the_suite_resolves_it_to_this_tests_tmp_path(tmp_path: Path) -> None:
    configured = Path(os.environ["BLUESKY_SESSION_PLAN_DIR"])
    assert not configured.exists()

    resolved = resolve_session_plan_dir()

    assert resolved == configured
    assert resolved.is_relative_to(tmp_path)
    assert not resolved.is_relative_to(Path(session_dir.__file__).parent)
    assert resolved.is_dir()


def test_the_variable_names_the_directory_and_resolution_creates_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "mounted" / "plans"
    monkeypatch.setenv("BLUESKY_SESSION_PLAN_DIR", str(target))

    assert resolve_session_plan_dir() == target
    assert target.is_dir()


@pytest.mark.parametrize("unset", ["deleted", "empty"])
def test_without_the_variable_the_in_image_default_is_used_and_created(
    unset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default = tmp_path / "in-image-default"
    monkeypatch.setattr(session_dir, "_DEFAULT_SESSION_PLAN_DIR", default)
    if unset == "deleted":
        monkeypatch.delenv("BLUESKY_SESSION_PLAN_DIR")
    else:
        monkeypatch.setenv("BLUESKY_SESSION_PLAN_DIR", "")

    assert resolve_session_plan_dir() == default
    assert default.is_dir()


def test_an_undone_monkeypatch_leaves_the_suite_value_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_value = os.environ["BLUESKY_SESSION_PLAN_DIR"]
    monkeypatch.delenv("BLUESKY_SESSION_PLAN_DIR")

    monkeypatch.undo()

    assert os.environ["BLUESKY_SESSION_PLAN_DIR"] == suite_value


@pytest.fixture
def module_level_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "module-override"
    monkeypatch.setenv("BLUESKY_SESSION_PLAN_DIR", str(target))
    return target


def test_a_fixture_override_wins_over_the_suite_value(module_level_override: Path) -> None:
    assert resolve_session_plan_dir() == module_level_override
