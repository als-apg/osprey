"""What the unknown-root-entry warning has to say, beyond listing the names.

Two causes land an operator here and they need different remedies: a
misspelled convention directory, and a build pointed at a facility repo rather
than at the ``profile/`` directory nested inside it. The warning cannot tell
them apart, so it must name both — these tests pin that it does, because the
second remedy is the one nothing else in the build output hints at.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from osprey.cli.profile_conventions import warn_unknown_root_entries


@pytest.fixture
def facility_repo(tmp_path: Path) -> Path:
    """A facility repo mistaken for a profile root.

    The realistic shape: repo-owned directories at the top level and the
    profile one level down, so every top-level entry reads as unrecognized.
    """
    repo = tmp_path / "my-facility"
    (repo / "profile").mkdir(parents=True)
    (repo / "profile" / "profile.yml").write_text("name: my-facility\n")
    for owned in ("ioc", "nginx", "tests"):
        (repo / owned).mkdir()
    return repo


def test_warning_names_the_nested_profile_directory(
    facility_repo: Path, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.WARNING):
        unknown = warn_unknown_root_entries(facility_repo)

    assert unknown == ["ioc", "nginx", "profile", "tests"]
    assert "profile/profile.yml" in caplog.text
    assert "facility repo" in caplog.text


def test_warning_names_the_build_interview(facility_repo: Path, caplog: pytest.LogCaptureFixture):
    """The remedy for a repo that has no `profile/` yet is not an edit but a layout."""
    with caplog.at_level(logging.WARNING):
        warn_unknown_root_entries(facility_repo)

    assert "/osprey-build-interview" in caplog.text


def test_warning_names_the_directory_it_is_judging(
    facility_repo: Path, caplog: pytest.LogCaptureFixture
):
    """Without the path, an operator running a build over several profiles
    cannot tell which one the advice is about."""
    with caplog.at_level(logging.WARNING):
        warn_unknown_root_entries(facility_repo)

    assert str(facility_repo) in caplog.text


def test_typo_remedy_survives_alongside_the_nested_layout_remedy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """The original cause still gets its own answer: the convention list."""
    profile = tmp_path / "my-profile"
    profile.mkdir()
    (profile / "profile.yml").write_text("name: my-profile\n")
    (profile / "rule").mkdir()

    with caplog.at_level(logging.WARNING):
        assert warn_unknown_root_entries(profile) == ["rule"]

    assert "check for a typo" in caplog.text
    assert "rules/" in caplog.text
