"""The retired-spellings sunset: one release line, one phrase, three windows.

The retired spellings are read through the stable release line
:data:`RETIRED_SPELLINGS_LAST_RELEASE_LINE` and refused in the next stable
release, whatever its number (owner ruling 2026-11-13). The first stable
release of a later line fails ``test_the_write_key_window_is_still_open`` and
``test_the_approval_policy_window_is_still_open``: that release must turn every
write-posture warning window into its refusal (drop the write-key rewrite,
refuse a governed tool's policy instead of dropping it), then remove these
tests.
"""

from __future__ import annotations

import pytest
from packaging.version import Version

from osprey.cli.build_profile_load import (
    _PROFILE_SCHEMA_MIN_OSPREY,
    RETIRED_APPROVAL_POLICY_REFUSED_FROM,
    RETIRED_ROSTER_KEYS_REFUSED_FROM,
    RETIRED_SPELLINGS_LAST_RELEASE_LINE,
    RETIRED_WRITE_KEYS_REFUSED_FROM,
    retired_spellings_window_open,
)
from osprey.version import get_release_version


def test_the_three_windows_close_together():
    assert RETIRED_WRITE_KEYS_REFUSED_FROM is RETIRED_ROSTER_KEYS_REFUSED_FROM
    assert RETIRED_APPROVAL_POLICY_REFUSED_FROM is RETIRED_ROSTER_KEYS_REFUSED_FROM


def test_the_sunset_phrase_names_no_version_and_no_date():
    assert RETIRED_ROSTER_KEYS_REFUSED_FROM == "the next stable OSPREY release"


def test_the_last_reading_line_is_the_one_after_the_vocabulary_release():
    floor = Version(_PROFILE_SCHEMA_MIN_OSPREY)
    assert RETIRED_SPELLINGS_LAST_RELEASE_LINE == (floor.major, floor.minor + 1)


@pytest.mark.parametrize(
    "release",
    [
        "2026.9.0b3",  # main before the stable tag
        "2026.11.0b1",
        "2026.11.0",
        "2026.11.4",  # a patch of the last reading line
        "2026.12.0b1",  # pre-releases of the refusing release still read
        "2026.12.0rc1",
        "2027.1.0b2",
        "0.0.0",  # no version resolved
    ],
)
def test_the_window_is_open_through_the_last_line_and_on_pre_releases(release):
    assert retired_spellings_window_open(release)


@pytest.mark.parametrize("release", ["2026.12.0", "2026.12.1", "2027.1.0", "2027.2.3"])
def test_the_window_is_closed_on_a_stable_release_of_a_later_line(release):
    assert not retired_spellings_window_open(release)


def test_the_write_key_window_is_still_open():
    """Fails on the sunset release: refuse what the window still reads."""
    assert retired_spellings_window_open(get_release_version())


def test_the_approval_policy_window_is_still_open():
    """Fails on the sunset release: refuse a governed tool's policy instead of dropping it."""
    assert retired_spellings_window_open(get_release_version())
