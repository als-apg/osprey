"""The write-posture sunset: one constant, bound to the roster sunset.

The release that reaches :data:`RETIRED_WRITE_KEYS_REFUSED_FROM` fails
``test_the_write_key_window_is_still_open``: that release must turn every
write-posture warning window into its refusal (and drop the write-key rewrite),
then remove this test.

The retired per-tool policies of the governed tools have their own window,
:data:`RETIRED_APPROVAL_POLICY_REFUSED_FROM`, which since the owner's 2026-11-02
ruling closes on the same release as the write window: the release that
reaches it fails ``test_the_approval_policy_window_is_still_open`` and must turn
the drop-with-a-warning into a refusal.
"""

from __future__ import annotations

from packaging.version import Version

from osprey.cli.build_profile_load import (
    _PROFILE_SCHEMA_MIN_OSPREY,
    RETIRED_APPROVAL_POLICY_REFUSED_FROM,
    RETIRED_ROSTER_KEYS_REFUSED_FROM,
    RETIRED_WRITE_KEYS_REFUSED_FROM,
)
from osprey.version import get_release_version


def test_the_write_sunset_is_the_roster_sunset():
    assert RETIRED_WRITE_KEYS_REFUSED_FROM is RETIRED_ROSTER_KEYS_REFUSED_FROM


def test_the_write_window_closes_two_minors_after_the_vocabulary_release():
    floor = Version(_PROFILE_SCHEMA_MIN_OSPREY)
    end = Version(RETIRED_WRITE_KEYS_REFUSED_FROM)
    assert end > floor
    assert (end.major, end.minor) == (floor.major, floor.minor + 2)
    assert end.micro == 0


def test_the_write_key_window_is_still_open():
    """Fails on the sunset release: refuse what the window still reads."""
    assert Version(get_release_version()) < Version(RETIRED_WRITE_KEYS_REFUSED_FROM)


def test_the_approval_policy_window_closes_with_the_write_window():
    assert RETIRED_APPROVAL_POLICY_REFUSED_FROM == "2026.12.0"
    assert Version(RETIRED_APPROVAL_POLICY_REFUSED_FROM) == Version(RETIRED_WRITE_KEYS_REFUSED_FROM)


def test_the_approval_policy_window_is_still_open():
    """Fails on the sunset release: refuse a governed tool's policy instead of dropping it."""
    assert Version(get_release_version()) < Version(RETIRED_APPROVAL_POLICY_REFUSED_FROM)
