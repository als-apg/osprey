"""The facility zone predicate and its close-match hint."""

from __future__ import annotations

import pytest

from osprey.utils import facility
from osprey.utils.facility import closest_zone_name, is_zone_name


def test_utc_is_a_zone_name():
    assert is_zone_name("UTC")


def test_a_real_zone_is_a_zone_name():
    assert is_zone_name("Europe/Berlin")


def test_a_misspelt_zone_is_not():
    assert not is_zone_name("Amerika/Los_Angeles")


def test_a_wrongly_cased_zone_is_not():
    """A case-insensitive filesystem opens this name; a Linux container does not."""
    assert not is_zone_name("america/los_angeles")


def test_an_empty_zone_database_judges_nothing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(facility, "available_timezones", lambda: set())
    assert is_zone_name("Amerika/Los_Angeles")


def test_closest_zone_name_fixes_case():
    assert closest_zone_name("america/los_angeles") == "America/Los_Angeles"


def test_closest_zone_name_fixes_a_typo():
    assert closest_zone_name("Amerika/Los_Angeles") == "America/Los_Angeles"
    assert closest_zone_name("Europe/Berln") == "Europe/Berlin"


def test_closest_zone_name_gives_none_for_nonsense():
    assert closest_zone_name("Mars/Olympus") is None
