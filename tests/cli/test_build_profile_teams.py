"""Tests only the ``teams_bridge:`` block of the build-profile schema has.

The contract every chat bridge shares (trigger default, parsing, the
:meth:`BuildProfile.validate` dispatch and trigger checks) is parametrized over
all four bridges in ``test_build_profile_nextcloud.py``.
"""

from __future__ import annotations

from osprey.cli.build_profile import (
    BuildProfile,
    TeamsBridgeProfileConfig,
    _parse_profile,
)


def test_teams_bridge_custom_trigger_overrides_the_default() -> None:
    """An explicit trigger replaces the default on the dataclass itself."""
    assert TeamsBridgeProfileConfig(trigger="desy-question").trigger == "desy-question"


def test_no_teams_bridge_leaves_the_field_none() -> None:
    """A profile that declares no block leaves the field None (opt-in key)."""
    assert BuildProfile(name="x").teams_bridge is None


def test_teams_bridge_parses_independently_of_the_other_bridges() -> None:
    """The three bridge keys are independent: declaring one leaves the others None."""
    profile = _parse_profile({"name": "x", "teams_bridge": {}})
    assert profile.teams_bridge is not None
    assert profile.gchat_bridge is None
    assert profile.nextcloud_bridge is None
