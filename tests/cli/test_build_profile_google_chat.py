"""Tests only the ``gchat_bridge:`` block of the build-profile schema has.

The contract every chat bridge shares (trigger default, parsing, the
:meth:`BuildProfile.validate` dispatch and trigger checks) is parametrized over
all four bridges in ``test_build_profile_nextcloud.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_profile import (
    BuildProfile,
    DispatchConfig,
    GChatBridgeProfileConfig,
    _parse_profile,
)
from osprey_connectors.errors import BuildProfileError

pytestmark = pytest.mark.usefixtures("facility_data_tree")


def _bridge_profile(**kwargs: object) -> BuildProfile:
    """A profile declaring the bridge plus a dispatch block, overridable per test."""
    fields: dict = {
        "name": "x",
        "data": "data",
        "dispatch": DispatchConfig(triggers="triggers.yml"),
        "gchat_bridge": GChatBridgeProfileConfig(),
    }
    fields.update(kwargs)
    return BuildProfile(**fields)


def test_gchat_bridge_parses_independently_of_nextcloud() -> None:
    """The two bridge keys are independent: declaring one leaves the other None."""
    profile = _parse_profile({"name": "x", "gchat_bridge": {}})
    assert profile.gchat_bridge is not None
    assert profile.nextcloud_bridge is None


def test_gchat_bridge_triggers_file_with_a_non_mapping_part_reports_parse_failure(
    tmp_path: Path,
) -> None:
    """A trigger whose action is not a mapping surfaces as the same legible
    'fix that file first' error, naming the trigger."""
    (tmp_path / "triggers.yml").write_text(
        "dispatcher:\n  dispatch_target: x\ntriggers:\n"
        "  - name: beam-loss\n    source: webhook\n    action: do it\n",
        encoding="utf-8",
    )
    with pytest.raises(BuildProfileError) as exc:
        _bridge_profile().validate(tmp_path)
    message = str(exc.value)
    assert "gchat_bridge.trigger cannot be checked" in message
    assert "beam-loss" in message
