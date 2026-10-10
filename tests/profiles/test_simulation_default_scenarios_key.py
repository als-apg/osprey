"""The ``simulation.default_scenarios`` key: the set a never-chosen deployment starts in.

``resolve_default_scenarios`` reads the key from a rendered config; the
control-assistant preset states it and the other root presets leave it out.
"""

from __future__ import annotations

import pytest

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_document
from osprey.simulation.apply import resolve_default_scenarios


def test_the_control_assistant_preset_starts_in_rf_thermal() -> None:
    config = _expand_dotted(resolve_build_document(None, "control-assistant").profile.config)
    assert resolve_default_scenarios(config) == ("rf-thermal",)


@pytest.mark.parametrize("preset", ["hello-world", "ariel-standalone", "channel-finder-standalone"])
def test_the_other_presets_state_no_start_set(preset: str) -> None:
    config = _expand_dotted(resolve_build_document(None, preset).profile.config)
    assert "default_scenarios" not in config.get("simulation", {})
    assert resolve_default_scenarios(config) == ()


@pytest.mark.parametrize("value", [None, []])
def test_absent_or_empty_states_none(value) -> None:
    assert resolve_default_scenarios({"simulation": {"default_scenarios": value}}) == ()


def test_names_are_kept_in_order_once() -> None:
    config = {"simulation": {"default_scenarios": ["b", "a", "b"]}}
    assert resolve_default_scenarios(config) == ("b", "a")
