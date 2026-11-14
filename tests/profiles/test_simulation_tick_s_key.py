"""The ``simulation.tick_s`` key: the simulator's tick period.

``resolve_tick_s`` reads the key from a rendered config; every shipped preset
states it, and a render without it resolves the default.
"""

from __future__ import annotations

import pytest

from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_document
from osprey_connectors.simulation import (
    DEFAULT_FAST_TICK_S,
    DEFAULT_TICK_S,
    resolve_fast_tick_s,
    resolve_tick_s,
)

#: The shipped presets every all-templates key must appear in.
_PRESETS = ("hello-world", "demo")


def test_a_render_without_the_key_resolves_the_default() -> None:
    config = _expand_dotted(resolve_build_document(None, "hello-world").profile.config)
    del config["simulation"]["tick_s"]
    assert "tick_s" not in config["simulation"]
    assert resolve_tick_s(config) == DEFAULT_TICK_S == 1.0


@pytest.mark.parametrize("preset", _PRESETS)
def test_every_preset_states_the_default(preset: str) -> None:
    config = _expand_dotted(resolve_build_document(None, preset).profile.config)
    assert config["simulation"]["tick_s"] == DEFAULT_TICK_S
    assert resolve_tick_s(config) == DEFAULT_TICK_S


@pytest.mark.parametrize("preset", _PRESETS)
def test_no_preset_states_the_fast_period(preset: str) -> None:
    """``simulation.fast_tick_s`` is not all-templates: the resolver holds its default."""
    config = _expand_dotted(resolve_build_document(None, preset).profile.config)
    assert "fast_tick_s" not in (config.get("simulation") or {})
    assert resolve_fast_tick_s(config) == DEFAULT_FAST_TICK_S == 0.1
