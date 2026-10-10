"""Which shipped presets carry the LATTICE tab.

The lattice dashboard is part of the control-assistant preset: the host profile
lists ``lattice`` in ``web_panels:``, the readonly, readwrite and admin personas
inherit it, and the knowledge and logbook personas, which inherit the same list,
subtract it through ``exclude: web_panels:``. The single-purpose presets never
list it.

Each preset is resolved the way ``osprey init`` resolves it, so ``extends:``,
list unions and ``exclude:`` are all applied before the list is read; the
persona deltas are the ones the materializer emits beside the host profile.
"""

from __future__ import annotations

import pytest

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.profile_cmd import _parsed_persona_deltas, _persona_profile_texts

_WITH_LATTICE = (
    "control-assistant",
    "control-assistant-readonly",
    "control-assistant-readwrite",
    "control-assistant-admin",
)

_WITHOUT_LATTICE = (
    "control-assistant-knowledge",
    "control-assistant-logbook",
    "hello-world",
    "ariel-standalone",
    "channel-finder-standalone",
)


def _panels(preset: str) -> list[str]:
    resolved, _preset_dir = resolve_build_profile(None, preset, (), ())
    return resolved.web_panels


@pytest.mark.parametrize("preset", _WITH_LATTICE)
def test_the_control_room_presets_carry_the_lattice_tab(preset: str) -> None:
    assert "lattice" in _panels(preset)


@pytest.mark.parametrize("preset", _WITHOUT_LATTICE)
def test_the_other_presets_leave_the_lattice_tab_out(preset: str) -> None:
    assert "lattice" not in _panels(preset)


def test_the_emitted_knowledge_and_logbook_deltas_exclude_the_lattice_tab() -> None:
    resolved, _preset_dir = resolve_build_profile(None, "control-assistant", (), ())
    texts = _persona_profile_texts(resolved, "Exemplar", "", "control-assistant")
    deltas = _parsed_persona_deltas(texts)

    for persona in ("knowledge", "logbook"):
        assert "lattice" in deltas[persona]["exclude"]["web_panels"]
    for persona in ("admin", "readonly", "readwrite"):
        excluded = (deltas[persona].get("exclude") or {}).get("web_panels") or []
        assert "lattice" not in excluded
