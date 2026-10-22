"""Which shipped presets carry the LATTICE tab.

The lattice dashboard is part of the control-assistant preset: the host profile
lists ``lattice`` in ``web_panels:``; the readonly, readwrite and admin tiers it
emits keep it, and the knowledge and logbook apps, which state their own
panel list, leave it out. hello-world never lists it.

Each preset is resolved the way ``osprey init`` resolves it, so ``extends:``,
list unions and ``exclude:`` are all applied before the list is read; the
tier and app files are the ones the materializer emits beside the host profile.
"""

from __future__ import annotations

import pytest

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.profile_cmd import _layer_profile_texts, _parsed_layer_deltas

_WITH_LATTICE = ("control-assistant",)

_WITHOUT_LATTICE = ("hello-world",)


def _panels(preset: str) -> list[str]:
    resolved, _preset_dir = resolve_build_profile(None, preset, (), ())
    return resolved.web_panels


@pytest.mark.parametrize("preset", _WITH_LATTICE)
def test_the_control_room_presets_carry_the_lattice_tab(preset: str) -> None:
    assert "lattice" in _panels(preset)


@pytest.mark.parametrize("preset", _WITHOUT_LATTICE)
def test_the_other_presets_leave_the_lattice_tab_out(preset: str) -> None:
    assert "lattice" not in _panels(preset)


def test_the_emitted_knowledge_and_logbook_apps_leave_the_lattice_tab_out() -> None:
    resolved, _preset_dir = resolve_build_profile(None, "control-assistant", (), ())
    texts = _layer_profile_texts(resolved, "Exemplar", "")
    deltas = _parsed_layer_deltas(texts)

    for app in ("knowledge", "logbook"):
        assert "lattice" not in (deltas[app].get("web_panels") or [])
    for render in ("admin", "readonly", "readwrite"):
        excluded = (deltas[render].get("exclude") or {}).get("web_panels") or []
        assert "lattice" not in excluded
