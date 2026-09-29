"""Byte identity of the demo's SR deck across the preset trees.

The simulation lattice under ``control_assistant/data/simulation/`` and the
facility deck ``data/facility/decks/SR.json`` in each preset tree are the same
pyAT JSON file; every generator mode reads the committed deck, so a change to
one must land in all of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

APPS_DIR = Path(__file__).resolve().parents[2] / "src/osprey/templates/apps"

#: The pyAT JSON lattice the deck copies are pinned to.
SOURCE_LATTICE = APPS_DIR / "control_assistant/data/simulation/lattice.json"

#: The preset trees that ship the SR deck.
PRESET_TREES = ("control_assistant", "ariel_standalone", "channel_finder_standalone")


@pytest.mark.parametrize("tree", PRESET_TREES)
def test_sr_deck_is_the_simulation_lattice_byte_for_byte(tree: str) -> None:
    deck = APPS_DIR / tree / "data/facility/decks/SR.json"
    assert deck.is_file(), f"{deck} is missing"
    assert deck.read_bytes() == SOURCE_LATTICE.read_bytes()
