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

#: The facility trees that ship the SR deck.
FACILITY_TREES = {
    "example": APPS_DIR.parent / "facilities/example",
    "ariel_standalone": APPS_DIR / "ariel_standalone/data/facility",
    "channel_finder_standalone": APPS_DIR / "channel_finder_standalone/data/facility",
}


@pytest.mark.parametrize("tree", sorted(FACILITY_TREES))
def test_sr_deck_is_the_simulation_lattice_byte_for_byte(tree: str) -> None:
    deck = FACILITY_TREES[tree] / "decks/SR.json"
    assert deck.is_file(), f"{deck} is missing"
    assert deck.read_bytes() == SOURCE_LATTICE.read_bytes()
