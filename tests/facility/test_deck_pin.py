"""Byte identity of the demo's SR deck and the simulation lattice.

The simulation lattice under ``control_assistant/data/simulation/`` and the
example facility's deck ``decks/SR.json`` are the same pyAT JSON file; every
generator mode reads the committed deck, so a change to one must land in both.
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
}


@pytest.mark.parametrize("tree", sorted(FACILITY_TREES))
def test_sr_deck_is_the_simulation_lattice_byte_for_byte(tree: str) -> None:
    deck = FACILITY_TREES[tree] / "decks/SR.json"
    assert deck.is_file(), f"{deck} is missing"
    assert deck.read_bytes() == SOURCE_LATTICE.read_bytes()
