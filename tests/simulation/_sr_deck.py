"""The example facility's committed SR deck, loaded the way the engine loads it."""

from __future__ import annotations

from pathlib import Path

import at

#: The pyAT JSON deck the example facility's ``SR`` model serves.
SR_DECK = (
    Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example/decks/SR.json"
)


def load_sr_deck() -> at.Lattice:
    """A fresh, independent copy of the SR deck."""
    return at.load_lattice(SR_DECK)


def load_sr_deck_4d() -> at.Lattice:
    """A fresh copy of the SR deck with radiation and the cavity off."""
    ring = load_sr_deck()
    ring.disable_6d()  # mutates in place; returns None
    return ring
