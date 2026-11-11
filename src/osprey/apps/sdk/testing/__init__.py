"""Test support for an application's own test suite: a synthetic deployment, built for real.

Importable from an installed ``osprey-framework``; nothing here reaches into
OSPREY's own test tree.
"""

from osprey.apps.sdk.testing._facility import (
    BPM,
    LINE_TWISS,
    QUAD,
    READING,
    SETTING,
    Deck,
    deck_tree,
    line_deck,
    model,
    plain_tree,
    sr_deck,
    write_tree,
)

__all__ = [
    "BPM",
    "LINE_TWISS",
    "QUAD",
    "READING",
    "SETTING",
    "Deck",
    "deck_tree",
    "line_deck",
    "model",
    "plain_tree",
    "sr_deck",
    "write_tree",
]
