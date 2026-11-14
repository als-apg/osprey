"""Minimal synthetic ``data/facility`` trees for the facility build's tests.

The trees ship with the application SDK's test support,
:mod:`osprey.apps.sdk.testing`; this module names them where OSPREY's own tests
import them.
"""

from osprey.apps.sdk.testing import (
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
