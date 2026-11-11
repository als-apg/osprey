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
from osprey.apps.sdk.testing._measured import (
    LINE_GROUPS,
    SR_GROUPS,
    calibrated_setting,
    measured_sr_deck,
    measured_tree,
    with_correctors,
    with_rf,
)

__all__ = [
    "BPM",
    "LINE_GROUPS",
    "LINE_TWISS",
    "QUAD",
    "READING",
    "SETTING",
    "SR_GROUPS",
    "Deck",
    "calibrated_setting",
    "deck_tree",
    "line_deck",
    "measured_sr_deck",
    "measured_tree",
    "model",
    "plain_tree",
    "sr_deck",
    "with_correctors",
    "with_rf",
    "write_tree",
]
