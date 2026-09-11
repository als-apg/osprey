"""Registry ↔ stylesheet parity: every built-in panel has a rail glyph.

``panel-rail.js`` sets ``data-icon`` to the panel's own id for every entry it
renders, and ``terminal.css`` maps that id to a glyph. A built-in panel with no
rule falls to the generic ``◈``, so the rail shows two or three panels wearing
the same placeholder — a silent regression, since nothing errors and the label
still reads. This reads the CSS source so it fails the moment a panel joins the
registry without a glyph, and the moment a rule outlives its panel.
"""

import re
from pathlib import Path

from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS

_CSS = (
    Path(__file__).parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "web_terminal"
    / "static"
    / "css"
    / "terminal.css"
)

#: The session tile's rail entry. It wears a ``data-icon`` like a panel but is
#: dock-layout state rather than a service panel, so it is not in the panel
#: registry and its rule is not stale (``panel-catalog.TERMINAL_RAIL_ID``).
_TERMINAL_RAIL_ID = "terminal"


def _glyph_ids() -> set[str]:
    """Every id ``terminal.css`` writes a ``.panel-rail-icon`` rule for."""
    return set(re.findall(r'\.panel-rail-icon\[data-icon="([\w-]+)"\]', _CSS.read_text()))


def test_every_builtin_panel_has_a_rail_glyph() -> None:
    missing = set(BUILTIN_PANEL_LABELS) - _glyph_ids()
    assert not missing, (
        "built-in panels falling back to the generic rail glyph: "
        f"{sorted(missing)} — add a .panel-rail-icon[data-icon=...] rule"
    )


def test_no_stale_rail_glyph_rules() -> None:
    """A rule naming no built-in panel is dead weight — the rename trap."""
    stale = _glyph_ids() - set(BUILTIN_PANEL_LABELS) - {_TERMINAL_RAIL_ID}
    assert not stale, f"rail glyph rules naming no built-in panel: {sorted(stale)}"
