"""Registry ↔ every browser-side copy of the built-in panel roster.

``osprey.profiles.web_panels`` owns the built-in panels and their display
labels. Four places in the browser depend on that roster, and each can drift
from it on its own:

- ``panel-catalog.js`` declares one descriptor per built-in panel — the id,
  the config endpoint, and whether the panel is health-polled. A panel added
  to the registry and not to the catalog has no descriptor; one removed from
  the registry but left here is a rail entry for a panel that no longer
  exists. The catalog states no label of its own.
- The page stamps the labels on ``<html data-panel-labels>``, which is where
  the catalog reads them. ``/api/panels`` answers for the ENABLED panels
  only, so it cannot seed the full roster before the catalog's first render.
- ``terminal.css`` maps each rail entry's ``data-icon`` (the panel id) to a
  glyph. A built-in panel with no rule falls to the generic ``◈``: a silent
  regression, since nothing errors and the label still reads.
- The vitest fixture that stamps the roster for the suites that assert a
  label is pinned to the same dict, so a renamed label cannot leave the
  fixture behind.

The JS and CSS checks read the shipped sources, so either direction of drift
fails immediately.
"""

import json
import re
from pathlib import Path

from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS

_ROOT = Path(__file__).parents[3]
_STATIC = _ROOT / "src" / "osprey" / "interfaces" / "web_terminal" / "static"
_CATALOG = _STATIC / "js" / "panel-catalog.js"
_CSS = _STATIC / "css" / "terminal.css"
_FIXTURE = Path(__file__).parent / "panel-labels-fixture.mjs"

#: The session tile's rail entry. It wears a ``data-icon`` like a panel but is
#: dock-layout state rather than a service panel, so it is not in the panel
#: registry and its rule is not stale (``panel-catalog.TERMINAL_RAIL_ID``).
_TERMINAL_RAIL_ID = "terminal"


def _catalog_ids() -> set[str]:
    """Every panel id the shipped ``PANELS`` array declares."""
    return set(re.findall(r"\bid: '([\w-]+)'", _CATALOG.read_text()))


def _glyph_ids() -> set[str]:
    """Every id ``terminal.css`` writes a ``.panel-rail-icon`` rule for."""
    return set(re.findall(r'\.panel-rail-icon\[data-icon="([\w-]+)"\]', _CSS.read_text()))


def test_the_catalog_declares_exactly_the_builtin_panels() -> None:
    ids = _catalog_ids()
    assert ids == set(BUILTIN_PANEL_LABELS), (
        "panel-catalog.js and BUILTIN_PANEL_LABELS disagree about the built-in "
        f"roster: only in the catalog {sorted(ids - set(BUILTIN_PANEL_LABELS))}, "
        f"only in the registry {sorted(set(BUILTIN_PANEL_LABELS) - ids)}"
    )


def test_the_catalog_states_no_label_of_its_own() -> None:
    """The labels come from the page's stamp, never from a literal here."""
    literals = set(re.findall(r"\blabel: '([^']*)'", _CATALOG.read_text()))
    assert literals == set(), f"panel-catalog.js restates panel labels: {sorted(literals)}"


def test_the_builtin_panel_roster_is_stamped_on_html(bar_items_app) -> None:
    """The page carries the whole registry roster on ``<html>`` before any script runs.

    Matched on the opening tag rather than the whole document, which also
    mentions the attribute in prose.
    """
    with bar_items_app() as client:
        resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    start = body.index("<html ")
    html_tag = body[start : body.index(">", start)]

    match = re.search(r"data-panel-labels='([^']*)'", html_tag)
    assert match, "the built-in panel roster is not stamped on <html>"
    assert json.loads(match.group(1)) == BUILTIN_PANEL_LABELS


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


def test_the_vitest_label_fixture_matches_the_registry() -> None:
    pairs = dict(re.findall(r"'([\w-]+)': '([A-Z-]+)'", _FIXTURE.read_text()))
    assert pairs == BUILTIN_PANEL_LABELS
