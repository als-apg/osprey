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
- ``PANEL_ICONS`` in ``panel-catalog.js`` maps each panel id to the name of an
  ``icons.js`` glyph for its sidebar row. A built-in panel it does not name
  falls to the generic icon, and a name ``icons.js`` does not draw makes
  ``icon()`` throw. The map also names panels the registry does not know (the
  Agent row, and config panels such as ``events`` and ``bluesky``), so only
  the registry-to-map direction is checked.
- The vitest fixture that stamps the roster for the suites that assert a
  label is pinned to the same dict, so a renamed label cannot leave the
  fixture behind.

The catalog and icon checks read the shipped sources, so either direction of drift
fails immediately.
"""

import json
import re
from pathlib import Path

from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS

_ROOT = Path(__file__).parents[3]
_INTERFACES = _ROOT / "src" / "osprey" / "interfaces"
_CATALOG = _INTERFACES / "web_terminal" / "static" / "js" / "panel-catalog.js"
_ICONS = _INTERFACES / "design_system" / "static" / "js" / "icons.js"
_FIXTURE = Path(__file__).parent / "panel-labels-fixture.mjs"

#: The Agent row's id in ``PANEL_ICONS`` (``panel-catalog.TERMINAL_RAIL_ID``). It
#: is dock-layout state rather than a service panel, so the registry does not
#: name it.
_TERMINAL_ROW_ID = "terminal"


def _catalog_ids() -> set[str]:
    """Every panel id the shipped ``PANELS`` array declares."""
    return set(re.findall(r"\bid: '([\w-]+)'", _CATALOG.read_text()))


def _panel_icons() -> dict[str, str]:
    """The ``PANEL_ICONS`` map of ``panel-catalog.js``, panel id to icon name.

    Read from the literal between ``PANEL_ICONS = Object.freeze({`` and its
    closing ``})``; keys are bare identifiers or quoted ids.
    """
    source = _CATALOG.read_text()
    opener = "PANEL_ICONS = Object.freeze({"
    start = source.index(opener) + len(opener)
    body = source[start : source.index("})", start)]
    icons = dict(re.findall(r"""^\s*['"]?([\w-]+)['"]?\s*:\s*['"]([\w-]+)['"]""", body, re.M))
    assert icons, "no PANEL_ICONS entries found — the pattern no longer matches panel-catalog.js"
    return icons


def _icon_names() -> set[str]:
    """Every name the design system's ``SHAPES`` table draws."""
    source = _ICONS.read_text()
    opener = "const SHAPES = Object.freeze({"
    start = source.index(opener) + len(opener)
    body = source[start : source.index("\n});", start)]
    names = set(re.findall(r"""^  ['"]?([\w-]+)['"]?: \[""", body, re.M))
    assert names, "no icon names found — the pattern no longer matches icons.js"
    return names


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


def test_the_builtin_panel_roster_is_stamped_on_html(isolated_app) -> None:
    """The page carries the whole registry roster on ``<html>`` before any script runs.

    Matched on the opening tag rather than the whole document, which also
    mentions the attribute in prose.
    """
    with isolated_app() as client:
        resp = client.get("/")
    assert resp.status_code == 200
    body = resp.text
    start = body.index("<html ")
    html_tag = body[start : body.index(">", start)]

    match = re.search(r"data-panel-labels='([^']*)'", html_tag)
    assert match, "the built-in panel roster is not stamped on <html>"
    assert json.loads(match.group(1)) == BUILTIN_PANEL_LABELS


def test_every_builtin_panel_has_an_icon() -> None:
    missing = set(BUILTIN_PANEL_LABELS) - set(_panel_icons())
    assert not missing, (
        "built-in panels falling back to the generic sidebar icon: "
        f"{sorted(missing)} — add them to PANEL_ICONS in panel-catalog.js"
    )


def test_every_panel_icon_is_a_design_system_icon() -> None:
    """``icon()`` throws on a name ``SHAPES`` lacks, so a typo blanks the sidebar row."""
    known = _icon_names()
    unknown = {panel: name for panel, name in _panel_icons().items() if name not in known}
    assert not unknown, f"PANEL_ICONS names icons icons.js does not draw: {unknown}"


def test_the_agent_row_has_an_icon() -> None:
    """The session tile's row is not a service panel, so the registry check cannot see it."""
    assert _TERMINAL_ROW_ID in _panel_icons()


def test_the_vitest_label_fixture_matches_the_registry() -> None:
    pairs = dict(re.findall(r"'?([\w-]+)'?: '([^']*)'", _FIXTURE.read_text()))
    assert pairs == BUILTIN_PANEL_LABELS
