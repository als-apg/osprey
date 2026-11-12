"""Registry ↔ every browser-side copy of the built-in panel roster.

``osprey.profiles.web_panels`` owns the built-in panels and their display
labels. Four places in the browser depend on that roster, and each can drift
from it on its own:

- ``panel-catalog.js`` declares one descriptor per companion panel — the id,
  the config endpoint, and whether the panel is health-polled. A panel added
  to the registry and not to the catalog has no descriptor; one removed from
  the registry but left here is a sidebar row for a panel that no longer
  exists. The catalog states no label of its own, and names no sidecar panel:
  those it builds from the descriptors the page stamps on
  ``<html data-sidecar-panels>``.
- The page stamps the labels on ``<html data-panel-labels>``, which is where
  the catalog reads them. ``/api/panels`` answers for the ENABLED panels
  only, so it cannot seed the full roster before the catalog's first render.
- ``PANEL_ICONS`` in ``panel-catalog.js`` maps each companion panel id to the
  name of an ``icons.js`` glyph for its sidebar row. A companion panel it does
  not name falls to the generic icon, and a name ``icons.js`` does not draw
  makes ``icon()`` throw. The map also names panels the registry does not
  know (the Agent row, and config panels such as ``events`` and ``bluesky``),
  so only the registry-to-map direction is checked. A sidecar panel's glyph
  is its registry entry's ``icon``, held to the same table of names.
- The vitest fixture that stamps the roster for the suites that assert a
  label is pinned to the same dict and the same descriptors, so a renamed
  label cannot leave the fixture behind.

The catalog and icon checks read the shipped sources, so either direction of drift
fails immediately.
"""

import json
import re
from pathlib import Path
from unittest.mock import patch

from osprey.interfaces.design_system.icons import icon_names
from osprey.profiles.web_panels import (
    BUILTIN_PANEL_LABELS,
    SIDECAR_PANELS,
    sidecar_panel_descriptors,
)

_ROOT = Path(__file__).parents[3]
_INTERFACES = _ROOT / "src" / "osprey" / "interfaces"
_CATALOG = _INTERFACES / "web_terminal" / "static" / "js" / "panel-catalog.js"
_ICONS = _INTERFACES / "design_system" / "static" / "js" / "icons.js"
_FIXTURE = Path(__file__).parent / "panel-labels-fixture.mjs"

#: The Agent row's id in ``PANEL_ICONS`` (``panel-catalog.TERMINAL_RAIL_ID``). It
#: is dock-layout state rather than a service panel, so the registry does not
#: name it.
_TERMINAL_ROW_ID = "terminal"


#: The built-in panels the catalog lists itself: every one that is not a sidecar.
_COMPANION_IDS = set(BUILTIN_PANEL_LABELS) - set(SIDECAR_PANELS)


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


def test_the_catalog_declares_exactly_the_companion_panels() -> None:
    ids = _catalog_ids()
    assert ids == _COMPANION_IDS, (
        "panel-catalog.js and the registry disagree about the companion "
        f"roster: only in the catalog {sorted(ids - _COMPANION_IDS)}, "
        f"only in the registry {sorted(_COMPANION_IDS - ids)}"
    )


def test_the_catalog_names_no_sidecar_panel() -> None:
    """Sidecar rows come from the page's stamp, so the bundle spells none."""
    source = _CATALOG.read_text()
    named = {panel_id for panel_id in SIDECAR_PANELS if f"'{panel_id}'" in source}
    assert named == set(), f"panel-catalog.js names sidecar panels: {sorted(named)}"
    assert not set(SIDECAR_PANELS) & set(_panel_icons()), "PANEL_ICONS names a sidecar panel"


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


def test_every_companion_panel_has_an_icon() -> None:
    missing = _COMPANION_IDS - set(_panel_icons())
    assert not missing, (
        "built-in panels falling back to the generic sidebar icon: "
        f"{sorted(missing)} — add them to PANEL_ICONS in panel-catalog.js"
    )


def test_every_sidecar_panel_names_its_icon() -> None:
    """A sidecar with no ``icon`` wears the generic glyph: the same silent regression."""
    missing = sorted(panel_id for panel_id, panel in SIDECAR_PANELS.items() if not panel.icon)
    assert not missing, f"sidecar panels with no registry icon: {missing}"


def test_every_panel_icon_is_a_design_system_icon() -> None:
    """``icon()`` throws on a name ``SHAPES`` lacks, so a typo blanks the sidebar row."""
    known = _icon_names()
    named = {**_panel_icons(), **{pid: panel.icon for pid, panel in SIDECAR_PANELS.items()}}
    unknown = {panel: name for panel, name in named.items() if name not in known}
    assert not unknown, f"panel icons name glyphs icons.js does not draw: {unknown}"


def test_the_agent_row_has_an_icon() -> None:
    """The session tile's row is not a service panel, so the registry check cannot see it."""
    assert _TERMINAL_ROW_ID in _panel_icons()


def _fixture_literal(name: str) -> str:
    """The source of the object or array the fixture assigns to *name*."""
    source = _FIXTURE.read_text()
    start = source.index(f"export const {name} = ")
    return source[start : source.index(";\n", start)]


def test_the_vitest_label_fixture_matches_the_registry() -> None:
    pairs = dict(re.findall(r"'?([\w-]+)'?: '([^']*)'", _fixture_literal("BUILTIN_PANEL_LABELS")))
    assert pairs == BUILTIN_PANEL_LABELS


def test_the_vitest_sidecar_fixture_matches_the_registry() -> None:
    """Each fixture descriptor, field for field, is the one the page stamps."""
    literal = _fixture_literal("SIDECAR_PANELS")
    entries = [
        dict(re.findall(r"(\w+): '([^']*)'", block))
        for block in re.findall(r"Object\.freeze\(\{(.*?)\}\)", literal, flags=re.DOTALL)
    ]
    assert entries == sidecar_panel_descriptors()


def test_the_python_icon_reader_agrees_with_the_shipped_table() -> None:
    """The build judges ``web.panels.<id>.icon`` against the names the browser draws."""
    assert icon_names() == _icon_names()


def _html_stamp(body: str, attribute: str) -> object:
    start = body.index("<html ")
    html_tag = body[start : body.index(">", start)]
    match = re.search(rf"{attribute}='([^']*)'", html_tag)
    assert match, f"{attribute} is not stamped on <html>"
    return json.loads(match.group(1))


def test_an_unconfigured_page_stamps_no_panel_icon(isolated_app) -> None:
    with isolated_app() as client:
        body = client.get("/").text
    assert _html_stamp(body, "data-panel-icons") == {}


def test_the_page_stamps_the_deployments_own_labels_and_icons(isolated_app) -> None:
    """``web.panels.<id>.label`` and ``.icon`` reach the page before any script runs."""
    from osprey.interfaces.web_terminal import app as web_terminal_app

    sidecar_id = next(iter(SIDECAR_PANELS))
    panels_config = {
        "ariel": {"label": "Logbook", "icon": "book-open"},
        sidecar_id: {"label": "Notes", "icon": "monitor"},
        "grafana": {"label": "GRAFANA", "url": "http://x:3000", "icon": "chart-column"},
    }
    runtime = web_terminal_app._PanelRuntimeConfig(
        allow_runtime_panels=False,
        runtime_panel_allowlist=None,
        visible_panels=["artifacts"],
        panels_config=panels_config,
    )
    with (
        patch.object(web_terminal_app, "_load_panel_runtime_config", return_value=runtime),
        isolated_app() as client,
    ):
        body = client.get("/").text
        api_labels = client.get("/api/panels").json()["labels"]

    assert _html_stamp(body, "data-panel-labels") == {
        **BUILTIN_PANEL_LABELS,
        "ariel": "Logbook",
        sidecar_id: "Notes",
    }
    assert _html_stamp(body, "data-panel-icons") == {
        "ariel": "book-open",
        sidecar_id: "monitor",
        "grafana": "chart-column",
    }
    assert _html_stamp(body, "data-sidecar-panels") == sidecar_panel_descriptors(panels_config)
    assert api_labels == {"artifacts": BUILTIN_PANEL_LABELS["artifacts"]}
