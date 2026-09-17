"""Registry ↔ catalog parity: the browser's panel roster is the registry's.

``panel-catalog.js`` declares one descriptor per built-in panel — the id, the
config endpoint, and whether the panel is health-polled. The DISPLAY LABEL is
not the browser's to decide: ``osprey.profiles.web_panels`` owns it, the page
stamps it on ``<html>``, and the catalog reads it from there. What can still
drift is the ROSTER — a panel added to the registry and not to the catalog is
a panel with no descriptor, and one removed from the registry but left here is
a rail entry for a panel that no longer exists. This reads the JS source so
either direction fails immediately.

The vitest fixture that stamps the roster for the suites that assert a label is
pinned to the same dict, so a renamed label cannot leave the fixture behind.
"""

import re
from pathlib import Path

from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS

_ROOT = Path(__file__).parents[3]
_CATALOG = (
    _ROOT / "src" / "osprey" / "interfaces" / "web_terminal" / "static" / "js" / "panel-catalog.js"
)
_FIXTURE = Path(__file__).parent / "panel-labels-fixture.mjs"


def _catalog_ids() -> set[str]:
    """Every panel id the shipped ``PANELS`` array declares."""
    return set(re.findall(r"\bid: '([\w-]+)'", _CATALOG.read_text()))


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


def test_the_vitest_label_fixture_matches_the_registry() -> None:
    pairs = dict(re.findall(r"'([\w-]+)': '([A-Z-]+)'", _FIXTURE.read_text()))
    assert pairs == BUILTIN_PANEL_LABELS
